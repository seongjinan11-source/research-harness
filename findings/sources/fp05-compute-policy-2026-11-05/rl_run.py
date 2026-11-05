"""FP05 one-worker training/evaluation. Never uses final labels for routing."""
import sklearn
import argparse
import gc
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from rl_util import ROOT, WORK, cfg, setup, dump, sha, digest, metrics, lock, design

TRACES = list(itertools.product([0, 1], repeat=3))


def data():
    import rl_data
    return rl_data


def model_class():
    from rl_model import Model
    return Model


def set_arm(model, arm):
    for p in model.policy.parameters():
        p.requires_grad_(arm in ['direct', 'group_pg'])
    return model.eval()


def parameters(model):
    return list(model.adapter_parameters()), [p for p in model.policy.parameters() if p.requires_grad]


def snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad or n.startswith('policy.')}


def restore(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            assert name in params and params[name].shape == value.shape, name
            params[name].copy_(value.to(params[name]))


def schedule(seed):
    rows = data().rows('train'); full = [i for i, r in enumerate(rows)
                                          if r['setting'] == 'full' and r['regime'] == 'natural']
    assert full
    rng = np.random.default_rng(cfg()['train']['schedule_seed'] + seed)
    out = []
    for update in range(cfg()['train']['updates']):
        index = int(rng.choice(full)) if update % 8 == 0 else int(rng.integers(len(rows)))
        out.append(dict(index=index, forward_seed=int(rng.integers(1, 2**30)),
                        action_seed=int(rng.integers(1, 2**30))))
    return out


def make_optimizer(model):
    t = cfg()['train']; theta, phi = parameters(model)
    groups = [dict(params=theta, lr=t['adapter_lr'], base_lr=t['adapter_lr'])]
    if phi:
        groups.append(dict(params=phi, lr=t['policy_lr'], base_lr=t['policy_lr']))
    return torch.optim.AdamW(groups, weight_decay=t['weight_decay'])


def fit_group(model, arm, inp, y, action_seed, opt, scaler, update):
    """All branches use pre-update parameters; θ conditional CE, φ separate actor."""
    t = cfg()['train']; total = t['updates']; cost_weight = cfg()['cost_weight']
    lr_factor = min((update + 1) / t['warmup'], 1.) * (.1 + .9 * .5 * (1 + math.cos(math.pi * update / total)))
    for group in opt.param_groups:
        group['lr'] = group['base_lr'] * lr_factor
    theta, phi = parameters(model); target = torch.as_tensor(y, device='cuda', dtype=torch.long)
    retries = []
    for attempt in range(16):
        opt.zero_grad(set_to_none=True)
        rng = np.random.default_rng(action_seed)
        records = []; leaf_p = []; actor_values = []
        count = 1 if arm == 'fixed' else t['traces']
        started = time.perf_counter()
        for index in range(count):
            forced = (0, 0, 0) if arm == 'fixed' else TRACES[index] if arm == 'direct' else None
            if arm == 'uniform':
                forced = tuple(int(v) for v in rng.integers(0, 2, 3))
            with torch.autocast('cuda', dtype=torch.float16):
                result = model(inp, trace=forced, mode='sample' if arm == 'group_pg' else 'greedy', action_rng=rng)
                loss = F.cross_entropy(result['logits'].float(), target)
            probability = result['logp'].float().exp()
            weight = probability.detach() if arm == 'direct' else loss.new_tensor(1. / count)
            scaler.scale(weight * loss).backward()
            if arm == 'direct':
                leaf_p.append(probability)
                actor_values.append(probability * (loss.detach() + cost_weight * result['cost']))
            elif arm == 'group_pg':
                actor_values.append(result['logp'].float())
            records.append(dict(ce=float(loss.detach()), trace=list(result['trace']), cost=int(result['cost']),
                                logp=float(result['logp'].detach()),
                                probability=float(probability.detach())))
        actor = None; reward_sd = None; zero_advantage = None
        if arm == 'direct':
            assert abs(float(torch.stack(leaf_p).sum().detach()) - 1) < 2e-4, 'Trace probabilities do not sum1'
            actor = torch.stack(actor_values).sum()  # NO /8; p already normalized.
        elif arm == 'group_pg':
            rewards = torch.tensor([-r['ce'] - cost_weight * r['cost'] for r in records], device='cuda')
            reward_sd = float(rewards.std(unbiased=False))
            advantages = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)).detach()
            zero_advantage = bool(torch.count_nonzero(advantages) == 0)
            actor = -(advantages * torch.stack(actor_values)).mean()
        if actor is not None:
            scaler.scale(actor).backward()
        scaler.unscale_(opt)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in theta + phi)
        finite = finite and all(np.isfinite(r['ce']) for r in records)
        if finite:
            theta_norm = float(torch.nn.utils.clip_grad_norm_(theta, 1., error_if_nonfinite=True))
            phi_norm = float(torch.nn.utils.clip_grad_norm_(phi, 1., error_if_nonfinite=True)) if phi else 0.
            scaler.step(opt); scaler.update()
            p = np.asarray([r['probability'] for r in records])
            return dict(records=records, actor=float(actor.detach()) if actor is not None else None,
                        mean_ce=float(np.mean([r['ce'] for r in records])),
                        expected_ce=float(sum(r['probability'] * r['ce'] for r in records)) if arm == 'direct' else None,
                        theta_grad_norm=theta_norm, phi_grad_norm=phi_norm,
                        reward_sd=reward_sd, zero_advantage=zero_advantage,
                        unique_traces=len({tuple(r['trace']) for r in records}),
                        actual_forwards=count, transformer_backwards=count,
                        logical_trace_slots=t['traces'], extra_adapter_pairs=sum(r['cost'] for r in records),
                        seconds=time.perf_counter() - started, retries=retries,
                        entropy_estimate=float(-sum(p * np.log(np.maximum(p, 1e-12)))) if arm == 'direct' else None)
        retries.append(dict(attempt=attempt, scaler=float(scaler.get_scale()),
                            ce=[r['ce'] if np.isfinite(r['ce']) else None for r in records],
                            ce_finite=[bool(np.isfinite(r['ce'])) for r in records], actual_forwards=count))
        scaler.update(new_scale=max(float(scaler.get_scale()) / 2, 1e-6))
    raise RuntimeError('FP05 finite-gradient retries exhausted; preserve partial artifacts')


def train(seed, arm):
    setup(); source = lock(); folder = ROOT / 'results' / f'{arm}_s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'training.json').exists():
        raise FileExistsError('No silent retraining/restart')
    if (folder / 'progress.jsonl').exists():
        raise FileExistsError('Incomplete run requires explicit repair record')
    model = set_arm(model_class()(seed), arm)
    initial = digest(model); frozen = model.frozen_digest()
    opt = make_optimizer(model); scaler = torch.amp.GradScaler('cuda', init_scale=64.)
    sched = schedule(seed); dump(folder / 'schedule.json', sched)
    schedule_sha = sha(folder / 'schedule.json'); rows = data().rows('train')
    torch.cuda.reset_peak_memory_stats(); begun = time.perf_counter(); exposures = []; counts = {}
    with (folder / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for update, item in enumerate(sched):
            row = rows[item['index']]; inp, y = data().read(row)
            inp = dict(inp, forward_seed=item['forward_seed'])
            result = fit_group(model, arm, inp, y, item['action_seed'], opt, scaler, update)
            extra = data().extra(row)
            actual_missing = ~np.asarray(inp['observed'])[np.asarray(inp['query_ids'])].any(-1)
            key = row['setting'] + '/' + row['regime']
            rec = counts.setdefault(key, dict(groups=0, observed_queries=0, missing_queries=0))
            rec['groups'] += 1; rec['observed_queries'] += int((~actual_missing).sum()); rec['missing_queries'] += int(actual_missing.sum())
            entry = dict(update=update + 1, episode=row['id'], base=row['base'],
                         input_sha256=row['sha256'], forward_seed=item['forward_seed'], **result)
            stream.write(json.dumps(entry, allow_nan=False) + '\n'); stream.flush()
            exposures.append(row['id'])
            if (update + 1) % 8 == 0:
                print('TRAIN', arm, seed, update + 1, round(result['mean_ce'], 5),
                      'cost', round(np.mean([r['cost'] for r in result['records']]), 2), flush=True)
    assert model.frozen_digest() == frozen
    state = snapshot(model)
    torch.save(dict(parameters=state, lock_sha256=sha(ROOT / 'results/lock.json'), arm=arm, seed=seed,
                    schedule_sha256=schedule_sha), folder / 'checkpoint.pt')
    dump(folder / 'training.json', dict(arm=arm, seed=seed, initial_digest=initial,
         final_digest=digest(model), frozen_digest=frozen, frozen_unchanged=True,
         checkpoint_sha256=sha(folder / 'checkpoint.pt'), schedule_sha256=schedule_sha,
         updates=len(sched), task_groups=len(sched), logical_trace_slots=len(sched) * cfg()['train']['traces'],
         unique_episodes=len(set(exposures)), unique_bases=len({r['base'] for r in rows if r['id'] in set(exposures)}),
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         adapter_parameters=sum(p.numel() for p in model.adapter_parameters()),
         policy_parameters=sum(p.numel() for p in model.policy.parameters()),
         query_exposures=counts, target_updates=0,
         seconds=time.perf_counter() - begun, peak_mib=torch.cuda.max_memory_allocated() / 2**20))
    lock(); del model, opt, scaler; gc.collect(); torch.cuda.empty_cache()


def load(seed, arm):
    folder = ROOT / 'results' / f'{arm}_s{seed}'
    info = json.loads((folder / 'training.json').read_text(encoding='utf8'))
    assert sha(folder / 'checkpoint.pt') == info['checkpoint_sha256']
    model = set_arm(model_class()(seed), arm)
    assert digest(model) == info['initial_digest']
    state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['lock_sha256'] == sha(ROOT / 'results/lock.json')
    restore(model, state['parameters'])
    assert digest(model) == info['final_digest']; assert model.frozen_digest() == info['frozen_digest']
    return model


def predict(model, inp, arm, forced=None):
    traces = TRACES if arm == 'uniform' and forced is None else [forced]
    predictions = []; costs = []; selected = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for trace in traces:
            if arm in ['fixed', 'frozen_pcfi', 'frozen_nan'] and trace is None:
                trace = (0, 0, 0)
            out = model(inp, trace=trace, mode='greedy')
            predictions.append(out['logits'].float().softmax(-1).cpu().numpy())
            costs.append(int(out['cost'])); selected.append(list(out['trace']))
    p = np.mean(predictions, axis=0)
    assert np.isfinite(p).all()
    return p, dict(forwards=len(traces), extra_adapter_pairs=sum(costs),
                   traces=selected, seconds=time.perf_counter() - started)


def calibrate(model, arm, seed, folder):
    path = folder / 'global_trace.json'
    if path.exists():
        return tuple(json.loads(path.read_text(encoding='utf8'))['trace'])
    totals = np.zeros(8); records = []; begun = time.perf_counter()
    for row in data().rows('dev'):
        inp, y = data().read(row)
        for index, trace in enumerate(TRACES):
            p, timing = predict(model, inp, arm, forced=trace)
            met = metrics(p, y); totals[index] += met['nll']
            prediction = folder / 'calibration' / f"{row['id']}_trace{index}.npz"
            prediction.parent.mkdir(parents=True, exist_ok=True); assert not prediction.exists()
            np.savez_compressed(prediction, p=p, y=y, query=inp['query_ids'],
                                context=inp['context_ids'], yc=inp['context_labels'])
            records.append(dict(episode=row['id'], input_sha256=row['sha256'], trace=trace,
                                prediction=prediction.relative_to(ROOT).as_posix(), prediction_sha256=sha(prediction),
                                **met, **timing))
    means = totals / len(data().rows('dev'))
    selected = min(range(8), key=lambda i: (means[i], sum(TRACES[i]), TRACES[i]))
    dump(folder / 'global_trace.json', dict(trace=TRACES[selected], selection='dev8 graphs x7settings, mean CE; tie lower cost then lexical',
             scores=means.tolist(), records=records, seconds=time.perf_counter() - begun,
             actual_forwards=len(records), seed=seed, arm=arm))
    return TRACES[selected]


def evaluate(seed, arm, split):
    setup(); lock(); folder = ROOT / 'results' / f'{arm}_s{seed}'
    output = folder / (split + '_evaluation.json')
    if output.exists():
        raise FileExistsError('No silent evaluation overwrite')
    model = load(seed, arm); selected = calibrate(model, arm, seed, folder) if arm in ['uniform', 'group_pg'] else None
    records = []; begun = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for index, row in enumerate(data().rows(split)):
        inp, y = data().read(row); extra = data().extra(row)
        variants = [('main', None)]
        if arm in ['direct', 'group_pg']:
            variants.append(('replay000', (0, 0, 0)))
        if selected is not None:
            variants.append(('global', selected))
        for variant, forced in variants:
            p, cost = predict(model, inp, arm, forced=forced)
            path = folder / 'predictions' / split / f"{row['id']}_{variant}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists(), path
            own = np.asarray(extra['standard_own_missing'], bool)
            np.savez_compressed(path, p=p, y=y, own_missing=own,
                                query=np.asarray(inp['query_ids']), context=np.asarray(inp['context_ids']),
                                yc=np.asarray(inp['context_labels']))
            records.append(dict(**row, arm=arm, seed=seed, variant=variant,
                 prediction=path.relative_to(ROOT).as_posix(), prediction_sha256=sha(path),
                 checkpoint_sha256=sha(folder / 'checkpoint.pt'),
                 metrics=metrics(p, y), missing_metrics=metrics(p[own], y[own]) if own.any() else None,
                 **cost))
        if (index + 1) % 96 == 0:
            print('EVAL', arm, seed, split, index + 1, len(data().rows(split)), flush=True)
    assert model.frozen_digest() == json.loads((folder / 'training.json').read_text())['frozen_digest']
    dump(output, dict(records=records, seconds=time.perf_counter() - begun,
                     peak_mib=torch.cuda.max_memory_allocated() / 2**20))
    del model; gc.collect(); torch.cuda.empty_cache()


def references(split):
    setup(); lock(); output = ROOT / 'results' / f'{split}_references.json'
    assert not output.exists()
    model = model_class()(11).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    records = []
    for row in data().rows(split):
        inp, y = data().read(row); extra = data().extra(row)
        for arm in ['frozen_pcfi', 'frozen_nan', 'prior', 'lp']:
            if arm.startswith('frozen'):
                visible = np.asarray(inp['values'])
                current = inp
                if arm == 'frozen_nan':
                    # Data supplies normalized visible x before PCFI, never hidden truth.
                    assert 'visible_values' in extra
                    current = dict(inp, values=np.where(inp['observed'], extra['visible_values'], np.nan).astype(np.float32))
                p, cost = predict(model, current, arm)
            else:
                p = data().simple_reference(inp, arm)
                cost = dict(forwards=0, extra_adapter_pairs=0, traces=[], seconds=0.)
            own = np.asarray(extra['standard_own_missing'], bool)
            path = ROOT / 'results/references' / split / f"{row['id']}_{arm}.npz"
            path.parent.mkdir(parents=True, exist_ok=True); assert not path.exists()
            np.savez_compressed(path, p=p, y=y, own_missing=own, query=inp['query_ids'],
                                context=inp['context_ids'], yc=inp['context_labels'])
            records.append(dict(**row, arm=arm, seed=None, variant='main',
                 prediction=path.relative_to(ROOT).as_posix(), prediction_sha256=sha(path),
                 metrics=metrics(p, y), missing_metrics=metrics(p[own], y[own]) if own.any() else None, **cost))
    dump(output, dict(records=records))
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'evaluate', 'references'])
    parser.add_argument('--seed', type=int); parser.add_argument('--arm', choices=cfg()['arms'])
    parser.add_argument('--split', choices=['final', 'real'], default='final')
    args = parser.parse_args()
    if args.action == 'train':
        train(args.seed, args.arm)
    elif args.action == 'evaluate':
        evaluate(args.seed, args.arm, args.split)
    else:
        references(args.split)
