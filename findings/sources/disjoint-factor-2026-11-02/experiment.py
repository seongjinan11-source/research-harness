"""Locked fresh-bank training/evaluation of the disjoint factor PFN pilot."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import itertools
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import ROOT, WORK, C, NATIVE, cfg, dump, sha, read, rows, tensors, public_input, intervention, feature_only, prepare, observation
from model import FactorPFN, belief_propagation


def sources():
    files = {p.name: sha(p) for p in ROOT.glob('*.py')}
    for name, path in [('protocol.json', ROOT / 'protocol.json'), ('native_data.py', WORK / 'direct_marginal_pilot/data.py'),
                       ('public_common.py', WORK / 'joint_imputation_pfn/common.py'), ('public_core.py', WORK / 'regret_prior_pilot/core.py'),
                       ('official_imputation.py', WORK / 'fisf_graphpfn/imputation.py')]:
        files[name] = sha(path)
    return files


def lock():
    state = {'sources': sources(), 'bank_manifest': sha(ROOT / 'bank/manifest.json'), 'references': sha(ROOT / 'references/completed.json')}
    path = ROOT / 'results/lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == state, 'Source/config/bank changed after lock'
    else:
        dump(path, state)
    return state


def preflight():
    C.setup()
    torch.set_num_threads(2)
    errors = {}
    j = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
    for a, b, value in [(0, 1, .3), (1, 2, -.6), (2, 4, 1.2), (4, 3, .9)]:
        j[0, 0, a, b] = j[0, 0, b, a] = value
    context = torch.tensor([[0, -1, -1, -1, -1]])
    p, z = belief_propagation(j, j[:, 0] != 0, context, 80, 1.)
    states = torch.tensor(list(itertools.product([-1., 1.], repeat=5)), dtype=torch.float64)
    score = .5 * torch.einsum('si,ij,sj->s', states, j[0, 0], states)
    score = score.masked_fill(states[:, 0] != -1, -torch.inf)
    errors['tree_marginal'] = float((p[0, 0, :, 1] - score.softmax(0) @ ((states + 1) / 2)).abs().max())
    errors['tree_logz'] = float((z[0, 0] - score.logsumexp(0)).abs())
    assert max(errors.values()) < 1e-10
    master = NATIVE.master(12999001, 'independent')
    inp, query, y = observation(master, 2, 6, 'disjoint')
    assert not inp['observed'][inp['context'] >= 0].any()
    batch = tensors([inp], 'cpu', torch.float64)
    torch.manual_seed(101)
    model = FactorPFN().double()
    p, weights, logz = model(batch, True)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]))
    flip = copy.deepcopy(batch)
    known = flip['context'] >= 0
    flip['context'][known] = 1 - flip['context'][known]
    errors['class_permutation'] = float((model(flip) - p.flip(-1)).abs().max())
    signed = copy.deepcopy(batch)
    signed['x'][:, :, 0] *= -1
    errors['feature_sign'] = float((model(signed) - p).abs().max())
    columns = copy.deepcopy(batch)
    columns['x'] = columns['x'].flip(-1)
    columns['observed'] = columns['observed'].flip(-1)
    errors['feature_permutation'] = float((model(columns) - p).abs().max())
    relations = copy.deepcopy(batch)
    relations['adj'] = relations['adj'].flip(1)
    relations['pair_valid'] = relations['pair_valid'].flip(1)
    errors['relation_permutation'] = float((model(relations) - p).abs().max())
    poison = copy.deepcopy(batch)
    poison['x'][~poison['observed']] = 10000
    errors['hidden_value_poison'] = float((model(poison) - p).abs().max())
    # Node permutation preserves type and never exposes withheld labels.
    order = np.random.default_rng(101).permutation(len(inp['x']))
    perm = copy.deepcopy(batch)
    for key in ['x', 'observed', 'context', 'type']:
        perm[key] = perm[key][:, order]
    for key in ['adj', 'pair_valid']:
        perm[key] = perm[key][:, :, order][:, :, :, order]
    errors['node_permutation'] = float((model(perm) - p[:, order]).abs().max())
    assert max(errors.values()) < 1e-7, errors
    empty = copy.deepcopy(batch)
    empty['observed'][:] = False
    empty['x'][:] = 0
    assert torch.isfinite(model(empty)).all()
    free = copy.deepcopy(empty)
    free['context'][:] = -1
    errors['unanchored_symmetry'] = float((model(free) - .5).abs().max())
    assert errors['unanchored_symmetry'] < 1e-7
    # Both branch marginals and self-derived role weights affect predictive CE.
    loss = -p[0, query, torch.as_tensor(y)].clamp_min(1e-10).log().mean()
    g_role = torch.autograd.grad(loss, weights, retain_graph=True)[0]
    assert torch.isfinite(g_role).all()
    loss.backward()
    grads = [v.grad for v in model.parameters() if v.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert float(model.feature_factor[-1].bias.grad.abs().sum()) > 0
    # Empty-observation PCFI, as used by the public intervention, is finite.
    none = intervention(inp, 'no_observed', 101)
    assert np.isfinite(none['pcfi']).all()
    public = C.Model('mask_plain', 11)
    probability = C.predict(public, public_input(inp, query, 101))
    assert probability.shape == (len(query), 2) and np.isfinite(probability).all()
    del public
    gc.collect()
    torch.cuda.empty_cache()
    dump(ROOT / 'results/preflight.json', {'errors': errors, 'finite_gradients': True, 'gradient_through_self_weights': True,
        'gradient_through_feature_factors': True, 'public_forward': True, 'empty_pcfi': True, 'sources': sources(),
        'scope': 'Numerical checks before locking/training. BP on the cyclic production model remains approximate.'})
    print('PREFLIGHT_OK', errors, flush=True)


def make_model(arm, seed):
    torch.manual_seed(seed)
    if arm == 'factor_pfn':
        return FactorPFN(iterations=cfg()['bp_iterations']).cuda()
    return C.Model('pcfi_plain' if arm == 'pcfi_pfn' else 'mask_plain', seed)


def predict(model, arm, inp, query, seed, *, diagnostic=None):
    model.eval()
    if arm in ('mask_pfn', 'pcfi_pfn'):
        return C.predict(model, public_input(inp, query, seed))
    with torch.inference_mode():
        out = model(tensors([inp]), feature_off=diagnostic == 'feature_off', feedback_off=diagnostic == 'feedback_off')[0, query].cpu().numpy()
    assert np.isfinite(out).all()
    return out


def modes(row, arm):
    out = ['standard']
    if row['case'] == 'matched' and row['coverage'] == 'disjoint' and row['context_total'] == 2 and row['observed_nodes'] == 6:
        out += ['no_observed', 'shuffled', 'graph_permuted']
        if arm == 'factor_pfn':
            out += ['feature_off', 'feedback_off']
    return out


def evaluate(model, arm, seed, out):
    begun = time.perf_counter()
    pred = out / 'predictions'
    pred.mkdir(exist_ok=True)
    records = []
    for i, row in enumerate(rows('test')):
        inp, query, y = read(row['path'])
        for mode in modes(row, arm):
            path = pred / f'{i}_{mode}.npz'
            if path.exists():
                with np.load(path) as z:
                    prob = z['p']
                    assert np.array_equal(y, z['y'])
            else:
                given = intervention(inp, mode, 88000000 + i) if mode in ['no_observed', 'shuffled', 'graph_permuted'] else inp
                prob = predict(model, arm, given, query, 88000000 + i, diagnostic=mode)
                np.savez_compressed(path, p=prob, y=y)
            records.append({**row, 'arm': arm, 'seed': seed, 'mode': mode, **C.metrics(prob, y), 'prediction': path.relative_to(ROOT).as_posix()})
        if (i + 1) % 192 == 0:
            print('EVAL', arm, seed, i + 1, flush=True)
    dump(out / 'evaluation.json', records)
    dump(out / 'evaluation_time.json', {'seconds': time.perf_counter() - begun})
    print('EVALUATED', arm, seed, len(records), flush=True)


def run(seed, only_arm=None):
    C.setup()
    torch.set_num_threads(2)
    state = lock()
    assert (ROOT / 'results/preflight.json').exists()
    c = cfg()
    schedule = np.random.default_rng(seed + 42000).permutation(c['train_episodes'])
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    train = rows('train')
    cache = [read(r['path']) for r in train]
    for arm in ([only_arm] if only_arm else c['arms']):
        out = ROOT / 'results' / f'{arm}_s{seed}'
        out.mkdir(exist_ok=True, parents=True)
        if (out / 'evaluation.json').exists():
            continue
        model = make_model(arm, seed)
        if (out / 'training.json').exists():
            saved = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)['state_dict']
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(saved[name].to(p.device))
            evaluate(model, arm, seed, out)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        assert not (out / 'progress.json').exists(), 'No silent partial training restart'
        frozen = C.parameter_digest(model, trainable=False)
        initial = C.parameter_digest(model, trainable=True)
        public = arm != 'factor_pfn'
        if public:
            groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('base.')], 'base_lr': c['public_base_lr']},
                      {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('base.')], 'base_lr': c['public_new_lr']}]
        else:
            groups = [{'params': list(model.parameters()), 'base_lr': c['lr']}]
        opt = torch.optim.AdamW(groups, weight_decay=c['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', init_scale=64., enabled=public)
        begun = time.perf_counter()
        logs, retries = [], []
        model.train()
        for step in range(c['updates']):
            ids = schedule[step * c['batch_size']:(step + 1) * c['batch_size']]
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * min((step + 1) / 16, 1) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / c['updates'])))
            for attempt in range(24):
                opt.zero_grad(set_to_none=True)
                if public:
                    values = []
                    for i in ids:
                        inp, q, y = cache[int(i)]
                        with torch.autocast('cuda', dtype=torch.float16):
                            loss = F.cross_entropy(model(public_input(inp, q, 44000000 + int(i) + seed)), torch.as_tensor(y, device='cuda')) / len(ids)
                        scaler.scale(loss).backward()
                        values.append(float(loss.detach()))
                    value = sum(values)
                    scaler.unscale_(opt)
                else:
                    probability = model(tensors([cache[int(i)][0] for i in ids]))
                    losses = []
                    for local, i in enumerate(ids):
                        _, q, y = cache[int(i)]
                        losses.append(-probability[local, q, torch.as_tensor(y, device='cuda')].clamp_min(1e-10).log().mean())
                    loss = torch.stack(losses).mean()
                    value = float(loss.detach())
                    loss.backward()
                parameters = [p for p in model.parameters() if p.requires_grad]
                finite = math.isfinite(value) and all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
                if finite:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, c['clip'])
                    if public:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    break
                if not public:
                    raise RuntimeError(f'Nonfinite factor model at seed{seed}/step{step}')
                retries.append({'step': step, 'attempt': attempt, 'scale': scaler.get_scale()})
                scaler.update(max(scaler.get_scale() / 2, 1e-6))
            else:
                raise RuntimeError('Finite-gradient retries exhausted')
            logs.append({'step': step + 1, 'loss': value, 'gradient_norm': float(norm), 'seconds': time.perf_counter() - begun})
            if (step + 1) % 16 == 0:
                dump(out / 'progress.json', logs[-1])
                print('TRAIN', arm, seed, step + 1, round(np.mean([r['loss'] for r in logs[-16:]]), 5), flush=True)
        assert C.parameter_digest(model, trainable=False) == frozen
        torch.save({'state_dict': {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, 'metadata': state}, out / 'checkpoint.pt')
        dump(out / 'train_log.json', logs)
        dump(out / 'overflow_retries.json', retries)
        dump(out / 'training.json', {'arm': arm, 'seed': seed, 'updates': len(logs), 'examples': c['train_episodes'], 'schedule_sha256': schedule_sha,
            'initial_sha256': initial, 'final_sha256': C.parameter_digest(model, trainable=True), 'frozen_sha256': frozen,
            'frozen_weights_unchanged': True, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.perf_counter() - begun, 'checkpoint_sha256': sha(out / 'checkpoint.pt')})
        evaluate(model, arm, seed, out)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'preflight', 'run'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--arm')
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'preflight':
        preflight()
    else:
        run(args.seed, args.arm)


if __name__ == '__main__':
    main()
