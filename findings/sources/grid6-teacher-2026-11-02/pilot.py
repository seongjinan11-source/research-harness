"""Exact-Grid6 predictive distillation; immutable historical students and fresh evaluation."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'disjoint_factor_pilot'
sys.path.insert(0, str(OLD))
import data as D
import experiment as E
from model import FactorPFN
import benchmark as B
C, NATIVE = D.C, D.NATIVE
NEW_ARMS = ['mask_pfn', 'factor_pfn']


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def dump(path, value):
    D.dump(path, value)


def read_test(row):
    with np.load(ROOT / 'bank' / row['path']) as z:
        return {key: z[key] for key in D.INPUT_KEYS}, z['query'], z['labels_query']


def test_rows():
    return json.loads((ROOT / 'bank/manifest.json').read_text())['episodes']


def sources():
    own = {p.name: D.sha(p) for p in ROOT.glob('*.py')}
    own['protocol.json'] = D.sha(ROOT / 'protocol.json')
    own['historical_lock'] = D.sha(OLD / 'results/lock.json')
    return own


def lock():
    E.lock()
    value = {'sources': sources(), 'train_manifest': D.sha(OLD / 'bank/manifest.json'),
             'new_test_manifest': D.sha(ROOT / 'bank/manifest.json'),
             'teacher_manifest': D.sha(ROOT / 'teachers/manifest.json'),
             'preflight': D.sha(ROOT / 'results/preflight.json')}
    path = ROOT / 'results/lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Source or bank changed after training lock'
    else:
        dump(path, value)
    return value


def prepare():
    E.lock()
    assert not (ROOT / 'bank/manifest.json').exists(), 'No silent bank overwrite'
    c = cfg()
    C.setup()
    torch.set_num_threads(2)
    records, seeds = [], []
    start = time.perf_counter()
    for pid, prior in enumerate(NATIVE.NAMES):
        for local in range(c['test_graphs_per_prior']):
            gid = pid * c['test_graphs_per_prior'] + local
            seed = c['test_seed_start'] + gid
            assert not D.cfg()['train_seed_start'] <= seed < D.cfg()['train_seed_start'] + c['train_episodes']
            assert not D.cfg()['test_seed_start'] <= seed < D.cfg()['test_seed_start'] + 96
            seeds.append(seed)
            case = ['matched', 'matched', 'density_low', 'signal_high'][local % 4]
            master = NATIVE.master(seed, prior, case)
            for k in c['contexts']:
                for count in c['observed_nodes']:
                    for coverage in ['disjoint', 'paired_overlap']:
                        inp, query, y = D.observation(master, k, count, coverage)
                        name = f'g{gid}_k{k}_o{count}_{coverage}.npz'
                        path = ROOT / 'bank/test' / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, **inp, query=query, labels_query=y)
                        records.append({'path': 'test/' + name, 'graph': gid, 'seed': seed, 'prior': prior, 'case': case,
                                        'context_total': k, 'observed_nodes': count, 'coverage': coverage, 'sha256': D.sha(path)})
            if (gid + 1) % 16 == 0:
                print('PREPARE_TEST', gid + 1, flush=True)
    dump(ROOT / 'bank/manifest.json', {'episodes': records, 'graph_seeds': seeds,
         'seconds': time.perf_counter() - start, 'historical_train_manifest': D.sha(OLD / 'bank/manifest.json')})
    engine = NATIVE.GridReference('cuda')
    start = time.perf_counter()
    teacher_records = []
    train = D.rows('train')
    for begin in range(0, len(train), 16):
        chunk = train[begin:begin + 16]
        # D.read returns query truth separately. It is explicitly never passed to the teacher.
        items = [D.read(row['path'])[0] for row in chunk]
        with torch.inference_mode():
            ref = engine.infer(items, with_bp=False)['grid6_exact']
        for i, row in enumerate(chunk):
            path = ROOT / 'teachers/train' / (Path(row['path']).stem + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, p=ref[i])
            teacher_records.append({'index': begin + i, 'input_sha256': row['sha256'],
                                    'path': path.relative_to(ROOT).as_posix(), 'sha256': D.sha(path)})
        if (begin + len(chunk)) % 256 == 0:
            print('TEACHER_TRAIN', begin + len(chunk), flush=True)
    dump(ROOT / 'teachers/manifest.json', {'episodes': teacher_records, 'seconds': time.perf_counter() - start,
         'teacher': 'exact prior predictive conditional on visible inputs; six parameters integrated, true parameters withheld'})
    start = time.perf_counter()
    references = []
    for begin in range(0, len(records), 16):
        chunk = records[begin:begin + 16]
        items = [read_test(r)[0] for r in chunk]
        with torch.inference_mode():
            values = engine.infer(items)
        for i, row in enumerate(chunk):
            path = ROOT / 'references' / (Path(row['path']).stem + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: v[i] for k, v in values.items()})
            references.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': D.sha(path)})
        if (begin + len(chunk)) % 192 == 0:
            print('REFERENCE_TEST', begin + len(chunk), flush=True)
    dump(ROOT / 'references/manifest.json', {'episodes': references, 'seconds': time.perf_counter() - start})
    print('PREPARED', flush=True)


def preflight():
    E.lock()
    C.setup()
    torch.set_num_threads(2)
    engine = NATIVE.GridReference('cuda')
    # Fresh non-test seed; reference invariances do not inspect scored test outcomes.
    inp, query, y = D.observation(NATIVE.master(12199991, 'independent'), 2, 6, 'disjoint')
    with torch.inference_mode():
        p = engine.infer([inp], with_bp=False)['grid6_exact'][0]
        poison = copy.deepcopy(inp)
        poison['x'][~poison['observed']] = 1234.
        bad = engine.infer([poison], with_bp=False)['grid6_exact'][0]
        flipped = copy.deepcopy(inp)
        known = flipped['context'] >= 0
        flipped['context'][known] = 1 - flipped['context'][known]
        pf = engine.infer([flipped], with_bp=False)['grid6_exact'][0]
        empty = copy.deepcopy(inp)
        empty['observed'][:] = False
        empty['x'][:] = 0
        empty['context'][:] = -1
        pe = engine.infer([empty], with_bp=False)['grid6_exact'][0]
        native_bp = engine.infer([inp])['grid6_bp']
        fast_bp = B.bp_only(engine, [inp])
    errors = {'normalization': float(np.max(abs(p.sum(-1) - 1))), 'hidden_value_poison': float(abs(p - bad).max()),
              'class_flip': float(abs(pf - p[:, ::-1]).max()), 'no_evidence_symmetry': float(abs(pe - .5).max()),
              'context_clamp': float(abs(p[known, 1] - inp['context'][known]).max()),
              'bp_only_parity': float(abs(native_bp - fast_bp).max())}
    assert max(errors.values()) < 1e-10, errors
    torch.manual_seed(137)
    logits = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    teacher = torch.softmax(torch.randn_like(logits), -1).detach()
    loss = F.kl_div(logits.log_softmax(-1), teacher, reduction='batchmean')
    soft = -(teacher * logits.log_softmax(-1)).sum(-1).mean()
    errors['kl_soft_ce_gradient'] = float(abs(torch.autograd.grad(loss, logits, retain_graph=True)[0] - torch.autograd.grad(soft, logits)[0]).max())
    assert errors['kl_soft_ce_gradient'] < 1e-12
    # A hard one-hot teacher recovers the original sampled-label CE gradient.
    hard = F.one_hot(torch.arange(7) % 2, 2).double()
    a = F.kl_div(logits.log_softmax(-1), hard, reduction='batchmean')
    b = F.cross_entropy(logits, torch.arange(7) % 2)
    errors['hard_ce_gradient'] = float(abs(torch.autograd.grad(a, logits, retain_graph=True)[0] - torch.autograd.grad(b, logits)[0]).max())
    assert errors['hard_ce_gradient'] < 1e-12
    masked_logits = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
    masked_teacher = torch.tensor(p, dtype=torch.float64)
    masked_loss = F.kl_div(masked_logits[query].log_softmax(-1), masked_teacher[query], reduction='batchmean')
    altered = masked_teacher.clone()
    nonquery = np.setdiff1d(np.arange(12), query)
    altered[nonquery] = altered[nonquery].flip(-1)
    altered_loss = F.kl_div(masked_logits[query].log_softmax(-1), altered[query], reduction='batchmean')
    masked_gradient = torch.autograd.grad(masked_loss, masked_logits)[0]
    errors['nonquery_teacher_invariance'] = float(abs(masked_loss.detach() - altered_loss.detach()))
    errors['nonquery_logit_gradient'] = float(masked_gradient[nonquery].abs().max())
    assert errors['nonquery_teacher_invariance'] == 0 and errors['nonquery_logit_gradient'] == 0
    matched = []
    for arm in NEW_ARMS:
        for seed in cfg()['seeds']:
            model = E.make_model(arm, seed)
            meta = json.loads((OLD / f'results/{arm}_s{seed}/training.json').read_text())
            assert C.parameter_digest(model, trainable=True) == meta['initial_sha256']
            assert C.parameter_digest(model, trainable=False) == meta['frozen_sha256']
            schedule = np.random.default_rng(seed + 42000).permutation(cfg()['train_episodes'])
            assert hashlib.sha256(schedule.tobytes()).hexdigest() == meta['schedule_sha256']
            if seed == 11:
                model.train()
                qteacher = torch.tensor(p[query], device='cuda', dtype=torch.float32)
                if arm == 'mask_pfn':
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = model(D.public_input(inp, query, 9991))
                        value = F.kl_div(output.float().log_softmax(-1), qteacher, reduction='batchmean')
                else:
                    output = model(D.tensors([inp]))[0, query]
                    value = (qteacher * (qteacher.clamp_min(1e-30).log() - output.clamp_min(1e-10).log())).sum(-1).mean()
                value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in model.parameters() if p.requires_grad)
            matched.append({'arm': arm, 'seed': seed, 'initial_sha256': meta['initial_sha256'], 'schedule_sha256': meta['schedule_sha256']})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    dump(ROOT / 'results/preflight.json', {'errors': errors, 'paired_original_initializations': matched,
         'teacher_only_visible_inputs': True, 'teacher_detached_query_rows_only': True,
         'finite_student_gradients': True, 'sources': sources()})
    lock()
    print('PREFLIGHT_OK', errors, flush=True)


def train(seed):
    C.setup()
    torch.set_num_threads(2)
    state = lock()
    c, oldcfg = cfg(), D.cfg()
    schedule = np.random.default_rng(seed + 42000).permutation(c['train_episodes'])
    sh = hashlib.sha256(schedule.tobytes()).hexdigest()
    rr = D.rows('train')
    cache = [D.read(r['path']) for r in rr]
    teachers = []
    for row in rr:
        with np.load(ROOT / 'teachers/train' / (Path(row['path']).stem + '.npz')) as z:
            teachers.append(z['p'])
    for arm in NEW_ARMS:
        out = ROOT / 'results' / f'{arm}_distill_s{seed}'
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'training.json').exists():
            continue
        assert not (out / 'progress.json').exists(), 'No silent partial training restart'
        model = E.make_model(arm, seed)
        previous = json.loads((OLD / f'results/{arm}_s{seed}/training.json').read_text())
        initial, frozen = C.parameter_digest(model, trainable=True), C.parameter_digest(model, trainable=False)
        assert initial == previous['initial_sha256'] and frozen == previous['frozen_sha256'] and sh == previous['schedule_sha256']
        public = arm == 'mask_pfn'
        if public:
            groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('base.')], 'base_lr': oldcfg['public_base_lr']},
                      {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('base.')], 'base_lr': oldcfg['public_new_lr']}]
        else:
            groups = [{'params': list(model.parameters()), 'base_lr': oldcfg['lr']}]
        opt = torch.optim.AdamW(groups, weight_decay=oldcfg['weight_decay'])
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
                        inp, q, _ = cache[int(i)]
                        target = torch.tensor(teachers[int(i)][q], device='cuda', dtype=torch.float32).detach()
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits = model(D.public_input(inp, q, 44000000 + int(i) + seed))
                            loss = F.kl_div(logits.float().log_softmax(-1), target, reduction='batchmean') / len(ids)
                        scaler.scale(loss).backward()
                        values.append(float(loss.detach()))
                    value = sum(values)
                    scaler.unscale_(opt)
                else:
                    probability = model(D.tensors([cache[int(i)][0] for i in ids]))
                    losses = []
                    for loc, i in enumerate(ids):
                        _, q, _ = cache[int(i)]
                        target = torch.tensor(teachers[int(i)][q], device='cuda', dtype=torch.float32).detach()
                        losses.append((target * (target.clamp_min(1e-30).log() - probability[loc, q].clamp_min(1e-10).log())).sum(-1).mean())
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    value = float(loss.detach())
                parameters = [p for p in model.parameters() if p.requires_grad]
                finite = math.isfinite(value) and all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
                if finite:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, oldcfg['clip'])
                    if public:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    break
                assert public, 'Nonfinite Factor gradient'
                retries.append({'step': step, 'attempt': attempt, 'scale': scaler.get_scale()})
                scaler.update(max(scaler.get_scale() / 2, 1e-6))
            else:
                raise RuntimeError('Finite-gradient retries exhausted')
            logs.append({'step': step + 1, 'loss': value, 'gradient_norm': float(norm), 'seconds': time.perf_counter() - begun})
            if (step + 1) % 16 == 0:
                dump(out / 'progress.json', logs[-1])
                print('TRAIN', arm, seed, step + 1, round(float(np.mean([r['loss'] for r in logs[-16:]])), 6), flush=True)
        assert C.parameter_digest(model, trainable=False) == frozen
        torch.save({'state_dict': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, 'metadata': state}, out / 'checkpoint.pt')
        dump(out / 'train_log.json', logs)
        dump(out / 'overflow_retries.json', retries)
        meta = {'arm': arm + '_distill', 'seed': seed, 'updates': len(logs), 'examples': c['train_episodes'], 'schedule_sha256': sh,
                'initial_sha256': initial, 'final_sha256': C.parameter_digest(model, trainable=True), 'frozen_sha256': frozen,
                'frozen_weights_unchanged': True, 'trainable_parameters': sum(p.numel() for p in parameters),
                'total_parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.perf_counter() - begun,
                'checkpoint_sha256': D.sha(out / 'checkpoint.pt'), 'paired_ce_metadata': str((OLD / f'results/{arm}_s{seed}/training.json').relative_to(ROOT.parent))}
        assert meta['trainable_parameters'] == previous['trainable_parameters']
        dump(out / 'training.json', meta)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    print('TRAINED_SEED', seed, flush=True)


def load_model(arm, seed, distilled):
    model = E.make_model(arm, seed)
    folder = ROOT / f'results/{arm}_distill_s{seed}' if distilled else OLD / f'results/{arm}_s{seed}'
    meta = json.loads((folder / 'training.json').read_text())
    assert D.sha(folder / 'checkpoint.pt') == meta['checkpoint_sha256']
    assert C.parameter_digest(model, trainable=True) == meta['initial_sha256']
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(saved['state_dict'][name].to(p.device))
    assert C.parameter_digest(model, trainable=True) == meta['final_sha256']
    assert C.parameter_digest(model, trainable=False) == meta['frozen_sha256']
    model.eval()
    return model


def eval_spec():
    spec = []
    for i, r in enumerate(test_rows()):
        spec.append((i, 'standard'))
        if r['case'] == 'matched' and r['coverage'] == 'disjoint' and r['context_total'] == 2 and r['observed_nodes'] == 6:
            spec.extend((i, m) for m in ['no_observed', 'shuffled', 'graph_permuted'])
    return spec


def evaluate(seed):
    C.setup()
    torch.set_num_threads(2)
    lock()
    rr, spec = test_rows(), eval_spec()
    cache = [read_test(r) for r in rr]
    for arm, distilled in [('mask_pfn', False), ('pcfi_pfn', False), ('factor_pfn', False), ('mask_pfn', True), ('factor_pfn', True)]:
        name = arm + ('_distill' if distilled else '_ce')
        folder = ROOT / 'evaluation' / f'{name}_s{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'completed.json').exists():
            continue
        model = load_model(arm, seed, distilled)
        begun = time.perf_counter()
        probs = np.full((len(spec), 12, 2), .5, np.float64)
        for j, (i, mode) in enumerate(spec):
            inp, q, y = cache[i]
            given = inp if mode == 'standard' else D.intervention(inp, mode, 88000000 + i)
            p = E.predict(model, arm, given, q, 88000000 + i)
            assert p.shape == (len(q), 2) and np.isfinite(p).all()
            probs[j, q] = p
            if (j + 1) % 192 == 0:
                print('EVAL', name, seed, j + 1, flush=True)
        path = folder / 'predictions.npz'
        np.savez_compressed(path, p=probs, episode=np.array([i for i, m in spec]), mode=np.array([m for i, m in spec]))
        dump(folder / 'completed.json', {'arm': name, 'seed': seed, 'records': len(spec), 'seconds': time.perf_counter() - begun,
                                         'prediction_sha256': D.sha(path)})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print('EVALUATED_SEED', seed, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'preflight', 'train', 'evaluate'])
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'preflight':
        preflight()
    elif args.action == 'train':
        train(args.seed)
    else:
        evaluate(args.seed)
