"""Same-budget CE training; final checkpoints only; isolated query labels."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import gc
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT, C, D, cfg, dump, sha, read, rows, lock
from operators import make_model


def is_public(arm):
    return arm in ['mask_pfn', 'pcfi_pfn']


def load_model(arm, seed):
    folder = ROOT / 'results' / f'{arm}_s{seed}'
    info = json.loads((folder / 'training.json').read_text())
    assert sha(folder / 'checkpoint.pt') == info['checkpoint_sha256']
    model = make_model(arm, seed)
    assert C.parameter_digest(model, True) == info['initial_sha256']
    assert C.parameter_digest(model, False) == info['frozen_sha256']
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['metadata'] == lock()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(checkpoint['state_dict'][name].to(p.device))
    assert C.parameter_digest(model, True) == info['final_sha256']
    return model.eval()


def predict(model, arm, inp, query, index):
    if is_public(arm):
        return C.predict(model, D.public_input(inp, query, 88000000 + index))
    with torch.inference_mode():
        return model(D.tensors([inp]))[0, query].cpu().numpy()


def evaluate(model, arm, seed, folder):
    model.eval()
    begun = time.perf_counter()
    test = rows('test')
    result = np.full((len(test), 12, 2), .5, np.float32)
    if is_public(arm):
        for i, row in enumerate(test):
            inp, q, _ = read(row)
            result[i, q] = predict(model, arm, inp, q, i)
            if (i + 1) % 192 == 0:
                print('EVAL', arm, seed, i + 1, flush=True)
    else:
        for start in range(0, len(test), 16):
            chunk = test[start:start + 16]
            items = [read(r) for r in chunk]
            with torch.inference_mode():
                probs = model(D.tensors([v[0] for v in items])).cpu().numpy()
            for i, (_, q, _) in enumerate(items):
                result[start + i, q] = probs[i, q]
    assert np.isfinite(result).all() and np.allclose(result.sum(-1), 1, atol=2e-5)
    path = folder / 'predictions.npz'
    np.savez_compressed(path, p=result)
    dump(folder / 'evaluation.json', {'arm': arm, 'seed': seed, 'episodes': len(test),
         'seconds': time.perf_counter() - begun, 'prediction_sha256': sha(path), 'lock': lock()})
    print('EVALUATED', arm, seed, flush=True)


def train(seed, only_arm=None):
    C.setup()
    torch.set_num_threads(2)
    state, c = lock(), cfg()
    schedule = np.random.default_rng(seed + 42000).permutation(c['train_episodes'])
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    cache = [read(r) for r in rows('train')]
    for arm in ([only_arm] if only_arm else c['arms']):
        folder = ROOT / 'results' / f'{arm}_s{seed}'
        folder.mkdir(exist_ok=True, parents=True)
        if (folder / 'evaluation.json').exists():
            continue
        if (folder / 'training.json').exists():
            model = load_model(arm, seed)
            evaluate(model, arm, seed, folder)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        assert not (folder / 'progress.json').exists(), 'No silent partial restart'
        model = make_model(arm, seed)
        frozen, initial = C.parameter_digest(model, False), C.parameter_digest(model, True)
        public = is_public(arm)
        if public:
            groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('base.')], 'base_lr': .00003},
                      {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('base.')], 'base_lr': .001}]
        else:
            groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'base_lr': .001}]
        opt = torch.optim.AdamW(groups, weight_decay=.0001)
        scaler = torch.amp.GradScaler('cuda', init_scale=64., enabled=public)
        begun, logs, retries = time.perf_counter(), [], []
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
                            loss = F.cross_entropy(model(D.public_input(inp, q, 44000000 + int(i) + seed)), torch.as_tensor(y, device='cuda')) / len(ids)
                        scaler.scale(loss).backward()
                        values.append(float(loss.detach()))
                    value = sum(values)
                    scaler.unscale_(opt)
                else:
                    p = model(D.tensors([cache[int(i)][0] for i in ids]))
                    terms = []
                    for local, i in enumerate(ids):
                        _, q, y = cache[int(i)]
                        terms.append(-p[local, q, torch.as_tensor(y, device='cuda')].clamp_min(1e-10).log().mean())
                    loss = torch.stack(terms).mean()
                    value = float(loss.detach())
                    loss.backward()
                params = [p for p in model.parameters() if p.requires_grad]
                finite = math.isfinite(value) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                if finite:
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                    if public:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    break
                if not public:
                    raise RuntimeError(f'Nonfinite {arm} seed{seed} step{step}')
                retries.append({'step': step, 'attempt': attempt, 'scale': scaler.get_scale()})
                scaler.update(max(scaler.get_scale() / 2, 1e-6))
            else:
                raise RuntimeError('Finite-gradient retries exhausted')
            logs.append({'step': step + 1, 'loss': value, 'gradient_norm': float(norm), 'seconds': time.perf_counter() - begun})
            if (step + 1) % 16 == 0:
                dump(folder / 'progress.json', logs[-1])
                print('TRAIN', arm, seed, step + 1, round(np.mean([r['loss'] for r in logs[-16:]]), 5), flush=True)
        assert C.parameter_digest(model, False) == frozen
        torch.save({'state_dict': {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, 'metadata': state}, folder / 'checkpoint.pt')
        dump(folder / 'train_log.json', logs)
        dump(folder / 'overflow_retries.json', retries)
        dump(folder / 'training.json', {'arm': arm, 'seed': seed, 'updates': len(logs), 'examples': len(schedule),
             'schedule_sha256': schedule_sha, 'initial_sha256': initial, 'final_sha256': C.parameter_digest(model, True),
             'frozen_sha256': frozen, 'frozen_weights_unchanged': True, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'total_parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.perf_counter() - begun, 'checkpoint_sha256': sha(folder / 'checkpoint.pt')})
        evaluate(model, arm, seed, folder)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arm')
    args = ap.parse_args()
    train(args.seed, args.arm)
