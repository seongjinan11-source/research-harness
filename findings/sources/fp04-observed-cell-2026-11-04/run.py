"""FP04 root-owned sequential training, native transfer and provenance lock."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from util import ROOT, config, dump, sha, sources, metrics, array_hash


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()


def tensor(inp, device='cuda'):
    out = {}
    for key in ['x', 'pcfi', 'observed', 'src', 'dst', 'context']:
        dtype = torch.bool if key == 'observed' else torch.long if key in ['src', 'dst', 'context'] else torch.float32
        out[key] = torch.as_tensor(inp[key], device=device, dtype=dtype)
    out['classes'] = int(inp['classes'])
    return out


def digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make(seed):
    from model import ObservedCellPFN
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    c = config()['model']
    return ObservedCellPFN(width=c['width'], rank=c['rank'], rounds=c['rounds']).cuda()


def lock(create=False):
    from data import rows
    banks = {}
    for split in ['train', 'dev', 'final', 'real']:
        rr = rows(split)
        for row in rr:
            assert sha(ROOT / row['path']) == row['sha256'], row['path']
        banks[split] = dict(count=len(rr), manifest=sha(ROOT / 'bank' / split / 'manifest.json'))
    state = dict(sources=sources(), banks=banks,
                 checks=sha(ROOT / 'results/checks.json'), preflight=sha(ROOT / 'results/preflight.json'))
    checked=json.loads((ROOT/'results/checks.json').read_text(encoding='utf8'))
    assert checked['status']=='PASS'
    for path,expected in checked['tested_sources'].items():
        assert sha(path)==expected, 'Re-run numerical checks after source changes: '+path
    assert json.loads((ROOT/'results/preflight.json').read_text(encoding='utf8'))['sources']==sources(), 'Native preflight source receipt stale'
    path = ROOT / 'results/lock.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == state, 'Scientific source/bank changed after lock'
    else:
        assert create
        assert json.loads((ROOT / 'results/checks.json').read_text(encoding='utf8'))['status'] == 'PASS'
        dump(path, state)
    return state


def preflight():
    """Discarded dev smoke; actual native resource smoke has no metric computation."""
    from data import rows, read
    setup()
    config_data = config()
    records = []
    sample = next(r for r in rows('dev') if r['setting'] == 'node95')
    inp, query, y = read(sample)
    batch = tensor(inp)
    initials = []
    for arm in config_data['arms']:
        model = make(918)
        initials.append(digest(model))
        assert all(p.requires_grad for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_data['train']['learning_rate'])
        losses = []
        for _ in range(config_data['dev']['smoke_updates']):
            optimizer.zero_grad(set_to_none=True)
            z = model(batch, arm=arm, class_chunk=batch['classes'])
            loss = F.cross_entropy(z[query], torch.as_tensor(y, device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        records.append(dict(arm=arm, discarded_updates=len(losses), losses=losses,
                            parameters=sum(p.numel() for p in model.parameters()), final=digest(model)))
        del model, optimizer
    assert len(set(initials)) == 1
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    resource = []
    model = make(918).eval()
    real = rows('real')
    for mode in ['95', '995', 'none']:
        row = next(r for r in real if r.get('graph') == 'CiteSeer' and r.get('repetition') == 0
                   and r.get('mode', r['setting']) == mode)
        inp, query, _ = read(row)
        batch = tensor(inp)
        for arm in config_data['arms']:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                z = model(batch, arm=arm, class_chunk=1, feature_chunk=config_data['model']['feature_chunk'])
                p = z.softmax(-1)
            torch.cuda.synchronize()
            assert z.shape == (len(inp['x']), int(inp['classes'])) and torch.isfinite(p).all()
            resource.append(dict(graph='CiteSeer', mode=mode, arm=arm, nodes=len(inp['x']),
                                 features=inp['x'].shape[1], seconds=time.perf_counter() - start,
                                 peak_mib=torch.cuda.max_memory_allocated()/2**20,
                                 input_path=row['path'],input_sha256=row['sha256'],
                                 scope='one cold native forward, no query metrics inspected'))
        del batch, inp, p, z
        gc.collect()
        torch.cuda.empty_cache()
    del model
    dump(ROOT / 'results/preflight.json', dict(status='PASS', discarded=records,
         same_initialization=True, resource=resource, sources=sources(),
         discarded_total_updates=sum(r['discarded_updates'] for r in records)))
    print('PREFLIGHT_PASS', json.dumps(resource), flush=True)


def predict_and_save(model, arm, seed, row, out, batch=None):
    from data import read
    began=time.perf_counter()
    name = hashlib.sha256((row['split']+'|'+row['path']).encode()).hexdigest()[:20]
    path = out / 'predictions' / (name + '.npz')
    sidecar = path.with_suffix('.json')
    association = dict(arm=arm, seed=seed, input_sha256=row['sha256'],
                       checkpoint_sha256=sha(out/'checkpoint.pt'),
                       lock_sha256=sha(ROOT/'results/lock.json'))
    inp, query, y = read(row)
    if path.exists():
        old = json.loads(sidecar.read_text(encoding='utf8'))
        assert old['association'] == association and old['prediction_sha256'] == sha(path)
        timing=old['timing']
        with np.load(path) as z:
            p = z['p']
            assert np.array_equal(z['query'], query) and np.array_equal(z['y'], y)
            assert np.array_equal(z['context'], inp['context'])
    else:
        t = tensor(inp) if batch is None else batch
        torch.cuda.synchronize()
        base_alloc=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        infer_began=time.perf_counter()
        with torch.inference_mode():
            logits = model(t, arm=arm, class_chunk=1, feature_chunk=config()['model']['feature_chunk'])
            p = logits[query].softmax(-1).cpu().numpy()
        torch.cuda.synchronize()
        timing=dict(inference_seconds=time.perf_counter()-infer_began,
                    end_to_end_seconds=time.perf_counter()-began,
                    peak_mib=torch.cuda.max_memory_allocated()/2**20,
                    additional_peak_mib=max(0,torch.cuda.max_memory_allocated()-base_alloc)/2**20,
                    scope='inference includes probability CPU transfer; end-to-end also input read/device transfer; PCFI precomputed separately')
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('xb') as handle:
            np.savez_compressed(handle, p=p, query=query, y=y, context=inp['context'])
        dump(sidecar,dict(association=association,prediction_sha256=sha(path),timing=timing))
    return {**row, 'arm':arm, 'seed':seed, 'prediction':path.relative_to(ROOT).as_posix(),
            'prediction_sha256':sha(path), 'association':association,'timing':timing,'metrics':metrics(p, y)}


def evaluate(model, arm, seed, split, out):
    from data import rows, read
    path = out / (split + '_evaluation.json')
    existing = json.loads(path.read_text(encoding='utf8')) if path.exists() else None
    begin = time.perf_counter()
    records = []
    model.eval()
    for i, row in enumerate(rows(split)):
        records.append(predict_and_save(model, arm, seed, row, out))
        if (i+1) % 96 == 0 or split == 'real' and (i+1) % 5 == 0:
            print('EVAL', arm, seed, split, i+1, flush=True)
    if existing is not None:
        assert existing['records'] == records, 'Completed evaluation provenance changed'
    else:
        dump(path, dict(records=records, seconds=time.perf_counter()-begin))
    print('EVALUATED', arm, seed, split, len(records), flush=True)


def train(seeds):
    from data import rows, read
    setup()
    state = lock()
    cfg = config()
    c = cfg['train']
    train_rows = rows('train')
    cache = []
    print('CACHE', len(train_rows), flush=True)
    for row in train_rows:
        inp, query, y = read(row)
        cache.append((tensor(inp), torch.as_tensor(query, device='cuda'), torch.as_tensor(y, device='cuda')))
    print('CACHED_MIB', torch.cuda.memory_allocated()/2**20, flush=True)
    for seed in seeds:
        schedule = np.random.default_rng(c['schedule_seed'] + seed).integers(len(cache), size=(c['updates'], c['batch']))
        schedule_sha = array_hash(schedule)
        common_init = None
        for arm in cfg['arms']:
            out = ROOT / 'results' / f'{arm}_s{seed}'
            out.mkdir(parents=True, exist_ok=True)
            model = make(seed)
            initial = digest(model)
            if common_init is None:
                common_init = initial
            assert initial == common_init
            checkpoint = out / 'checkpoint.pt'
            if (out / 'training.json').exists():
                ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
                assert ck['lock_sha256'] == sha(ROOT/'results/lock.json') and ck['initial'] == initial
                model.load_state_dict(ck['state_dict'])
            else:
                assert not (out / 'progress.json').exists(), 'Partial fit is preserved, no silent restart'
                optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
                logs = []
                begin = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                model.train()
                for step, indices in enumerate(schedule):
                    mult = min((step+1)/c['warmup'], 1.) * (.1+.9*.5*(1+math.cos(math.pi*step/c['updates'])))
                    optimizer.param_groups[0]['lr'] = c['learning_rate']*mult
                    optimizer.zero_grad(set_to_none=True)
                    running = 0.
                    for index in indices:
                        batch, query, target = cache[int(index)]
                        logits = model(batch, arm=arm, class_chunk=batch['classes'], feature_chunk=cfg['model']['feature_chunk'])
                        loss = F.cross_entropy(logits[query], target) / c['batch']
                        assert torch.isfinite(loss), (arm, seed, step, int(index))
                        loss.backward()
                        running += float(loss.detach())
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    logs.append(dict(step=step+1, loss=running, gradient_norm=float(norm), seconds=time.perf_counter()-begin))
                    if (step+1) % 64 == 0:
                        dump(out/'progress.json', logs[-1])
                        print('TRAIN',arm,seed,step+1,round(np.mean([r['loss'] for r in logs[-64:]]),5),flush=True)
                torch.cuda.synchronize()
                torch.save(dict(state_dict=model.cpu().state_dict(), initial=initial,
                                lock_sha256=sha(ROOT/'results/lock.json')),checkpoint)
                model.cuda()
                dump(out/'schedule.json', dict(indices=schedule.tolist(), sha256=schedule_sha))
                dump(out/'train_log.json',logs)
                dump(out/'training.json',dict(arm=arm, seed=seed, updates=len(logs),
                     presentations=int(schedule.size), unique_episodes=int(len(np.unique(schedule))),
                     schedule_sha256=schedule_sha, initial_sha256=initial, final_sha256=digest(model),
                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     total_parameters=sum(p.numel() for p in model.parameters()),
                     seconds=time.perf_counter()-begin, peak_mib=torch.cuda.max_memory_allocated()/2**20,
                     checkpoint_sha256=sha(checkpoint), target_updates=0))
                del optimizer
            evaluate(model, arm, seed, 'final', out)
            # Real is unconditional on the synthetic result. One GPU worker.
            evaluate(model, arm, seed, 'real', out)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    lock()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'preflight', 'lock', 'train'])
    p.add_argument('--split', choices=['train','dev','final','real'])
    p.add_argument('--seed',type=int, nargs='+')
    args = p.parse_args()
    if args.action == 'prepare':
        from data import prepare
        prepare(args.split)
    elif args.action == 'preflight':
        preflight()
    elif args.action == 'lock':
        lock(create=True)
        print('LOCKED',sha(ROOT/'results/lock.json'),flush=True)
    else:
        train(args.seed or config()['seeds'])


if __name__ == '__main__':
    main()
