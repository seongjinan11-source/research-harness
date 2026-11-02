"""Post-worker visible-input timing, with BP-only reference output."""
import sys
sys.dont_write_bytecode = True
import sklearn
import gc
import json
import math
import time
import numpy as np
import torch
from data import ROOT, C, NATIVE, cfg, dump, rows, read, pcfi, tensors
from experiment import make_model, predict, lock


def bp_only(engine, items):
    b = tensors(items, engine.device, torch.float64)
    e, count = b['incidence'].flatten(2), len(items)
    ne = 1 - e
    pri = torch.as_tensor(np.stack([NATIVE.PRIORS[NATIVE.NAMES[int(x['prior'].argmax())]] for x in items]), device=engine.device, dtype=torch.float64)
    xi = torch.tensor(NATIVE.PARAMETERS, device=engine.device)[None].expand(count, -1, -1)
    mass = torch.tensor(NATIVE.MASS, device=engine.device)[None].expand(count, -1)
    d, s = xi.unbind(-1)
    plus, minus = d * (1 + s), d * (1 - s)
    ce = .5 * (plus.log() + minus.log()) - d.log()
    cn = .5 * (torch.log1p(-plus) + torch.log1p(-minus)) - torch.log1p(-d)
    je = .5 * (plus.log() - minus.log())
    jn = .5 * (torch.log1p(-plus) - torch.log1p(-minus))
    ea, na = e.sum(-1) @ engine.roles.T, ne.sum(-1) @ engine.roles.T
    es = torch.einsum('bre,se,hr->bhs', e, engine.products, engine.roles)
    ns = torch.einsum('bre,se,hr->bhs', ne, engine.products, engine.roles)
    constant = e.sum((1, 2))[:, None] * d.log() + ne.sum((1, 2))[:, None] * torch.log1p(-d)
    logg = constant[:, :, None, None] + ea[:, None, :, None] * ce[:, :, None, None] + na[:, None, :, None] * cn[:, :, None, None] + es[:, None] * je[:, :, None, None] + ns[:, None] * jn[:, :, None, None]
    mu, sd = .8, .6
    obs, x = b['observed'].double(), b['x']
    field = mu / (sd * sd) * torch.einsum('bnf,af->ban', obs * x, engine.orient)
    fx = torch.einsum('ban,sn->bas', field, engine.spin) + (-.5 * ((x * x + mu * mu) / (sd * sd) + math.log(2 * math.pi * sd * sd)) * obs).sum((1, 2))[:, None, None]
    c = b['context']
    allowed = ((c[:, None] < 0) | (c[:, None] == engine.labels[None])).all(-1)
    score = logg[:, :, :, None, :] + fx[:, None, None] + mass.log()[:, :, None, None, None] + pri.log()[:, None, :, None, None] - math.log(len(engine.orient))
    score = score.masked_fill(~allowed[:, None, None, None], -torch.inf)
    weight = torch.logsumexp(score, -1).flatten(1).softmax(-1)
    jrel = e[:, None] * je[:, :, None, None] + ne[:, None] * jn[:, :, None, None]
    couplings = torch.einsum('bgre,hr->bghe', jrel, engine.roles).reshape(count, 6, 4, engine.nt, engine.nh)
    j = logg.new_zeros(count, 6, 4, engine.n, engine.n)
    j[:, :, :, :engine.nt, engine.nt:] = couplings
    j[:, :, :, engine.nt:, :engine.nt] = couplings.transpose(-1, -2)
    a = j[:, :, :, None].expand(-1, -1, -1, len(engine.orient), -1, -1).flatten(1, 3)
    u, t = torch.zeros_like(a), a.tanh()
    unary = field[:, None, None].expand(-1, 6, 4, -1, -1).flatten(1, 3)
    known, spin = (c >= 0)[:, None, :, None], (2 * c - 1)[:, None, :, None]
    for _ in range(64):
        cavity = (unary[..., None] + u.sum(-2)[..., :, None] - u.transpose(-1, -2)).tanh()
        cavity = torch.where(known, spin, cavity)
        u = .5 * u + .5 * torch.atanh((t * cavity).clamp(-1 + 1e-15, 1 - 1e-15))
    means = (unary + u.sum(-2)).tanh()
    means = torch.where((c >= 0)[:, None], (2 * c - 1)[:, None], means)
    prob = (means + 1) / 2
    return torch.einsum('bh,bhnc->bnc', weight, torch.stack((1 - prob, prob), -1)).cpu().numpy()


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(30):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(1000 * (time.perf_counter() - start))
    return {'median_ms': float(np.median(values)), 'samples_ms': values,
            'additional_working_mb': max(0, torch.cuda.max_memory_allocated() - before) / 2 ** 20}


def main():
    C.setup()
    torch.set_num_threads(2)
    state = lock()
    engine = NATIVE.GridReference('cuda')
    sample = rows('test')[::193]
    inputs = [read(r['path'])[0] for r in sample]
    with torch.inference_mode():
        expected = engine.infer(inputs)['grid6_bp']
        actual = bp_only(engine, inputs)
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-10, error
    row = next(r for r in rows('test') if r['case'] == 'matched' and r['coverage'] == 'disjoint' and r['context_total'] == 2 and r['observed_nodes'] == 6)
    inp, query, _ = read(row['path'])
    timings = []
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            model = make_model(arm, seed)
            saved = torch.load(ROOT / f'results/{arm}_s{seed}/checkpoint.pt', map_location='cpu', weights_only=False)['state_dict']
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(saved[name].to(p.device))
            def fn():
                given = inp
                if arm == 'pcfi_pfn':
                    given = dict(inp)
                    given['pcfi'] = pcfi(given)
                return predict(model, arm, given, query, 111)
            timings.append({'arm': arm, 'seed': seed, **measure(fn)})
            del model, saved
            gc.collect()
            torch.cuda.empty_cache()
    with torch.inference_mode():
        timings.append({'arm': 'grid6_bp', 'seed': 0, **measure(lambda: bp_only(engine, [inp]))})
        timings.append({'arm': 'grid6_exact', 'seed': 0, **measure(lambda: engine.infer([inp], with_bp=False))})
    report = {'device': torch.cuda.get_device_name(), 'measurements': timings, 'row': row, 'sources': state['sources'],
        'bp_only_equivalence_max_error': error, 'scope': 'Same-device visible input to probabilities after our workers finish. PCFI timing includes imputation. Grid6-BP skips unused exact query moments. Setup and bank I/O excluded; external GPU load uncontrolled.'}
    dump(ROOT / 'results/benchmark.json', report)
    decision = json.loads((ROOT / 'results/decision.json').read_text())
    factor = float(np.median([r['median_ms'] for r in timings if r['arm'] == 'factor_pfn']))
    baseline = next(r['median_ms'] for r in timings if r['arm'] == 'grid6_bp')
    quality = next(r['grid6_kl'] for r in json.loads((ROOT / 'results/summary.json').read_text()) if r['subset'] == 'primary' and r['arm'] == 'factor_pfn')
    decision.update(competitive_gate=bool(decision['mechanism_gate'] and quality <= .01 and factor <= baseline / 2),
        matched_grid6_kl=quality, factor_median_ms=factor, grid6_bp_median_ms=baseline)
    dump(ROOT / 'results/decision.json', decision)
    with (ROOT / 'REPORT_KO.md').open('a', encoding='utf-8') as f:
        f.write(f'\n## 완료 후 비용\n\nFactor median {factor:.3f}ms, Grid6-BP {baseline:.3f}ms. BP-only reference 예측 일치 최대오차 {error:.2g}. Competitive gate={decision["competitive_gate"]}. 외부 GPU 부하는 통제하지 않았다.\n')
    print('BENCHMARKED', decision, flush=True)


if __name__ == '__main__':
    main()
