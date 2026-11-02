"""Graph-paired primary accuracy and separate probabilistic-quality analysis."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import torch
from data import ROOT, C, cfg, dump, sha, read, rows, tensors, feature_only
from model import FactorPFN


def primary(r):
    return r['case'] == 'matched' and r['coverage'] == 'disjoint'


def ci(values):
    x = np.asarray(values, float)
    rng = np.random.default_rng(2026110217)
    draws = x[rng.integers(len(x), size=(5000, len(x)))].mean(1)
    return {'difference': float(x.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(), 'graphs': len(x)}


def kl(a, b):
    a, b = np.clip(a, 1e-10, 1), np.clip(b, 1e-10, 1)
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    return float((a * (np.log(a) - np.log(b))).sum(-1).mean())


def baseline_records():
    torch.set_num_threads(2)
    model = FactorPFN(iterations=cfg()['bp_iterations'], constant=True).cuda()
    model.eval()
    test = rows('test')
    records = []
    folder = ROOT / 'results/baseline_predictions'
    folder.mkdir(exist_ok=True)
    for start in range(0, len(test), 32):
        chunk = test[start:start + 32]
        values = [read(r['path']) for r in chunk]
        with torch.inference_mode():
            fixed = model(tensors([v[0] for v in values])).cpu().numpy()
        for i, r in enumerate(chunk):
            inp, query, y = values[i]
            with np.load(ROOT / 'references' / (Path(r['path']).stem + '.npz')) as z:
                ref = {key: z[key][query] for key in z.files}
            known = inp['context'] >= 0
            a = inp['adj'].sum(0) + np.eye(len(inp['x']))
            a /= a.sum(-1, keepdims=True)
            lp = np.full((len(a), 2), .5)
            for _ in range(40):
                lp = a @ lp
                lp[known] = np.eye(2)[inp['context'][known]]
            prior = (np.bincount(inp['context'][known], minlength=2) + 1) / (known.sum() + 2)
            predictions = {**ref, 'constant_factor': fixed[i, query], 'feature_only': feature_only(inp)[query],
                           'lp': lp[query], 'class_prior': np.tile(prior, (len(query), 1)), 'uniform': np.full((len(query), 2), .5)}
            path = folder / f'{start + i}.npz'
            np.savez_compressed(path, **predictions, y=y)
            for arm, prob in predictions.items():
                free = ~inp['observed'][query].any(-1)
                extra = {'query_no_feature_' + m: v for m, v in C.metrics(prob[free], y[free]).items()} if free.any() else {}
                records.append({**r, 'arm': arm, 'seed': 0, 'mode': 'standard', **C.metrics(prob, y), **extra,
                    'grid6_kl': kl(ref['grid6_exact'], prob), 'prediction': path.relative_to(ROOT).as_posix()})
    dump(ROOT / 'results/baselines.json', records)
    return records


def main():
    c = cfg()
    records = baseline_records()
    training = []
    files = 0
    for arm in c['arms']:
        for seed in c['seeds']:
            folder = ROOT / 'results' / f'{arm}_s{seed}'
            meta = json.loads((folder / 'training.json').read_text())
            assert meta['updates'] == c['updates'] and meta['examples'] == c['train_episodes'] and meta['frozen_weights_unchanged']
            assert meta['checkpoint_sha256'] == sha(folder / 'checkpoint.pt')
            assert meta['final_sha256'] != meta['initial_sha256']
            training.append(meta)
            for r in json.loads((folder / 'evaluation.json').read_text()):
                inp, query, y = read(r['path'])
                with np.load(ROOT / r['prediction']) as z:
                    prob = z['p']
                    assert np.array_equal(z['y'], y)
                for m, value in C.metrics(prob, y).items():
                    assert abs(value - r[m]) < 1e-9, (arm, seed, r['prediction'], m)
                if r['mode'] == 'standard':
                    with np.load(ROOT / 'references' / (Path(r['path']).stem + '.npz')) as z:
                        r['grid6_kl'] = kl(z['grid6_exact'][query], prob)
                    free = ~inp['observed'][query].any(-1)
                    if free.any():
                        r.update({'query_no_feature_' + m: v for m, v in C.metrics(prob[free], y[free]).items()})
                records.append(r)
                files += 1
    for seed in c['seeds']:
        mm = [r for r in training if r['seed'] == seed]
        assert len({r['schedule_sha256'] for r in mm}) == 1
        assert len({r['initial_sha256'] for r in mm if r['arm'] != 'factor_pfn'}) == 1
    def vector(arm, metric, select=primary, mode='standard'):
        groups = defaultdict(list)
        for r in records:
            if r['arm'] == arm and r['mode'] == mode and select(r) and metric in r:
                groups[r['graph']].append(r[metric])
        return {g: float(np.mean(rr)) for g, rr in sorted(groups.items())}
    metrics = ['accuracy', 'nll', 'brier', 'grid6_kl']
    arms = c['arms'] + ['constant_factor', 'grid6_exact', 'grid6_bp', 'feature_only', 'lp', 'class_prior', 'uniform']
    summary = []
    for subset, selector in [('primary', primary), ('overlap', lambda r: r['case'] == 'matched' and r['coverage'] == 'paired_overlap'),
                             ('density_low', lambda r: r['case'] == 'density_low' and r['coverage'] == 'disjoint'),
                             ('signal_high', lambda r: r['case'] == 'signal_high' and r['coverage'] == 'disjoint')]:
        for arm in arms:
            values = {m: vector(arm, m, selector) for m in metrics}
            summary.append({'subset': subset, 'arm': arm, 'graphs': len(values['accuracy']), **{m: float(np.mean(list(v.values()))) for m, v in values.items()}})
    paired = {}
    for baseline in [a for a in arms if a != 'factor_pfn']:
        for metric in metrics:
            a, b = vector('factor_pfn', metric), vector(baseline, metric)
            assert set(a) == set(b)
            paired[f'factor_pfn-{baseline}/{metric}'] = ci([a[g] - b[g] for g in a])
    intervention_selector = lambda r: primary(r) and r['context_total'] == 2 and r['observed_nodes'] == 6
    interventions = []
    for arm in c['arms']:
        for mode in ['no_observed', 'shuffled', 'graph_permuted'] + (['feature_off', 'feedback_off'] if arm == 'factor_pfn' else []):
            for metric in ['accuracy', 'nll', 'brier']:
                a, b = vector(arm, metric, intervention_selector, mode), vector(arm, metric, intervention_selector)
                assert set(a) == set(b)
                interventions.append({'arm': arm, 'mode': mode, 'metric': metric, **ci([a[g] - b[g] for g in a])})
    query_free = []
    for arm in arms:
        query_free.append({'arm': arm, **{m: float(np.mean(list(vector(arm, 'query_no_feature_' + m).values()))) for m in ['accuracy', 'nll', 'brier']}})
    neural = all(paired[f'factor_pfn-{a}/accuracy']['ci95'][0] > 0 for a in ['mask_pfn', 'pcfi_pfn'])
    removal = next(r for r in interventions if r['arm'] == 'factor_pfn' and r['mode'] == 'no_observed' and r['metric'] == 'accuracy')['ci95'][1] < 0
    potential = paired['factor_pfn-constant_factor/accuracy']['ci95'][0] > 0
    decision = {'mechanism_gate': bool(neural and removal), 'neural_accuracy_gate': bool(neural), 'feature_removal_gate': bool(removal),
        'learned_potential_accuracy_increment': bool(potential), 'competitive_gate': None, 'scope': 'Fresh known-family disjoint test. Fixed seeds; secondary partitions separate; no novelty claim.'}
    dump(ROOT / 'results/summary.json', summary)
    dump(ROOT / 'results/paired.json', paired)
    dump(ROOT / 'results/interventions.json', interventions)
    dump(ROOT / 'results/query_without_features.json', query_free)
    dump(ROOT / 'results/decision.json', decision)
    dump(ROOT / 'results/audit.json', {'models': len(training), 'successful_updates': sum(r['updates'] for r in training),
        'task_exposures': sum(r['examples'] for r in training), 'rescored_probability_files': files, 'schedule_matched_per_seed': True,
        'public_initializations_matched_per_seed': True, 'frozen_weights_unchanged': True, 'checkpoint_digests_checked': True, 'training': training})
    lines = ['# Disjoint feature–label factor PFN 결과', '', '**새 모델 학습을 포함한 소형 known-family pilot.** 최종 판정은 아래 gate와 별도 비용 결과에 한정한다.', '',
        'Primary: 새 matched48 graphs, feature 노드와 label context의 교집합0, context1/2/4·관측노드3/6을 graph 내 평균, fixed3 seeds 평균. Accuracy가 primary이며 NLL/Brier와 Grid6 KL은 별도다.', '',
        '| 방법 | Accuracy % | NLL | Brier | Grid6 exact KL |', '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['subset'] == 'primary':
            lines.append(f"| {r['arm']} | {100*r['accuracy']:.4f} | {r['nll']:.6f} | {r['brier']:.6f} | {r['grid6_kl']:.6f} |")
    lines += ['', '## Paired 차이', '', '| Factor PFN − 대조 | Accuracy %p [95% CI] | NLL [95% CI] |', '|---|---|---|']
    for baseline in ['mask_pfn', 'pcfi_pfn', 'constant_factor', 'grid6_bp']:
        a, n = paired[f'factor_pfn-{baseline}/accuracy'], paired[f'factor_pfn-{baseline}/nll']
        lines.append(f"| {baseline} | {100*a['difference']:+.4f} [{100*a['ci95'][0]:+.4f}, {100*a['ci95'][1]:+.4f}] | {n['difference']:+.6f} [{n['ci95'][0]:+.6f}, {n['ci95'][1]:+.6f}] |")
    lines += ['', '## 판정과 범위', '', f'Neural accuracy gate={neural}; feature-removal gate={removal}; mechanism gate={neural and removal}; learned potential accuracy increment={potential}.', '',
        'Feature latent state와 node class 사이의 관측값 factor를 통해 두 evidence source를 연결한다. PCFI처럼 완성 feature 행렬을 만들지 않으며 별도 role posterior head가 없다. 다만 CSBM/attributed-graph BP 자체는 선행연구가 있고, 고정2-feature·2-relation·binary family의 결과를 새 schema나 실제 graph 전이로 확대하지 않는다.', '',
        'Factor 모델은 finite loopy BP와 Bethe score를 사용하는 conditional structured model이다. 원래 데이터 posterior의 exact 추론이나 연속 X의 정규화된 생성 likelihood를 학습한 것은 아니다. Feature-return 차단은 고정된 계산을 깨는 보조 개입이며 새 방법으로 선택하지 않는다.', '',
        f"실제 {len(training)} models, {sum(r['updates'] for r in training)} successful updates, {files} saved predictions 재채점. Query 정답은 query CE 및 평가에만 사용한다. Test-side parameter tuning 없음.", '',
        '세부: protocol.json, results/summary.json, paired.json, interventions.json, query_without_features.json, audit.json. 비용·competitive gate는 별도 benchmark 후 확정한다.']
    (ROOT / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('ANALYZED', decision, flush=True)


if __name__ == '__main__':
    main()
