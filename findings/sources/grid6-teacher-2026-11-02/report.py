"""Rescore all saved outputs with paired base-graph uncertainty, no model selection."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import torch
from pilot import ROOT, OLD, D, C, NATIVE, FactorPFN, cfg, dump, lock, read_test, test_rows

STUDENTS = ['mask_pfn_ce', 'pcfi_pfn_ce', 'factor_pfn_ce', 'mask_pfn_distill', 'factor_pfn_distill']
BASELINES = ['constant_factor', 'grid6_exact', 'grid6_bp', 'feature_only', 'lp', 'class_prior', 'uniform']


def kl(a, b):
    a, b = np.clip(a, 1e-10, 1), np.clip(b, 1e-10, 1)
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    return float((a * (np.log(a) - np.log(b))).sum(-1).mean())


def ci(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2026110251)
    means = x[rng.integers(len(x), size=(5000, len(x)))].mean(1)
    return {'difference': float(x.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(), 'graphs': len(x)}


def primary(row):
    return row['case'] == 'matched' and row['coverage'] == 'disjoint'


def reference(row):
    with np.load(ROOT / 'references' / (Path(row['path']).stem + '.npz')) as z:
        return {k: z[k] for k in z.files}


def baseline_predictions():
    folder = ROOT / 'evaluation/baselines'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'predictions.npz'
    if path.exists():
        with np.load(path) as z:
            return {key: z[key] for key in z.files}
    rr = test_rows()
    outputs = {arm: np.full((len(rr), 12, 2), .5, np.float64) for arm in BASELINES}
    model = FactorPFN(iterations=D.cfg()['bp_iterations'], constant=True).cuda().eval()
    for begin in range(0, len(rr), 32):
        chunk = rr[begin:begin + 32]
        items = [read_test(r) for r in chunk]
        with torch.inference_mode():
            fixed = model(D.tensors([item[0] for item in items])).cpu().numpy()
        for local, (r, item) in enumerate(zip(chunk, items)):
            inp, q, y = item
            ref = reference(r)
            a = inp['adj'].sum(0) + np.eye(12)
            a /= a.sum(-1, keepdims=True)
            lp = np.full((12, 2), .5)
            known = inp['context'] >= 0
            for _ in range(40):
                lp = a @ lp
                lp[known] = np.eye(2)[inp['context'][known]]
            prior = (np.bincount(inp['context'][known], minlength=2) + 1) / (known.sum() + 2)
            values = {**ref, 'constant_factor': fixed[local], 'feature_only': D.feature_only(inp),
                      'lp': lp, 'class_prior': np.tile(prior, (12, 1)), 'uniform': np.full((12, 2), .5)}
            for arm, p in values.items():
                outputs[arm][begin + local] = p
    np.savez_compressed(path, **outputs)
    dump(folder / 'completed.json', {'probability_sha256': D.sha(path), 'episodes': len(rr), 'arms': BASELINES})
    return outputs


def main():
    lock()
    torch.set_num_threads(2)
    rr, records, training = test_rows(), [], []
    for arm in STUDENTS:
        base = arm.replace('_distill', '').replace('_ce', '')
        for seed in cfg()['seeds']:
            folder = ROOT / 'evaluation' / f'{arm}_s{seed}'
            completion = json.loads((folder / 'completed.json').read_text())
            assert D.sha(folder / 'predictions.npz') == completion['prediction_sha256']
            with np.load(folder / 'predictions.npz') as z:
                probabilities, episode, mode = z['p'], z['episode'], z['mode']
            assert len(probabilities) == completion['records']
            assert np.isfinite(probabilities).all() and np.max(abs(probabilities.sum(-1) - 1)) < 1e-5
            src = ROOT / f'results/{base}_distill_s{seed}' if arm.endswith('_distill') else OLD / f'results/{base}_s{seed}'
            meta = json.loads((src / 'training.json').read_text())
            assert meta['updates'] == cfg()['updates'] and meta['examples'] == cfg()['train_episodes']
            assert D.sha(src / 'checkpoint.pt') == meta['checkpoint_sha256'] and meta['initial_sha256'] != meta['final_sha256']
            training.append({'evaluation_arm': arm, **meta})
            for index, (p, i, m) in enumerate(zip(probabilities, episode, mode)):
                row = rr[int(i)]
                inp, q, y = read_test(row)
                ref = reference(row)['grid6_exact']
                values = C.metrics(p[q], y)
                if m == 'standard':
                    values['grid6_kl'] = kl(ref[q], p[q])
                    free = ~inp['observed'][q].any(-1)
                    if free.any():
                        values.update({'query_no_feature_' + k: v for k, v in C.metrics(p[q][free], y[free]).items()})
                records.append({**row, 'arm': arm, 'training_seed': seed, 'mode': str(m), **values})
    baseline = baseline_predictions()
    for arm, probabilities in baseline.items():
        for row, p in zip(rr, probabilities):
            inp, q, y = read_test(row)
            ref = reference(row)['grid6_exact']
            free = ~inp['observed'][q].any(-1)
            extra = {'query_no_feature_' + k: v for k, v in C.metrics(p[q][free], y[free]).items()} if free.any() else {}
            records.append({**row, 'arm': arm, 'mode': 'standard', **C.metrics(p[q], y), 'grid6_kl': kl(ref[q], p[q]), **extra})
    for base in ['mask_pfn', 'factor_pfn']:
        for seed in cfg()['seeds']:
            a = next(t for t in training if t['evaluation_arm'] == base + '_distill' and t['seed'] == seed)
            b = next(t for t in training if t['evaluation_arm'] == base + '_ce' and t['seed'] == seed)
            for key in ['initial_sha256', 'frozen_sha256', 'schedule_sha256', 'trainable_parameters', 'examples', 'updates']:
                assert a[key] == b[key], (base, seed, key)
    def vector(arm, metric, select=primary, mode='standard'):
        groups = defaultdict(list)
        for r in records:
            if r['arm'] == arm and r['mode'] == mode and select(r) and metric in r:
                groups[r['graph']].append(r[metric])
        return {g: float(np.mean(values)) for g, values in sorted(groups.items())}
    def difference(a, b, metric, select=primary, mode='standard'):
        x, y = vector(a, metric, select, mode), vector(b, metric, select)
        assert set(x) == set(y) and x, (a, b, metric)
        return ci([x[g] - y[g] for g in x])
    selectors = [('primary', primary), ('overlap', lambda r: r['case'] == 'matched' and r['coverage'] == 'paired_overlap'),
                 ('density_low', lambda r: r['case'] == 'density_low' and r['coverage'] == 'disjoint'),
                 ('signal_high', lambda r: r['case'] == 'signal_high' and r['coverage'] == 'disjoint')]
    metrics = ['accuracy', 'nll', 'brier', 'grid6_kl']
    arms = STUDENTS + BASELINES
    summary = []
    for subset, select in selectors:
        for arm in arms:
            vv = {m: vector(arm, m, select) for m in metrics}
            summary.append({'subset': subset, 'arm': arm, 'graphs': len(vv['accuracy']),
                            **{m: float(np.mean(list(v.values()))) for m, v in vv.items()}})
    paired = []
    pairs = [('mask_pfn_distill', a) for a in ['mask_pfn_ce', 'pcfi_pfn_ce', 'factor_pfn_ce', 'constant_factor', 'grid6_bp', 'grid6_exact']]
    pairs += [('factor_pfn_distill', a) for a in ['factor_pfn_ce', 'constant_factor', 'grid6_bp', 'grid6_exact', 'mask_pfn_distill']]
    for subset, select in selectors:
        for a, b in pairs:
            for metric in metrics:
                paired.append({'subset': subset, 'a': a, 'b': b, 'metric': metric, **difference(a, b, metric, select)})
    interventions = []
    selected = lambda r: primary(r) and r['context_total'] == 2 and r['observed_nodes'] == 6
    for arm in STUDENTS:
        for mode in ['no_observed', 'shuffled', 'graph_permuted']:
            for metric in ['accuracy', 'nll', 'brier']:
                interventions.append({'arm': arm, 'mode': mode, 'metric': metric, **difference(arm, arm, metric, selected, mode)})
    query_free = []
    for arm in arms:
        query_free.append({'arm': arm, **{m: float(np.mean(list(vector(arm, 'query_no_feature_' + m).values()))) for m in ['accuracy', 'nll', 'brier']}})
    query_paired = []
    for a, b in pairs:
        for metric in ['accuracy', 'nll', 'brier']:
            query_paired.append({'a': a, 'b': b, 'metric': metric, **difference(a, b, 'query_no_feature_' + metric)})
    key = next(r for r in paired if r['subset'] == 'primary' and r['a'] == 'mask_pfn_distill' and r['b'] == 'mask_pfn_ce' and r['metric'] == 'grid6_kl')
    value = next(r['grid6_kl'] for r in summary if r['subset'] == 'primary' and r['arm'] == 'mask_pfn_distill')
    cost = json.loads((ROOT / 'results/benchmark.json').read_text()) if (ROOT / 'results/benchmark.json').exists() else None
    speed = cost['grid6_bp_median_ms'] / cost['mask_distill_median_ms'] if cost else None
    decision = {'primary_teacher_effect_gate': bool(key['ci95'][1] < 0), 'primary_grid6_kl': value,
                'primary_quality_gate': value <= .01, 'speed_ratio_grid6_over_mask': speed,
                'competitive_compression_gate': bool(key['ci95'][1] < 0 and value <= .01 and speed >= 2) if speed is not None else None,
                'primary_effect': key, 'scope': 'Exact known-family teacher on unchanged train tasks; fresh same-family test graphs. No method novelty or real/unseen-schema transfer claim.'}
    dump(ROOT / 'results/summary.json', summary)
    dump(ROOT / 'results/paired.json', paired)
    dump(ROOT / 'results/interventions.json', interventions)
    dump(ROOT / 'results/query_without_features.json', {'summary': query_free, 'paired': query_paired})
    dump(ROOT / 'results/decision.json', decision)
    dump(ROOT / 'results/training_audit.json', {'models_reevaluated': len(training), 'new_models_trained': 6,
         'new_successful_updates': sum(t['updates'] for t in training if t['evaluation_arm'].endswith('_distill')),
         'new_task_exposures': sum(t['examples'] for t in training if t['evaluation_arm'].endswith('_distill')),
         'paired_initialization_schedule_budget_and_parameters_match': True, 'training': training,
         'episode_metric_records_rescored': len(records)})
    dump(ROOT / 'results/metrics.json', records)
    def fdelta(r, percent=False):
        scale = 100 if percent else 1
        a, b = r['ci95']
        return f"{r['difference'] * scale:+.6f} [{a * scale:+.6f}, {b * scale:+.6f}]"
    lines = ['# Grid6 exact teacher — 새 disjoint bank 비교', '',
             '2026-11-02. 사전 primary는 공개 Mask GraphPFN의 exact-Grid6 predictive KL이다. 동일 생성 prior에서 sampled-label CE를 soft predictive KL로 바꿨다. Factor 학생은 보조 비교이고 primary 실패를 대체하지 않는다.', '',
             '## 실제 학습과 평가', '',
             '- 새 모델6개: Mask/Factor 각3 seeds. 각2048 기존 train tasks,256 successful updates. 원래 초기화·task order·optimizer·learning-rate schedule·trainable parameter 수를 CE arm과 일치시켰다.',
             '- 기존 CE Mask/PCFI/Factor9개 checkpoint를 새96 graph bank에서 재평가했다. 이전 학습 결과에 추가 학습한 것이 아니다. Train tasks는 재사용했고 test seed만 새로 만들었다.',
             '- Teacher는 실제 task parameter를 받지 않는 exact Grid6다. 네 role·여섯 parameter·네 feature orientation과 모든 미공개 node labels를 주변화한다. Student와 teacher는 graph/관측값/mask/context/family만 받는다.',
             '- 총1152 standard episodes. Primary는 matched48 graphs의 feature–label disjoint 조건이며 context 총1/2/4개, 관측노드3/6개를 graph 내 평균한다. 세 fixed training seed도 graph 내 평균하고5000 base-graph bootstrap을 사용한다.', '',
             '## Primary 결과', '', '| 방법 | Accuracy % | NLL | Brier | Grid6 exact KL |', '|---|---:|---:|---:|---:|']
    for arm in arms:
        r = next(r for r in summary if r['subset'] == 'primary' and r['arm'] == arm)
        lines.append(f"| {arm} | {100 * r['accuracy']:.4f} | {r['nll']:.6f} | {r['brier']:.6f} | {r['grid6_kl']:.6f} |")
    lines += ['', '## Paired 결과', '', '각 값은 distill−대조, 괄호는 paired95%CI다. Accuracy만 %p이며 나머지는 원 단위다. CI가0을 포함하는 것은 동등성이나 무효의 증명이 아니다.', '']
    for a, b in [('mask_pfn_distill', 'mask_pfn_ce'), ('factor_pfn_distill', 'factor_pfn_ce'), ('mask_pfn_distill', 'grid6_bp'), ('factor_pfn_distill', 'grid6_bp')]:
        lines.append(f'### {a} − {b}')
        lines.append('')
        for metric in metrics:
            r = next(r for r in paired if r['subset'] == 'primary' and r['a'] == a and r['b'] == b and r['metric'] == metric)
            lines.append(f'- {metric}: {fdelta(r, metric == "accuracy")}')
        lines.append('')
    lines += ['## Secondary와 intervention', '', '| 분할 | 학생 | Accuracy % | NLL | Grid6 KL |', '|---|---|---:|---:|---:|']
    for subset in ['overlap', 'density_low', 'signal_high']:
        for arm in ['mask_pfn_ce', 'mask_pfn_distill', 'factor_pfn_ce', 'factor_pfn_distill', 'grid6_bp']:
            r = next(r for r in summary if r['subset'] == subset and r['arm'] == arm)
            lines.append(f"| {subset} | {arm} | {100 * r['accuracy']:.4f} | {r['nll']:.6f} | {r['grid6_kl']:.6f} |")
    lines += ['', 'Query 자체도 feature가 없는 분할은 `results/query_without_features.json`에 별도로 기록했다. 아래 intervention은 matched disjoint context2/observed6의 보조 진단이다. Feature 제거/셔플은 새로운 재학습이나 선택된 hybrid가 아니다.', '']
    for arm in ['mask_pfn_distill', 'factor_pfn_distill']:
        for mode in ['no_observed', 'shuffled', 'graph_permuted']:
            r = next(r for r in interventions if r['arm'] == arm and r['mode'] == mode and r['metric'] == 'accuracy')
            lines.append(f'- {arm}, {mode}−standard accuracy: {fdelta(r, True)}%p')
    lines += ['', '## 비용과 판정', '', f"Primary teacher-effect gate={decision['primary_teacher_effect_gate']}; KL<=.01 gate={decision['primary_quality_gate']}; competitive compression gate={decision['competitive_compression_gate']}.", '']
    teacher_cost = json.loads((ROOT / 'teachers/manifest.json').read_text())['seconds']
    newcost = [t['seconds'] for t in training if t['evaluation_arm'].endswith('_distill')]
    lines.append(f'Teacher2048 tasks의 offline 생성 wall time={teacher_cost:.3f}s. 새6개 학생 학습 wall time 합={sum(newcost):.3f}s이며 일부 worker가 동시에 실행됐으므로 이 합은 전체 elapsed time이나 통제된 CE 대비 속도 비교가 아니다. Teacher 입력·출력 bank I/O를 포함한 준비 비용이고 배포시 teacher 호출은 없다.')
    if cost:
        lines.append(f"완료 후 동일GPU 단일 graph median: Mask distill{cost['mask_distill_median_ms']:.3f}ms, Factor distill{cost['factor_distill_median_ms']:.3f}ms, Grid6-BP{cost['grid6_bp_median_ms']:.3f}ms. Grid6/Mask 비율={speed:.3f}배. PCFI·baseline 구성 비용은 비교 범위에 따라 별도이며 teacher 생성 비용을 inference 절감에 숨기지 않는다. 외부GPU 부하는 통제하지 않았다.")
    lines += ['', '## 해석 범위', '',
              'Distillation 자체는 기존 기법이며 예전 feature-free JR에서도 시험했다. 이번의 다른 범위는 연속 Gaussian 관측과 label context가 분리된 과제에서, unknown-parameter six-support predictive를 직접 입력형 학생에 가르친다는 것이다. 이 단일 비교를 새로운 PFN 메소드의 novelty로 주장하지 않는다.',
              '정확한 matched prior predictive는 해당 family 안의 기준이다. Density/signal OOD에서 Grid6 KL은 Grid6 reference에 대한 일치도이며 참 OOD Bayes posterior의 KL이 아니다. Accuracy·NLL·Brier·teacher KL을 분리한다.',
              '12-node binary/2-relation/2-Gaussian-feature,50/75% whole-node missing에 한정한다. 실제 graph,99.5% missing, 새로운 feature law·semantic schema·큰 graph를 검증하지 않았다. Full-grid teacher의 exponential label enumeration을 확장 가능한 방법으로 부르지 않는다.',
              '고유 실패 원인은 이 실험만으로 식별하지 않는다. Teacher를 바꿔도 개선이 작으면 표현력·최적화·입력 경로·prior 중 무엇 때문인지 추가 분리 없이 확정할 수 없다. 결과에 맞춰 loss·temperature·학습량·primary를 바꾸지 않았다.']
    (ROOT / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('REPORTED', decision, flush=True)


if __name__ == '__main__':
    main()
