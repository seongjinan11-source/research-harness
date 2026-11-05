"""FP05 independent saved-array rescore and prespecified graph-level inference."""
import sklearn
from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np
from rl_util import ROOT, cfg, sha, dump, lock, metrics


def grouped_mean(rows, key, field='metrics'):
    groups = defaultdict(list)
    for r in rows:
        if r.get(field) is not None:
            groups[key(r)].append(r[field])
    return {k: {m: float(np.mean([v[m] for v in vs])) for m in vs[0]} for k, vs in groups.items()}


def interval(values, strata, index=0):
    ids = sorted(values); assert ids
    groups = defaultdict(list)
    for gid in ids:
        groups[tuple(strata[gid])].append(gid)
    rng = np.random.default_rng(cfg()['statistics']['seed'] + index)
    draws = []
    for _ in range(cfg()['statistics']['bootstrap']):
        selected = [gid for group in groups.values() for gid in rng.choice(group, len(group), replace=True)]
        draws.append(np.mean([values[gid] for gid in selected]))
    alpha = (1 - cfg()['statistics']['confidence']) / 2
    lo, hi = np.quantile(draws, [alpha, 1 - alpha])
    return dict(mean=float(np.mean(list(values.values()))), ci=[float(lo), float(hi)],
                n_graphs=len(ids), n_strata=len(groups), confidence=cfg()['statistics']['confidence'])


def main():
    locked = lock(); c = cfg(); all_rows = []; fits = []; evaluations = []; error = 0.; artifacts = {}
    canonical_rows = {split: {r['id']: r for r in json.loads((ROOT / 'bank' / split / 'manifest.json').read_text(encoding='utf8'))['episodes']}
                      for split in ['train', 'dev', 'final', 'real']}
    train_rows = list(canonical_rows['train'].values())
    full_indices = [i for i, r in enumerate(train_rows) if r['setting'] == 'full' and r['regime'] == 'natural']
    schedules = {}
    for arm, seed in itertools.product(c['arms'], c['seeds']):
        folder = ROOT / 'results' / f'{arm}_s{seed}'
        info = json.loads((folder / 'training.json').read_text(encoding='utf8'))
        assert sha(folder / 'checkpoint.pt') == info['checkpoint_sha256']
        assert sha(folder / 'schedule.json') == info['schedule_sha256']
        rng = np.random.default_rng(c['train']['schedule_seed'] + seed)
        expected = []
        for update in range(c['train']['updates']):
            idx = int(rng.choice(full_indices)) if update % 8 == 0 else int(rng.integers(len(train_rows)))
            expected.append(dict(index=idx, forward_seed=int(rng.integers(1, 2**30)), action_seed=int(rng.integers(1, 2**30))))
        actual = json.loads((folder / 'schedule.json').read_text())
        assert actual == expected
        progress = [json.loads(line) for line in (folder / 'progress.jsonl').read_text().splitlines()]
        assert len(progress) == c['train']['updates']
        for log, scheduled in zip(progress, expected):
            original = train_rows[scheduled['index']]
            assert log['episode'] == original['id'] and log['input_sha256'] == original['sha256']
            assert log['forward_seed'] == scheduled['forward_seed']
        assert info['updates'] == c['train']['updates'] and info['target_updates'] == 0
        if seed in schedules:
            assert schedules[seed] == info['schedule_sha256']
        schedules[seed] = info['schedule_sha256']; fits.append(info)
        for split in ['final', 'real']:
            path = folder / f'{split}_evaluation.json'
            record = json.loads(path.read_text(encoding='utf8'))
            assert len([r for r in record['records'] if r['variant'] == 'main']) == (96 * 11 if split == 'final' else 45)
            by_variant = defaultdict(list)
            for row in record['records']:
                by_variant[row['variant']].append(row['id'])
            for variant, ids in by_variant.items():
                assert len(ids) == len(set(ids)) and set(ids) == set(canonical_rows[split]), (arm, seed, split, variant)
            all_rows += record['records']; evaluations.append(dict(arm=arm, seed=seed, split=split,
                seconds=record['seconds'], peak_mib=record['peak_mib']))
            artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
        artifacts[(folder / 'checkpoint.pt').relative_to(ROOT).as_posix()] = info['checkpoint_sha256']
    for split in ['final', 'real']:
        all_rows += json.loads((ROOT / 'results' / f'{split}_references.json').read_text())['records']
    identity = {}; canonical = {}
    for r in all_rows:
        original_row = canonical_rows[r['split']][r['id']]
        for key in ['path', 'sha256', 'base', 'setting', 'intervention', 'standard_source_nodes']:
            assert r[key] == original_row[key], ('Canonical episode metadata mismatch', r['id'], key)
        path = ROOT / r['prediction']; assert sha(path) == r['prediction_sha256']
        artifacts[r['prediction']] = r['prediction_sha256']
        input_path = ROOT / r['path']; assert sha(input_path) == r['sha256']
        with np.load(path, allow_pickle=False) as d:
            with np.load(input_path, allow_pickle=False) as source:
                for saved, original in [('y', 'y'), ('query', 'query_ids'), ('context', 'context_ids'),
                                        ('yc', 'context_labels'), ('own_missing', 'standard_own_missing')]:
                    assert np.array_equal(d[saved], source[original]), ('Canonical bank mismatch', path, saved)
            p, y, own = d['p'], d['y'], d['own_missing'].astype(bool)
            met = metrics(p, y)
            for k in met:
                error = max(error, abs(met[k] - r['metrics'][k]))
            assert error < 1e-10
            mm = metrics(p[own], y[own]) if own.any() else None
            assert (mm is None) == (r['missing_metrics'] is None)
            if mm:
                for k in mm:
                    error = max(error, abs(mm[k] - r['missing_metrics'][k]))
            key = r['split'], str(r['base']), r['setting']
            current = tuple((k, d[k].tobytes()) for k in ['query', 'y', 'context', 'yc', 'own_missing'])
            if key in identity:
                assert identity[key] == current, ('Paired identity mismatch', key)
            identity[key] = current
            assert np.isfinite(p).all() and np.allclose(p.sum(-1), 1, atol=2e-5)
    assert error < 1e-10

    final = [r for r in all_rows if r['split'] == 'final']
    real = [r for r in all_rows if r['split'] == 'real']
    strata = {str(r['base']): (r['shape'], r['k'], r['regime']) for r in final}
    meta = {str(r['base']): r for r in final}
    # First fixed training seeds within physical case, then settings within base.
    case = grouped_mean(final, lambda r: (r['arm'], r['variant'], str(r['base']), r['setting'], r['intervention']), 'missing_metrics')
    whole = grouped_mean(final, lambda r: (r['arm'], r['variant'], str(r['base']), r['setting'], r['intervention']))
    bases = sorted(meta)
    primary = {}; gates = {}; graph_values = {}
    for arm in c['arms']:
        values = {}
        for gid in bases:
            available = [case[(arm, 'main', gid, rate, 'standard')]['accuracy']
                         for rate in c['primary_settings'] if (arm, 'main', gid, rate, 'standard') in case]
            if available:
                values[gid] = float(np.mean(available))
        graph_values[arm] = values
        primary[arm] = dict(accuracy=float(np.mean(list(values.values()))) if values else None, graphs=len(values))
    for j, control in enumerate(['fixed', 'direct']):
        valid = graph_values['group_pg'].keys() & graph_values[control].keys()
        values = {g: graph_values['group_pg'][g] - graph_values[control][g] for g in valid}
        result = interval(values, strata, j) if values else dict(n_graphs=0)
        result['verdict'] = 'PASS' if len(valid) == 96 and result['ci'][0] > 0 else 'FAIL' if len(valid) == 96 else 'UNDEFINED'
        gates['group_pg_minus_' + control] = result
    eligible = defaultdict(list)
    for r in final:
        if r['arm'] == 'group_pg' and r['seed'] == 11 and r['variant'] == 'main' and r['intervention'] == 'standard' and r['setting'] in c['primary_settings']:
            if r['regime'] == 'disjoint' and r['standard_source_nodes'] >= 2 and r['missing_metrics'] is not None:
                eligible[str(r['base'])].append(r['setting'])
    mechanism = {}
    for j, control in enumerate(['none', 'shuffle'], start=2):
        values = {}; did = {}
        for gid, rates in eligible.items():
            deltas = []; differences = []
            for rate in rates:
                delta = case[('group_pg', 'main', gid, rate, 'standard')]['accuracy'] - case[('group_pg', 'main', gid, rate, control)]['accuracy']
                delta_fixed = case[('fixed', 'main', gid, rate, 'standard')]['accuracy'] - case[('fixed', 'main', gid, rate, control)]['accuracy']
                deltas.append(delta); differences.append(delta - delta_fixed)
            values[gid] = float(np.mean(deltas)); did[gid] = float(np.mean(differences))
        if values:
            result = interval(values, strata, j); result['verdict'] = 'PASS' if result['ci'][0] > 0 else 'FAIL'
            gates['feature_utility_' + control] = result
            mechanism['feature_did_' + control] = interval(did, strata, j + 10)
        else:
            gates['feature_utility_' + control] = dict(verdict='UNDEFINED', n_graphs=0)
    guard = {}
    for gid in bases:
        if meta[gid]['regime'] == 'natural':
            # full_natural is actually all-query observed; full disjoint excluded.
            guard[gid] = whole[('group_pg', 'main', gid, 'full', 'standard')]['accuracy'] - whole[('fixed', 'main', gid, 'full', 'standard')]['accuracy']
    result = interval(guard, strata, 4); result['verdict'] = 'PASS' if result['ci'][0] > -.02 else 'FAIL'
    gates['full_natural_guard'] = result
    for j, arm in enumerate(['direct', 'group_pg'], start=20):
        values = {}
        for gid in bases:
            deltas = [case[(arm, 'main', gid, rate, 'standard')]['accuracy'] - case[(arm, 'replay000', gid, rate, 'standard')]['accuracy']
                      for rate in c['primary_settings'] if (arm, 'main', gid, rate, 'standard') in case]
            if deltas:
                values[gid] = float(np.mean(deltas))
        mechanism[arm + '_routing_minus_replay000'] = interval(values, strata, j)

    real_summary = []
    for key, vals in grouped_mean(real, lambda r: (r['arm'], r['variant'], r['graph'], r['setting'])).items():
        real_summary.append(dict(arm=key[0], variant=key[1], graph=key[2], setting=key[3], **vals))
    budget = dict(fits=len(fits), optimizer_groups=sum(r['updates'] for r in fits),
                  logical_trace_slots=sum(r['logical_trace_slots'] for r in fits),
                  train_seconds=sum(r['seconds'] for r in fits), fits_metadata=fits,
                  eval_seconds=sum(r['seconds'] for r in evaluations), evaluations=evaluations,
                  saved_records=len(all_rows), actual_evaluation_forwards=sum(r['forwards'] for r in all_rows),
                  actual_extra_adapter_pairs=sum(r['extra_adapter_pairs'] for r in all_rows))
    verdict = 'PASS' if all(r['verdict'] == 'PASS' for r in gates.values()) else 'FAIL'
    historical_path = ROOT.parent / 'observed_cell_cycle/results/summary.json'
    historical = json.loads(historical_path.read_text(encoding='utf8'))['external']
    final_secondary = [dict(arm=key[0], variant=key[1], setting=key[2], intervention=key[3], **vals)
                       for key, vals in grouped_mean(final, lambda r: (r['arm'], r['variant'], r['setting'], r['intervention'])).items()]
    summary = dict(cycle='FP05', gate=verdict, primary=primary, gates=gates,
                   mechanism=mechanism, utility=dict(planned_disjoint_graphs=48,
                   eligible_graphs=len(eligible), eligible_cases=sum(map(len, eligible.values()))),
                   final_secondary=final_secondary, real=real_summary, historical_real=historical,
                   historical_source=dict(path=str(historical_path), sha256=sha(historical_path),
                       scope='previously independently audited native/public references; representation differs; not new forwards'),
                   scope='restricted official synthetic prior; reused3 real development graphs; no target updates')
    dump(ROOT / 'results/audit.json', dict(status='PASS', max_rescore_error=error,
          records=len(all_rows), source_lock_sha256=sha(ROOT / 'results/lock.json'),
          paired_identity=True, matched_schedules=True, no_target_updates=True))
    dump(ROOT / 'results/artifact_manifest.json', artifacts)
    dump(ROOT / 'results/summary.json', summary); dump(ROOT / 'results/budget.json', budget)
    lines = ['# FP05: missing-graph computation policy', '',
        'Audit PASS; prespecified gate ' + verdict + '. Controller policy gradient + adapter CE; no new RL algorithm or heterogeneous claim.', '',
        '| Arm | Missing-query accuracy % | Base graphs |', '|---|---:|---:|']
    for arm, value in primary.items():
        label = f"{100*value['accuracy']:.3f}" if value['accuracy'] is not None else 'undefined'
        lines.append(f"| {arm} | {label} | {value['graphs']} |")
    lines += ['', '| Comparison | Difference pp | 97.5% CI pp | Verdict |', '|---|---:|---|---|']
    for name, value in gates.items():
        if 'mean' in value:
            lines.append(f"| {name} | {100*value['mean']:+.3f} | [{100*value['ci'][0]:+.3f}, {100*value['ci'][1]:+.3f}] | {value['verdict']} |")
        else:
            lines.append(f"| {name} | undefined | undefined | UNDEFINED |")
    lines += ['', 'Rates and fixed seeds are averaged inside base graph before stratified bootstrap. Utility is a separate eligible-disjoint estimand. See summary.json for all secondary mechanism tests and real results.', '',
              'Real graphs are reused development data; representation is fixed visible-row RP16→RMS→PCFI. Uniform inference uses8 traces; greedy policies1 trace. Historical native public baselines have different preprocessing/budget.', '',
              'No tuning after these locked results. Full posterior/Transformer RL training and new imputer claims are outside this pilot.']
    (ROOT / 'results/REPORT_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps(dict(audit='PASS', gate=verdict, primary=primary, gates=gates, records=len(all_rows))))


if __name__ == '__main__':
    main()
