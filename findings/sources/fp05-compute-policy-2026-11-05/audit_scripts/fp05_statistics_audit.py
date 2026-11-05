"""Independent CPU-only FP05 saved-probability audit; never imports rl_*.

Run after the root evaluation, references and locked analyzer have completed:
    python research_cycles/fp05_statistics_audit.py
Toy checks only, with no result reads/writes:
    python research_cycles/fp05_statistics_audit.py --selfcheck

The only created artifact is pfn_rl_cycle/results/statistics_audit.json, using
exclusive creation. No model/checkpoint is executed or deserialized.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

METRICS = ('accuracy', 'nll', 'brier', 'balanced_accuracy', 'macro_f1')
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / 'pfn_rl_cycle'
ANALYZER_SHA = '03343d1972ae2b65a06551d04eee98b43edfd232f5abcf4ab77d81dd4184fb04'
OFFSETS = {'group_pg_minus_fixed': 0, 'group_pg_minus_direct': 1,
           'feature_utility_none': 2, 'feature_utility_shuffle': 3,
           'full_natural_guard': 4, 'feature_did_none': 12, 'feature_did_shuffle': 13,
           'direct_routing_minus_replay000': 20, 'group_pg_routing_minus_replay000': 21}
VARIANTS = {'fixed': ('main',), 'uniform': ('main', 'global'),
            'direct': ('main', 'replay000'), 'group_pg': ('main', 'replay000', 'global')}
REFERENCES = ('frozen_pcfi', 'frozen_nan', 'prior', 'lp')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def score(probability, truth):
    """Confusion-matrix definitions, independent of sklearn/production helpers."""
    p = np.asarray(probability, np.float64)
    y = np.asarray(truth, np.int64)
    require(p.ndim == 2 and len(y) == len(p) and len(y) > 0, 'Invalid score axes/empty target')
    require(np.isfinite(p).all() and np.all(p >= 0), 'Invalid probability values')
    require(np.max(np.abs(p.sum(1) - 1)) < 2e-5, 'Probability rows not normalized')
    k = p.shape[1]
    require(np.all((y >= 0) & (y < k)), 'Truth outside supplied class vocabulary')
    p = np.clip(p, 1e-8, 1.)
    p /= p.sum(1, keepdims=True)
    pred = np.argmax(p, axis=1)
    table = np.bincount(k * y + pred, minlength=k*k).reshape(k, k)
    support, guessed, correct = table.sum(1), table.sum(0), np.diag(table)
    f1_den = support + guessed
    f1 = np.divide(2 * correct, f1_den, out=np.zeros(k, np.float64), where=f1_den > 0)
    brier = (p*p).sum(1) - 2*p[np.arange(len(y)), y] + 1
    return dict(accuracy=float(correct.sum()/len(y)),
                nll=float((-np.log(p[np.arange(len(y)), y])).mean()),
                brier=float(brier.mean()),
                balanced_accuracy=float((correct[support > 0]/support[support > 0]).mean()),
                macro_f1=float(f1.mean()))


def average(metrics):
    require(bool(metrics), 'Cannot average an empty metric set')
    return {k: float(np.mean([m[k] for m in metrics])) for k in METRICS}


def bootstrap(values, strata, seed, confidence=.975, repetitions=5000):
    """Match the locked analyzer's rep-major RNG order without calling it.

    Strata enter in order of their first sorted base ID, not lexical stratum
    order. RNG draws are one stratum at a time INSIDE each replicate. Reordering
    these draws would be a valid bootstrap but would not reproduce its exact CI.
    """
    ids = sorted(values)
    if not ids:
        return dict(mean=None, ci=None, n_graphs=0, n_strata=0, confidence=confidence)
    positions = {key:i for i, key in enumerate(ids)}
    groups = {}
    for key in ids:
        groups.setdefault(tuple(strata[key]), []).append(positions[key])
    x = np.asarray([values[key] for key in ids], np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty(repetitions, np.float64)
    for repeat in range(repetitions):
        selected = []
        for group in groups.values():
            sample = generator.integers(0, len(group), size=len(group))
            selected.extend(group[int(i)] for i in sample)
        draws[repeat] = x[selected].mean()
    tail = (1-confidence)/2
    return dict(mean=float(x.mean()), ci=np.quantile(draws, [tail, 1-tail]).tolist(),
                n_graphs=len(ids), n_strata=len(groups), confidence=confidence)


def selfcheck():
    p = np.array([[.75, .25, 0], [.25, .75, 0], [.5, .5, 0]])
    y = np.array([0, 1, 1])
    m = score(p, y)
    require(abs(m['accuracy']-2/3) < 1e-12, 'Accuracy toy failed')
    require(abs(m['balanced_accuracy']-.75) < 1e-12, 'Balanced accuracy toy failed')
    require(abs(m['macro_f1']-4/9) < 1e-12, 'Supplied-K macro-F1 toy failed')
    # Independent direct Brier expression after the declared clipping convention.
    clipped = np.clip(p, 1e-8, 1.); clipped /= clipped.sum(1, keepdims=True)
    require(abs(m['brier']-np.square(clipped-np.eye(3)[y]).sum(1).mean()) < 1e-12, 'Brier toy failed')
    values = {'g1':-.2, 'g2':.4, 'g3':.9}
    strata = {'g1':('b',), 'g2':('b',), 'g3':('a',)}
    actual = bootstrap(values, strata, 83, repetitions=29)
    oracle_rng = np.random.default_rng(83); oracle = []
    for _ in range(29):
        chosen = list(oracle_rng.choice(['g1', 'g2'], 2, replace=True))
        chosen += list(oracle_rng.choice(['g3'], 1, replace=True))
        oracle.append(np.mean([values[key] for key in chosen]))
    require(np.allclose(actual['ci'], np.quantile(oracle, [.0125, .9875]), atol=1e-15), 'Bootstrap RNG oracle failed')
    require(bootstrap({}, {}, 1)['ci'] is None, 'Empty interval toy failed')
    # Average two seeds within case, then eligible rates within graph: this is
    # deliberately different from pooling an unequal number of eligible cases.
    per_graph = [np.mean([.2, .4]), np.mean([.8])]
    require(abs(np.mean(per_graph)-.55) < 1e-12, 'Graph weighting toy failed')
    return dict(status='PASS', gpu_used=False, model_results_read=False, files_created=0)


def audit(root):
    root = Path(root).resolve()
    output = root / 'results/statistics_audit.json'
    require(not output.exists(), f'Refuse existing audit output: {output}')
    config = read_json(root/'protocol.json')
    locked = read_json(root/'results/lock.json')
    stats = config['statistics']
    require((stats['seed'], stats['confidence'], stats['bootstrap']) == (51079000, .975, 5000),
            'This independent recipe was reviewed for the locked FP05 statistics only')
    # Source is READ and hashed only; no production module is imported/executed.
    analyzer_text = (root/'rl_analyze.py').read_text(encoding='utf8')
    require(sha(root/'rl_analyze.py') == ANALYZER_SHA, 'Locked analyzer recipe changed; review audit recipe first')
    require('start=20' in analyzer_text and 'j + 10' in analyzer_text,
            'Expected replay/DiD offset recipe absent from locked analyzer')
    for path, digest in locked['sources'].items():
        require(sha(path) == digest, f'Locked source changed: {path}')
    for path, digest in locked['bank'].items():
        require(sha(path) == digest, f'Locked manifest changed: {path}')
    for path, digest in locked['checkpoints'].items():
        require(sha(path) == digest, f'Locked inherited checkpoint changed: {path}')
    require(sha(root/'results/checks.json') == locked['checks_sha256'], 'Checks receipt changed')
    manifests = {split: read_json(root/f'bank/{split}/manifest.json') for split in ('final', 'real')}
    bank = {}; lookup = {}; canonical = {}; artifacts = {}
    for split, manifest in manifests.items():
        require(len(manifest['episodes']) == (1056 if split == 'final' else 45), 'Bank count drift')
        lookup[split] = {row['id']:row for row in manifest['episodes']}
        require(len(lookup[split]) == len(manifest['episodes']), 'Duplicate canonical row ID')
        bank[split] = manifest['episodes']
        for row in bank[split]:
            source = root/row['path']
            require(sha(source) == row['sha256'], f'Canonical NPZ changed: {source}')
            with np.load(source, allow_pickle=False) as z:
                keys = ('y', 'query_ids', 'context_ids', 'context_labels', 'standard_own_missing',
                        'observed', 'classes', 'standard_source_nodes')
                a = {key:z[key].copy() for key in keys}
            require(int(a['classes']) == row['classes'], 'Canonical K mismatch')
            require(int(a['standard_source_nodes']) == row['standard_source_nodes'], 'Canonical source count mismatch')
            canonical[split, row['id']] = a
        for row in bank[split]:
            a = canonical[split, row['id']]
            if split == 'final':
                original = next(r for r in bank[split] if r['base']==row['base'] and
                                r['setting']==row['setting'] and r['intervention']=='standard')
                standard = canonical[split, original['id']]
            else:
                standard = a
            for key in ('y', 'query_ids', 'context_ids', 'context_labels', 'standard_own_missing'):
                require(np.array_equal(a[key], standard[key]), f'Intervention paired identity drift: {row["id"]}/{key}')
            own = ~standard['observed'][a['query_ids']].any(1)
            require(np.array_equal(own, a['standard_own_missing']), 'Standard subset drift')
            require(int(standard['observed'].any(1).sum()) == row['standard_source_nodes'], 'Eligibility drift')
    final_rows = bank['final']
    base_meta = {row['base']:row for row in final_rows if row['setting']=='full' and row['intervention']=='standard'}
    require(len(base_meta)==96, 'Final base count drift')
    strata = {b:(r['shape'], r['k'], r['regime']) for b,r in base_meta.items()}
    final_index = {(r['base'],r['setting'],r['intervention']):r for r in final_rows}
    require(len(final_index)==len(final_rows), 'Duplicate physical final case')
    valid_rates = {b:[rate for rate in config['primary_settings'] if
        canonical['final',final_index[b,rate,'standard']['id']]['standard_own_missing'].any()]
        for b in base_meta}
    eligible = {b:[rate for rate in valid_rates[b] if
        final_index[b,rate,'standard']['standard_source_nodes'] >= 2]
        for b in base_meta if base_meta[b]['regime']=='disjoint'}
    eligible = {b:rates for b,rates in eligible.items() if rates}
    natural = [b for b in base_meta if base_meta[b]['regime']=='natural']
    stratum_counts = defaultdict(int)
    utility_stratum_counts = defaultdict(int)
    for b in base_meta:
        stratum_counts[strata[b]] += 1
    for b in eligible:
        utility_stratum_counts[strata[b]] += 1
    require(len(stratum_counts)==12 and set(stratum_counts.values())=={8}, 'Primary12x8 stratum drift')
    require(len(utility_stratum_counts)==6 and set(utility_stratum_counts.values())=={8}, 'Utility6x8 stratum drift')
    require(sum(bool(v) for v in valid_rates.values())==96 and sum(map(len,valid_rates.values()))==192,
            'Primary eligibility differs from frozen input audit')
    require(len(eligible)==48 and sum(map(len,eligible.values()))==58, 'Utility48/58 denominator drift')
    require(len(natural)==48, 'Full-natural graph denominator drift')
    full_queries = 0
    for b in natural:
        row = next(r for r in final_rows if r['base']==b and r['setting']=='full' and r['intervention']=='standard')
        a = canonical['final',row['id']]
        require(a['observed'][a['query_ids']].all(), 'Full guard includes non-observed query')
        full_queries += len(a['query_ids'])
    require(full_queries==768, 'Full-natural target denominator drift')

    scored = []; errors = []; evaluations = []; checkpoint_hashes = {}
    def consume(record, split):
        require(record['split']==split, 'Evaluation record split mismatch')
        row = lookup[split][record['id']]
        for key in ('path', 'sha256', 'base', 'setting', 'intervention', 'standard_source_nodes'):
            require(record[key] == row[key], f'Canonical record metadata mismatch: {record["id"]}/{key}')
        a = canonical[split,row['id']]
        path = root/record['prediction']; digest = sha(path)
        require(digest == record['prediction_sha256'], f'Prediction hash mismatch: {path}')
        artifacts[record['prediction']] = digest
        with np.load(path, allow_pickle=False) as z:
            for saved, original in [('y','y'), ('query','query_ids'), ('context','context_ids'),
                                    ('yc','context_labels'), ('own_missing','standard_own_missing')]:
                require(np.array_equal(z[saved],a[original]), f'Canonical prediction array mismatch: {path}/{saved}')
            p = z['p']; y = z['y']; own = a['standard_own_missing'].astype(bool)
            require(p.shape == (len(y),row['classes']), f'Prediction class shape mismatch: {path}')
            all_metrics = score(p,y)
            missing_metrics = score(p[own],y[own]) if own.any() else None
        require((record['missing_metrics'] is None)==(missing_metrics is None), 'Missing-metric null mismatch')
        for key in METRICS:
            errors.append(abs(all_metrics[key]-record['metrics'][key]))
            if missing_metrics is not None:
                errors.append(abs(missing_metrics[key]-record['missing_metrics'][key]))
        scored.append(dict(split=split, row=row, arm=record['arm'], variant=record['variant'],
                           seed=record['seed'], all=all_metrics, missing=missing_metrics))

    for arm, seed in itertools.product(config['arms'],config['seeds']):
        folder = root/f'results/{arm}_s{seed}'
        info = read_json(folder/'training.json')
        require(info['arm']==arm and info['seed']==seed, 'Training identity mismatch')
        ck_sha = sha(folder/'checkpoint.pt')
        require(ck_sha==info['checkpoint_sha256'], 'Checkpoint receipt mismatch')
        checkpoint_hashes[f'{arm}_s{seed}'] = ck_sha
        for split in ('final','real'):
            path = folder/f'{split}_evaluation.json'; packet = read_json(path)
            evaluations.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path)))
            groups = defaultdict(list)
            for record in packet['records']:
                require(record['arm']==arm and record['seed']==seed, 'Evaluation arm/seed mismatch')
                require(record['checkpoint_sha256']==ck_sha, 'Prediction/checkpoint identity mismatch')
                groups[record['variant']].append(record['id'])
                consume(record,split)
            require(set(groups)==set(VARIANTS[arm]), 'Missing/extra evaluation variant')
            for variant, ids in groups.items():
                require(len(ids)==len(set(ids)) and set(ids)==set(lookup[split]), 'Incomplete/duplicate variant rows')
    for split in ('final','real'):
        path = root/f'results/{split}_references.json'; packet = read_json(path)
        evaluations.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path)))
        groups = defaultdict(list)
        for record in packet['records']:
            require(record['seed'] is None and record['variant']=='main', 'Reference pseudo-seed mismatch')
            groups[record['arm']].append(record['id']); consume(record,split)
        require(set(groups)==set(REFERENCES), 'Missing/extra frozen/simple reference arm')
        for ids in groups.values():
            require(len(ids)==len(set(ids)) and set(ids)==set(lookup[split]), 'Incomplete reference rows')
    require(max(errors,default=0) < 1e-10, 'Independent metric differs from saved evaluation')
    require(len(scored)==30828, 'Expected29568 final +1260 real saved records')

    # Each case is first averaged over the fixed training seeds. Reference cases
    # have one frozen prediction; they are never replicated as three new seeds.
    cases = defaultdict(list)
    for r in scored:
        key = (r['split'],r['arm'],r['variant'],r['row']['base'],r['row']['setting'],r['row']['intervention'])
        cases[key].append(r)
    case_metrics = {}
    for key, records in cases.items():
        arm = key[1]
        require(len(records)==(3 if arm in config['arms'] else 1), 'Case seed multiplicity mismatch')
        if arm in config['arms']:
            require({r['seed'] for r in records}==set(config['seeds']), 'Case fixed seed set mismatch')
        valid = [r['missing'] for r in records if r['missing'] is not None]
        require(len(valid) in (0,len(records)), 'Missing subset varied by seed')
        case_metrics[key] = dict(all=average([r['all'] for r in records]),
                                 missing=average(valid) if valid else None)
    def case(arm,variant,base,rate,intervention='standard',field='missing'):
        return case_metrics['final',arm,variant,base,rate,intervention][field]
    pairs = sorted({(key[1],key[2]) for key in case_metrics if key[0]=='final'})
    graph_metrics = {}; primary = {}
    for arm,variant in pairs:
        values = {b:average([case(arm,variant,b,rate) for rate in valid_rates[b]])
                  for b in sorted(base_meta) if valid_rates[b]}
        graph_metrics[arm,variant] = values
        primary[f'{arm}/{variant}'] = dict(metrics=average(list(values.values())),n_graphs=len(values))
    def interval(name,values,threshold=None):
        out = bootstrap(values,strata,stats['seed']+OFFSETS[name],stats['confidence'],stats['bootstrap'])
        if threshold is not None:
            out['verdict'] = 'PASS' if out['ci'] is not None and out['ci'][0]>threshold else 'FAIL' if values else 'UNDEFINED'
        out['graph_deltas'] = values
        return out
    gates = {}; mechanism = {}
    for other in ('fixed','direct'):
        name = 'group_pg_minus_'+other
        delta = {b:graph_metrics['group_pg','main'][b]['accuracy']-graph_metrics[other,'main'][b]['accuracy']
                 for b in sorted(base_meta)}
        gates[name] = interval(name,delta,0.)
    for control in ('none','shuffle'):
        delta = {}; did = {}
        for b,rates in eligible.items():
            pg = [case('group_pg','main',b,rate)['accuracy']-case('group_pg','main',b,rate,control)['accuracy'] for rate in rates]
            fixed = [case('fixed','main',b,rate)['accuracy']-case('fixed','main',b,rate,control)['accuracy'] for rate in rates]
            delta[b] = float(np.mean(pg)); did[b] = float(np.mean(np.asarray(pg)-np.asarray(fixed)))
        name = 'feature_utility_'+control; gates[name] = interval(name,delta,0.)
        name = 'feature_did_'+control; mechanism[name] = interval(name,did)
    guard = {b:case('group_pg','main',b,'full',field='all')['accuracy']-
               case('fixed','main',b,'full',field='all')['accuracy'] for b in natural}
    gates['full_natural_guard'] = interval('full_natural_guard',guard,-.02)
    for arm in ('direct','group_pg'):
        name = arm+'_routing_minus_replay000'
        delta = {b:float(np.mean([case(arm,'main',b,rate)['accuracy']-
                                  case(arm,'replay000',b,rate)['accuracy'] for rate in valid_rates[b]]))
                 for b in sorted(base_meta)}
        mechanism[name] = interval(name,delta)
    all_secondary = []; primary_subset_secondary = []; real_secondary = []
    for split in ('final','real'):
        groups_all = defaultdict(list); groups_primary = defaultdict(list)
        for key,mm in case_metrics.items():
            if key[0]!=split: continue
            _,arm,variant,base,setting,intervention=key
            if split=='final':
                group = (arm,variant,setting,intervention)
                groups_all[group].append(mm['all'])
                if setting in config['primary_settings'] and mm['missing'] is not None:
                    groups_primary[group].append(mm['missing'])
            else:
                graph = next(r['graph'] for r in bank['real'] if r['base']==base)
                groups_all[arm,variant,graph,setting].append(mm['all'])
        for key,values in sorted(groups_all.items()):
            if split=='final':
                all_secondary.append(dict(arm=key[0],variant=key[1],setting=key[2],intervention=key[3],
                                          n_graph_cases=len(values),metrics=average(values)))
            else:
                require(len(values)==3,'Real repetition count mismatch')
                real_secondary.append(dict(arm=key[0],variant=key[1],graph=key[2],setting=key[3],
                                           repetitions=3,metrics=average(values)))
        for key,values in sorted(groups_primary.items()):
            primary_subset_secondary.append(dict(arm=key[0],variant=key[1],setting=key[2],intervention=key[3],
                n_graph_cases=len(values),metrics=average(values),
                subset='original STANDARD own-missing query subset; all96, not conditional utility48/58'))

    # Compare with the root result, rather than invoking its implementation.
    reported_path = root/'results/summary.json'; reported = read_json(reported_path)
    differences = []
    def close(actual,expected,label):
        if actual is None or expected is None:
            require(actual is expected,'Null comparison mismatch: '+label);return
        error=float(np.max(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))))
        differences.append(error);require(error<1e-10,'Summary comparison mismatch: '+label)
    for name,actual in gates.items():
        expected=reported['gates'][name]
        for key in ('mean','ci','n_graphs','n_strata','confidence'):
            close(actual[key],expected[key],name+'/'+key)
        require(actual['verdict']==expected['verdict'],'Gate verdict mismatch: '+name)
    for name,actual in mechanism.items():
        for key in ('mean','ci','n_graphs','n_strata','confidence'):
            close(actual[key],reported['mechanism'][name][key],name+'/'+key)
    for arm in config['arms']:
        close(primary[arm+'/main']['metrics']['accuracy'],reported['primary'][arm]['accuracy'],'primary/'+arm)
    require(reported['utility']['eligible_graphs']==48 and reported['utility']['eligible_cases']==58,'Reported utility denominator drift')
    reported_secondary={(r['arm'],r['variant'],r['setting'],r['intervention']):r for r in reported['final_secondary']}
    for r in all_secondary:
        ref=reported_secondary[r['arm'],r['variant'],r['setting'],r['intervention']]
        for key in METRICS:close(r['metrics'][key],ref[key],'all-query secondary/'+key)
    reported_real={(r['arm'],r['variant'],r['graph'],r['setting']):r for r in reported['real']}
    for r in real_secondary:
        ref=reported_real[r['arm'],r['variant'],r['graph'],r['setting']]
        for key in METRICS:close(r['metrics'][key],ref[key],'real secondary/'+key)
    verdict='PASS' if all(g['verdict']=='PASS' for g in gates.values()) else 'FAIL'
    require(verdict==reported['gate'],'Composite verdict mismatch')
    digest=hashlib.sha256()
    for path,value in sorted(artifacts.items()):
        digest.update((path+'\0'+value+'\n').encode('utf8'))
    result=dict(status='PASS',cycle='FP05',scientific_verdict=verdict,
        auditor=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),production_modules_called=False,
                     gpu_used=False,metrics='independent confusion matrix, direct NLL and Brier'),
        recipe=dict(analyzer_sha256=ANALYZER_SHA,confidence=stats['confidence'],bootstrap=stats['bootstrap'],
                    seed=stats['seed'],offsets=OFFSETS,rng_order='replicate then first-sorted-base stratum',
                    unit='fixed train seeds -> eligible rates -> equally weighted base graphs; within-stratum bootstrap'),
        identity=dict(lock_sha256=sha(root/'results/lock.json'),summary_sha256=sha(reported_path),
                      records=len(scored),final_records=sum(r['split']=='final' for r in scored),
                      real_records=sum(r['split']=='real' for r in scored),
                      canonical_bank_rows={s:len(rows) for s,rows in bank.items()},
                      prediction_index_sha256=digest.hexdigest(),checkpoint_sha256=checkpoint_hashes,
                      evaluation_indices=evaluations,max_saved_metric_error=max(errors,default=0),
                      max_summary_comparison_error=max(differences,default=0)),
        eligibility=dict(primary_graphs=96,primary_cases=192,primary_strata=12,
                         utility_graphs=48,utility_cases=58,utility_strata=6,
                         utility_rates_by_base=eligible,
                         utility_cases_by_rate={rate:sum(rate in rates for rates in eligible.values())
                                                for rate in config['primary_settings']},
                         strata=[dict(shape=s[0],k=s[1],regime=s[2],primary_graphs=n,
                                      utility_graphs=utility_stratum_counts.get(s,0))
                                 for s,n in sorted(stratum_counts.items())],
                         full_natural_graphs=48,full_natural_queries=768),
        primary=primary,gates=gates,mechanism=mechanism,
        secondary_all_query=all_secondary,
        secondary_primary_same_missing_subset=primary_subset_secondary,
        real_descriptive_no_ci=real_secondary,
        interpretation='Audit PASS means saved metrics/paired statistics agree. Scientific gate may FAIL. '
                       'No four-way simultaneous95% claim; real graphs are reused development; '
                       'all-query secondary is distinct from original missing-query and conditional utility estimands.')
    serialized = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with output.open('x',encoding='utf8',newline='\n') as stream:
        stream.write(serialized)
    return dict(status='PASS',scientific_verdict=verdict,records=len(scored),output=str(output),
                max_saved_metric_error=max(errors,default=0),max_summary_comparison_error=max(differences,default=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--selfcheck',action='store_true')
    args=parser.parse_args()
    try:
        result=selfcheck() if args.selfcheck else audit(args.root)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print(json.dumps(dict(status='ERROR',error=str(exc),output_created=False)),file=sys.stderr)
        return 1
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':
    raise SystemExit(main())
