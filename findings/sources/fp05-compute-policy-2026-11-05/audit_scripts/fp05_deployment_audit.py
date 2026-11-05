"""Read-only FP05 CPU artifact audit. Run ONLY after all fits/evaluations finish.

Usage: <research-python> -B -X utf8 research_cycles/fp05_deployment_audit.py
Output: pfn_rl_cycle/results/deployment_audit.json, create-only, PASS or FAIL.
No scientific forward, optimizer, GPU, source/bank/lock edits, or score-based
method choices. CPU construction of Model is used only for parameter digests.

Tolerances: saved metrics 1e-10 absolute; probabilities sum 2e-5; direct leaf
sum 2e-4 (same as locked runner); exp(logp) 3e-7+2e-6 relative; scalar direct
actor/CE 3e-5*(1+abs(expected)). PG rewards are rebuilt as FP32, population SD
checked within 2e-6*(1+maxabsreward). PG actor tolerance also includes a bounded
FP32 reduction term proportional to eps*maxabsreward/(SD+1e-6), since group
centering near zero variance amplifies float32 summation differences. It is
reported per group, not silently treated as exact float64 equality.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This process must have no visible GPU.
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows DLL import order.
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

WORK = Path(__file__).resolve().parent.parent
ROOT = WORK / 'pfn_rl_cycle'
TRACES = tuple(itertools.product((0, 1), repeat=3))
VARIANTS = {'fixed': ('main',), 'uniform': ('main', 'global'),
            'direct': ('main', 'replay000'), 'group_pg': ('main', 'replay000', 'global')}
REFS = ('frozen_pcfi', 'frozen_nan', 'prior', 'lp')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, atol=1e-10, rtol=0., name='scalar'):
    a, b = float(actual), float(expected)
    require(np.isfinite(a) and np.isfinite(b), f'{name}: nonfinite {a}/{b}')
    error = abs(a-b)
    require(error <= atol + rtol*abs(b), f'{name}: {a} vs {b}; error={error}, tol={atol+rtol*abs(b)}')
    return error


def artifact(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), f'artifact escapes cycle: {relative}')
    return path


def parameter_digest(module, frozen_only=False):
    """Independent implementation of the locked name/dtype/shape/bytes format."""
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        if frozen_only and p.requires_grad:
            continue
        h.update(name.encode()); h.update(str(p.dtype).encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def independent_metrics(probability, truth):
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(truth, dtype=np.int64)
    require(p.ndim == 2 and len(p) == len(y) and len(y) > 0, 'invalid/empty prediction shape')
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1.00002).all(), 'invalid probability range')
    require(np.max(np.abs(p.sum(1)-1)) <= 2e-5, 'unnormalized probabilities')
    require(y.ndim == 1 and y.min() >= 0 and y.max() < p.shape[1], 'truth outside supplied vocabulary')
    p = np.clip(p, 1e-8, 1.)
    p /= p.sum(1, keepdims=True)
    prediction = p.argmax(1)
    recalls, f1 = [], []
    for c in range(p.shape[1]):
        actual, guessed = y == c, prediction == c
        tp = int(np.sum(actual & guessed))
        if actual.any():
            recalls.append(tp / int(actual.sum()))
        denominator = int(actual.sum()+guessed.sum())
        f1.append(2*tp/denominator if denominator else 0.)
    return dict(accuracy=float(np.mean(prediction == y)),
                nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                brier=float(np.mean(np.sum(p*p, axis=1)-2*p[np.arange(len(y)), y]+1)),
                balanced_accuracy=float(np.mean(recalls)), macro_f1=float(np.mean(f1)))


def compare_metrics(got, recorded, name):
    require(set(got) == set(recorded), f'{name}: metric names differ')
    return max(near(got[k], recorded[k], name=f'{name}/{k}') for k in got)


def check_trace(trace):
    trace = tuple(trace)
    require(len(trace) == 3 and all(x in (0, 1) for x in trace), f'invalid trace: {trace}')
    return trace


class Audit:
    def __init__(self):
        self.config = read_json(ROOT / 'protocol.json')
        self.design = read_json(ROOT / 'results/design.json')
        self.lock = read_json(ROOT / 'results/lock.json')
        self.lock_sha = sha(ROOT / 'results/lock.json')
        self.input_cache = {}
        self.case = {}
        self.fits = {}
        self.calibration = {}
        self.source_before = {}
        self.deployment_forwards = self.deployment_extra = 0
        self.calibration_forwards = self.calibration_extra = 0
        self.calibration_seconds = self.evaluation_seconds = self.reference_record_seconds = 0.
        self.metric_error = self.actor_error = self.pg_actor_tolerance = 0.
        self.prediction_count = 0
        self.training_cost = dict(successful_groups=0, logical_slots=0, successful_forwards=0,
             retry_forwards=0, successful_transformer_backwards=0, retry_transformer_backwards=0,
             successful_extra_pairs=0, retry_extra_pairs_lower=0, retry_extra_pairs_upper=0,
             retry_groups=0, retry_attempts=0, wall_seconds=0.)
        self.report = dict(cycle='FP05', status='RUNNING', scientific_forwards=0,
             optimizer_updates_performed_by_audit=0, gpu_used=False,
             script_sha256=sha(__file__), python=sys.version, torch=torch.__version__, numpy=np.__version__,
             tolerances=dict(metric_absolute=1e-10, probability_rowsum=2e-5, leaf_sum=2e-4,
                 exp_logp_absolute=3e-7, exp_logp_relative=2e-6,
                 direct_scalar='3e-5*(1+abs(expected))',
                 pg_sd='2e-6*(1+maxabsreward)',
                 pg_actor='3e-5+16*float32_eps*max(1,maxabsreward)*max(1,meanabslogp)/(recorded_sd+1e-6)'),
             limitations=[
                 'No forward or training replay: logged PG conditional probabilities cannot be rederived without per-update states/checkpoints.',
                 'Calibration association is tied to its immutable fit folder/checkpoint; calibration records do not themselves store checkpoint SHA.',
                 'Failed retry extra-adapter counts are bounded where retry traces were not saved; actual forward/backward counts are exact.',
                 'Reference wall times exclude unrecorded prior/LP CPU time; reported wall sums are component totals, not critical-path time.',
                 'Recorded training/evaluation wall times are actual workload records under possible concurrent CPU audits/hash work, not controlled robust latency benchmarks.'])

    def anchors(self):
        require(self.design['sources'] == self.lock['sources'], 'design/scientific source sets differ')
        for path, expected in self.lock['sources'].items():
            require(sha(path) == expected, f'locked source changed: {path}')
            self.source_before[path] = expected
        require(self.design['checkpoints'] == self.lock['checkpoints'], 'public checkpoint anchors differ')
        for path, expected in self.lock['checkpoints'].items():
            require(sha(path) == expected, f'public checkpoint changed: {path}')
        require(sha(ROOT/'results/checks.json') == self.lock['checks_sha256'] == self.design['checks_sha256'], 'checks hash changed')
        require(read_json(ROOT/'results/checks.json')['status'] == 'PASS', 'checks not PASS')
        self.manifests, self.rows = {}, {}
        for split in ('train', 'dev', 'final', 'real'):
            path = ROOT/'bank'/split/'manifest.json'
            expected = self.lock['bank'][str(path)] if str(path) in self.lock['bank'] else self.lock['bank'][str(path.resolve())]
            require(sha(path) == expected, f'{split} manifest differs from lock')
            manifest = read_json(path); rows = manifest['episodes']
            require(len(rows) == len({r['id'] for r in rows}), f'duplicate {split} episode IDs')
            self.manifests[split], self.rows[split] = manifest, {r['id']: r for r in rows}
            for row in rows:
                require(sha(artifact(row['path'])) == row['sha256'], f'input hash {split}/{row["id"]}')
        expected_counts = dict(train=self.config['train']['graphs']*7,
                               dev=self.config['dev']['graphs']*7,
                               final=self.config['final']['graphs']*11, real=45)
        require({s:len(r) for s,r in self.rows.items()} == expected_counts, 'bank cardinality mismatch')
        self.report['anchors'] = dict(design_sha256=sha(ROOT/'results/design.json'), lock_sha256=self.lock_sha,
            sources=len(self.source_before), public_checkpoints=self.lock['checkpoints'], bank_counts=expected_counts)
        self.anchor_files={str(ROOT/'results/design.json'):sha(ROOT/'results/design.json'),
                           str(ROOT/'results/lock.json'):self.lock_sha}
        self.anchor_files.update(self.lock['checkpoints'])
        self.anchor_files.update(self.lock['bank'])
        required=[]
        for arm,seed in itertools.product(self.config['arms'],self.config['seeds']):
            folder=ROOT/'results'/f'{arm}_s{seed}'
            required += [folder/name for name in ('training.json','checkpoint.pt','schedule.json','progress.jsonl','final_evaluation.json','real_evaluation.json')]
            if arm in ('uniform','group_pg'):required.append(folder/'global_trace.json')
        required += [ROOT/'results'/f'{split}_references.json' for split in ('final','real')]
        require(all(p.exists() for p in required), 'Audit requires completed artifacts; missing: '+', '.join(str(p) for p in required if not p.exists()))

    def canonical(self, split, row_id):
        key = split, row_id
        if key in self.input_cache:
            return self.input_cache[key]
        row = self.rows[split][row_id]
        with np.load(artifact(row['path']), allow_pickle=False) as z:
            d = {k:z[k].copy() for k in ('y', 'query_ids', 'context_ids', 'context_labels',
                     'standard_own_missing', 'observed', 'classes', 'standard_source_nodes')}
        d['classes'] = int(d['classes']); d['standard_source_nodes'] = int(d['standard_source_nodes'])
        require(d['y'].shape == d['query_ids'].shape == d['standard_own_missing'].shape, 'canonical query shapes')
        require(d['context_ids'].shape == d['context_labels'].shape, 'canonical context shapes')
        require(not np.intersect1d(d['query_ids'], d['context_ids']).size, 'canonical query/context overlap')
        require(row['standard_source_nodes'] == d['standard_source_nodes'], 'standard source metadata mismatch')
        if row['intervention'] == 'standard':
            expected_own = ~d['observed'][d['query_ids']].any(1)
            require(np.array_equal(expected_own, d['standard_own_missing']), 'standard missing-query membership wrong')
            require(int(d['observed'].any(1).sum()) == d['standard_source_nodes'], 'standard source count wrong')
        else:
            standard_id = next(r['id'] for r in self.rows[split].values()
                if r['base'] == row['base'] and r['setting'] == row['setting'] and r['intervention'] == 'standard')
            standard = self.canonical(split, standard_id)
            for field in ('y', 'query_ids', 'context_ids', 'context_labels', 'standard_own_missing'):
                require(np.array_equal(d[field], standard[field]), f'intervention identity/{field}')
            require(d['standard_source_nodes'] == standard['standard_source_nodes'], 'intervention source eligibility changed')
        self.input_cache[key] = d
        return d

    def rescore(self, record, split, calibration=False):
        row_id = record['episode'] if calibration else record['id']
        row, d = self.rows[split][row_id], self.canonical(split, row_id)
        if calibration:
            require(record['input_sha256'] == row['sha256'], 'calibration input association')
        else:
            for k in ('path', 'sha256', 'split', 'base', 'setting', 'intervention', 'standard_source_nodes', 'classes', 'shape', 'k', 'regime'):
                require(record[k] == row[k], f'canonical metadata {row_id}/{k}')
        path = artifact(record['prediction'])
        require(sha(path) == record['prediction_sha256'], f'prediction changed: {path}')
        with np.load(path, allow_pickle=False) as z:
            for saved, canonical in (('y','y'), ('query','query_ids'), ('context','context_ids'), ('yc','context_labels')):
                require(np.array_equal(z[saved], d[canonical]), f'prediction identity {path}/{saved}')
            if not calibration:
                require(np.array_equal(z['own_missing'], d['standard_own_missing']), f'prediction own subset {path}')
            p = z['p'].copy()
        require(p.shape == (len(d['y']), d['classes']), f'probability shape {path}')
        met = independent_metrics(p, d['y'])
        recorded = {k:record[k] for k in met} if calibration else record['metrics']
        self.metric_error = max(self.metric_error, compare_metrics(met, recorded, str(path)))
        missing = d['standard_own_missing'].astype(bool)
        mm = independent_metrics(p[missing], d['y'][missing]) if missing.any() else None
        if not calibration:
            require((mm is None) == (record['missing_metrics'] is None), 'missing metric validity')
            if mm is not None:
                self.metric_error = max(self.metric_error, compare_metrics(mm, record['missing_metrics'], str(path)+'/missing'))
            self.prediction_count += 1
            if record['arm'] in self.config['arms']:
                key = record['arm'],record['seed'],str(row['base']),row['setting'],row['intervention'],record['variant']
                require(key not in self.case, f'duplicate case {key}')
                self.case[key] = dict(whole=met, missing=mm)
        return met

    def expected_schedule(self, seed):
        rows = list(self.rows['train'].values())
        natural = [i for i,r in enumerate(rows) if r['setting']=='full' and r['regime']=='natural']
        require(bool(natural), 'no full-natural training rows')
        rng = np.random.default_rng(self.config['train']['schedule_seed']+seed)
        answer = []
        for update in range(self.config['train']['updates']):
            index = int(rng.choice(natural)) if update%8==0 else int(rng.integers(len(rows)))
            answer.append(dict(index=index, forward_seed=int(rng.integers(1,2**30)), action_seed=int(rng.integers(1,2**30))))
        return answer

    def progress(self, folder, arm, seed, info, scheduled):
        logs = [json.loads(line) for line in (folder/'progress.jsonl').read_text(encoding='utf8').splitlines()]
        require(len(logs) == self.config['train']['updates'], 'progress group count')
        rows = list(self.rows['train'].values()); exposures = defaultdict(lambda:dict(groups=0, observed_queries=0, missing_queries=0))
        for update,(log,item) in enumerate(zip(logs,scheduled), start=1):
            row = rows[item['index']]; d = self.canonical('train',row['id'])
            require(log['update']==update and log['episode']==row['id'] and log['base']==row['base'], 'progress schedule identity')
            require(log['input_sha256']==row['sha256'] and log['forward_seed']==item['forward_seed'], 'progress input/epsilon')
            records = log['records']; count = 1 if arm=='fixed' else 8
            require(len(records)==count==log['actual_forwards']==log['transformer_backwards'], 'actual group forward/backward count')
            require(log['logical_trace_slots']==8, 'logical slots')
            traces = [check_trace(r['trace']) for r in records]
            if arm=='fixed': require(traces==[(0,0,0)], 'fixed action changed')
            if arm=='direct': require(traces==list(TRACES), 'direct is not exact lexicographic8 enumeration')
            if arm=='uniform':
                rng=np.random.default_rng(item['action_seed'])
                wanted=[tuple(int(x) for x in rng.integers(0,2,3)) for _ in range(8)]
                require(traces==wanted, 'uniform actions differ from separate scheduled RNG')
            for r,t in zip(records,traces):
                require(r['cost']==sum(t), 'logged trace cost')
                require(np.isfinite(r['ce']) and r['ce']>=-1e-6 and np.isfinite(r['logp']) and r['logp']<=1e-6, 'invalid CE/logp')
                near(r['probability'], np.exp(r['logp']), 3e-7,2e-6,'exp(logp)')
            require(log['unique_traces']==len(set(traces)) and log['extra_adapter_pairs']==sum(map(sum,traces)), 'multiplicity/cost count')
            near(log['mean_ce'], np.mean([r['ce'] for r in records]), 1e-8, name='mean training CE')
            require(np.isfinite(log['theta_grad_norm']) and np.isfinite(log['phi_grad_norm']), 'nonfinite successful gradient norms')
            probability=np.array([r['probability'] for r in records]); ce=np.array([r['ce'] for r in records]); cost=np.array([sum(t) for t in traces])
            if arm=='direct':
                near(probability.sum(),1.,2e-4,name='direct leaf sum')
                expected=float(np.sum(probability*ce))
                near(log['expected_ce'],expected,3e-5*(1+abs(expected)),name='direct expected CE')
                actor=float(np.sum(probability*(ce+self.config['cost_weight']*cost)))
                self.actor_error=max(self.actor_error,near(log['actor'],actor,3e-5*(1+abs(actor)),name='direct actor/no extra eighth'))
                near(log['entropy_estimate'],-np.sum(probability*np.log(np.maximum(probability,1e-12))),3e-5,name='direct entropy')
            elif arm=='group_pg':
                reward=np.asarray([-float(l)-self.config['cost_weight']*int(c) for l,c in zip(ce,cost)],dtype=np.float32)
                sd=float(np.std(reward.astype(np.float64),ddof=0)); maxreward=float(np.max(np.abs(reward),initial=0))
                near(log['reward_sd'],sd,2e-6*(1+maxreward),name='PG population reward SD')
                advantage=(reward.astype(np.float64)-reward.astype(np.float64).mean())/(float(log['reward_sd'])+1e-6)
                actor=-float(np.mean(advantage*np.array([r['logp'] for r in records])))
                tolerance=float(3e-5+16*np.finfo(np.float32).eps*max(1.,maxreward)*max(1.,float(np.mean(np.abs([r['logp'] for r in records]))))/(float(log['reward_sd'])+1e-6))
                self.pg_actor_tolerance=max(self.pg_actor_tolerance,tolerance)
                self.actor_error=max(self.actor_error,near(log['actor'],actor,tolerance,name='PG normalized actor'))
                require(bool(log['zero_advantage'])==bool(np.all(reward==reward[0])), 'PG zero-group flag')
            else:
                require(log['actor'] is None and log['expected_ce'] is None, 'fixed/uniform unexpected actor')
                near(log['phi_grad_norm'],0.,name='frozen policy gradient')
            retries=log['retries']; self.training_cost['retry_groups']+=int(bool(retries)); self.training_cost['retry_attempts']+=len(retries)
            for attempt,retry in enumerate(retries):
                require(retry['attempt']==attempt and retry['actual_forwards']==count, 'retry count/order')
                require(len(retry['ce'])==len(retry['ce_finite'])==count, 'retry CE fields')
                for value,finite in zip(retry['ce'],retry['ce_finite']):
                    require((value is None and not finite) or (value is not None and finite and np.isfinite(value)), 'retry nonfinite JSON handling')
                self.training_cost['retry_forwards']+=count; self.training_cost['retry_transformer_backwards']+=count
                # Fixed/direct/uniform retry actions are exact from forced traces.
                low=high=sum(map(sum,traces)) if arm!='group_pg' else 0
                if arm=='group_pg': high=3*count
                self.training_cost['retry_extra_pairs_lower']+=low; self.training_cost['retry_extra_pairs_upper']+=high
            for k,v in [('successful_groups',1),('logical_slots',8),('successful_forwards',count),('successful_transformer_backwards',count),('successful_extra_pairs',sum(map(sum,traces)))]:self.training_cost[k]+=v
            bucket=exposures[row['setting']+'/'+row['regime']]; own=d['standard_own_missing'].astype(bool)
            bucket['groups']+=1;bucket['observed_queries']+=int((~own).sum());bucket['missing_queries']+=int(own.sum())
        require(dict(exposures)==info['query_exposures'], 'query exposure counts')
        self.training_cost['wall_seconds']+=float(info['seconds'])

    def checkpoints_and_training(self):
        require(not torch.cuda.is_initialized(), 'CUDA initialized before CPU audit')
        sys.path.insert(0,str(ROOT))
        from rl_model import Model  # Construction only, NEVER call forward.
        common_frozen=None; checkpoint_reports=[]
        for seed in self.config['seeds']:
            model=Model(seed,device='cpu')
            require(not torch.cuda.is_initialized(), 'CPU construction unexpectedly initialized CUDA')
            named=dict(model.named_parameters())
            expected={n for n,p in named.items() if p.requires_grad or n.startswith('policy.')}
            initial={n:named[n].detach().clone() for n in expected}
            initial_digest=parameter_digest(model); frozen=parameter_digest(model.base,frozen_only=True)
            if common_frozen is None:common_frozen=frozen
            require(frozen==common_frozen,'public frozen parameters differ across seeds')
            for arm in self.config['arms']:
                folder=ROOT/'results'/f'{arm}_s{seed}'; info=read_json(folder/'training.json')
                require(info['arm']==arm and info['seed']==seed and info['target_updates']==0,'training metadata identity')
                require(info['updates']==self.config['train']['updates'] and info['task_groups']==info['updates'],'training group metadata')
                require(info['logical_trace_slots']==info['updates']*8,'training logical slots')
                require(sha(folder/'checkpoint.pt')==info['checkpoint_sha256'] and sha(folder/'schedule.json')==info['schedule_sha256'],'checkpoint/schedule file hash')
                schedule=self.expected_schedule(seed);require(read_json(folder/'schedule.json')==schedule,'realized schedule differs from RNG replay')
                checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                for key,want in [('arm',arm),('seed',seed),('lock_sha256',self.lock_sha),('schedule_sha256',info['schedule_sha256'])]:require(checkpoint[key]==want,f'checkpoint internal {key}')
                values=checkpoint['parameters'];require(set(values)==expected,'checkpoint parameter key coverage')
                adapter_count=sum(p.numel() for n,p in named.items() if n in expected and not n.startswith('policy.'))
                policy_count=sum(p.numel() for n,p in named.items() if n.startswith('policy.'))
                require(info['adapter_parameters']==adapter_count and info['policy_parameters']==policy_count,'parameter count metadata')
                require(info['trainable_parameters']==adapter_count+(policy_count if arm in ('direct','group_pg') else 0),'trainable parameter count')
                with torch.no_grad():
                    for n,v in values.items():
                        require(torch.is_tensor(v) and v.device.type=='cpu' and v.shape==named[n].shape,'checkpoint tensor shape/device')
                        require(v.dtype==torch.float32==named[n].dtype and bool(torch.isfinite(v).all()),'checkpoint dtype/finite')
                    for n,v in initial.items():named[n].copy_(v)
                require(parameter_digest(model)==info['initial_digest']==initial_digest,'initial public/policy identity')
                with torch.no_grad():
                    for n,v in values.items():named[n].copy_(v)
                require(parameter_digest(model)==info['final_digest'],'full final tensor digest')
                require(parameter_digest(model.base,frozen_only=True)==info['frozen_digest']==frozen and info['frozen_unchanged'],'public frozen digest')
                if arm in ('fixed','uniform'):
                    require(all(torch.equal(values[n],initial[n]) for n in expected if n.startswith('policy.')),'nonlearned policy changed')
                self.progress(folder,arm,seed,info,schedule)
                self.fits[arm,seed]=info
                checkpoint_reports.append(dict(arm=arm,seed=seed,checkpoint_sha256=info['checkpoint_sha256'],
                    final_digest=info['final_digest'],parameter_tensors=len(values),all_trainable_float32_finite=True))
            del model
        self.report['checkpoints']=checkpoint_reports;self.report['public_frozen_digest']=common_frozen

    def dev_selection(self):
        expected={(row_id,trace) for row_id in self.rows['dev'] for trace in TRACES}
        for arm in ('uniform','group_pg'):
            for seed in self.config['seeds']:
                folder=ROOT/'results'/f'{arm}_s{seed}'; document=read_json(folder/'global_trace.json')
                require(document['arm']==arm and document['seed']==seed,'calibration fit identity')
                seen=set(); totals=defaultdict(list)
                for record in document['records']:
                    trace=check_trace(record['trace']); key=record['episode'],trace
                    require(key in expected and key not in seen,'dev branch coverage');seen.add(key)
                    met=self.rescore(record,'dev',calibration=True);totals[trace].append(met['nll'])
                    require(record['forwards']==1 and tuple(map(tuple,record['traces']))==(trace,) and record['extra_adapter_pairs']==sum(trace),'calibration forced trace/cost')
                    self.calibration_forwards+=1;self.calibration_extra+=sum(trace)
                require(seen==expected,'calibration missing branch')
                means=[float(np.mean(totals[t])) for t in TRACES]
                for a,b in zip(document['scores'],means):near(a,b,1e-10,name='dev mean CE/no cost term')
                selected=min(range(8),key=lambda i:(means[i],sum(TRACES[i]),TRACES[i]))
                require(check_trace(document['trace'])==TRACES[selected],'dev selected trace/tie break')
                require(document['actual_forwards']==len(expected),'calibration forward metadata')
                self.calibration[arm,seed]=TRACES[selected];self.calibration_seconds+=float(document['seconds'])
        self.report['calibration']=dict(fits=len(self.calibration),branches_per_fit=len(expected),criterion='independent mean CE; lower-cost/lexical ties',
            selected=[dict(arm=a,seed=s,trace=list(t),checkpoint_sha256=self.fits[a,s]['checkpoint_sha256']) for (a,s),t in self.calibration.items()])

    def deployment(self):
        for arm in self.config['arms']:
            for seed in self.config['seeds']:
                for split in ('final','real'):
                    folder=ROOT/'results'/f'{arm}_s{seed}'; doc=read_json(folder/f'{split}_evaluation.json')
                    expected={(row_id,v) for row_id in self.rows[split] for v in VARIANTS[arm]}; seen=set()
                    for record in doc['records']:
                        key=record['id'],record['variant'];require(key in expected and key not in seen,'deployment expected variant matrix');seen.add(key)
                        require(record['arm']==arm and record['seed']==seed and record['split']==split,'deployment fit/split identity')
                        require(record['checkpoint_sha256']==self.fits[arm,seed]['checkpoint_sha256'],'deployment checkpoint association')
                        self.rescore(record,split)
                        traces=tuple(check_trace(t) for t in record['traces'])
                        if record['variant']=='global':wanted=(self.calibration[arm,seed],)
                        elif record['variant']=='replay000' or arm=='fixed':wanted=((0,0,0),)
                        elif arm=='uniform':wanted=TRACES
                        else:wanted=None
                        require(wanted is None and len(traces)==1 or traces==wanted,'deployment forced/uniform/greedy trace shape')
                        require(record['forwards']==len(traces) and record['extra_adapter_pairs']==sum(map(sum,traces)),'deployment forward/cost')
                        self.deployment_forwards+=len(traces);self.deployment_extra+=sum(map(sum,traces))
                    require(seen==expected,'missing deployment variant/episode');self.evaluation_seconds+=float(doc['seconds'])
        for split in ('final','real'):
            doc=read_json(ROOT/'results'/f'{split}_references.json')
            expected={(row_id,arm) for row_id in self.rows[split] for arm in REFS};seen=set()
            for record in doc['records']:
                key=record['id'],record['arm'];require(key in expected and key not in seen,'reference matrix');seen.add(key)
                require(record['seed'] is None and record['variant']=='main' and record['split']==split,'reference identity')
                self.rescore(record,split)
                count=1 if record['arm'].startswith('frozen') else 0
                wanted=[[0,0,0]] if count else []
                require(record['forwards']==count and record['extra_adapter_pairs']==0 and record['traces']==wanted,'reference trace/cost')
                self.deployment_forwards+=count;self.reference_record_seconds+=float(record['seconds'])
            require(seen==expected,'missing reference/episode')
        self.report['deployment']=dict(records=self.prediction_count,max_independent_metric_error=self.metric_error,
             exact_expected_variants={k:list(v) for k,v in VARIANTS.items()},reference_arms=list(REFS))

    def interval(self,values,strata,index):
        if not values:return dict(mean=None,ci=None,n_graphs=0,n_strata=0)
        ids=sorted(values);groups=defaultdict(list)
        for gid in ids:groups[tuple(strata[gid])].append(gid)
        rng=np.random.default_rng(self.config['statistics']['seed']+index);draws=[]
        for _ in range(self.config['statistics']['bootstrap']):
            samples=[gid for group in groups.values() for gid in rng.choice(group,len(group),replace=True)]
            draws.append(float(np.mean([values[g] for g in samples])))
        tail=(1-self.config['statistics']['confidence'])/2
        return dict(mean=float(np.mean(list(values.values()))),ci=[float(x) for x in np.quantile(draws,[tail,1-tail])],n_graphs=len(ids),n_strata=len(groups))

    def scientific_aggregation(self):
        """Independent primary/utility/guard reconstruction; never changes gates."""
        base={str(r['base']):r for r in self.rows['final'].values()}; strata={b:(r['shape'],r['k'],r['regime']) for b,r in base.items()}
        seeds=self.config['seeds'];rates=self.config['primary_settings']
        def get(arm,seed,b,rate,intervention='standard',metric='accuracy',whole=False):
            record=self.case[arm,seed,b,rate,intervention,'main']['whole' if whole else 'missing']
            return None if record is None else record[metric]
        primary={};graph={};secondary={}
        for arm in self.config['arms']:
            graph[arm]={};secondary[arm]={}
            for metric in ('accuracy','nll','brier','balanced_accuracy','macro_f1'):
                values={}
                for b in base:
                    valid=[r for r in rates if get(arm,seeds[0],b,r,metric=metric) is not None]
                    if valid:values[b]=float(np.mean([get(arm,s,b,r,metric=metric) for s in seeds for r in valid]))
                secondary[arm][metric]=float(np.mean(list(values.values()))) if values else None
                if metric=='accuracy':graph[arm]=values
            primary[arm]=dict(accuracy=secondary[arm]['accuracy'],graphs=len(graph[arm]))
        gates={}
        for i,control in enumerate(('fixed','direct')):
            valid=graph['group_pg'].keys()&graph[control].keys()
            v={b:graph['group_pg'][b]-graph[control][b] for b in valid};ans=self.interval(v,strata,i)
            ans['verdict']='UNDEFINED' if len(valid)!=96 else 'PASS' if ans['ci'][0]>0 else 'FAIL'
            gates['group_pg_minus_'+control]=ans
        eligible={}
        for b,r in base.items():
            if r['regime']!='disjoint':continue
            valid=[]
            for rate in rates:
                row=next(x for x in self.rows['final'].values() if str(x['base'])==b and x['setting']==rate and x['intervention']=='standard')
                d=self.canonical('final',row['id'])
                if d['standard_source_nodes']>=2 and d['standard_own_missing'].any():valid.append(rate)
            if valid:eligible[b]=valid
        for index,control in enumerate(('none','shuffle'),start=2):
            v={b:float(np.mean([get('group_pg',s,b,r)-get('group_pg',s,b,r,control) for s in seeds for r in valid])) for b,valid in eligible.items()}
            ans=self.interval(v,strata,index);ans['verdict']='UNDEFINED' if not v else 'PASS' if ans['ci'][0]>0 else 'FAIL';gates['feature_utility_'+control]=ans
        guard={}
        for b,r in base.items():
            if r['regime']!='natural':continue
            row=next(x for x in self.rows['final'].values() if str(x['base'])==b and x['setting']=='full' and x['intervention']=='standard')
            require(not self.canonical('final',row['id'])['standard_own_missing'].any(),'full_natural queries not observed')
            guard[b]=float(np.mean([get('group_pg',s,b,'full',whole=True)-get('fixed',s,b,'full',whole=True) for s in seeds]))
        ans=self.interval(guard,strata,4);ans['verdict']='UNDEFINED' if not guard else 'PASS' if ans['ci'][0]>-.02 else 'FAIL';gates['full_natural_guard']=ans
        summary_path=ROOT/'results/summary.json'
        if summary_path.exists():
            summary=read_json(summary_path)
            for arm in primary:
                require(primary[arm]['graphs']==summary['primary'][arm]['graphs'],'primary graph denominator')
                if primary[arm]['accuracy'] is not None:near(primary[arm]['accuracy'],summary['primary'][arm]['accuracy'],1e-12,name='primary accuracy')
            for key,got in gates.items():
                recorded=summary['gates'][key];require(got['verdict']==recorded['verdict'] and got['n_graphs']==recorded['n_graphs'],'gate status/denominator')
                if got['ci'] is not None:
                    near(got['mean'],recorded['mean'],1e-12,name=key+'/mean')
                    for a,b in zip(got['ci'],recorded['ci']):near(a,b,1e-12,name=key+'/CI')
        self.report['independent_scientific_rescore']=dict(primary=primary,primary_matched_secondary=secondary,gates=gates,
            utility_eligible_graphs=len(eligible),utility_eligible_cases=sum(map(len,eligible.values())),existing_summary_compared=summary_path.exists())

    def finish(self):
        preparation={s:{k:float(m[k]) for k in ('prepare_wall_seconds','prepare_cpu_seconds','pcfi_wall_seconds') if k in m} for s,m in self.manifests.items()}
        prepwall=sum(m.get('prepare_wall_seconds',0.) for m in preparation.values())
        self.report['cost']=dict(training=self.training_cost,preparation=preparation,
            preparation_wall_seconds=prepwall,
            pcfi_wall_seconds_subset_of_preparation=sum(m.get('pcfi_wall_seconds',0.) for m in preparation.values()),
            calibration=dict(actual_forwards=self.calibration_forwards,extra_adapter_pairs=self.calibration_extra,wall_seconds=self.calibration_seconds),
            deployment=dict(actual_forwards=self.deployment_forwards,extra_adapter_pairs=self.deployment_extra,
                trained_evaluation_wall_seconds=self.evaluation_seconds,reference_record_seconds=self.reference_record_seconds),
            total_actual_forwards=self.training_cost['successful_forwards']+self.training_cost['retry_forwards']+self.calibration_forwards+self.deployment_forwards,
            wall_component_sum_seconds=prepwall+self.training_cost['wall_seconds']+self.calibration_seconds+self.evaluation_seconds+self.reference_record_seconds,
            wall_scope='PCFI already included in preparation; no double count. Reference LP/prior timing is unrecorded. Checks/dev smoke not included. Workload times may include CPU-audit/hash contention and are not controlled robust latency benchmarks.')
        require(self.training_cost['successful_groups']==len(self.config['arms'])*len(self.config['seeds'])*self.config['train']['updates'],'total successful groups')
        require(self.training_cost['logical_slots']==12288,'total logical slots')
        for path,value in self.source_before.items():require(sha(path)==value,f'source changed during audit: {path}')
        for path,value in self.anchor_files.items():require(sha(path)==value,f'lock/checkpoint/manifest changed during audit: {path}')
        require(not torch.cuda.is_initialized(),'audit unexpectedly initialized CUDA')
        self.report.update(status='PASS',max_actor_scalar_error=self.actor_error,max_pg_actor_tolerance=self.pg_actor_tolerance,
            source_hashes_unchanged=True,cpu_parameter_reconstruction_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    output=ROOT/'results/deployment_audit.json'
    if output.exists():raise FileExistsError(f'Create-only audit already exists: {output}')
    start=time.perf_counter();audit=None
    try:
        audit=Audit();audit.anchors();print('ANCHORS_PASS',flush=True)
        audit.checkpoints_and_training();print('CPU_CHECKPOINT_PROGRESS_PASS',flush=True)
        audit.dev_selection();print('DEV_SELECTION_PASS',flush=True)
        audit.deployment();print('CANONICAL_DEPLOYMENT_RESCORE_PASS',flush=True)
        audit.scientific_aggregation();audit.finish();result=audit.report
    except Exception as exc:
        result=audit.report if audit is not None else dict(cycle='FP05')
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
    result['audit_wall_seconds']=time.perf_counter()-start
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8',newline='\n') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('DEPLOYMENT_AUDIT_'+result['status'],str(output),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
