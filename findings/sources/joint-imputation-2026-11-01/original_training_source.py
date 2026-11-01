"""prepare -> preflight -> run -> analyze. No target-graph fitting."""
import argparse
import copy
from common import *


def generator_config():
    from mgpfn import upstream as U
    c=U.prior_config(); v=c['prior']['values'][0]
    v['graph']['n_nodes']=config()['nodes']
    v['graph']['avg_degree']={'_distribution_':'uniform','min':4.,'max':12.}
    v['graph']['sampler']['values'][0]['min_first_level_n_nodes']=16
    v['scm']['base']['n_features']['min']=8
    v['scm']['base']['n_features']['max']=16
    v['task']['n_classes']['max']=4
    v['task']['quantile']['min']=.3; v['task']['quantile']['max']=.7
    return c


def prepare():
    from mgpfn import prior_adapter as PA
    c=config(); bank=ROOT/'bank'
    if (bank/'manifest.json').exists():
        raise RuntimeError('Bank exists; preserve it. Use another directory for a different experiment.')
    bank.mkdir(exist_ok=True); (ROOT/'results').mkdir(exist_ok=True)
    cfg=generator_config(); write_json(bank/'generator.json',cfg)
    rows=[]; started=time.perf_counter()
    for split in ('train','test'):
        (bank/split).mkdir(exist_ok=True)
        for gid in range(c[split+'_graphs']):
            requested=c[split+'_seed_start']+gid
            for attempt in range(20):
                try:
                    task=PA.sample_official_task(cfg,requested+attempt*100000000,
                        allowed_task_types=('binclass','multiclass'))
                    break
                except AssertionError:
                    pass
            else:
                raise RuntimeError(('Invalid generator draws',split,gid))
            _,task['y']=np.unique(task['y'],return_inverse=True)
            n,d=task['X_full'].shape; classes=len(np.unique(task['y']))
            rng=np.random.default_rng(requested+777)
            anchors={cls:rng.permutation(np.flatnonzero(task['y'][:task['n_train']]==cls)) for cls in range(classes)}
            assert all(len(z)>0 for z in anchors.values())
            query=rng.permutation(np.arange(task['n_train'],n))[:c['queries']]
            assert len(query)>=4
            PA.save_task(bank/split/f'graph_{gid}.npz',task)
            settings=[(m,r,c['context_per_class']) for m,r in c['train_settings']] if split=='train' else c['test_settings']
            for sid,(mechanism,rate,k) in enumerate(settings):
                context=np.concatenate([anchors[cls][:k] for cls in range(classes)])
                # k=1 secondary reuses the k=4 uniform-99.5 mask for an actual paired context intervention.
                mask_id=2 if sid==6 else sid
                mask_seed=c['mask_seed_start']+(0 if split=='train' else 100000)+gid*100+mask_id
                observed=BASE.mask(task,mechanism,rate,mask_seed)
                values=observed_scale(task['X_full'],observed)
                poisoned=task['X_full'].copy(); poisoned[~observed]=1e12
                assert np.array_equal(values,observed_scale(poisoned,observed))
                raw=np.where(observed,values,np.nan)
                filled=BASE.fill(raw,task['src'],task['dst'],'pcfi',mechanism).clip(-10,10)
                if d%2:
                    values=np.pad(values,((0,0),(0,1)))
                    filled=np.pad(filled,((0,0),(0,1)))
                    observed=np.pad(observed,((0,0),(0,1)),constant_values=True)
                path=bank/split/f'e{gid}_{sid}.npz'
                np.savez_compressed(path, observed_values=values,observed=observed,pcfi=filled,
                    src=task['src'],dst=task['dst'],context=context,labels_context=task['y'][context],
                    query=query,labels_query=task['y'][query],classes=classes,forward_seed=mask_seed+10000000)
                rows.append({'split':split,'graph':gid,'setting':sid,'mechanism':mechanism,'rate':rate,'k':k,
                    'path':path.relative_to(bank).as_posix(),'nodes':n,'features':d,'queries':len(query),
                    'classes':classes,'context':len(context),'observed_rows':int(observed[:,:d].any(1).sum()),
                    'observed_cells':int(observed[:,:d].sum()),'mask_seed':mask_seed,
                    'graph_seed':requested+attempt*100000000,'invalid_retries':attempt,'sha256':sha(path)})
            if (gid+1)%8==0:
                print('BANK',split,gid+1,round(time.perf_counter()-started,1),flush=True)
    write_json(bank/'manifest.json',{'episodes':rows,'sources':sources(),'seconds':time.perf_counter()-started,
        'hidden_value_poisoning_invariance':True})


def rowset(split):
    return [r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']==split]


def counts(model):
    return {'total':sum(p.numel() for p in model.parameters()),
        'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'new':sum(p.numel() for name,p in model.named_parameters() if not name.startswith('base.')),
        'common':model.groups}


def preflight():
    train=rowset('train'); masked=next(r for r in train if r['mechanism']=='uniform_nodes' and r['observed_rows']>0)
    inp,y=read_episode(masked); checks={}; inits={}; models={}
    for arm in config()['arms']:
        model=Model(arm,11); models[arm]=counts(model)
        inits[arm]=parameter_digest(model.base,trainable=True)
        p=predict(model,inp); repeat=predict(model,copy.deepcopy(inp))
        assert np.array_equal(p,repeat)
        assert not hasattr(inp,'labels_query') and not hasattr(inp,'hidden_values')
        if arm in ('frontend','interleaved'):
            assert model.last_diagnostics['refinement_steps']==4
            assert model.last_diagnostics['observed_clamp_max']==0.
            off=predict(model,inp,disable_refinement=True)
            checks[arm+'_initial_refinement_max_delta']=float(np.max(np.abs(p-off)))
        model.train(); model.zero_grad(set_to_none=True)
        t0=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(inp); loss=F.cross_entropy(logits,torch.as_tensor(y,device='cuda'))
        loss.backward(); torch.cuda.synchronize()
        gradients={group:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
            if name.startswith(group) and p.grad is not None) for group in ('extra','mask_embedding','base')}
        assert all(z>0 and math.isfinite(z) for z in gradients.values()),gradients
        checks[arm]={'parameters':counts(model),'gradients':gradients,'step_seconds':time.perf_counter()-t0,
            'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'probability_sum_max_error':float(np.abs(p.sum(1)-1).max())}
        if arm=='interleaved':
            # A real downstream dependence on the post-PFN states, not a detached logging path.
            no_feedback=predict(model,inp,disable_feedback=True)
            checks['feedback_forward_max_delta']=float(np.max(np.abs(p-no_feedback)))
            assert checks['feedback_forward_max_delta']>1e-7
            # Node relabeling + a matching edge/context/query permutation must preserve probabilities.
            order=np.random.default_rng(77).permutation(len(inp.observed)); inv=np.argsort(order)
            renamed=copy.deepcopy(inp)
            for key in ('observed_values','observed','pcfi'):
                setattr(renamed,key,getattr(inp,key)[order])
            renamed.src=inv[inp.src]; renamed.dst=inv[inp.dst]
            renamed.context=inv[inp.context]; renamed.query=inv[inp.query]
            renamed_prob=predict(model,renamed)
            checks['node_permutation_max_error']=float(np.abs(renamed_prob-p).max())
            assert checks['node_permutation_max_error']<.004
            full=read_episode(train[0])[0]
            normal=predict(model,full); disabled=predict(model,full,disable_refinement=True)
            checks['full_observed_refinement_max_delta']=float(np.abs(normal-disabled).max())
            assert checks['full_observed_refinement_max_delta']<.003
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(set(inits.values()))==1
    assert models['frontend']['new']==models['interleaved']['new']
    checks['common_initialization_identical']=True
    checks['frontend_interleaved_parameter_counts_identical']=True
    checks['source_files']=sources()
    write_json(ROOT/'results/preflight.json',checks)
    print('PREFLIGHT_OK',json.dumps(checks),flush=True)


def lock():
    from mgpfn import upstream as U
    expected={'sources':sources(),'bank_manifest_sha256':sha(ROOT/'bank/manifest.json'),
        'public_weights':{p.name:sha(p) for p in (U.ADAPTERS,U.LIMIX)},'upstream_commit':U.COMMIT}
    path=ROOT/'run_manifest.json'
    if path.exists():
        assert json.loads(path.read_text())==expected,'Sources, bank, or weights changed after lock'
    else:
        write_json(path,expected)
    return expected


def evaluate(model,rows,arm,seed):
    directory=ROOT/'results/predictions';directory.mkdir(exist_ok=True)
    records=[]
    modes=['standard']+(['no_feedback','no_refinement'] if arm=='interleaved' else [])
    for row in rows:
        inp,y=read_episode(row)
        for mode in modes:
            # Mechanism ablations are diagnostic on the same development graphs, not extra model selection.
            t0=time.perf_counter()
            kwargs={'disable_feedback':mode=='no_feedback','disable_refinement':mode=='no_refinement'}
            p=predict(model,inp,**kwargs);torch.cuda.synchronize()
            elapsed=time.perf_counter()-t0
            path=directory/f'{arm}_s{seed}_g{row["graph"]}_e{row["setting"]}_{mode}.npz'
            np.savez_compressed(path,p=p,y=y)
            records.append({**row,'arm':arm,'seed':seed,'mode':mode,**metrics(p,y),
                'seconds':elapsed,'diagnostics':model.last_diagnostics,'prediction':path.relative_to(ROOT).as_posix()})
    return records


def run(only_arm=None,only_seed=None):
    c=config(); metadata=lock()
    train=rowset('train');test=rowset('test');cache=[read_episode(r) for r in train]
    if not (ROOT/'results/preflight.json').exists():
        raise RuntimeError('Run preflight first')
    for seed in ([only_seed] if only_seed is not None else c['seeds']):
        schedule=np.random.default_rng(seed+40000).integers(len(train),size=c['updates'])
        schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
        for arm in ([only_arm] if only_arm else c['arms']):
            out=ROOT/'results'/f'{arm}_s{seed}';out.mkdir(exist_ok=True)
            if (out/'evaluation.json').exists():
                print('SKIP_COMPLETE',arm,seed,flush=True);continue
            if (out/'training.json').exists():
                raise RuntimeError('Partial run; preserve it and explicitly resume evaluation instead of retraining')
            model=Model(arm,seed)
            initial=parameter_digest(model,trainable=True);frozen=parameter_digest(model,trainable=False)
            optimizer=torch.optim.AdamW([
                {'params':[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('base.')],'lr':c['base_lr'],'base_lr':c['base_lr']},
                {'params':[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('base.')],'lr':c['new_lr'],'base_lr':c['new_lr']}],weight_decay=c['weight_decay'])
            scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            logs=[];started=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
            for step,idx in enumerate(schedule):
                inp,y=cache[int(idx)]
                mult=min((step+1)/16,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/c['updates'])))
                for group in optimizer.param_groups:group['lr']=group['base_lr']*mult
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(inp);loss=F.cross_entropy(logits,torch.as_tensor(y,device='cuda'))
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['gradient_clip'],error_if_nonfinite=True)
                previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                assert scaler.get_scale()>=previous_scale,'Unexpected skipped update; stop rather than silently compare budgets'
                logs.append({'step':step+1,'episode':int(idx),'loss':float(loss.detach()),'gradient_norm':float(norm),'scale':scaler.get_scale(),
                    'seconds':time.perf_counter()-started})
                if (step+1)%32==0:
                    print('TRAIN',arm,seed,step+1,'loss',round(float(np.mean([z['loss'] for z in logs[-32:]])),5),'seconds',round(time.perf_counter()-started,1),flush=True)
                    write_json(out/'progress.json',logs[-1])
            torch.cuda.synchronize();train_seconds=time.perf_counter()-started
            assert parameter_digest(model,trainable=False)==frozen
            assert parameter_digest(model,trainable=True)!=initial
            torch.save({'state_dict':{name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad},
                'arm':arm,'seed':seed,'metadata':metadata},out/'trainable.pt')
            record={'arm':arm,'seed':seed,'updates':len(logs),'schedule_sha256':schedule_sha,
                'initial_trainable_sha256':initial,'frozen_sha256':frozen,'frozen_weights_unchanged':True,
                'seconds':train_seconds,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
                'parameters':counts(model),'checkpoint_sha256':sha(out/'trainable.pt'),'metadata':metadata}
            write_json(out/'training.json',record);write_json(out/'train_log.json',logs)
            print('TRAINED',arm,seed,'evaluation_started',flush=True)
            results=evaluate(model,test,arm,seed)
            write_json(out/'evaluation.json',results)
            print('EVALUATED',arm,seed,len(results),flush=True)
            del model,optimizer;gc.collect();torch.cuda.empty_cache()


def group_rows(rows,name):
    if name=='uniform': return [r for r in rows if r['mechanism']=='uniform_nodes' and r['k']==4]
    if name=='clustered': return [r for r in rows if r['mechanism']=='clustered_nodes']
    if name=='cell': return [r for r in rows if r['mechanism']=='uniform_cells']
    if name=='k1': return [r for r in rows if r['k']==1]
    if name=='full': return [r for r in rows if r['mechanism']=='full']
    return [r for r in rows if f"{r['mechanism']}:{r['rate']}:{r['k']}"==name]


def analyze():
    c=config();lock();rows=[];training=[]
    for seed in c['seeds']:
        for arm in c['arms']:
            out=ROOT/'results'/f'{arm}_s{seed}'
            rows.extend(json.loads((out/'evaluation.json').read_text()))
            training.append(json.loads((out/'training.json').read_text()))
    # Verify all saved probability files independently, including deployment ablations.
    for r in rows:
        with np.load(ROOT/r['prediction']) as d: measured=metrics(d['p'],d['y'])
        assert all(abs(measured[k]-r[k])<1e-10 for k in measured)
        assert r['diagnostics']['observed_clamp_max']==0.
    for seed in c['seeds']:
        subset=[r for r in training if r['seed']==seed]
        assert len({r['schedule_sha256'] for r in subset})==1
    summary=[];paired=[];rng=np.random.default_rng(20261101)
    boot=rng.integers(c['test_graphs'],size=(5000,c['test_graphs']))
    groups=['uniform','clustered','cell','k1','full']+[f'{m}:{r}:{k}' for m,r,k in c['test_settings']]
    for group in groups:
        subset=group_rows(rows,group)
        for arm in c['arms']:
            selected=[r for r in subset if r['arm']==arm and r['mode']=='standard']
            means={metric:float(np.mean([r[metric] for r in selected])) for metric in ('accuracy','nll','brier')}
            summary.append({'group':group,'arm':arm,**means,
                'per_seed_nll':{str(s):float(np.mean([r['nll'] for r in selected if r['seed']==s])) for s in c['seeds']}})
        def graph_values(arm,mode,metric):
            return np.array([np.mean([r[metric] for r in subset if r['arm']==arm and r['mode']==mode and r['graph']==g]) for g in range(c['test_graphs'])])
        for other,mode in [(a,'standard') for a in c['arms'] if a!='interleaved']+[('interleaved','no_feedback'),('interleaved','no_refinement')]:
            for metric in ('accuracy','nll','brier'):
                diff=graph_values('interleaved','standard',metric)-graph_values(other,mode,metric)
                low,high=np.quantile(diff[boot].mean(1),[.025,.975])
                paired.append({'group':group,'contrast':'interleaved - '+other+':'+mode,'metric':metric,
                    'delta':float(diff.mean()),'low':float(low),'high':float(high)})
    passed=all(next(r for r in paired if r['group']==group and r['contrast']==f'interleaved - {arm}:standard' and r['metric']=='nll')['high']<0
        for group in ('uniform','clustered') for arm in ('mask_plain','pcfi_plain','frontend'))
    decision='Conditional Go: primary pilot gate passed; real graphs and new prior families remain untested.' if passed else 'No-Go for scaling this implementation: prespecified primary gate not passed.'
    write_json(ROOT/'results/summary.json',summary);write_json(ROOT/'results/paired.json',paired)
    write_json(ROOT/'results/decision.json',{'decision':decision,'primary_gate_passed':passed,'source_lock':lock()})
    # Simple sanity predictors are included from the first protocol, not added after a favorable comparison.
    sanity=[]
    for row in rowset('test'):
        inp,y=read_episode(row);freq=np.bincount(inp.labels_context,minlength=int(inp.classes))+1
        prior=np.broadcast_to(freq/freq.sum(),(len(y),len(freq)))
        uniform=np.full_like(prior,1/len(freq))
        sanity.append({**row,'context_prior':metrics(prior,y),'uniform_prior':metrics(uniform,y)})
    write_json(ROOT/'results/sanity.json',sanity)
    lines=['# Joint Imputation + Public GraphPFN — matched pilot','',decision,'',
        '공개 GraphPFN 1.3의 adapter/feature encoder/label decoder를 합성 과제로 추가 학습했다. 원래 LimiX transformer weights는 고정했다. 실제 그래프 전이·새 schema·논문 novelty는 미검증이다.',
        '', '전체 graph를 사용했다. 부분그래프 선택은 변경하지 않았다. 공통 입력은 observed-only RMS, feature-pair numeric encoding과 원래 mask embedding이다. 공개 기본 전처리와 다르므로 이전 보고서 숫자를 전후 성능으로 비교하지 않는다.',
        '', 'Mask Plain: zero+mask와 일반 token residual. PCFI Plain: PCFI+mask와 같은 residual. Frontend: PFN 전에 공유 refiner 4회. Interleaved: 같은 refiner를 layer 0/3/6/9 앞에 넣는다. 두 refiner 모델은 파라미터·갱신 횟수가 같고 PFN은 모두 한 번(12 layers) 실행한다.',
        '', '## 결과', '', '| 조건 | 모델 | Accuracy % | NLL ↓ | Brier ↓ |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['group'] in ('uniform','clustered','cell','k1','full'):
            lines.append(f"| {r['group']} | {r['arm']} | {100*r['accuracy']:.2f} | {r['nll']:.5f} | {r['brier']:.5f} |")
    lines+=['','## Interleaved의 paired 차이','','| 조건 | 대조군 | Δ NLL [95% CI] |','|---|---|---:|']
    for r in paired:
        if r['metric']=='nll' and r['group'] in ('uniform','clustered','cell','k1','full'):
            lines.append(f"| {r['group']} | {r['contrast']} | {r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}] |")
    lines+=['','## 단순 확률 대조','','| 조건 | Context prior NLL | Uniform prior NLL |','|---|---:|---:|']
    for group in ('uniform','clustered','cell','k1','full'):
        ss=group_rows(sanity,group)
        lines.append(f"| {group} | {np.mean([r['context_prior']['nll'] for r in ss]):.5f} | {np.mean([r['uniform_prior']['nll'] for r in ss]):.5f} |")
    lines+=['','## 학습·검증 범위','',f"64 train / 32 fresh test graphs, seeds {c['seeds']}, {c['updates']} updates/arm/seed. 동일 seed에서 task schedule·common initialization·forward RNG를 맞췄다. 마지막 checkpoint만 사용했다. Query CE만 학습했고 feature reconstruction/auxiliary loss는 사용하지 않았다.",
        '', 'Graph별로 결측 조건과 3 seed 지표를 평균한 뒤 5,000회 paired bootstrap했다. CI는 graph 변동이며 training-seed 모집단 CI나 다중 검정 보정을 포함하지 않는다. Per-seed와 개별 mask 결과는 summary.json에 모두 남겼다. Zero-observation episode를 제외하거나 다시 뽑지 않았다.',
        '', 'Frontend/Interleaved의 정확한 동일 파라미터 비교와 Plain의 유사 용량 비교를 구분한다. General token residual과 explicit numeric refiner는 서로 다른 함수족이다. No-feedback와 no-refinement는 이미 학습한 checkpoint의 기전 진단이며 별도 재학습 대조가 아니다.',
        '', '| 모델 | Trainable parameters | New parameters | Train sec, mean | Peak allocated MiB, max |','|---|---:|---:|---:|---:|']
    for arm in c['arms']:
        ss=[r for r in training if r['arm']==arm];r=ss[0]
        lines.append(f"| {arm} | {r['parameters']['trainable']} | {r['parameters']['new']} | {np.mean([z['seconds'] for z in ss]):.1f} | {max(z['peak_allocated_mb'] for z in ss):.1f} |")
    lines+=['','Validation: hidden-feature poisoning invariance; no query labels in input dataclass; observed-value clamp; node relabeling; nonzero refiner/encoder gradients; identical common initialization; identical frontend/interleaved parameter counts; unchanged frozen weights; all saved probabilities independently re-scored. Details: preflight.json / training.json / run_manifest.json.',
        '', '후속 방향은 이 구현의 결과 범위를 지킨다. Primary gate 미통과 시 test 결과를 보고 loss·반복 수·학습량을 바꿔 자동 구제하지 않는다. 성공하더라도 joint imputation 자체의 novelty를 주장하지 않는다.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'results/audit.json',{'saved_outputs_checked':len(rows),'shared_schedules':True,
        'all_frozen_parameters_unchanged':all(r['frozen_weights_unchanged'] for r in training),'final_sources':sources(),
        'preflight':json.loads((ROOT/'results/preflight.json').read_text()),'primary_gate_passed':passed})
    print('ANALYSIS_COMPLETE',decision,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','preflight','run','analyze'))
    ap.add_argument('--arm',choices=config()['arms']);ap.add_argument('--seed',type=int);args=ap.parse_args()
    setup();torch.cuda.set_per_process_memory_fraction(.7)
    if args.action=='prepare':prepare()
    elif args.action=='preflight':preflight()
    elif args.action=='run':run(args.arm,args.seed)
    else:analyze()


if __name__=='__main__':main()
