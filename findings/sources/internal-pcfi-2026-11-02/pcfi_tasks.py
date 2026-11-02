"""Train/dev preparation precedes a separate locked final graph bank."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import time
import numpy as np
from pcfi_shared import ROOT,C,cfg,dump,sha,sources,design,lock,prepare_geometry


def generator_config():
    from mgpfn import upstream as U
    c=U.prior_config();v=c['prior']['values'][0]
    v['graph']['n_nodes']=cfg()['nodes']
    v['graph']['avg_degree']={'_distribution_':'uniform','min':4.,'max':12.}
    v['graph']['sampler']['values'][0]['min_first_level_n_nodes']=16
    v['scm']['base']['n_features']['min']=8;v['scm']['base']['n_features']['max']=16
    v['task']['n_classes']['max']=4
    v['task']['quantile']['min']=.3;v['task']['quantile']['max']=.7
    return c


def generate(split):
    from mgpfn import prior_adapter as PA
    c=cfg();gc=generator_config();records=[]
    begun=time.perf_counter()
    (ROOT/'bank'/split).mkdir(parents=True,exist_ok=True)
    (ROOT/'bank/raw'/split).mkdir(parents=True,exist_ok=True)
    for gid in range(c[split+'_graphs']):
        requested=c[split+'_seed_start']+gid
        for attempt in range(20):
            try:
                task=PA.sample_official_task(gc,requested+attempt*100000000,allowed_task_types=('binclass','multiclass'))
                break
            except AssertionError:pass
        else:raise RuntimeError('Generator retries exhausted')
        _,task['y']=np.unique(task['y'],return_inverse=True)
        n,d=task['X_full'].shape;classes=len(np.unique(task['y']))
        rng=np.random.default_rng(requested+777)
        anchors={cl:rng.permutation(np.flatnonzero(task['y'][:task['n_train']]==cl)) for cl in range(classes)}
        assert all(len(a)>0 for a in anchors.values())
        query=rng.permutation(np.arange(task['n_train'],n))[:c['queries']]
        assert len(query)>=4
        PA.save_task(ROOT/'bank/raw'/split/f'g{gid}.npz',task)
        settings=c['train_settings'] if split=='train' else c['test_settings']
        for sid,(mechanism,rate,k) in enumerate(settings):
            context=np.concatenate([anchors[cl][:k] for cl in range(classes)])
            mid=2 if sid==6 else sid
            mask_seed=c['mask_seed_start']+{'train':0,'dev':100000,'test':200000}[split]+gid*100+mid
            observed=C.BASE.mask(task,mechanism,rate,mask_seed)
            value=C.observed_scale(task['X_full'],observed)
            poison=task['X_full'].copy();poison[~observed]=1e12
            assert np.array_equal(value,C.observed_scale(poison,observed))
            pcfi=C.BASE.fill(np.where(observed,value,np.nan),task['src'],task['dst'],'pcfi',mechanism).clip(-10,10)
            if d%2:
                value=np.pad(value,((0,0),(0,1)));pcfi=np.pad(pcfi,((0,0),(0,1)))
                observed=np.pad(observed,((0,0),(0,1)),constant_values=True)
            inp=C.Inputs(value,observed,pcfi,task['src'],task['dst'],context,task['y'][context],query,classes,mask_seed+10000000)
            extra=prepare_geometry(inp,d)
            path=ROOT/'bank'/split/f'g{gid}_e{sid}.npz'
            np.savez_compressed(path,**vars(inp),**extra,labels_query=task['y'][query])
            records.append({'split':split,'graph':gid,'setting':sid,'mechanism':mechanism,'rate':rate,'k':k,'nodes':n,'features':d,
                 'classes':classes,'queries':len(query),'context':len(context),'path':path.relative_to(ROOT/'bank').as_posix(),
                 'observed_cells':int(observed[:,:d].sum()),'observed_nodes':int(observed[:,:d].any(1).sum()),
                 'missing_queries':int((~observed[query,:d].any(1)).sum()),'invalid_retries':attempt,
                 'graph_seed':requested+attempt*100000000,'mask_seed':mask_seed,'sha256':sha(path)})
        if (gid+1)%8==0:print('BANK',split,gid+1,round(time.perf_counter()-begun,1),flush=True)
    return records,time.perf_counter()-begun


def prepare():
    assert not (ROOT/'bank/development.json').exists()
    C.setup();dump(ROOT/'bank/generator.json',generator_config())
    tr,t1=generate('train');dev,t2=generate('dev')
    dump(ROOT/'bank/development.json',{'episodes':tr+dev,'train_seconds':t1,'dev_seconds':t2})
    print('TRAIN_DEV_READY',flush=True)


def finalize():
    design();C.setup()
    assert not (ROOT/'bank/manifest.json').exists()
    import json
    previous=json.loads((ROOT/'bank/development.json').read_text(encoding='utf8'))
    final,seconds=generate('test')
    dump(ROOT/'bank/manifest.json',{'episodes':previous['episodes']+final,'test_seconds':seconds,'design_sha256':sha(ROOT/'results/design.json')})
    lock();print('FINAL_BANK_LOCKED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','finalize']);a=p.parse_args()
    prepare() if a.action=='prepare' else finalize()
