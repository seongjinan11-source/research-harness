"""Re-evaluate only the no-feedback intervention with cancellation-safe FP16 arithmetic.

Training and standard prediction are untouched. Original predictions/paired.json
are preserved; corrected predictions and contrasts are saved separately.
"""
import inspect
import runpy
import sys
import textwrap
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import common
from common import *
import experiment as E


def main():
    setup();torch.cuda.set_per_process_memory_fraction(.7);E.lock()
    assert (ROOT/'results/completed.json').exists()
    original=inspect.getsource(Model.forward)
    changed=original.replace('h0[:,:,:-1]+old_embedding-encoded','h0[:,:,:-1]+(old_embedding-encoded)')
    changed=changed.replace('h[:,:,:-1]+self.numeric_embedding(updated)-old_embedding',
        'h[:,:,:-1]+(self.numeric_embedding(updated)-old_embedding)')
    assert changed!=original
    namespace=dict(vars(common));exec(textwrap.dedent(changed),namespace)
    Model.forward=namespace['forward']
    directory=ROOT/'results/feedback_correction';directory.mkdir(exist_ok=True)
    (directory/'forward_corrected.py.txt').write_text(textwrap.dedent(changed),encoding='utf-8')
    corrected=[];replayed=[];parity=[];full_errors=[]
    for seed in config()['seeds']:
        model=Model('interleaved',seed)
        checkpoint=ROOT/'results'/f'interleaved_s{seed}/trainable.pt'
        stored=torch.load(checkpoint,map_location='cuda',weights_only=False)
        model.load_state_dict(stored['state_dict'],strict=False)
        trained=json.loads((checkpoint.parent/'training.json').read_text())
        assert parameter_digest(model,trainable=False)==trained['frozen_sha256']
        before=parameter_digest(model)
        for index,row in enumerate(E.rowset('test')):
            inp,y=read_episode(row)
            path=directory/f's{seed}_g{row["graph"]}_e{row["setting"]}.npz'
            p=predict(model,inp,disable_feedback=True)
            # Pair this intervention with a standard forward in this same process.
            # Cross-process FP16 replay is measured, never forced to be bitwise equal.
            standard=predict(model,inp)
            with np.load(ROOT/'results/predictions'/f'interleaved_s{seed}_g{row["graph"]}_e{row["setting"]}_standard.npz') as d:
                error=float(np.max(np.abs(standard-d['p'])))
            parity.append(error)
            if row['mechanism']=='full':
                full_error=float(np.max(np.abs(p-standard)));full_errors.append(full_error)
                assert full_error==0.
            np.savez_compressed(path,p=p,standard=standard,y=y)
            with np.load(path) as d:
                assert metrics(d['p'],d['y'])==metrics(p,y)
            corrected.append({**row,'arm':'interleaved','seed':seed,'mode':'no_feedback_corrected',**metrics(p,y),
                'prediction':path.relative_to(ROOT).as_posix()})
            replayed.append({**row,'arm':'interleaved','seed':seed,'mode':'standard',**metrics(standard,y),
                'prediction':path.relative_to(ROOT).as_posix()})
        assert parameter_digest(model)==before
        print('CORRECTED_FEEDBACK',seed,len(corrected),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(directory/'evaluation.json',corrected)
    write_json(directory/'replayed_standard.json',replayed)
    rows=replayed+corrected
    groups=['uniform','clustered','cell','k1','full']+[f'{m}:{r}:{k}' for m,r,k in config()['test_settings']]
    boot=np.random.default_rng(20261101).integers(config()['test_graphs'],size=(5000,config()['test_graphs']))
    new=[]
    for group in groups:
        subset=E.group_rows(rows,group)
        for metric in ('accuracy','nll','brier'):
            def values(mode):
                return np.array([np.mean([r[metric] for r in subset if r['mode']==mode and r['graph']==g]) for g in range(config()['test_graphs'])])
            diff=values('standard')-values('no_feedback_corrected')
            lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
            new.append({'group':group,'contrast':'interleaved - interleaved:no_feedback','metric':metric,
                'delta':float(diff.mean()),'low':float(lo),'high':float(hi),'diagnostic_version':'cancellation-safe'})
    original_paired=json.loads((ROOT/'results/paired.json').read_text())
    merged=[r for r in original_paired if r['contrast']!='interleaved - interleaved:no_feedback']+new
    write_json(ROOT/'results/paired_corrected.json',merged)
    report=ROOT/'REPORT.md';text=report.read_text(encoding='utf-8')
    (directory/'report_before_diagnostic_correction.md').write_text(text,encoding='utf-8')
    for row in new:
        if row['metric']=='nll' and row['group'] in ('uniform','clustered','cell','k1','full'):
            prefix=f"| {row['group']} | interleaved - interleaved:no_feedback |"
            replacement=f"{prefix} {row['delta']:+.5f} [{row['low']:+.5f}, {row['high']:+.5f}] |"
            text='\n'.join(replacement if line.startswith(prefix) else line for line in text.split('\n'))
    text+='\n## 피드백 제거 대조의 수치 보정\n\n'
    text+='최초 no-feedback 대조의 `h + new_embedding - old_embedding`은 FP16에서 cancellation 오차가 있었다. `h + (new_embedding - old_embedding)`으로 묶고 입력 token의 같은 표현도 일치시켜 해당 대조만 3 seed 모두 다시 평가했다. 모델 학습과 standard/no-refinement 예측은 바꾸지 않았다. 위 표의 no-feedback 행은 보정된 값이다. 원래 predictions·paired.json은 보존했고, 현재 대조는 paired_corrected.json과 feedback_correction/evaluation.json을 사용한다.\n\n'
    text+=f'보정 후 full-observed 96개 검사에서 no-feedback와 같은 process의 standard 차이는 정확히0이었다. 별도 process의 FP16 checkpoint replay와 원래 예측의 최대 확률 차이는 {max(parity):.8f}였다. 따라서 해당 진단은 같은 process에서 다시 계산한 standard와 짝지었다. 주 비교의 원래 예측·성공 기준·선택한 checkpoint는 보존했다.\n'
    report.write_text(text,encoding='utf-8')
    write_json(directory/'audit.json',{'corrected_predictions':len(corrected),'standard_replay_max_error':max(parity),
        'full_observed_prediction_max_error':max(full_errors),'training_unchanged':True,'standard_predictions_unchanged':True,
        'corrector_sha256':sha(__file__),'corrected_forward_sha256':sha(directory/'forward_corrected.py.txt'),
        'original_model_source_sha256':sha(ROOT/'common.py'),'primary_results_unchanged':True})
    runpy.run_path(str(ROOT/'runtime/finalize.py'),run_name='__main__')
    print('DIAGNOSTIC_CORRECTION_COMPLETE',flush=True)


if __name__=='__main__':main()
