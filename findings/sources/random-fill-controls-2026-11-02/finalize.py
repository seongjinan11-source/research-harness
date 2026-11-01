"""Score finished workers, write the scoped verdict and archive the evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as X
import shutil
import yaml


def main():
    X.analyze()
    root=X.ROOT;c=X.config()
    summary=X.json.loads((root/'results/summary.json').read_text())
    paired=X.json.loads((root/'results/paired.json').read_text())
    decision=X.json.loads((root/'results/decision.json').read_text())
    replay=X.json.loads((root/'results/replay.json').read_text())
    audit=X.json.loads((root/'results/audit.json').read_text())
    original_summary=X.json.loads((X.REFERENCE/'results/summary.json').read_text())
    def value(g,a):return next(r for r in summary if r['group']==g and r['arm']==a)
    def difference(g,a,metric='nll'):
        return next(r for r in paired if r['group']==g and r['contrast']=='conditional_joint - '+a and r['metric']==metric)
    def ci(g,a):
        p=difference(g,a);return f"{p['delta']:+.5f} [{p['low']:+.5f}, {p['high']:+.5f}]"
    lines=['# Frozen sampling — random-fill 대조 결과', '',
        '실행 완료: 2026-11-02. **새 neural PFN 학습이 아닌, 설계안 첫 단계의 실제 frozen inference 실험이다.**', '',
        f"판정: **{decision['decision']}**", '',
        '기존32개 개발 graph·3 frozen checkpoints·4 probability views를 유지했다. Uniform/Clustered는 각각 node missing90/99.5 평균, context 최대4/class다.', '',
        '| Input | Uniform NLL ↓ | Clustered NLL ↓ | Uniform acc | Clustered acc | 보조 k1 NLL ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for a in c['arms']:
        u,v,k=(value(g,a) for g in ('uniform','clustered','k1'))
        lines.append(f"| {a} | {u['nll']:.5f} | {v['nll']:.5f} | {u['accuracy']*100:.2f}% | {v['accuracy']*100:.2f}% | {k['nll']:.5f} |")
    lines+=['', '| Conditional − null NLL | Uniform [95% CI] | Clustered [95% CI] |','|---|---:|---:|']
    for a in c['arms'][1:]:lines.append(f'| {a} | {ci("uniform",a)} | {ci("clustered",a)} |')
    lines+=['',f"주 gate 통과: **{decision['primary_passed']}**. 두 variance-matched null과 두 주 group의 네 비교를 모두 만족해야 통과다. k1이나 개별 seed로 gate를 바꾸지 않는다.", '',
        '## 무엇을 통제했는가', '',
        '- 모두 같은 관측값·mask·graph·context labels·query·checkpoint·feature-position view schedule을 사용한다.',
        '- 두 variance-matched null은 원래 mixture의 analytical per-cell conditional variance를 받는다. 관측값에 의존한 scale 정보가 남으므로 완전 unconditional control이라고 주장하지 않는다.',
        '- Unit-noise의 missing proposal에는 실제 관측값을 쓰지 않는다. 실제 관측 위치에는 모든 방법이 원래 값을 그대로 넣는다.',
        '- Wrong-sign은 missing mean에만 개입하고 conditional covariance/rho/noise를 유지한 진단이다. 새 hybrid로 선택하지 않는다.',
        '- Primary conditional은 원래 probability 파일을 바로 붙이지 않고 같은 worker에서 다시 실행했다. Full-observed에서는 모든 입력이 같아 같은 probabilities를 공유했다.', '',
        '## 기존 point baseline과 실행 감사', '',
        '| 이전 동일 bank 입력 대조 | Uniform NLL | Clustered NLL |','|---|---:|---:|']
    for a in ('zero','pcfi','posterior_mean'):
        def ov(g):return next(r['nll'] for r in original_summary if r['group']==g and r['arm']==a)
        lines.append(f'| {a} | {ov("uniform"):.5f} | {ov("clustered"):.5f} |')
    lines+=['',
        '위 point 수치는 이전 실행의 저장 결과이며 새로운 null 주 gate와 구분한다. PCFI도 같은 zero-input-trained checkpoint의 입력 대조다. 별도 PCFI/FISF-trained 모델 비교로 확대하지 않는다.', '',
        f"- 재채점한 확률 파일 {audit['probability_files']}개, scored views{audit['scored_views']}개, optimizer update0회.",
        f"- Original conditional replay 최대 확률 차이 {replay['maximum_probability_delta']:.8f}, 평균 NLL 차이 {replay['mean_nll_delta']:+.8f}.",
        '- Source·input bank·completion bank·checkpoint hashes와 frozen weights를 검증했다. 원래 실험 파일을 변경하지 않았다.',
        '- Bootstrap은 graph32개를 paired resampling하며 rates와 fixed3 checkpoint seeds를 먼저 평균했다. Completion RNG와 checkpoint seeds 모집단의 불확실성은 포함하지 않고, 다중 비교 보정도 하지 않았다.', '',
        '## 해석의 경계', '',
        'Null과 구분되지 않는다면 이전 sampling 개선을 유효한 observed-value conditional completion이나 참 posterior 추론의 성과라고 해석할 근거가 부족하다. 이것이 모든 observed feature의 무용함을 의미하지 않는다. 실제 값의 직접 PFN 입력 경로는 유지돼 있다.', '',
        '이 결과는 context labels를 feature law에 사용하는 모델의 성공·실패 판정이 아니다. Shared encoder·hypothesis suffix와 coherence training도 아직 실행하지 않았다. 이 frozen operator의 gate를 통과하지 못하면 자동으로 추가 head/loss/규모를 붙여 확장하지 않는다. 새로운 conditional-law 학습은 별도 가설과 강한 대조를 가져야 한다.', '',
        '전체 condition·seed·accuracy/Brier CI는 `results/summary.json`, `paired.json`에 보관한다. `protocol.json`은 사전 설정, `experiment.py`는 실제 실행 source다.', '']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    harness=root.parent/'research-harness';archive=harness/'findings/sources/random-fill-controls-2026-11-02'
    archive.mkdir(parents=True,exist_ok=True)
    files=[('REPORT.md','REPORT.md'),('README.md','README.md'),('protocol.json','protocol.json'),
        ('experiment.py','experiment.py'),('runtime/finalize.py','finalize.py'),('bank/manifest.json','bank_manifest.json')]
    files += [('results/'+n,n) for n in ('summary.json','paired.json','replay.json','decision.json','audit.json','preflight.json','lock.json')]
    manifest=[]
    for relative,alias in files:
        source=root/relative;target=archive/alias;shutil.copyfile(source,target)
        assert X.E.C.sha(source)==X.E.C.sha(target)
        manifest.append({'alias':alias,'original':str(source),'snapshot':target.relative_to(harness).as_posix(),
            'sha256':X.E.C.sha(source),'bytes':source.stat().st_size})
    write=lambda path,data:X.write(path,data)
    write(archive/'manifest.json',{'date':'2026-11-02','sources':manifest,
        'note':'New frozen controls on the existing development bank. Probability files remain at the original run location.'})
    prefix=archive.relative_to(harness).as_posix()+'/'
    observation='; '.join(f"{g}: Conditional{value(g,'conditional_joint')['nll']:.5f}, iid-matched{value(g,'iid_variance_matched')['nll']:.5f}, graph-prior-matched{value(g,'graph_prior_variance_matched')['nll']:.5f}, iid-unit{value(g,'iid_unit')['nll']:.5f}, wrong-sign{value(g,'wrong_sign_mean')['nll']:.5f}" for g in ('uniform','clustered'))
    entries=[]
    if not decision['primary_passed']:
        entries.append({'id':'RFILL-F01','kind':'failure',
            'claim':'현재 frozen conditional Gaussian completion이 분산을 맞춘 random-fill/graph-prior null보다 안정적으로 좋은 예측을 제공한다.',
            'observation':observation+'; Conditional−iid matched '+ci('uniform','iid_variance_matched')+' / '+ci('clustered','iid_variance_matched')+'; Conditional−graph prior matched '+ci('uniform','graph_prior_variance_matched')+' / '+ci('clustered','graph_prior_variance_matched'),
            'verdict':'사전 random-control gate 미통과; 현재 frozen conditional의 독립 가치 미입증',
            'failure_reason':'동일 관측 입력·checkpoint·분산/forward-budget 대조에서 요구한 conditional law 우위가 없었다. 관측값에 조건화한 복원, 구조 random feature, 확률 smoothing의 원인별 기여를 완전히 식별한 결과는 아니다.',
            'reason_status':'unknown','excluded_explanations':['Null은 분산·동일 PFN 호출 수를 맞췄다. 이것이 정확한 true posterior 또는 모든 feature law에 대한 대조를 뜻하지는 않는다.'],
            'closed_scope':'현재 zero-input-trained PFN과 Gaussian surrogate를 실제 feature posterior의 메인 메소드로 확대하는 것',
            'not_closed':'실제 관측 피처의 직접 효용·context-conditioned feature-law 학습·shared suffix·coherence 일반은 평가하지 않았다. 분산 맞춤 null에는 second-moment 정보가 남고 unit null에는 실제 값 clamp가 남는다.',
            'reopen_when':'Context와 관측 피처가 joint feature law에 제공하는 추가 정보를 proper feature score와 실제 query improvement로 입증하고, 같은 예산의 random/구조 입력보다 좋은 연산이 나올 때.',
            'comparison_floor':'Variance-matched iid/graph prior, iid unit, zero/PCFI/FISF 입력 및 같은 예산·별도 학습의 강한 PFN. 실제 관측 입력 직접 경로와 proposal 경로를 구분.',
            'recheck':'새 학습 가설을 source/data/budget 고정 후 별도 bank에서 비교한다. Label-blind law 대조와 true-label-context conditional-law 증분을 분리하며 query labels를 forward에 사용하지 않는다.',
            'tags':['observation-mask','uncertainty','eval-protocol','graph-structure-use'],
            'keywords':['random fill','variance matched','conditional feature law','graph random features','frozen PFN'],
            'reopen_tags':['observation-mask','uncertainty','graph-structure-use'],
            'reopen_keywords':['context-conditioned feature posterior','joint feature law','random fill','conditional imputation'],
            'source_refs':[prefix+'REPORT.md',prefix+'paired.json'],'evidence':'기존 개발 bank 32 graphs×3 frozen checkpoints×4 views의 실제 대조',
            'revisit_status':'open'})
    else:
        entries.append({'id':'RFILL-P01','kind':'preserved','claim':'현재 frozen Gaussian conditional이 두 variance-matched null을 두 주 group 모두에서 이겼다.',
            'observation':observation,'do_not_claim':'새 neural model·context-conditioned law·실제 graph/schema 전이·최종 독립 검증·새 메소드 novelty는 미입증.',
            'tags':['observation-mask','uncertainty','graph-structure-use'],'keywords':['random fill','conditional feature law'],
            'source_refs':[prefix+'REPORT.md',prefix+'paired.json'],'evidence':'사전 paired graph gate'})
    entries += [{'id':'RFILL-L01','kind':'lesson','claim':'분산 맞춤 null을 완전히 무정보 random baseline이라고 부르지 않는다.',
        'observation':'Iid/graph-prior variance-matched null에는 원래 conditional variance 정보가 남는다. Unit-noise에는 missing proposal의 관측값 정보가 없지만 실제 관측 위치의 값은 여전히 PFN에 제공한다. Null 우위/동등성을 전체 observed feature의 무용함으로 확대하지 않는다.',
        'tags':['eval-protocol','observation-mask','uncertainty'],'keywords':['variance matching','observed clamp','proposal path','random control'],
        'source_refs':[prefix+'README.md',prefix+'protocol.json'],'evidence':'실행 source와 입력 경로 정의'},
        {'id':'RFILL-U01','kind':'untested','claim':'이 대조 이후의 context-conditioned neural feature law와 shared PFN 학습',
        'observation':'이번 실행의 optimizer update는0회다. 새 모델을 실제 학습한 결과가 아니며 CFB-U01의 미실행 항목을 완료나 실패로 바꾸지 않는다.',
        'tags':['amortized-inference','observation-mask','zero-shot-gap'],'keywords':['context-conditioned feature posterior','shared encoder','coherence training'],
        'source_refs':[prefix+'REPORT.md'],'evidence':'실제 실행 범위'}]
    target=harness/'findings/random-fill-controls.yaml'
    assert not target.exists(),'Preserve old verdict rather than overwrite it'
    target.write_text(yaml.safe_dump({'project':'Frozen feature sampling random-fill controls','date':'2026-11-02',
        'source':prefix+'REPORT.md','setting':'기존 개발32 graphs×6 conditions, 3 frozen checkpoints, 5 arms×4 views. Parameter updates0. 같은 process conditional replay와 null 비교.',
        'related':['GraphPFN','PEMix','MCSSL'],'entries':entries},allow_unicode=True,sort_keys=False,width=110),encoding='utf-8')
    write(root/'results/completed.json',{'primary_passed':decision['primary_passed'],'report_sha256':X.E.C.sha(root/'REPORT.md'),
        'archived_sources':len(manifest),'new_model_training':False})
    print('FINALIZED',decision['primary_passed'],flush=True)


if __name__=='__main__':main()
