"""Archive the finished experiment and append scoped findings without rewriting history."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiment as E
import hashlib
import json
import shutil
import yaml


def main():
    complete=json.loads((ROOT/'results/completed.json').read_text());assert E.C.sha(ROOT/'REPORT.md')==complete['report_sha256']
    c=E.cfg();decision=complete['decision'];paired=json.loads((ROOT/'results/paired.json').read_text())
    summary=json.loads((ROOT/'results/summary.json').read_text());audit=json.loads((ROOT/'results/audit.json').read_text())
    harness=ROOT.parent/'research-harness';finding=harness/'findings/mechanism-crossed-prior.yaml'
    assert not finding.exists(),'Preserve existing finding; no automatic overwrite'
    folder=harness/'findings/sources/mechanism-crossed-2026-11-02';folder.mkdir(parents=True,exist_ok=True)
    files={
        'REPORT.md':ROOT/'REPORT.md','README.md':ROOT/'README.md','protocol.json':ROOT/'protocol.json',
        'experiment.py':ROOT/'experiment.py','analyze.py':ROOT/'runtime/analyze.py','finalize.py':Path(__file__),
        'reference_common.py':E.REFERENCE/'common.py','reference_core.py':E.C.WORK/'regret_prior_pilot/core.py',
        'bank_manifest.json':ROOT/'bank/manifest.json','exposure.json':ROOT/'bank/exposure.json',
        'oracle_information.json':ROOT/'bank/oracle_information.json','official_generator.json':ROOT/'bank/official_generator.json',
        'summary.json':ROOT/'results/summary.json','paired.json':ROOT/'results/paired.json',
        'interventions.json':ROOT/'results/interventions.json','sampling.json':ROOT/'results/sampling.json',
        'decision.json':ROOT/'results/decision.json','audit.json':ROOT/'results/audit.json',
        'preflight.json':ROOT/'results/preflight.json','lock.json':ROOT/'results/lock.json',
        'analysis_lock.json':ROOT/'results/analysis_lock.json'}
    for seed in c['seeds']:
        for arm in c['arms']:files[f'training_{arm}_s{seed}.json']=ROOT/'results'/f'{arm}_s{seed}'/'training.json'
    manifest=[]
    for alias,original in files.items():
        target=folder/alias;shutil.copyfile(original,target)
        assert original.read_bytes()==target.read_bytes()
        manifest.append({'alias':alias,'original':str(original),'snapshot':target.relative_to(harness).as_posix(),
            'sha256':E.C.sha(target),'bytes':target.stat().st_size})
    E.write(folder/'manifest.json',{'date':'2026-11-02','sources':manifest,
        'note':'Verbatim experiment sources/results. Raw latent banks, probability outputs and weights remain at original run paths.'})
    ref=lambda name:(folder/name).relative_to(harness).as_posix()
    def item(group,other,metric='nll'):
        return next(r for r in paired if r['group']==group and r['contrast']==f'crossed_mask:standard - {other}:standard' and r['metric']==metric)
    def desc(r):return f"{r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}]"
    contrasts='; '.join(f"{g} vs {a}: {desc(item(g,a))}" for g,a in [
        ('controlled:offdiagonal:uniform','coupled_mask'),('controlled:offdiagonal:clustered','coupled_mask'),
        ('controlled:offdiagonal:uniform','coupled_pcfi'),('controlled:offdiagonal:clustered','coupled_pcfi'),
        ('official:any:uniform','official_mask'),('official:any:clustered','official_mask')])
    entries=[]
    if decision['primary_passed']:
        entries.append({'id':'MCP-P01','kind':'preserved','claim':'현재 mechanism-crossing pilot의 두 사전 gate 통과',
            'observation':contrasts,'do_not_claim':'독립 SCM의 novelty, 기존 공식 prior 실패 원인, real graph·unseen schema·full-scale PFN의 우위로 확대하지 않는다.',
            'tags':['prior-mismatch','pretraining-data','observation-mask'],'keywords':['mechanism crossed prior','controlled marginals','official transfer'],
            'source_refs':[ref('REPORT.md'),ref('paired.json')],'evidence':'3 training seeds와32 fresh base graphs의 실제 학습·평가'})
    else:
        entries.append({'id':'MCP-F01','kind':'failure',
            'claim':'동일 주변분포의 feature-style/label-rule crossing이 강한 입력 기준선을 넘어 공식-prior 결측 과제로 전이된다.',
            'observation':f"Gate A={decision['gate_a_controlled_offdiagonal']}, Gate B={decision['gate_b_official_transfer']}; "+contrasts,
            'verdict':'두 gate 동시 통과 실패; 현재 controlled prior를 메인 방법으로 확대하지 않음',
            'failure_reason':'정해진 초기화·256 성공 updates·함수족에서 요구한 predictive 우위를 확보하지 못했다. Prior의 한계, 학습량, preprocessing, context 식별 난도 중 원인을 분리한 실험은 아니다.',
            'reason_status':'unknown','excluded_explanations':['서로 다른 trainable parameter 수나 seed별 update schedule 차이로 대조를 설명할 수 없도록 검사했다.'],
            'closed_scope':'이 finite controlled family와 공개 Mask Plain continuation 설정에서 단순 mechanism crossing만으로 범용 missing GraphPFN 성과를 주장하는 것',
            'not_closed':'다른 그래프 prior·관측 정보가 충분한 task family·실제 graph·동적 learned generator 일반을 기각하지 않는다. 기존 공식 prior의 induced coupling 자체가 실패 원인이라고 입증하지 않았다.',
            'reopen_when':'어떤 관측 정보와 예측 rule의 조합이 전이에 필요한지를 구체적으로 바꾸는 generator/inference 원리가 나오고 동일 비용의 강한 대조를 넘을 근거가 생길 때.',
            'comparison_floor':'동일 구조·초기화·update의 Official/Coupled Mask, 입력으로 학습한 Coupled PCFI, context/uniform prior, LP, 4-view random-fill; 실제 graph에서는 PCFI/FISF-trained PFN·Ephris도 필요.',
            'recheck':'새 독립 bank에서 full/node/cell 조건과 actual-feature 제거·셔플을 고정해 검증한다. 같은 bank의 loss/epochs 튜닝으로 구제하지 않는다. Controlled/official family 변경과 coupling만의 효과를 분리한다.',
            'tags':['prior-mismatch','pretraining-data','observation-mask','distribution-shift'],
            'keywords':['mechanism crossed prior','independent SCM','same marginal distributions','controlled prior','official transfer'],
            'reopen_tags':['prior-mismatch','pretraining-data','compositional-generalization'],
            'reopen_keywords':['predictive mechanism prior','observation information','graph task generation','mechanism recombination'],
            'source_refs':[ref('REPORT.md'),ref('paired.json'),ref('protocol.json')],
            'evidence':'실제12 checkpoint 학습 및32 fresh base graph 단위 paired 분석','revisit_status':'open'})
    positive=[r for r in paired if r['group'] in ['controlled:offdiagonal:uniform','controlled:offdiagonal:clustered']
        and r['contrast']=='crossed_mask:standard - coupled_mask:standard' and r['metric']=='nll' and r['high']<0]
    if positive and not decision['primary_passed']:
        entries.append({'id':'MCP-P02','kind':'preserved','claim':'Constructed coupling을 바꾼 일부 off-diagonal 조건의 NLL 효과',
            'observation':'; '.join(r['group']+': '+desc(r) for r in positive),
            'do_not_claim':'다른 기준선·official 전이·accuracy·real graph까지 자동 확대하지 않는다. 전체 gate 판정은 유지한다.',
            'tags':['prior-mismatch','distribution-shift'],'keywords':['off diagonal','mechanism crossing','controlled effect'],
            'source_refs':[ref('paired.json'),ref('REPORT.md')],'evidence':'사전 대조의 부분 조건 효과'})
    entries.append({'id':'MCP-L01','kind':'lesson',
        'claim':'Mechanism-crossing의 실제 학습을 기존 finite-bank curriculum과 분리해 기록한다.',
        'observation':f"NEXT-U01의 첫 pilot을 실행했다. {audit['checkpoints']} checkpoints, {audit['successful_updates']} successful updates, {audit['probability_files_rescored']} probability files 재채점. Concrete feature/mask/query 및 rule-instance marginals를 맞췄고 model 구조·초기화·schedule을 유지했다. One graph/update이므로 paired loss는 없다.",
        'tags':['eval-protocol','pretraining-data','prior-mismatch'],'keywords':['matched marginal exposures','no paired loss','same initialization','synthetic task distribution'],
        'source_refs':[ref('audit.json'),ref('exposure.json'),ref('preflight.json')],'evidence':'실행·source·hash 감사'})
    entries.append({'id':'MCP-U01','kind':'untested','claim':'Real dense-feature graph, heterogeneous schema 및 본격적인 separable generative prior 검증',
        'observation':'현재 실행은 graph/function seed가 새로운 작은 binary synthetic family다. 원래 GraphPFN 실패의 causal 설명, 실제 데이터 전이, FISF/Ephris 비교, 모든 induced laws의 분리성 보장은 수행하지 않았다. Known-rule sign-only 정보량 진단은 배포 predictor의 성과가 아니다.',
        'tags':['zero-shot-gap','feature-heterogeneity','theory-gap'],'keywords':['real graph transfer','unseen schema','induced law separability','privileged information check'],
        'source_refs':[ref('REPORT.md'),ref('oracle_information.json')],'evidence':'명시한 미실행 범위'})
    finding.write_text(yaml.safe_dump({'project':'Mechanism-crossed GraphPFN controlled-prior pilot','date':'2026-11-02',
        'source':ref('REPORT.md'),'setting':'새64 train/32 test base graphs,4 arms×3 seeds×256 updates; 공개 Mask Plain continuation, 마지막 checkpoint 평가.',
        'related':['GraphPFN','PFNSeparability','GNNmim','Ephris'],'entries':entries},allow_unicode=True,sort_keys=False),encoding='utf8')
    E.write(ROOT/'results/finalization.json',{'archived_sources':len(manifest),'finding':finding.relative_to(harness).as_posix(),
        'decision':decision,'report_sha256':E.C.sha(ROOT/'REPORT.md')})
    print('FINALIZED',decision,flush=True)


if __name__=='__main__':main()
