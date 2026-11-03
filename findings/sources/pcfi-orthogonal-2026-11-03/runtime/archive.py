import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil,re
from collections import Counter
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'research-harness';DEST=REPO/'findings/sources/pcfi-orthogonal-2026-11-03'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def preserve():
    before=read(ROOT/'runtime/harness_before.json');current={}
    for p in (REPO/'findings').glob('*.yaml'):
        if p.name.startswith('_'):continue
        for r in yaml.safe_load(p.read_text(encoding='utf8'))['entries']:current[r['id']]=sha(json.dumps(r,sort_keys=True,ensure_ascii=False).encode())
    assert all(current[k]==v for k,v in before['entries'].items())
    assert all(sha((REPO/p).read_bytes())==v for p,v in before['files'].items())
    return {'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files'])}

def main():
    preservation=preserve();assert not DEST.exists();DEST.mkdir(parents=True)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'protocol.json',ROOT/'bank/manifest.json']+list((ROOT/'results').rglob('*.json'))+[Path(__file__),ROOT/'runtime/finish.py',ROOT/'runtime/numerical_probe.py']
    for p in files:
        q=DEST/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for rel in ['nsd_token_pfn/core.py','nsd_token_pfn/bank.py','nsd_token_pfn/protocol.json','joint_imputation_pfn/common.py',
                'regret_prior_pilot/core.py','sheaf_pcfi_pilot/core.py','sheaf_pcfi_pilot/data.py','fisf_graphpfn/imputation.py']:
        p=ROOT.parent/rel;q=DEST/'reference/workspace'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    (DEST/'ARCHIVE_SCOPE.md').write_text('Snapshot of code, protocol, report, manifests, training/evaluation metrics and audits. Raw feature/extra/probability NPZs and frontend/frozenPFN checkpoints remain in the workspace. Same frozenPFN checkpoint per seed is reused by all arms. Source paths retain workspace references; not a standalone training distribution. Historical source instructions do not authorize new runs.\n',encoding='utf8')
    s=read(ROOT/'results/statistics.json');a=read(ROOT/'results/audit.json');p=read(ROOT/'results/preflight.json');go=[x for x in ['edge','node'] if s['gates'][x]['expand']]
    tags=['graph-geometry','graph-structure-use','observation-mask','zero-shot-gap','eval-protocol']
    keywords=['PCFI spatial operator replacement','confidence weighted graph sheaf diffusion','numeric feature imputation','fixed downstream GraphPFN','orthogonal edge transport','orthogonal node frame','first-coordinate readout']
    refs=[(DEST/f).relative_to(REPO).as_posix() for f in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/audit.json','results/control_summary.json','results/preflight.json','results/cost.json']]
    obs='; '.join(f"{candidate}-{other} {100*s['comparisons'][candidate][other]['accuracy']['mean']:+.6f}%p CI{[100*v for v in s['comparisons'][candidate][other]['accuracy']['ci95']]}" for candidate in ['edge','node'] for other in ['pcfi','scalar'])
    obs+='; realhetero '+str({x:{k:100*v for k,v in s['real_hetero_differences'][x].items() if k in ['pcfi','scalar','native_pcfi','negative_lp']} for x in ['edge','node']})+'%p; gates='+str(s['gates'])
    common={'tags':tags,'keywords':keywords,'source_refs':refs,'evidence':'9new737parameterfrontends/9216queryCEupdates;allPFNweightsfixedthroughtraining/inference;3oldsynthetictrainedPFNcheckpoints reused;512same traininputs/schedule;64freshsame-familytestgraphs;4oldrealdevelopmentgraphs;5000graphbootstrap/3fixedseeds/unadjustedmultiplecontrasts.','revisit_status':'open'}
    first=dict(id='PCFIO-P01' if go else 'PCFIO-F01',kind='preserved' if go else 'failure',claim='PCFI의 confidence·경계값·100회·채널 보정을 유지하고 수치 공간 전파만 직교 Sheaf로 교체하면 고정 PFN에서 강한 대조보다 missing 분류가 좋아진다.',observation=obs,
               verdict='사전 확대 기준 통과 후보: '+','.join(go)+'. 동일family/개발real범위의결과로보존한다.' if go else '현재 두 후보는 사전 확대 기준을 통과하지 못했다. 일부 대조의 개선이 있으면 분리해 보존하고 같은 bank의 loss/차원/예산 조정으로 구제하지 않는다.',**common)
    if go:first['do_not_claim']='PCFI공간교체의제한된결과다. 원NSD전체재현/새sheaf원리/jointPFN학습/원본차원학습/미사용real또는schema확인으로확대하지않는다.'
    else:first.update(failure_reason='사전 PCFI/학습scalar/real pipeline floor 조건 중 일부가 충족되지 않았다. Map식별·SO2/static함수족·고정PFN적합성·RP16·prior와학습량의고유원인은분리하지않았다.',reason_status='unknown',
          closed_scope='현재staticSO2per-cellnumerictransport/737param/100steps/firstcoordinate/1channelcorrection/frozenPCFI-PFN/1024frontupdates/RP16/knownfamily/4realdevelopmentgraph의메인확대.',
          not_closed='JointPFN재학습,dynamicnumericmaps,일반O(d),원본physicalfeature학습과새family/real/schema는미검증이다. 이들이해결책이라는뜻은아니다.',
          reopen_when='극소관측에서transport를식별할추가정보또는현재함수족으로부족한피처관계가명확하고PCFI및같은예산scalar대조를넘는구체적연산이있을때.',
          reopen_tags=['graph-geometry','observation-mask','distribution-shift'],reopen_keywords=['confidence weighted sheaf imputation','transport identifiability'],
          comparison_floor='같은고정PFN의원PCFI,TorchI환원,동일737parameter/예산Scalar,Edge,Node;nativePCFI,고정negativeLP;full/source0동일입력과확률점수/실제비용.',
          recheck='새graph/family와미사용real입력을먼저고정한다. PCFI공간전파교체와hidden-token확산,추가warmup/반복수,PFN재학습을동시에바꾸지않는다. I환원/gradient/anchor/full/source0확률환원/numericalreplay와동일query개입을검증한다.')
    entries=[first,dict(id='PCFIO-P02',kind='preserved',claim='관측 숫자 피처의 confidence 가중 graph sheaf diffusion과 원PCFI I환원을 실제로 구현하고 고정 PFN을 통해 학습했다.',
          observation=f"NumPyPCFIerror{p['pcfi_numpy_max_abs']},Ireduction0,observedclamp0,source0output0;9fronts/9216updates/PFN0/real0;full&source0={a['full_and_zero_source_reduction']};syntheticaccuracy={ {k:100*v['accuracy'] for k,v in s['synthetic'].items()} }.",
          verdict='실제 대치 계산·정보 경계·고정 분류기와 평가 수치를 보존한다. 구현 성공을 성능 우위로 바꾸지 않는다.',
          do_not_claim='직교transport의개별norm보존은평균상쇄나정확한결측복원을보장하지않는다. SO2static숫자연산이며전체PCFI/NSD/PFN의새확률모형이나일반우위가아니다.',**common),
          dict(id='PCFIO-L01',kind='lesson',claim='PCFI 확산 교체, warm100+추가100, PFN hidden-token diffusion은 서로 다른 가설이다.',
          observation='이번은결측0시작/100spatial/1correction/고정PFN이다. 초기수치Sheaf는warm100+추가100symmetriccontraction과공동PFN학습,최근NSD는숫자대치없이hiddenfeature/label토큰전파였다.',
          verdict='다른입력표현·반복수·PFN학습경로의차이를Sheaf자체의단독효과로설명하지않는다. 원래사용자질문은관측숫자의confidence가중Sheaf결측추정임을유지한다.',**common),
          dict(id='PCFIO-L02',kind='lesson',claim='I 초기화와 I 환원 검사, 학습할 수 있는 map 초기화는 구분한다.',
          observation='Scalar를(x,0)으로lift하고첫좌표만읽으면정확한I에서angle의첫미분이0이될수있다. 작은비영초기화에서모든737paramfront의실제gradient/optimizerstep을검증했다. PFN은모든학습동안고정이었다.',
          verdict='정확한I로시작했다는이유만으로안전하게학습가능하다고가정하지않고,실제원PCFI환원과frontendgradient를함께검사한다.',**common)]
    project={'project':'Confidence weighted numeric Sheaf PCFI with fixed PFN','date':'2026-11-03','source':refs[0],
             'setting':'Replace only100-stepPCFIspatialmessageswithstaticSO2transport,keepconfidence/clamp/correlationandidenticalfrozenPFN.','related':['NeuralSheafDiffusion'],'entries':entries}
    (REPO/'findings/pcfi-orthogonal.yaml').write_text(yaml.safe_dump(project,allow_unicode=True,sort_keys=False,width=150),encoding='utf8')
    idea={'id':'pcfi-orthogonal-replacement','date':'2026-11-03','status':'completed-pilot-conditional-go' if go else 'completed-bounded-pilot-no-go','text':first['claim'],'tags':tags,'keywords':keywords,'note':refs[1],'followup':refs[0],'pilot_gates':s['gates']}
    (REPO/'ideas/pcfi-orthogonal-replacement.yaml').write_text(yaml.safe_dump(idea,allow_unicode=True,sort_keys=False),encoding='utf8')
    projects=[];allentries=[]
    for f in (REPO/'findings').glob('*.yaml'):
        if f.name.startswith('_'):continue
        projects.append(f);allentries+=yaml.safe_load(f.read_text(encoding='utf8'))['entries']
    counts=Counter(e['kind'] for e in allentries);n=len(allentries);ni=len(list((REPO/'ideas').glob('*.yaml')))
    section='## PCFI의 confidence 가중 공간 확산을 Sheaf로 교체\n\n2026-11-03. 원래 질문으로 돌아와 관측 숫자→confidence 가중 Sheaf 확산→결측 추정→고정 PFN을 비교했다. PCFI confidence·관측값 고정·100회·채널 보정을 유지했고,PFN은학습중에도고정했다. Scalar/Edge/Node를각3seed·1024updates로학습했다.\n\n'
    section+='| 방법 | Cora | CiteSeer | Texas | Wisconsin |\n|---|---:|---:|---:|---:|\n'
    for arm in ['pcfi','scalar','edge','node','native_pcfi','negative_lp']:
        section+='| '+arm+' | '+' | '.join(f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in ['Cora','CiteSeer','texas','wisconsin'])+' |\n'
    section+='\n'+obs+'\n\n'+first['verdict']+' 고유원인은미상으로남기고hidden-token실험과구분한다. [전체보고서]('+refs[0]+'), [계산정의]('+refs[1]+'), [구조화기록](findings/pcfi-orthogonal.yaml).\n\n'
    summary=REPO/'RESULTS_SUMMARY_KO.md';body=summary.read_text(encoding='utf8').replace('## 노드 직교 frame으로 제한한 PFN',section+'## 노드 직교 frame으로 제한한 PFN',1)
    body=re.sub(r'기록은\d+개 프로젝트 묶음의\d+항목이다\. 실패·중단\d+개, 보존\d+개, 교훈\d+개, 미실행\d+개이며 \*\*\d+번의 독립 실험을 했다는 뜻은 아니다\.\*\*',f"기록은{len(projects)}개 프로젝트 묶음의{n}항목이다. 실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개이며 **{n}번의 독립 실험을 했다는 뜻은 아니다.**",body)
    summary.write_text(body.replace('graphpfn_review_packet_2026-11-03_v17.zip','graphpfn_review_packet_2026-11-03_v18.zip'),encoding='utf8')
    hand=REPO/'RESEARCH_HANDOFF_KO.md';body=hand.read_text(encoding='utf8').replace('노드 직교 frame PFN pilot까지 포함한','confidence 가중 Sheaf-PCFI 공간 교체 pilot까지 포함한',1)
    body=body.replace('## 노드 직교 frame으로 제한한 PFN',section+'## 노드 직교 frame으로 제한한 PFN',1)
    body=re.sub(r'연구 기록 \d+항목, 아이디어 \d+개가 있다\. \d+개 프로젝트 묶음이며 실패·중단\d+, 보존\d+, 교훈\d+, 미실행\d+항목이다\. \d+번의 독립 실험이라는 뜻은 아니다\.',f"연구 기록 {n}항목, 아이디어 {ni}개가 있다. {len(projects)}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목이다. {n}번의 독립 실험이라는 뜻은 아니다.",body)
    body=body.replace('v17은노드별orthogonalframe제약','v18은고정PFN앞의confidence가중Sheaf숫자대치를추가했다. v17은노드별orthogonalframe제약',1)
    body=body.replace('첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.','첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.\n최신실험은관측숫자에confidence가중Sheaf확산을적용해결측을추정하고같은고정PFN에넣었어. Hidden-tokenNSD나warm100+추가100과다른계산이며이미실행한범위와미실행joint학습을구분해줘.',1);hand.write_text(body,encoding='utf8')
    f=REPO/'README.md';f.write_text((f.read_text(encoding='utf8').rstrip()+'\n\n'+section).rstrip()+'\n',encoding='utf8')
    f=REPO/'tools/build_review_packet.py';f.write_text(f.read_text(encoding='utf8').replace('VERSION = 17','VERSION = 18'),encoding='utf8')
    preservation=preserve();f=ROOT/'results/harness_preservation.json';f.write_text(json.dumps(preservation,indent=2),encoding='utf8');shutil.copyfile(f,DEST/'results/harness_preservation.json')
    manifest={p.relative_to(DEST).as_posix():sha(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('ARCHIVED',n,ni,len(manifest),preservation,flush=True)

if __name__=='__main__':main()
