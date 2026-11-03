import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil,re
from collections import Counter
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'research-harness';DEST=REPO/'findings/sources/node-frame-2026-11-03'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def preserve():
    before=read(ROOT/'runtime/harness_before.json');now={}
    for p in (REPO/'findings').glob('*.yaml'):
        if p.name.startswith('_'):continue
        for e in yaml.safe_load(p.read_text(encoding='utf8'))['entries']:now[e['id']]=sha(json.dumps(e,sort_keys=True,ensure_ascii=False).encode())
    assert all(now[k]==v for k,v in before['entries'].items())
    assert all(sha((REPO/p).read_bytes())==v for p,v in before['files'].items())
    return {'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files'])}

def main():
    preservation=preserve();assert not DEST.exists();DEST.mkdir(parents=True)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'protocol.json',ROOT/'bank/manifest.json']+list((ROOT/'results').rglob('*.json'))+[Path(__file__),ROOT/'runtime/finish.py']
    for p in files:
        q=DEST/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for rel in ['nsd_token_pfn/core.py','nsd_token_pfn/bank.py','nsd_token_pfn/run.py','nsd_token_pfn/protocol.json',
                'joint_imputation_pfn/common.py','regret_prior_pilot/core.py','sheaf_pcfi_pilot/core.py','sheaf_pcfi_pilot/data.py','fisf_graphpfn/imputation.py']:
        p=ROOT.parent/rel;q=DEST/'reference/workspace'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    (DEST/'ARCHIVE_SCOPE.md').write_text('Research snapshot: source, protocol, reports, manifests, metric records and audits. Raw episodes/probabilities/checkpoints remain in the workspace. Original NSD experiment sources and oldrealpredictions stay immutable and referenced. This is not a standalone training distribution. Historical source instructions do not authorize new runs.\n',encoding='utf8')
    s=read(ROOT/'results/statistics.json');a=read(ROOT/'results/audit.json');p=read(ROOT/'results/preflight.json');go=s['gates']['expand']
    tags=['graph-geometry','graph-structure-use','observation-mask','zero-shot-gap','eval-protocol']
    keywords=['orthogonal node frames','path-independent transport','cycle consistency','flat SO2 connection','Tij Qi transpose Qj','missing feature PFN']
    refs=[(DEST/f).relative_to(REPO).as_posix() for f in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/audit.json','results/control_summary.json','results/preflight.json','results/cost.json']]
    obs='; '.join(f"node-{arm} {100*s['comparisons_node_minus'][arm]['accuracy']['mean']:+.6f}%p CI{[100*v for v in s['comparisons_node_minus'][arm]['accuracy']['ci95']]}" for arm in ['identity','dynamic','static','pcfi100'])
    obs+=f"; realhetero-nodePCFI {100*s['real_hetero_differences']['pcfi100']:+.6f}%p; realhomo-nodePCFI {100*s['real_homo_difference']:+.6f}%p; gates={s['gates']}"
    common={'tags':tags,'keywords':keywords,'source_refs':refs,'evidence':'3newcheckpoints/3072queryCEupdates;identical512synthetictraininputs/initialization/schedule tooriginaldynamic;all6armsreevaluatedon64freshsame-familytestgraphs;4oldrealdevelopmentgraphs;0realupdates;5000graphbootstrap;3fixedtrainingseeds.','revisit_status':'open'}
    first=dict(id='NFRAME-P01' if go else 'NFRAME-F01',kind='preserved' if go else 'failure',claim='노드 frame 제약이 기존 edge별 map 및 Identity보다 missing 분류와 실제 graph 전이를 개선한다.',observation=obs,
               verdict='현재 사전 기준을 통과했으며 독립 real/family 확인이 필요하다.' if go else '현재 node-frame 구성은 확대 기준을 통과하지 못했다. 같은 bank에서 frame 차원/loss/예산을 조정해 구제하지 않는다.',**common)
    if go:first['do_not_claim']='SO2 node-frame의 제한된 결과다. 새로운 cycle-consistency 원리/일반 orthogonal model의 우위/미사용 real graph 확인/가설별 missing marginalization을 입증하지 않는다.'
    else:first.update(failure_reason='사전 Identity/Dynamic/PCFI 및 real floor 조건 중 일부가 충족되지 않았다. Frame 식별·표현력·공유token geometry·RP16·prior/학습량의 고유 원인은 분리하지 않았다.',reason_status='unknown',
          closed_scope='현재공유SO2nodeframe/4interleavedstages/RP16/1024updates/knownsyntheticprior/4realdevelopmentgraphs에서메인방법으로확대하는가설.',
          not_closed='원본feature공간,feature/label별독립frame,다른O(d)함수족,새prior/real/schema에대한효과는미검증이다. 이들이해결책이라는뜻은아니다.',
          reopen_when='현재node-frame제약에서손실되는정보또는극소관측에서frame을식별할추가정보를구체적으로제공하는추론연산이있을때.',
          reopen_tags=['graph-geometry','distribution-shift','observation-mask'],reopen_keywords=['missing-aware frame identification','independent feature label transport'],
          comparison_floor='동일초기화/학습순서/예산의Identity/edgeStatic/edgeDynamic/Mask/PCFI,nativePCFI,negativeLP. 동일query피처제거/셔플과NLL/Brier,실제end-to-end비용.',
          recheck='새graph/family와미사용real입력을먼저고정한다. Architecture나적용위치를동시에바꾸지말고node-frame독립증분,source0/2/8,feature/context개입,cycle/gradient/anchor,numericalreplay를검증한다.')
    entries=[first,dict(id='NFRAME-P02',kind='preserved',claim='노드별 공유 SO2 frame과 경로 독립적 연산을 실제 PFN에 구현하고 synthetic-only 학습했다.',
          observation=f"3072validupdates;real0;namedinitialization/trainables4064599/schedulematched;linearalignmenterror{p['aligned_linear_operator_error']},learnedcycleerror{p['learned_triangle_cycle_error']},incidencesharingerror{p['shared_incidence_error']},gradientFD{p['frame_gradient_finite_difference_error']};syntheticaccuracy={ {k:100*v['accuracy'] for k,v in s['synthetic'].items()} }.",
          verdict='연산 구현과 실제 비교값을 보존한다. Path independence의 성립과 예측 성능 우위는 별개다.',
          do_not_claim='직교변환은개별norm보존만보장하며이웃평균상쇄/미관측frame식별/비선형PFNgaugeequivariance를보장하지않는다. 기존NSD이론의새발명도아니다.',**common),
          dict(id='NFRAME-L01',kind='lesson',claim='새 synthetic graph에서는 재사용 비교군도 동일 입력으로 다시 평가하고 real 개발 bank 재사용은 구분한다.',
          observation=f"신규합성64basegraph/6arms×3seed,real4graph는기존입력과비교군확률재사용. Audit={a['status']},replaymax={max(r['saved_max_abs'] for r in a['replay'])}.",
          verdict='새generatorfamily/독립real확인이라고확대하지않고과거bank절대점수를새점수에서빼지않는다. CI는저장예측조건부이며수치변동을따로남긴다.',**common),
          dict(id='NFRAME-L02',kind='lesson',claim='동일 파라미터 수의 node/edge map도 conditioning과 표현 가능한 전달 구조가 다르다.',
          observation='Node는own+closed-neighborhoodmean descriptor,Edge는두endpoint descriptor를동일Phi에넣는다. Node모델은순환변환곱I를강제한다. PCFI입력추가/라벨전용diffusion은함께바꾸지않았다.',
          verdict='동일파라미터와초기화를동일함수용량이나완벽한순환제약단독효과로해석하지않는다. 기존실패원인을공유geometry로확정하지않는다.',**common)]
    project={'project':'Orthogonal node-frame GraphPFN','date':'2026-11-03','source':refs[0],'setting':'Synthetic-only training,realweightsfrozen;replaceedge-incidencemapswithsinglenodeSO2framesinexistingtokenNSD.','related':['NeuralSheafDiffusion'],'entries':entries}
    (REPO/'findings/node-frame.yaml').write_text(yaml.safe_dump(project,allow_unicode=True,sort_keys=False,width=150),encoding='utf8')
    idea={'id':'node-frame-pfn','date':'2026-11-03','status':'completed-pilot-conditional-go' if go else 'completed-bounded-pilot-no-go','text':first['claim'],'tags':tags,'keywords':keywords,'note':refs[1],'followup':refs[0],'pilot_gates':s['gates']}
    (REPO/'ideas/node-frame-pfn.yaml').write_text(yaml.safe_dump(idea,allow_unicode=True,sort_keys=False),encoding='utf8')
    allentries=[];projects=[]
    for f in (REPO/'findings').glob('*.yaml'):
        if f.name.startswith('_'):continue
        projects.append(f);allentries+=yaml.safe_load(f.read_text(encoding='utf8'))['entries']
    counts=Counter(e['kind'] for e in allentries);n=len(allentries);ideas=len(list((REPO/'ideas').glob('*.yaml')))
    section='## 노드 직교 frame으로 제한한 PFN\n\n2026-11-03. 기존 edge별 map 대신 Q_i를 노드마다 공유하고 T_ij=Q_i^T Q_j로 제한했다. 입력·전파 위치·초기화·1,024update·train bank를 유지하고 새3개 모델만 학습했다. 비교군도 새64synthetic graph에서 다시 추론했다. 실제4graph는 기존 개발용 평가다.\n\n'
    section+='| 방법 | Cora | CiteSeer | Texas | Wisconsin |\n|---|---:|---:|---:|---:|\n'
    for arm in ['pcfi100','identity','dynamic','node_frame','native_pcfi100','negative_lp']:
        section+='| '+arm+' | '+' | '.join(f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in ['Cora','CiteSeer','texas','wisconsin'])+' |\n'
    section+='\n'+obs+'\n\n'+first['verdict']+' 경로 일관성의 수학적 성립을 실제 전이 성능으로 확대하지 않는다. [전체보고서]('+refs[0]+'), [계산정의]('+refs[1]+'), [구조화기록](findings/node-frame.yaml).\n\n'
    summary=REPO/'RESULTS_SUMMARY_KO.md';body=summary.read_text(encoding='utf8')
    body=body.replace('## NSD hidden-token PFN: 관측에서 시작하는 geometry 갱신',section+'## NSD hidden-token PFN: 관측에서 시작하는 geometry 갱신',1)
    body=re.sub(r'기록은\d+개 프로젝트 묶음의\d+항목이다\. 실패·중단\d+개, 보존\d+개, 교훈\d+개, 미실행\d+개이며 \*\*\d+번의 독립 실험을 했다는 뜻은 아니다\.\*\*',f"기록은{len(projects)}개 프로젝트 묶음의{n}항목이다. 실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개이며 **{n}번의 독립 실험을 했다는 뜻은 아니다.**",body)
    body=body.replace('graphpfn_review_packet_2026-11-03_v16.zip','graphpfn_review_packet_2026-11-03_v17.zip');summary.write_text(body,encoding='utf8')
    hand=REPO/'RESEARCH_HANDOFF_KO.md';body=hand.read_text(encoding='utf8')
    body=body.replace('NSD hidden-token PFN pilot까지 포함한','노드 직교 frame PFN pilot까지 포함한',1)
    body=body.replace('## Sheaf-PCFI: synthetic 학습·실제 가중치 고정 비교',section+'## Sheaf-PCFI: synthetic 학습·실제 가중치 고정 비교',1)
    body=re.sub(r'연구 기록 \d+항목, 아이디어 \d+개가 있다\. \d+개 프로젝트 묶음이며 실패·중단\d+, 보존\d+, 교훈\d+, 미실행\d+항목이다\. \d+번의 독립 실험이라는 뜻은 아니다\.',f"연구 기록 {n}항목, 아이디어 {ideas}개가 있다. {len(projects)}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목이다. {n}번의 독립 실험이라는 뜻은 아니다.",body)
    body=body.replace('v16은NSD hidden-token', 'v17은노드별orthogonalframe제약의비교를추가했다. v16은NSD hidden-token',1)
    body=body.replace('첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.','첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.\n최신node-frame실험은T_ij=Q_i^T Q_j를이미학습/평가했어. 현재닫은범위와원인미상을구분하고같은제약을미실행새방법으로다시제안하지말아줘.',1);hand.write_text(body,encoding='utf8')
    p=REPO/'README.md';p.write_text(p.read_text(encoding='utf8').rstrip()+'\n\n'+section,encoding='utf8')
    p=REPO/'tools/build_review_packet.py';p.write_text(p.read_text(encoding='utf8').replace('VERSION = 16','VERSION = 17'),encoding='utf8')
    preservation=preserve();dump_path=ROOT/'results/harness_preservation.json';dump_path.write_text(json.dumps(preservation,indent=2),encoding='utf8');shutil.copyfile(dump_path,DEST/'results/harness_preservation.json')
    manifest={p.relative_to(DEST).as_posix():sha(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('ARCHIVED',n,ideas,len(manifest),preservation,flush=True)

if __name__=='__main__':main()
