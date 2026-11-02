import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil,re
from collections import Counter
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'research-harness';DEST=REPO/'findings/sources/nsd-token-2026-11-03'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def preserve():
    before=read(ROOT/'runtime/harness_before.json');current={}
    for p in (REPO/'findings').glob('*.yaml'):
        if p.name.startswith('_'):continue
        for e in yaml.safe_load(p.read_text(encoding='utf8'))['entries']:current[e['id']]=sha(json.dumps(e,sort_keys=True,ensure_ascii=False).encode())
    assert all(current[k]==v for k,v in before['entries'].items())
    assert all(sha((REPO/p).read_bytes())==v for p,v in before['files'].items())
    return {'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files'])}
def main():
    preservation=preserve();assert not DEST.exists();DEST.mkdir(parents=True)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'protocol.json',ROOT/'bank/manifest.json']+[p for p in (ROOT/'reference').iterdir() if p.is_file()]+list((ROOT/'results').rglob('*.json'))+[Path(__file__),ROOT/'runtime/finish.py']
    for p in files:
        q=DEST/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for rel in ['joint_imputation_pfn/common.py','regret_prior_pilot/core.py','sheaf_pcfi_pilot/core.py','sheaf_pcfi_pilot/data.py','fisf_graphpfn/imputation.py']:
        p=ROOT.parent/rel;q=DEST/'reference/workspace'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    (DEST/'ARCHIVE_SCOPE.md').write_text('Research snapshot: code, protocol, original report, input/metric manifests, train logs, statistics, controls and audits. Raw feature/episode/probability NPZs, checkpoints and full environment stay in workspace and are excluded. Scripts retain original workspace references; this is not a standalone training distribution. Historical instructions in source documents do not authorize new runs. Official NSD reference files retain their copyright/license headers.\n',encoding='utf8')
    s=read(ROOT/'results/statistics.json');a=read(ROOT/'results/audit.json');go=s['gates']['expand'];tags=['graph-geometry','graph-structure-use','observation-mask','zero-shot-gap','eval-protocol']
    keywords=['Neural Sheaf Diffusion','hidden-token sheaf PFN','dynamic restriction maps','SO2 connection Laplacian','observed token anchoring','static dynamic geometry','synthetic-only real frozen']
    refs=[(DEST/p).relative_to(REPO).as_posix() for p in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/audit.json','results/control_summary.json','results/cost.json','results/preflight.json']]
    comp=s['comparisons_dynamic_minus'];obs='; '.join(f"dynamic-{arm} {100*comp[arm]['accuracy']['mean']:+.6f}%p CI{[100*v for v in comp[arm]['accuracy']['ci95']]}" for arm in ['mask','pcfi100','identity','static'])
    obs+='; realhetero '+str({k:100*v for k,v in s['real_hetero_differences'].items()})+'%p; gates='+str(s['gates'])
    common={'tags':tags,'keywords':keywords,'source_refs':refs,'evidence':'15newcheckpoints/15360validupdates,64trainbasegraphs×8cases×2passes,64freshsynthetictestgraphs;sameknownfamily;real4developmentgraphs/3context-maskrepetitions,0realupdates;graph-pairedbootstrap5000;10selectedreplays;no post-test rescue.','revisit_status':'open'}
    first=dict(id='NSDT-P01' if go else 'NSDT-F01',kind='preserved' if go else 'failure',claim='현재 token에서 geometry를 갱신하는 PFN이 PCFI/identity/static보다 missing 분류가 좋고 실제 heterophily의 강한 floor와 경쟁한다.',observation=obs,
        verdict='사전 gate 통과. 독립 실제 graph·다른 family·강한 선행 모델과의 검증이 다음 단계다.' if go else '현재 SO2 token adapter는 사전 gate를 통과하지 못했으며 자동 확대나 추가 loss/head/budget 구제를 하지 않는다.',**common)
    if go:first['do_not_claim']='원본NSD전체재현·새sheaf원리·독립realconfirmation·unseenheterogeneousschema·완전missingmarginalization이아니다. 3seed조건부CI와현재RP16/development한계를유지한다.'
    else:first.update(failure_reason='동일budget의strongarm/realfloor조건중일부가충족되지않았다. 아래수치로확인한현상이며architecture최적화·학습량·RP16·prior차이중고유한원인은분리하지않았다.',reason_status='unknown',
        closed_scope='2D SO2 orthogonalmaps,현재token평균+labeltoken요약,4interleavedstages,기존GraphPFNadapter유지,RP16,1024updates,현재8caseprior와4realdevelopmentgraphs.',
        not_closed='원본NSDtarget-graph학습재현,일반O(d)/generalrestriction,GraphPFNgraphmodule의완전교체,원본physicalfeature공간,새heterogeneousschema는미검증이다. 일부대조의이득은별도로보존한다.',
        reopen_when='새근거가현재함수족/geometry식별/전이에서구체적으로다른정보나추론을제공하고nativePCFI·identity/static·고정negativeLP의floor를넘을이유를보일때.',
        reopen_tags=['graph-geometry','graph-structure-use','distribution-shift'],reopen_keywords=['context-conditioned restriction learning','missing sheaf transport transfer'],
        comparison_floor='동일초기화·taskorder·updates의Mask/PCFI100/identity/static/dynamic;nativePCFI→같은PFN;publicsingleforward;양/음LP;NLL/Brier/macroF1·fullfeature·source2/8·no-source·전체추론비용.',
        recheck='새basegraph와미사용realgraph를먼저고정하고dynamic-static/identity의독립증분,관측anchor,layergeometry변화와signedcos실사용,feature/context개입및numericalreplay를함께평가한다. Secondary를primary로대체하지않는다.')
    entries=[first,dict(id='NSDT-P02',kind='preserved',claim='PCFI warmup이나scalar completion 없이관측input/context에서시작하는미분가능hidden-tokenNSD를실제로학습했다.',
        observation=f"15checkpoints/15360updates,real0. Denseincidenceerror4.44e-16,gaugeoperatorerror6.66e-16,mapgradientFDerror1.82e-9;negativecos표현가능. Geometrysummary={s['geometry']}. Syntheticaccuracy={ {k:100*v['accuracy'] for k,v in s['synthetic'].items()} }.",
        verdict='연산·gradient·고정anchor와실제성능수치를보존한다. Geometry갱신의유용성은static대비결과와구분한다.',
        do_not_claim='OperatorPSD/gaugeidentity는전체학습encoder의gaugeequivariance나예측우위를증명하지않는다. 같은container개수를동일유효용량으로쓰지않고원NSD의분류성과를우리의zero-shot성과로이전하지않는다.',**common),
        dict(id='NSDT-L01',kind='lesson',claim='Static 대 dynamic과identity 대 learned geometry를같은hiddenstate계산에서분리해야한다.',
        observation=obs,verdict='맵이변하거나mapgradient가있다는사실만으로갱신이좋다고판정하지않는다. Static/dynamic은파라미터를정확히맞췄고같은dynamiccheckpoint의추론시freeze진단도별도보존했다.',**common),
        dict(id='NSDT-L02',kind='lesson',claim='새synthetic seed·기존realdevelopment입력·고정source수secondary와원래결측primary를구분한다.',
        observation=f"64freshsynthetictestgraphs/samefamily;oldreal4graphsreused;source2/8secondary;zero-sourcepartition={s['zero_source_partition']};audit={a['status']},maxreplay={max(r['max_probability_difference'] for r in a['replay'])}.",
        verdict='이전384updatepilot과새1024update모델의절대차이를구조효과로계산하지않고이번동일예산arm끼리비교한다. Fullfeature/개입/확률점수/비용을함께남기고numericalvariation도공개한다.',**common)]
    project={'project':'Interleaved NSD hidden-token GraphPFN','date':'2026-11-03','source':refs[0],'setting':'Synthetic-only1024updates,realweightsfrozen;2Dorthogonalendpointrestrictions inPFNlayers0/3/6/9;noPCFIwarmup,mask-awaretokenanchors.','related':['NeuralSheafDiffusion'],'entries':entries}
    (REPO/'findings/nsd-token.yaml').write_text(yaml.safe_dump(project,allow_unicode=True,sort_keys=False,width=155),encoding='utf8')
    idea={'id':'nsd-token-pfn','date':'2026-11-03','status':'completed-pilot-conditional-go' if go else 'completed-bounded-pilot-no-go','text':first['claim'],'tags':tags,'keywords':keywords,'note':refs[1],'followup':refs[0],'pilot_gates':s['gates']}
    (REPO/'ideas/nsd-token-pfn.yaml').write_text(yaml.safe_dump(idea,allow_unicode=True,sort_keys=False),encoding='utf8')
    allentries=[];projects=[]
    for p in (REPO/'findings').glob('*.yaml'):
        if p.name.startswith('_'):continue
        projects.append(p);allentries+=yaml.safe_load(p.read_text(encoding='utf8'))['entries']
    counts=Counter(e['kind'] for e in allentries);n=len(allentries)
    section='## NSD hidden-token PFN: 관측에서 시작하는 geometry 갱신\n\n'
    section+='2026-11-03. PCFI warmup/숫자completion대신PFNhiddenstate에서endpointrestriction을학습했다. Mask/PCFI100/identity/static/dynamic을각3seed·1024updates로새로학습했고실제graph학습은0이다. 새64syntheticbasegraph지만family는같고실제4graph는기존developmentbank다.\n\n'
    section+='| 방법 | Cora | CiteSeer | Texas | Wisconsin |\n|---|---:|---:|---:|---:|\n'
    for arm in ['pcfi100','identity','static','dynamic','native_pcfi100','negative_lp']:section+='| '+arm+' | '+' | '.join(f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in ['Cora','CiteSeer','texas','wisconsin'])+' |\n'
    section+='\n'+obs+'\n\n'
    section+=('현재pilot은사전gate를통과했으나독립검증과novelty는미확정이다.' if go else '현재pilot은확대기준미통과다. 이특정SO2adapter의범위로보존하고모든NSD를기각하지않는다.')
    section+=' Numericreplay상태는'+a['status']+'이며초기화·taskorder·고정가중치와정보경계를검사했다. [전체보고서](findings/sources/nsd-token-2026-11-03/REPORT_KO.md),[계산정의](findings/sources/nsd-token-2026-11-03/METHOD_KO.md),[구조화기록](findings/nsd-token.yaml).\n\n'
    summary=REPO/'RESULTS_SUMMARY_KO.md';body=summary.read_text(encoding='utf8');body=body.replace('## Sheaf-PCFI: synthetic 학습·실제 가중치 고정 비교',section+'## Sheaf-PCFI: synthetic 학습·실제 가중치 고정 비교',1)
    body=re.sub(r'기록은38개 프로젝트 묶음의160항목이다\. 실패·중단48개, 보존36개, 교훈53개, 미실행23개이며 \*\*160번의 독립 실험을 했다는 뜻은 아니다\.\*\*',f"기록은{len(projects)}개 프로젝트 묶음의{n}항목이다. 실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개이며 **{n}번의 독립 실험을 했다는 뜻은 아니다.**",body)
    body=body.replace('[160개 전체 기록]',f'[{n}개 전체 기록]').replace('graphpfn_review_packet_2026-11-03_v15.zip','graphpfn_review_packet_2026-11-03_v16.zip');summary.write_text(body,encoding='utf8')
    hand=REPO/'RESEARCH_HANDOFF_KO.md';body=hand.read_text(encoding='utf8').replace('Sheaf-PCFI synthetic-to-real pilot까지','NSD hidden-token PFN pilot까지',1)
    body=body.replace('## 먼저 읽을 자료',section+'## 먼저 읽을 자료',1)
    body=body.replace('연구 기록 160항목, 아이디어 12개가 있다. 38개 프로젝트 묶음이며 실패·중단48, 보존36, 교훈53, 미실행23항목이다. 160번의 독립 실험이라는 뜻은 아니다.',f"연구 기록 {n}항목, 아이디어 13개가 있다. {len(projects)}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목이다. {n}번의 독립 실험이라는 뜻은 아니다.")
    body=body.replace('v15는 Sheaf-PCFI','v16은NSD hidden-token의동일예산학습과static/dynamic비교를추가했다. v15는 Sheaf-PCFI',1)
    body=body.replace('첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.','첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.\n최신NSDtokenpilot은warmPCFI없이hiddenrepresentation과restrictionmap을함께학습했어. 결과와닫은범위를확인하고단순tokenNSD추가를미실행새방법으로다시제안하지말아줘.',1);hand.write_text(body,encoding='utf8')
    readme=REPO/'README.md';readme.write_text(readme.read_text(encoding='utf8').rstrip()+'\n\n'+section.rstrip()+'\n',encoding='utf8')
    builder=REPO/'tools/build_review_packet.py';builder.write_text(builder.read_text(encoding='utf8').replace('VERSION = 15','VERSION = 16'),encoding='utf8')
    preservation=preserve();(ROOT/'results/harness_preservation.json').write_text(json.dumps(preservation,indent=2),encoding='utf8');shutil.copyfile(ROOT/'results/harness_preservation.json',DEST/'results/harness_preservation.json')
    manifest={p.relative_to(DEST).as_posix():sha(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('ARCHIVED',n,len(manifest),preservation,flush=True)
if __name__=='__main__':main()
