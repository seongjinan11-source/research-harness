import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil
from collections import Counter
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'research-harness'
DEST=REPO/'findings/sources/sheaf-pcfi-2026-11-03'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(b):return hashlib.sha256(b).hexdigest()
def verify_before():
    before=read(ROOT/'runtime/harness_before.json');current={}
    for p in (REPO/'findings').glob('*.yaml'):
        if p.name.startswith('_'):continue
        for r in yaml.safe_load(p.read_text(encoding='utf8'))['entries']:current[r['id']]=digest(json.dumps(r,sort_keys=True,ensure_ascii=False).encode())
    assert all(current[k]==v for k,v in before['entries'].items())
    assert all(digest((REPO/k).read_bytes())==v for k,v in before['files'].items())
    return {'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files'])}
def main():
    preservation=verify_before();assert not DEST.exists();DEST.mkdir(parents=True)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'protocol.json',ROOT/'bank/manifest.json']+list((ROOT/'results').rglob('*.json'))+[Path(__file__)]
    for p in files:
        q=DEST/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for p in [ROOT.parent/'joint_imputation_pfn/common.py',ROOT.parent/'regret_prior_pilot/core.py',ROOT.parent/'fisf_graphpfn/imputation.py']:
        q=DEST/'reference'/p.parent.name/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    old=ROOT.parent/'sheaf_pcfi_pilot_design_confound_v0'
    for p in (old/'results').glob('mask_s*/training.json'):
        q=DEST/'discarded_design'/p.parent.name/'training.json';q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    (DEST/'ARCHIVE_SCOPE.md').write_text('Protocol, source, report, statistics, metric manifests, train logs and audits are preserved as bytes. Checkpoints, raw feature/episode/probability NPZ banks and full environment remain in workspace and are excluded from the review packet. Snapshot paths in scripts reference the execution workspace; this is a research record, not a standalone reproducible environment. Source documents preserve historical instructions; they do not authorize reviewers to run experiments.\n',encoding='utf8')
    s=read(ROOT/'results/statistics.json');u=read(ROOT/'results/supplement.json');a=read(ROOT/'results/audit.json')
    tags=['graph-structure-use','graph-geometry','observation-mask','zero-shot-gap','eval-protocol']
    keywords=['Sheaf PCFI','homophily heterophily','signed transport','confidence propagation','synthetic-only frozen real PFN','RP16','negative LP']
    refs=[(DEST/p).relative_to(REPO).as_posix() for p in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/supplement.json','results/audit.json','results/cost.json','DESIGN_CORRECTION_KO.md']]
    comp=s['synthetic_comparisons']['sheaf - pcfi100']['accuracy']
    observation=f"6arms×3seeds/6912validupdates,32newtestgraphs. SyntheticPCFI100 {100*s['synthetic']['pcfi100']['accuracy']:.6f}%,sheaf {100*s['synthetic']['sheaf']['accuracy']:.6f}%,difference{100*comp['mean']:+.6f}%p CI{[100*v for v in comp['ci95']]}. Realheterophily sheaf-PCFI {100*s['real_hetero_differences']['pcfi100']:+.6f}%p,nativePCFI {100*s['real_hetero_differences']['native_pcfi100']:+.6f}%p. Gates allFalse. NegativeLP Texas{100*s['real']['texas']['negative_lp']['accuracy']:.4f}%,Wisconsin{100*s['real']['wisconsin']['negative_lp']['accuracy']:.4f}%."
    evidence='384syntheticCEupdates/arm/seed;4selectedrealdevelopmentgraphs,90/99.5%uniform/clustered whole-node masks,missing-query primary;0realupdates;graph-pairedbootstrap5000;original savedprobabilities rescored. Audit checkpoints/hash/info-boundaries pass with numerical replay variation disclosed, not exact-reproduction PASS.'
    common={'tags':tags,'keywords':keywords,'source_refs':refs,'evidence':evidence,'revisit_status':'open'}
    entries=[dict(id='SPCFI-F01',kind='failure',claim='Synthetic-trained2D sheaf transport를PCFI에넣으면양수전파및signed scalar보다강하고실제homo/heterophily missing을함께개선한다.',
        observation=observation,verdict='현재RP16·symmetric-contraction·384updates구성은확대하지않는다. 일부CiteSeer증분과signed대비이득을범용성공으로승격하지않는다.',
        closed_scope='WarmPCFI100+learned100의특정2Dper-cell수축함수족,공유459parameterrouter,RP16,4selecteddevelopmentgraphs,동일syntheticbank의현재학습구성.',
        not_closed='원본NSD/FSD-CAP/CPGNN/CLP재현,physicalfeature공간에서의새학습,일반sheaf표현력,heterogeneousgraph는미검증이다. 원본피처추론firstrep는별도진단이며그곳의양수신호도보존한다.',
        failure_reason='PCFI100/200/weighted대비syntheticaccuracy CI가0을포함하며realheterophily평균도낮다. 고유원인은미상이다. 첫좌표직접전달계수a<0비율은0이고cross-stalk사용은관측됐다. 행렬의다른성분·고유값·반복경로의signed효과는이통계로판정할수없다. Warm/RP16/학습량/prior차이를확인된원인으로단정하지않는다.',reason_status='unknown',
        excluded_explanations='관측source clamp·PCFIidentity환원·hidden/query입력경계·routergradient기본검사는통과했다. 이것으로모든구현문제를배제한것은아니다.',
        reopen_when='새근거가현재함수족/geometry식별/전이의구체적제약을바꾸고,원본피처PCFI와weighted및고정negativeLP보다유리한정보·계산을설명할때.',
        reopen_tags=['graph-geometry','graph-structure-use','distribution-shift'],reopen_keywords=['missing-aware sheaf transport','heterophily feature geometry identification'],
        comparison_floor='동일prior·초기화·updates의Mask/PCFI100/200/weighted/signed;원본피처PCFI→동일PFN;public초기화;양/음LP;관측0source포함;학습/추론비용과수치재현.',
        recheck='구조/예산을새bank전에고정하고feature/label법칙을분리한다. 새로운syntheticgraph및미사용realgraph에서missing-queryaccuracy와확률점수,피처제거/셔플,negative実사용과numericalreplay를함께평가한다.',**common),
        dict(id='SPCFI-P01',kind='preserved',claim='Confidence·source clamp·inter-channel correction을보존한미분가능2Dtransport와PFN의synthetic-only학습을구현했다.',
            observation=f"18validcheckpoints,6912updates,0realupdates. PCFI/SciPymax3.58e-7,identityerrors<7.8e-7,routerweightupdate>6e-5. Hidden/querypoisonPASS. Sheafcross-stalkmean{u['transport_diagnostics_synthetic_missing']['sheaf']['mean_abs_cross_stalk']:.6f};first_diagonal_a_negative_fraction0;other_entries/eigenvalues_not_claimed. Nativefirstrep:Texas{100*u['native_first_repetition_seed11']['texas']['native_sheaf']['accuracy']:.4f}%/Wisconsin{100*u['native_first_repetition_seed11']['wisconsin']['native_sheaf']['accuracy']:.4f}%,nativePCFI{100*u['native_first_repetition_seed11']['texas']['native_pcfi100']['accuracy']:.4f}%/{100*u['native_first_repetition_seed11']['wisconsin']['native_pcfi100']['accuracy']:.4f}%.",
            verdict='구현·gradient·정보경계와제한된native추론신호를보존한다.',
            do_not_claim='유효operator구현을성능우위·새sheaf원리·원본NSD재현으로쓰지않는다. Nativefirstrep/seed11은독립확인도원본피처학습도아니며Texas에서는native_signed가더높다. Strictnumericalreplay는완전통과하지않았다.',**common),
        dict(id='SPCFI-L01',kind='lesson',claim='Labelheterophily·featureenergy·projection순서·source수·학습된transport부호는각각실측해야한다.',
            observation='RP16primary에서sheaf의heterophily이득은없다. 원본피처추론firstrep에는일부개선이있다. Synthetictest의첫좌표직접전달계수a<0비율은0이며2D행렬전체의부호와구분한다. Texas99.5%mask의0source도유지했다. 고정negativeLP가현재PFN보다Texas/Wisconsin에서높다.',
            verdict='부호허용을부호학습과동일시하지않고LP와feature전파도구분한다. Feature제거/shuffle는원래query집합을고정하며약한compressedPCFI하나만floor로쓰지않는다.',**common),
        dict(id='SPCFI-L02',kind='lesson',claim='설계교정과수치재현의실패는메소드성능과별도로공개한다.',
            observation=f"초기class수-featurelaw결합을비교성능판독전에발견해1152updates를폐기하고모든arm을다시학습했다. Oldfolder보존. Numericalreplay .005criterion failed;max{max(r['max_probability_difference'] for r in a['replay']):.8f},selectedepisodeargmaxmax{100*max(r['replay_argmax_change_fraction'] for r in a['replay']):.4f}%. Savedpredictions unchanged;CIconditionalonstoredpredictions.",
            verdict='기존checkpoint연장이나실제validationtuning은없다. Audit상태는PASS_WITH_REPLAY_VARIATION이며범위를기록했다. 수치원인은미확인이고일반적CUDA문제라고단정하지않는다.',**common)]
    finding={'project':'Confidence-weighted Sheaf-PCFI GraphPFN synthetic-to-real pilot','date':'2026-11-03','source':refs[0],
        'setting':'Homophily/heterophily only,notheterogeneous;publicGraphPFNinit;6arms×3seeds;syntheticonlyCE,realweightsfixed.','related':['NeuralSheafDiffusion'],'entries':entries}
    (REPO/'findings/sheaf-pcfi.yaml').write_text(yaml.safe_dump(finding,allow_unicode=True,sort_keys=False,width=150),encoding='utf8')
    idea={'id':'sheaf-pcfi-pfn','date':'2026-11-03','status':'completed-bounded-pilot-no-go',
        'text':'PCFI confidence를보존하고2Dtransport를synthetic-only학습한PFN. 현재primarygate불통과,원본NSD전체를기각하지않는다.',
        'tags':tags,'keywords':keywords,'note':refs[1],'followup':refs[0],'pilot_gates':s['gates']}
    (REPO/'ideas/sheaf-pcfi-pfn.yaml').write_text(yaml.safe_dump(idea,allow_unicode=True,sort_keys=False),encoding='utf8')
    section='''## Sheaf-PCFI: synthetic 학습·실제 가중치 고정 비교

2026-11-03에 homophily/heterophily만 남긴 새 PFN pilot을 완료했다. 6arm×3seed,6,912유효updates,64train/32testsyntheticgraphs,실제Cora/CiteSeer/Texas/Wisconsin은가중치고정이다. 주지표는90/99.5% uniform/clustered whole-node missing의결측queryaccuracy다.

| 방법 | Cora | CiteSeer | Texas | Wisconsin |
|---|---:|---:|---:|---:|
| PCFI100+PFN | 50.78 | 32.40 | 28.81 | 22.69 |
| Sheaf-PCFI+PFN | 50.46 | 33.64 | 28.32 | 22.40 |
| 원본피처 PCFI100→RP16+PFN | 52.68 | 33.92 | 32.42 | 24.92 |
| 고정 negative LP | 32.03 | 24.76 | 63.54 | 42.73 |

Synthetic sheaf−PCFI +0.132%p CI[-0.585,+0.968],heterophily실제평균−0.392%p로primarygate를통과하지못했다. **현재2D symmetric-contraction/RP16/384updates구성은확대하지않는다.** Synthetictest의첫좌표직접전달계수a<0비율은0이다. 이통계는2D행렬의다른성분·고유값·반복경로의signed효과를판정하지못하며음의geometry학습의성공/실패로확대하지않는다. Native추론firstrep/seed11에서Texas/Wisconsin일부개선은보존하되primary로교체하지않았다. 원본NSD재현과원본피처공간학습은미검증이다.

초기생성기confound때문에1152updates를폐기하고전부다시학습했다. Checkpoint재실행최대확률차이.007497/선택episodeargmax2.48%변경을공개하며엄격수치재현PASS라고하지않는다. 기존156항목과844paper/source파일은그대로보존했다. [전체보고서](findings/sources/sheaf-pcfi-2026-11-03/REPORT_KO.md),[계산정의](findings/sources/sheaf-pcfi-2026-11-03/METHOD_KO.md),[구조화기록](findings/sheaf-pcfi.yaml).

'''
    summary=REPO/'RESULTS_SUMMARY_KO.md';body=summary.read_text(encoding='utf8').replace('기준일은2026-11-02다.','기준일은2026-11-03다.',1)
    body=body.replace('기록은37개 프로젝트 묶음의156항목이다. 실패·중단47개, 보존35개, 교훈51개, 미실행23개이며 **156번의 독립 실험을 했다는 뜻은 아니다.**','기록은38개 프로젝트 묶음의160항목이다. 실패·중단48개, 보존36개, 교훈53개, 미실행23개이며 **160번의 독립 실험을 했다는 뜻은 아니다.**')
    body=body.replace('## TSFM 공동 정보·방해 변수 frozen 비교',section+'## TSFM 공동 정보·방해 변수 frozen 비교',1)
    body=body.replace('[144개 전체 기록]','[160개 전체 기록]').replace('공유용 `graphpfn_review_packet_2026-11-02_v11.zip`','공유용 `graphpfn_review_packet_2026-11-03_v15.zip`')
    summary.write_text(body,encoding='utf8')
    handoff=REPO/'RESEARCH_HANDOFF_KO.md';body=handoff.read_text(encoding='utf8')
    body=body.replace('기준일: 2026-11-02. 최근 업데이트: 누적56편의 문헌과 별도 TSFM 공동정보 frozen 비교까지 포함한','기준일: 2026-11-03. 최근 업데이트: 누적57편의 문헌과 Sheaf-PCFI synthetic-to-real pilot까지 포함한',1)
    body=body.replace('## 먼저 읽을 자료',section+'## 먼저 읽을 자료',1)
    body=body.replace('논문 56편, 연구 기록 156항목, 아이디어 11개가 있다. 37개 프로젝트 묶음이며 실패·중단47, 보존35, 교훈51, 미실행23항목이다. 156번의 독립 실험이라는 뜻은 아니다.',
        '논문 57편, 연구 기록 160항목, 아이디어 12개가 있다. 38개 프로젝트 묶음이며 실패·중단48, 보존36, 교훈53, 미실행23항목이다. 160번의 독립 실험이라는 뜻은 아니다.')
    body=body.replace('v14는 별도TSFM', 'v15는 Sheaf-PCFI의synthetic-only학습·실제가중치고정pilot과NSD문헌을추가한2026-11-03 snapshot이다. v14는 별도TSFM',1)
    body=body.replace('첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.','첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.\n최신 Sheaf-PCFI pilot의bounded No-Go와negativeLP floor,첫좌표직접전달계수a<0비율0,수치재현한계도함께봐줘. 단순PCFI내부화나signed허용을새아이디어로다시제안하지말아줘.',1)
    handoff.write_text(body,encoding='utf8')
    with (REPO/'README.md').open('a',encoding='utf8') as f:f.write('\n'+section)
    builder=REPO/'tools/build_review_packet.py';body=builder.read_text(encoding='utf8').replace("DATE = '2026-11-02'","DATE = '2026-11-03'").replace('VERSION = 14','VERSION = 15');builder.write_text(body,encoding='utf8')
    preservation=verify_before();dump_path=ROOT/'results/harness_preservation.json';dump_path.write_text(json.dumps(preservation,indent=2),encoding='utf8');shutil.copyfile(dump_path,DEST/'results/harness_preservation.json')
    manifest={p.relative_to(DEST).as_posix():digest(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('ARCHIVED',len(manifest),preservation,flush=True)
if __name__=='__main__':main()
