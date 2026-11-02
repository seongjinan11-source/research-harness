"""Archive literature review and supplementary comparisons without rewriting old evidence."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil
from collections import Counter
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
root=Path(__file__).resolve().parents[1];repo=root.parent/'research-harness'
def read(p):return json.loads((root/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read('PRIMARY_COMPARISONS.json');old=read('runtime/previous_harness.json')
assert r['all_nonprivileged_accuracy_pointwise_ci_lower_positive'] and r['all_nonprivileged_nll_brier_pointwise_ci_upper_negative']
assert len(r['nonprivileged_comparators'])==11 and r['new_updates']==0
def delta(d,m='accuracy'):
    scale=100 if m=='accuracy' else 1
    a,lo,hi=[x*scale for x in [d[m]['delta'],*d[m]['ci95']]]
    return f'{a:+.6f} [95% CI {lo:+.6f}, {hi:+.6f}]'
names={'mask':'Mask PFN','identity':'일반 PCFI→PFN','parity':'고정 parity PCFI→PFN','uniform_two':'50:50 두 복원 평균',
    'feature_adapt':'Feature-only 적응 PCFI→PFN','joint_adapt':'Context 적응 PCFI→PFN',
    'positive_lp':'양의 LP','signed_lp':'Signed LP','adaptive_lp':'Context 적응 LP',
    'gaussian_label_only':'같은 Gaussian/probit·피처 제거','uniform':'Uniform',
    'known_g_reference':'참 feature-law g 복원 reference','known_t_lp_reference':'참 label-law t LP reference'}
ssgp='https://www.ijcai.org/Proceedings/07/Papers/171.pdf'
ggp='https://proceedings.neurips.cc/paper/2018/hash/1fc214004c9481e4c8073e85323bfd4b-Abstract.html'
body='# Gaussian/probit 결과와 SSGP·GGP 문헌 검토\n\n'
body+='이번 기록은 기존 adaptive-transport bank의 추가 비교와 논문 정리다. 새 graph·모델·학습 update는0이며, 새 독립 검증이나 SSGP/GGP 재현 실험이 아니다. 기존 ADTR-F01/P01/L01 판정과 source snapshots는 유지한다.\n\n'
body+='## 문헌에서 확인한 계산과 우리 구현의 차이\n\n'
body+='| 항목 | SSGP, IJCAI 2007 | GGP, NeurIPS 2018 | 현재 직접 Gaussian/probit |\n|---|---|---|---|\n'
body+='| Covariance | Graph regularizer로 수정한 kernel | P K(X,X) Pᵀ | Graph low-pass Gaussian covariance의 sign-gauged 가설 |\n'
body+='| Label 처리 | Probit와 EP | Robust-max와 inducing-point variational inference | Context의 sign 제약과 고정 Sobol 적분 |\n'
body+='| Feature 처리 | 입력 좌표에서 kernel 계산 | 이웃의 feature kernel 결합 | 관측행만으로 singular score를 만들고 Gaussian 관측으로 조건화 |\n'
body+='| 과제 적응 | Evidence 기반 모델 선택 | ELBO로 target 데이터의 파라미터 최적화 | 고정24가설을 관측 evidence로 주변화; 새 optimizer 학습0 |\n'
body+='| 이번에 재현했나? | 아니오 | 아니오 | 기존 adaptive-transport 실험의 대조군 |\n\n'
body+='[SSGP 공식 논문]('+ssgp+')과 [GGP 공식 논문]('+ggp+')의 구조·가정·강점·한계·인용은 papers/SSGP.yaml과 papers/GGP.yaml에 정리했다. GGP의 [공식 코드](https://github.com/yincheng/GGP) README도 확인했다. 한계의 self와 analyst를 구분했고, 읽지 않은 후속 논문이나 리뷰의 비판은 만들지 않았다.\n\n'
body+='우리 구현은 두 논문의 재현도, 새로운 GP 원리의 입증도 아니다. 겹치는 선행 원리는 graph covariance를 통한 확률 분류다. 관측 score와 분리된 context를 함께 조건화하고 g/t를 주변화하는 현재 구성의 차별성은 별도로 평가해야 하며, 기존 구성요소의 조합만으로 novelty가 확정되지는 않는다. 이 판단은 조사자 해석이다.\n\n'
body+='## 기존 primary 조건에서의 추가 paired 비교\n\n'
body+='새64 base graphs,192nodes,8correlated columns,source8/192=95.83% whole-node missing,k4 labels/class다. Label/feature 네 법칙과 기존 세 고정 training seeds를 graph 안에서 평균했다. 현재 Gaussian/probit 정확도는56.154827%,NLL0.672120,Brier0.479871이다. Original statistics와 모든 입력 SHA가 일치하는지 확인한 뒤 저장된 metric/probability를 재집계했다.\n\n'
body+='다음은 Gaussian−대조의 차이다. Bootstrap은 기존과 같은 graph 단위5,000회다. 이 비교들은 같은 bank를 본 뒤 추가한 보조 분석이며 pointwise95CI를 사용한다. 11개 비교 전체에 대한 family-wise 유의성이나 새로운 사전 primary 성공을 주장하지 않는다.\n\n'
body+='| 비특권적 대조 | Accuracy 차이%p [95% CI] | NLL 차이 [95% CI] | Brier 차이 [95% CI] |\n|---|---:|---:|---:|\n'
for method in r['nonprivileged_comparators']:
    d=r['gaussian_minus'][method];body+=f"| {names[method]} | {delta(d)} | {delta(d,'nll')} | {delta(d,'brier')} |\n"
body+='\n이 primary 조건의 평균에서는 모든 비특권적 대조에 대해 accuracy CI 하한>0,NLL/Brier CI 상한<0였다. 따라서 Gaussian 대조의 우위를 유의하지 않은 점추정으로 축소하지 않는다. 정확히 같은 primary missing-query 집합의 피처 제거 비교는 기존 ADTR-P01의+2.466964 [95% CI +0.884420,+4.166715]%p를 유지한다.\n\n'
d=r['gaussian_minus']['known_t_lp_reference']
body+='## 모든 조건과 이전 방법을 이긴 것은 아니다\n\n'
body+='참 label-law t를 받은 LP reference는57.135862%로 Gaussian보다 점추정이 높았다. Gaussian−reference 차이는 '+delta(d)+'%p이며 CI는0을 포함한다. 이 reference는 privileged information을 받으므로 비특권적 대조와 분리한다.\n\n'
body+='동일 source8/k4에서도 t0/g0의 양의 LP57.254319%가 Gaussian55.815288%보다 높았고, t1/g1의 signed LP57.017405%가 Gaussian56.494366%보다 높았다. 이는 보조 cell의 점추정 반례이며 해당 차이의 유의성을 주장하지 않는다. 전체32개 세부 cell 중 비특권적 방법의 점추정이 더 높은 cell은'+str(len(r['point_estimate_cell_counterexamples']))+'개였다. 전체 목록은 PRIMARY_COMPARISONS.json에 있다.\n\n'
body+='현재 Gaussian covariance는 생성 family의 저주파 구조를 활용하며, feature는 한 잠재 신호를 공유한다. 비교한 PFN은 다른 prior로 학습된 frozen checkpoint다. 그러므로 같은 family로 재학습한 PFN, 기존 Grid6, 실제 graph, arbitrary heterophily나 새로운 schema까지 모두 이겼다는 뜻은 아니다. 숨겨진 feature 값·query 정답과 참 g/t는 실행 가능한 Gaussian 추론에 추가 공개하지 않았다.\n\n'
body+='## 메소드로서의 판정\n\n'
body+='현재 Gaussian/probit는 성능이 확인된 고정 통계 기준선이자 후속 연구에서 사용할 추론 자산으로 보존한다. 메인 메소드의 novelty와 범용성이 확정됐다는 판정은 보류한다. 신경망이 없다는 이유로 배제하는 것이 아니라, 선행 GP 원리와 구별되는 연산 및 현재 family 밖의 근거가 아직 부족하다는 판단이다. 이는 Gaussian 추론 전체에 대한 No-Go가 아니다.\n\n'
body+='SSGP/GGP 원문은 우리 missing-feature PFN 문제를 이미 해결했다는 증거가 아니다. 반대로 우리 구현이 두 원 논문보다 우월하다는 실험도 아직 없다. 두 논문 자체의 재현은 미실행으로 명시하고 학습 실패로 세지 않는다. 이번 기록 요청으로 새 모델이나 GPU 학습을 시작하지 않았다.\n\n'
body+='## 보존과 재검토\n\n'
body+='Papers의 verified_from에는 공식 원문/학회 metadata를 넣었다. SSGP 페이지1059–1064는 공식 PDF footer에서 확인했고, GGP 페이지1683–1694는 공식 metadata를 따랐다. 검색의 상충한 페이지 수는 채택하지 않았다. GGP 코드 링크는 저자 repository README로 확인했으며, SSGP의 전용 공식 코드는 확인하지 못해 링크를 만들지 않았다.\n\n'
body+='각 논문에 대해 revisit --paper를 실행한 결과도 보존했다. 자동 매칭은 재검토용 검색 단서이며 실패 해결이나 학습 재개의 판정이 아니다. 원142개 finding,699개 보존 source 파일,33개 기존 paper YAML의 hash를 비교한다.\n'
(root/'REVIEW_KO.md').write_text(body,encoding='utf8')

verification={'date':'2026-11-02','SSGP':{'official_pdf':ssgp,'toc':'https://www.ijcai.org/Proceedings/07/Papers/486.pdf',
    'sections_read':['2','3.1','3.2','3.3','4','5'],'verified_authors':['Vikas Sindhwani','Wei Chu','S. Sathiya Keerthi'],
    'pages_from_pdf_footer':'1059--1064','venue_from_footer':'IJCAI-07','dedicated_official_code':'not verified; no URL fabricated'},
    'GGP':{'official_page':ggp,'official_pdf':'https://proceedings.neurips.cc/paper/2018/file/1fc214004c9481e4c8073e85323bfd4b-Paper.pdf',
    'metadata':'https://proceedings.neurips.cc/paper_files/paper/2018/file/1fc214004c9481e4c8073e85323bfd4b-Metadata.json',
    'sections_read':['2','3','3.1','3.2','3.3','4','5.1','5.2','6'],'verified_authors':['Yin Cheng Ng','Nicolò Colombo','Ricardo Silva'],
    'pages_from_official_metadata':'1683--1694','volume_from_official_proceedings':31,'code':'https://github.com/yincheng/GGP',
    'readme_verified_with':'GitHub connector, https://github.com/yincheng/GGP/blob/master/README.md; GPflow-inter-domain dependency and ssl/al scripts'},
    'scope':'Metadata and section provenance, paraphrases only; no paper reproduction or new training. All limitations use self or analyst attribution.'}
(root/'SOURCE_VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
prefix='findings/sources/gaussian-probit-review-2026-11-02/';dest=repo/prefix;assert not dest.exists();dest.mkdir(parents=True)
paths=list(root.glob('*.py'))+list(root.glob('*.json'))+list(root.glob('*.md'))+list((root/'runtime').glob('*.py'))
paths += [repo/'papers/SSGP.yaml',repo/'papers/GGP.yaml']
manifest=[]
for path in sorted(set(paths)):
    rel=Path('papers')/path.name if path.parent==repo/'papers' else path.relative_to(root)
    q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,q)
    manifest.append({'alias':rel.as_posix(),'original':str(path),'snapshot':q.relative_to(repo).as_posix(),'sha256':sha(q),'bytes':q.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'date':'2026-11-02','scope':'Two verified GP papers and supplementary existing-bank Gaussian comparisons. No new graph/model/update. Verbatim source snapshots; original ADTR archive unchanged.','sources':manifest},ensure_ascii=False,indent=2),encoding='utf8')
refs=[prefix+x for x in ['REVIEW_KO.md','PRIMARY_COMPARISONS.json','SOURCE_VERIFICATION.json','papers/SSGP.yaml','papers/GGP.yaml']]
refs+=['findings/sources/adaptive-transport-2026-11-02/'+x for x in ['REPORT_KO.md','protocol.json','results/statistics.json','results/cost.json']]
comparisons='; '.join(names[m]+':accuracy%p '+delta(r['gaussian_minus'][m]) for m in r['nonprivileged_comparators'])
preserved={'id':'ADTR-P02','kind':'preserved','claim':'기존 primary 조건의 평균에서 직접 Gaussian/probit가 비특권적 대조11개보다 accuracy·NLL·Brier에서 우세했다.',
    'observation':'Gaussian accuracy56.154827%,NLL0.672120,Brier0.479871. '+comparisons+'. 11개 개별95CI에서 accuracy하한>0,NLL/Brier상한<0.',
    'do_not_claim':'기존 bank를 본 뒤 계산한 보조 비교이며 multiplicity 미보정 pointwise CI다. 새 독립 검증·사전 Gaussian primary 통과·모든 세부 cell·Grid6·SSGP/GGP 재현 대비 승리·같은 prior matched PFN 우위·메소드 novelty가 아니다. 32개 cell 중22개에서 비특권적 대조의 점추정이 더 높았다.',
    'tags':['model-based-baseline','graph-structure-use','observation-mask','eval-protocol'],
    'keywords':['Gaussian probit primary comparisons','graph Gaussian process','SSGP','GGP','pointwise confidence interval','known family baseline'],
    'source_refs':refs,'evidence':'기존64 base graphs/2048episodes,8sources/k4 per class/all4 laws,3 fixed historical training seeds. Graph-paired bootstrap5000. 저장 metric과 기존 primary 평균을1e-12 오차 이내로 재확인. 새graph/model/update0.'}
lesson={'id':'ADTR-L02','kind':'lesson','claim':'Gaussian 대조의 관측된 승리와 graph-GP 선행 원리·PFN 메소드의 기여를 구분한다.',
    'observation':'SSGP는 graph regularizer/probit/EP/evidence 선택, GGP는 P K(X,X) P^T/robust-max/variational ELBO를 사용한다. 현재 구현은 관측행 singular score와24 sign/amplitude 가설의 고정 Gaussian/probit 추론이며 두 논문의 재현이 아니다. 참 label-law LP57.135862%와 Gaussian56.154827%의 accuracy차이는 '+delta(r['gaussian_minus']['known_t_lp_reference'])+'%p였다.',
    'interpretation':'현재 family의 성능을 보존하되, Gaussian 조건화나 graph covariance 자체를 새 기여로 주장하지 않는다. Target fitting을 수행하는 선행 GP와0-update frozen 비교의 차이를 남긴다. 현재 모델이 신경망이 아니라는 이유로 배제하지 않으며, 메인 메소드 확정은 아직 보류다. 임의 missing/다중class/미지schema 일반화와 SSGP/GGP 직접 재현은 미실행이다. 자동 revisit는 해결·학습 재개를 뜻하지 않는다.',
    'tags':['model-based-baseline','zero-shot-gap','eval-protocol','uncertainty'],
    'keywords':['Gaussian process novelty boundary','SSGP GGP prior art','main method pending','not a PFN','unrun reproduction','no new training'],
    'source_refs':refs,'evidence':'공식 IJCAI PDF/목차,NeurIPS 원문/metadata,저자 code README와 현재 implementation을 대조했다. 선행연구와의 차별점 평가는 analyst 해석이며 미실행을 실패로 세지 않는다.'}
path=repo/'findings/adaptive-transport.yaml';project=yaml.safe_load(path.read_text(encoding='utf8'))
assert not any(e['id'] in ['ADTR-P02','ADTR-L02'] for e in project['entries'])
project['entries'] += [preserved,lesson];project['related'] += ['SSGP','GGP']
path.write_text(yaml.safe_dump(project,allow_unicode=True,sort_keys=False,width=170),encoding='utf8')
entries=[]
for path in (repo/'findings').glob('*.yaml'):
    if not path.name.startswith('_'):entries+=yaml.safe_load(path.read_text(encoding='utf8')).get('entries',[])
hashes={e['id']:hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for e in entries}
assert all(hashes[k]==v for k,v in old['entries'].items())
assert all(sha(repo/p)==v for p,v in old['sources'].items())
assert all(sha(repo/p)==v for p,v in old['papers'].items())
counts=Counter(e['kind'] for e in entries)
(root/'ARCHIVE_INTEGRITY.json').write_text(json.dumps({'old_findings_unchanged':len(old['entries']),'old_sources_unchanged':len(old['sources']),'old_papers_unchanged':len(old['papers']),
    'new_ids':['ADTR-P02','ADTR-L02'],'counts':dict(counts)},ensure_ascii=False,indent=2),encoding='utf8')
section='## Gaussian/probit의 비교 범위와 선행연구 기록\n\n'
section+='SSGP(IJCAI2007)와 GGP(NeurIPS2018)의 원문·인용·가정·강점·출처별 한계를 추가했다. 현재 Gaussian/probit는 두 논문의 재현이 아니며, 논문 코드로 실행한 대조는 아직 없다. [SSGP](papers/SSGP.yaml), [GGP](papers/GGP.yaml), [문헌과 결과 대조 보고서]('+prefix+'REVIEW_KO.md).\n\n'
section+='기존 primary의 보조 paired 비교에서 직접 Gaussian/probit56.154827%는 비특권적 대조11개보다 accuracy·NLL·Brier가 좋았다. PCFI 대비accuracy '+delta(r['gaussian_minus']['identity'])+'%p,적응 LP 대비 '+delta(r['gaussian_minus']['adaptive_lp'])+'%p다. CI는 개별95%이며 multiplicity 미보정이다. 모든 세부 조건·privileged reference·다른 family와의 승리로 넓히지 않는다. 현재 구현은 강한 고정 통계 기준선/추론 자산으로 보존하며 메인 메소드 확정은 보류한다. 기존 frozen 입력 선택기 No-Go와 직접 추론의 긍정 결과를 구분한다. 새학습0.\n\n'
for name in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
    path=repo/name;body=path.read_text(encoding='utf8');anchor='## 참 법칙 없는 적응 전달과 직접 통계 추론';assert anchor in body
    body=body.replace(anchor,section+anchor,1)
    body=body.replace('34개 프로젝트 묶음의142항목','34개 프로젝트 묶음의144항목').replace('연구 기록 142항목','연구 기록 144항목')
    body=body.replace('실패·중단42개, 보존31개, 교훈47개, 미실행22개','실패·중단42개, 보존32개, 교훈48개, 미실행22개')
    body=body.replace('실패·중단42, 보존31, 교훈47, 미실행22항목','실패·중단42, 보존32, 교훈48, 미실행22항목')
    body=body.replace('142번의 독립 실험','144번의 독립 실험').replace('[142개 전체 기록]','[144개 전체 기록]').replace('논문 33편','논문 35편')
    body=body.replace('graphpfn_review_packet_2026-11-02_v9.zip','graphpfn_review_packet_2026-11-02_v10.zip')
    body=body.replace('v9는 참 법칙 없는 적응 전달과 직접 통계 추론까지 포함한다.','v10은 SSGP/GGP 문헌과 Gaussian 추가 비교까지 포함한다. v9는 직전 참 법칙 없는 적응 전달 snapshot이다.')
    body=body.replace('과거 v1/v2/v3/v4/v5/v6/v7/v8 ZIP','과거 v1/v2/v3/v4/v5/v6/v7/v8/v9 ZIP')
    if name=='RESEARCH_HANDOFF_KO.md':
        body=body.replace('최근 업데이트: 참 법칙 없는 적응 전달까지 포함한','최근 업데이트: SSGP/GGP 문헌과 Gaussian 비교까지 포함한')
        body=body.replace('1. [최신 적응 전달 비교]','1. [최신 GP 문헌·결과 검토]('+prefix+'REVIEW_KO.md)와 [적응 전달 비교]',1)
        i=body.rfind('```text');assert i>=0;pre,tail=body[:i],body[i:];j=tail.rfind('\n```');assert j>=0
        tail=tail[:j]+'\nSSGP/GGP도 원문 기준으로 정리했어. 현재 Gaussian/probit가 기존 primary 평균의 비특권적 대조11개를 개별95CI 기준으로 이겼지만 모든 세부조건을 이긴 것은 아니야. SSGP/GGP 자체의 재현은 아직 하지 않았고, 같은 prior로 새학습한 PFN과도 비교하지 않았어. 기존 graph-GP 원리와 우리 연산의 추가 가치를 구분해 줘.\n'+tail[j:];body=pre+tail
    path.write_text(body,encoding='utf8')
path=repo/'README.md';body=path.read_text(encoding='utf8');marker='다른 연구자나 모델에게 검토를 요청할 때는'
body=body.replace(marker,'SSGP·GGP 선행연구와 직접 Gaussian/probit의 추가 비교·메소드 판정도 기록했다.\n'+marker,1);path.write_text(body,encoding='utf8')
path=repo/'tools/build_review_packet.py';body=path.read_text(encoding='utf8');assert 'VERSION = 9' in body;path.write_text(body.replace('VERSION = 9','VERSION = 10',1),encoding='utf8')
print('REVIEW_ARCHIVED',len(entries),dict(counts),'PAPERS35',flush=True)
