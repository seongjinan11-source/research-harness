# Joint count 온라인 검증: 주 비교 통과, 조건별 손해도 확인

2026-11-05. 사용자가 현재 연구의 의미를 묻고 진행을 지시하여, 고정 archive의 flexible-count 결과를 새 온라인 bank에서 검증했다. RESEARCH_GOAL.md의 문제–해결–실제 선택 연결을 기준으로 했다. 세 관점의 계획·코드·결과 리뷰, source/analysis freeze, 개발 검사와 원시 기록 전량 감사를 수행했다.

## 메타 판단

**이번에는 고정 자료의 예측 개선을 넘어, 강한 SSPd3+probe 대비 실제 online incumbent 개선의 근거를 얻었다.** 사전에 정한 비다항 size 혼합의 mean incumbent regret는0.01410415→0.00707694였다. 독립24 target 각각의12조건 평균에서도 모두 개선됐고, 주 효과·대조군 손해·비용·optimizer·audit 기준을 전부 통과했다.

따라서 ordinary flexible joint count 모델을 **다음 독립 외부/물리 과제 확인의 후보로 승격**한다. 모든 subset 문제의 기본값, 새로운 GP 원리 또는 ICML 준비 완료로 해석하지 않는다. 좁은 count support의 일부 조건에서는 더 나빴다. Cut의 이전 추가효용 실패도 그대로 유지한다.

## 무엇을 실행했는가

- 새 seed2026110561, target0..23. 이전 bank의 labels/predictions를 사용하지 않았다.
- n16, K2..12, 초기44 unique subset, 공통748개 finite catalog. 추가16개를 순차 평가하고 매번 refit하여17checkpoint를 저장했다. 각 후보의 고정 noise potential outcome을 모든 arm이 공유했다.
- Wide/narrow count support × uniform/aligned archive의4조건. Zero/linear/cubic/sigmoid/step/nonmonotone-wave의6shape. 비제로 size amplitude4, observation noiseSD.05. Composition은 공통 선형 item effect인 합성 문제다.
- SSPd3 mean/probe, joint count GP mean/probe의2×2비교. Probe는 기존과 동일한 t0/4/8/12의 farthest-observed-count rule이며 새로운 acquisition 기여가 아니다.
- Joint는 이전 flexible-count cycle에서 고정한 full EB recipe를 그대로 재사용한다. Zero-sum item prior와 ordinary Matern3/2 count covariance, proper intercept,4scale EB를 사용한다. SSPd3는 원본 wrapper/starts/bounds를 변경하지 않았다.
-2,304runs,39,168selected fits/78,336optimizer starts,36,864new queries. 초기 공유 관측을 arm별로 세면138,240observations.4CPU429.349초로600초 nominal budget 안에 완료했다. 외부 objective 호출/금전 지출은 없다.

Primary 추천은 **현재까지 관측한 noisy utility가 가장 큰 후보**다. 분석에서 그 선택의 latent utility를 평가한다. True-best-observed로 대체하거나, 사후 누적 min으로 다듬지 않는다. Regret reference는 공통 finite catalog의 latent 최선·최악이며,17checkpoint의 평균을 mean incumbent regret라 부른다. Hypervolume AUC나 posterior 추천 지표가 아니다.

## 사전 primary와 gates

Sigmoid/step/wave ×4regime의12context를 target 안에서 먼저 균등 평균했다. 독립 단위는24target이며 shape/regime/query를 별도 독립 표본으로 세지 않았다. Delta는 SSPd3_probe−joint_probe로, 양수가 joint의 이득이다.

| 항목 | 결과 |
|---|---:|
| SSPd3+probe mean incumbent regret | 0.0141041494 |
| Joint count GP+probe | 0.0070769447 |
| 평균 paired 감소량 | +0.0070272047 |
| Target paired bootstrap95% CI | [0.0060360818,0.0079987850] |
| Target별 개선 | 24/24 |
| 가장 작은 target 평균 개선 | +0.0022582093 |
| 구현의 fit+selection wall ratio | 0.09338434 |
| Primary selected finite-failure fraction | 11/9,792 =0.11234% |

감소량은 normalized regret의 **절대0.00703**, 즉 약0.703percentage point다. Baseline 평균에 대한 상대 감소는 약49.82%다. 이를 objective 자체의50% 향상이나 AUC+50%라고 쓰면 안 된다.

Zero/linear/cubic ×4regime의12control context에서 joint_probe−SSPd3_probe는 평균−0.0000246447, bootstrap95% CI [−0.0000840573,+0.0000302084]였다. 평균 손해 upper≤0.005의 사전 제한을 통과했다. 동등성이나 개별 조건 보장은 아니다.

사전 gates 모두 통과: mean≥0.003,CI lower>0,16/24승 이상,worst≥−0.02,control margin,cost ratio≤2,primary selected failure≤5%,source/raw/action audit.

## 조건별 성과와 손해

각 shape에서4regime을 평균한 mean incumbent regret다. 아래 shape별 비교는 사전 보조 분석이며 별개의 multiplicity-adjusted 확인 성공으로 세지 않는다.

| Shape | SSPd3 mean | SSPd3 probe | Joint mean | Joint probe |
|---|---:|---:|---:|---:|
| Zero | 0.006943 | 0.009412 | 0.006608 | 0.009314 |
| Linear | 0.012701 | 0.011507 | 0.014112 | 0.011530 |
| Cubic | 0.025593 | 0.014056 | 0.088179 | 0.014057 |
| Sigmoid | 0.013980 | 0.009349 | 0.006736 | 0.007887 |
| Step | 0.016408 | 0.017224 | 0.002023 | 0.003013 |
| Nonmonotone wave | 0.013147 | 0.015740 | 0.008006 | 0.010331 |

Step가 전체 primary 이득의 큰 부분을 만든다. Sigmoid의 online gap은 이전 static catastrophe 차이보다 훨씬 작다. 이는 SSPd3도 실제 feedback으로 회복한다는 기존 근거와 일치한다. 반대로 step의 final incumbent regret는 SSPd3 probe0.014861대joint probe0.000582로,16새 관측 후에도 실제 격차가 남았다. 같은 조건의 final same-count composition regret도0.211180대0.001589였다.

**모든 support 조건에서 이긴 것은 아니다.** 비다항12context 중 평균 손해가 있는5개도 그대로 보존한다.

| 비다항 regime 평균 | SSPd3 probe | Joint probe |
|---|---:|---:|
| Wide uniform | 0.019166 | 0.005331 |
| Wide aligned | 0.014233 | 0.005153 |
| Narrow uniform | 0.013830 | 0.007615 |
| Narrow aligned | **0.009187** | 0.010208 |

특히 wave에서는 narrow_uniform0.003958→0.004959, narrow_aligned0.014952→0.019175로 악화했다. Sigmoid도 wide_aligned와 두 narrow regime에서는 작게 악화했다. Global24/24승은 target별12context 평균의 결과이며, 모든 shape/regime/실행의 승리를 뜻하지 않는다. 전체24shape/regime strata와 모든 target 값은 SUMMARY/CSV에 있다.

## Probe는 언제 필요한가

비다항 혼합만 보면 joint_mean0.005588가joint_probe0.007077보다 좋았다. 강제 probe는 유용한 exploitation을 미루는 기회비용이 있다. 그러나 cubic에서 joint_mean0.088179는joint_probe0.014057보다 훨씬 나빴고, narrow cubic의 두 regime에서는약0.18946/0.16062가0.02667/0.02516으로 줄었다.

따라서 ‘probe가 항상 좋다’거나 사후에 joint_mean으로 기본 방법을 바꾸는 해석을 하지 않는다. 등록한 joint_probe package는 polynomial/control 손해 제한까지 통과했다. Shape/support에 따라 필요한 관측이 다르다는 점은 유효한 다음 질문이며, 새로운 adaptive query의 성공을 이미 입증한 것은 아니다.

같은 probe rule도 같은 query/count path를 보장하지 않는다. 모델에 따라 exploitation이 달라져 support가 바뀐다. 관측 count수만으로 전체 성능 차이의 원인을 설명하지 않는다. 또한 이 비교는 count covariance뿐 아니라 prior/intercept/normalization/EB가 다른 **전체 fitting recipe의 효과**다. Matern 하나만의 순수 인과 효과로 귀속할 수 없다.

## 정확성·실패 보존·그림

- 개발5tests 통과: 중첩 seed 두 곳의 독립 재생, catalog/potential noise, frozen joint/SSPd3 recipe 예측 동치, independent Gaussian posterior, probe tie/unseen 규칙, 유한 optimizer 실패 결과의 보존.
- 개발96runs/1,632fits의 raw audit와 실제CSV writer 통과. 조작한 query 기록은 audit에서 거부했다. 네 arm의 공통 초기 incumbent와 같은 모델의 두 정책 초기 posterior 일치도 확인했다.
- Main2,304runs의 모든 label·query·posterior recommendation·noisy incumbent를 재구성했다.6912개 checkpoint의 Gaussian posterior를 별도 precision/SVD 경로로 검증했다. 최대 label 오차8.88e−16, posterior 오차1.20e−10.
- 전체39,168selected fits에서 finite 실패76개: joint_mean42,joint_probe34,SSPd3양arm0. Primary는joint_probe11개이며 finite parameters/NLL/status를 모두 남겼다. 추가 restart·행 삭제·target 교체는 없다. Primary optimizer gate는 이를 포함해 통과했다.
- 현재 source 및 raw24파일 hash 유지. 과거 active source manifests8개/88source entries도 확인했다. 이전 tomo utility의 기록된 startup repair는 선언된MANIFEST_V2와 보존된v1source에 따라 검증했다. 과거 파일을 새 manifest에 맞춰 수정하지 않았다.
- [online_curves.png](online_curves.png) / [PDF](online_curves.pdf)는 audited 결과의 시각화다. Probe arms의 band는 target-bootstrap pointwise95% 설명용 구간이고 panel별 y축 scale은 다르다. Primary paired 효과의 CI와 구별한다. 그림을 실제 열어 label/legend/caption을 확인했다.

## 지금까지의 연구는 유의미한가?

**이전보다 논문으로서 의미가 분명히 강화됐다.** 이제 size/composition 문제를 주장하는 데서 그치지 않고 다음 연결을 갖는다.

1. 기존 이론과 통제 injection은 size 추정 실패가 같은 count의 composition 판단까지 오염시킬 수 있음을 정량화한다.
2. Flexible joint 추정은 일부 shape에서 충분한 단순 교정이었다. Cut나 별도 projection이 항상 필요하다는 전제는 실제 반증으로 제거했다.
3. 이번에는 강한 SSPd3+동일 probe와 같은 관측 예산에서 실제 noisy-observed incumbent의 회복까지 확인했다. Static 예측 개선만 보여주는 상태를 넘어섰다.
4. Narrow-support 손해와 probe 기회비용도 드러나, 곡선 표현 문제와 필요한 정보/관측 문제를 분리할 근거를 얻었다.

하지만 독창성의 중심은 ordinary count GP나 기존 space-filling rule 자체가 아니다. **Subset-specific failure의 정량화, 필요한 교정의 이유와 충분성, 실제 선택 결과 및 실패 조건을 연결하는 전체 기여**로 서술해야 한다. 기존 appendix의 다항식/matrix retention, cut/partialling-out, variable-cardinality BO 문헌을 새 발명으로 쓰지 않는다. 이번 합성 bank가 독립 seed라 해도 새로운 물리 objective나 nonlinear composition으로의 일반화 증거는 아니다.

## 다음 판단

- Joint count GP+probe를 다음 독립 외부/물리 과제 확인 후보로 유지한다. 현재 결과를 확인 bank로 삼아 kernel/bounds/probe frequency를 다시 맞추지 않는다.
- 우선순위는 실제 objective와 다른 composition 구조에서도 효과가 남는지 확인하는 것이다. 기존 tomography40 confirmation bank를 재튜닝/새 확인에 재사용하지 않는다. Corrected physical operator, strong SSPd3+동일 query policy, 비용과 모델 recipe scale을 명세한 별도 cycle이 필요하다.
- Query 개발은 narrow wave와 불필요한 probes의 손해를 줄이는 구체적 목표가 있을 때 진행한다. Generic experimental design/IDS와 대조하며, 평균 혼합 성공을 모든 count 함수에 대한 안전성으로 바꾸지 않는다.
- 새 kernel을 추가하는 복잡화보다, 현재 단순 교정의 효과와 한계를 더 직접적으로 입증하는 것이 다음 기여를 강화한다.

이번 cycle은 실행·감사·독립 결과 리뷰를 완료했다. 다음 물리 실험을 실행 중이거나 예약했다는 뜻은 아니다.

## 산출물

PLAN/REVISION_LOG, PLAN_REVIEW_*/CODE_REVIEW_*/RESULT_REVIEW_*, model/benchmarks/run/analyze/verify/test_online, NUMERICAL_CHECKS, MANIFEST/EXECUTION, runs/00..23.json, RAW_HASHES, SUMMARY, CURVES, per_run.csv, AUDIT_COMPLETE, ARCHIVE_LOCK_AUDIT, online_curves.png/pdf.
