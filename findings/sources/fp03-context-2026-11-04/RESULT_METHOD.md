# FP03 결과 방법 검토

2026-11-04. **FP03는 사전 개발 gate FAIL로 종료한다. Context weighting의 유효한 부분 효과는 보존하지만, 현재 degree-weighted Potts/MF family를 충분한 해결책이나 성공한 missing-feature PFN으로 확대하지 않는다.** 다음에 검증할 한 가설은 기존 learned full/edge의 실제 conditional predictor를 정확한 두 anchor로 포함했을 때 같은 context calibration이 유용한지다. 이는 별도 개발 cycle이며 FP03 결과를 정정하거나 구제하는 작업이 아니다.

검토 자료는 [보고서](results/REPORT_KO.md), `summary.json`, `artifact_manifest.json`, `audit.json`, `checks.json`, 두 plan과 protocol, 기존 `model.py`/FP01 graph core다. Manifest248개 파일의 SHA256을 직접 확인했고 누락·불일치는 없었다. Score/weight/control 재구성 오차는 감사 기록상 각각 최대3.55×10⁻¹⁵/1.67×10⁻¹⁶/2.98×10⁻⁸이며 none/95 weights는 bit-identical이다. 이 reviewer는 operators 구현자이므로 수치 구현의 독립성은 별도 NumPy/oracle 검증에 근거한다. 이번 검토에서 신규 모델 실행·평가·학습·branch 변경은 하지 않았고 결과 파일도 바꾸지 않았다.

## 1. 판정에 필요한 수치

Feature-none, 동일 rep0/seed11의5개 개발 graph에 동일 가중을 준 결과다.

| 방법 | Accuracy % | BAcc % | NLL | Brier |
|---|---:|---:|---:|---:|
| adaptive | 60.910 | 53.272 | 1.33772 | .62142 |
| uniform prior mixture | 35.269 | 25.635 | 1.38693 | .66661 |
| held-score MAP | 52.425 | 45.724 | 1.42016 | .62967 |
| fixed edge, a=log4,b=0 | 62.523 | 56.611 | 1.62348 | .56913 |
| inherited old full | 55.691 | 54.599 | 2.53411 | .70453 |
| inherited old edge | 63.750 | 57.432 | 1.62087 | .54456 |
| restart LP | 66.319 | 59.848 | 1.63741 | .76178 |
| harmonic LP | 50.689 | 43.137 | 1.23455 | .57337 |

**Adaptive−prior accuracy +25.641%p는 실재하는 효과이므로 보존한다.** 같은 candidate probabilities와 같은 입력에서 weight만 바꾼 비교라 이 family 내 context weighting의 직접적인 개발 근거다. 그러나 prior35.269%는 약한 비교 대상이다. Adaptive는 fixed edge보다 accuracy −1.612%p, BAcc −3.339%p이고, old edge보다 accuracy −2.840%p, BAcc −4.160%p이며 restart LP에도 못 미친다. 개선된 mean NLL로 balanced/minority 성능 실패를 대체하지 않는다.

열 가지 사전 gate 중5개가 통과하고5개가 실패했다. Physics old_full 대비 +37.352%p는 통과했지만, CS +3.032%p는 +5%p 기준에 못 미쳤다. CiteSeer accuracy −12.223%p 및 NLL +.23278, mean BAcc의 old_edge 미달, mean accuracy의 fixed-edge 미달도 실패다. Current10-gate는 그대로이며 일부 통과로 primary FAIL을 바꾸지 않는다.

Mean NLL은 fixed edge보다 .28576 낮았고 old_edge보다 .28316 낮았다. 그러나 Brier는 둘보다 나쁘며, per-graph NLL 이득도 균일하지 않다. 특히 PubMed에서 adaptive1.07710 대 fixed2.71817의 차이가 크다. 따라서 이를 “전반적인 calibration을 해결했다”는 단일 문장으로 압축하지 않는다. NLL은 calibration뿐 아니라 predictive discrimination도 반영한다.

## 2. Family 실패와 calibration 실패를 어디까지 분리할 수 있는가

### 보존되는 긍정 결과

Balanced folds는 단일 held label의4→3 count cue를 막았고, CS의13 extras도 calibration에서 차단했다. Context126 points/19 folds로 만든 score에서 weights를 재구성할 수 있으며, 실제 predictor는 query truth를 읽지 않았다. 직접 labeled edge가 적어도 unlabeled paths를 통한 held prediction으로15개 predictor의 weighting을 바꿀 수 있었다. 적어도 이 실험은 “context는 아무 정보도 주지 못한다”는 결론과 맞지 않는다.

### 남은 calibration 한계

Predictive weighting은 uniform prior보다 낫지만 고정된 좋은 predictor를 보존하지 못했다. 모든5개 graph의 none에서 fixed-edge accuracy보다 낮고, broad weights에는 chance/weak candidate mass가 남는다. 그렇다고 결과를 본 뒤 score temperature를 바꿔야 한다는 결론은 나오지 않는다. 현재 unit mean-score rule의 결과이며 그대로 종료해야 한다.

Held calibration과 final prediction의 label coverage가 다르다. 예를 들어 CS calibration은 class당2개·총28개의 anchor를 썼고 final은 원55개를 썼다. Macro held risk는 불균형 query accuracy/NLL의 직접 추정량도 아니다. CS의 held-score MAP는 index6인데 query accuracy45.221%로 fixed-edge73.171%보다 낮았다. 이는 calibration ranking을 full-context query 성능의 oracle로 볼 수 없다는 구체적 경고다. 작은/dependent folds, criterion 차이, coverage shift 중 하나를 유일 원인으로 식별한 결과는 아니다.

### Family coverage의 별도 문제

CiteSeer none은 adaptive40.674%, fixed40.991%, best **single** candidate41.834%인데 old_full은52.898%다. 새15개 family가 기존에 유용했던 full behavior를 회수하지 못했다는 강한 개발 신호다. PubMed도 best single accuracy48.389%가 restart LP71.598%에 못 미쳤다. 단순한 a/b predictive score가 실제 graph의 관계 정보를 충분히 표현한다고 볼 근거가 없다.

하지만 **best single은 convex probability mixture의 accuracy 상한이 아니다.** 서로 다른 query에서 실수하는 두 predictor를 평균하면 두 single보다 높은 accuracy를 낼 수 있다. 예를 들어 true class probability가 두 query에서 (.9,.4)와(.4,.9)인 두 binary predictor는 각각 절반만 맞지만 평균(.65,.65)은 둘 다 맞는다. Aggregate NLL도 mixture가 가장 좋은 single보다 나아질 수 있다. 따라서 41.834%를 “가능한 모든 mixture의 최대 정확도”로 읽거나 “weighting을 아무리 잘해도52.898%에 절대 도달하지 못한다”고 선언하지 않는다. Convex-hull oracle 최적화는 이번 검토에서 새로 수행하지 않았다.

더 확실한 문제는 **operator family가 old_full을 구조적으로 포함하지 않는다는 것**이다. 기존 model의 homophilic `J_full[i,j]`는 learned density/signal을 읽는 pair별 conditional Potts score이고, observed edge strength도 단일 a가 아니다. FP03의 nonedge score는 `b*d_i*d_j/sum(degree)`라는 rank-one degree form이며, old full의 nonedge BP도 product mean field로 바꿨다. Coefficient grid가 충분히 넓다는 것만으로 이 score/inference 차이는 없어지지 않는다. Saved predictions가 우연히 convex-combination으로 표현되는지와 실제 parent operator를 family에 포함하는지는 다른 질문이다.

이 결과가 식별하는 범위는 현재 scalar/degree-offset/MF family와 현재 weighting 규칙이다. Homophily에 따른 context adaptation 일반, conditional inference 전체, 모든 nonedge 활용 또는 calibrated mixtures 전체를 반박하지 않는다.

## 3. 추가 feature 효용은 이번에도 입증되지 않았다

Adaptive의95−none 평균은 accuracy **−.107%p**, BAcc **−.337%p**, NLL **+.000389**다. Harm이 fixed-edge의 평균−1.176%p보다 작았다는 사실과, 새 feature 정보가 query를 개선했다는 주장은 다르다. Graph-only weights를 none/95에 공유했으므로 이 차이는 candidate별 feature feedback의 합성 효과이며 feature-conditioned relation posterior를 추론한 결과도 아니다.

신규 shuffle이 없어 alignment 성공도 검증되지 않았다. Source0/1·hidden poison 검사는 허용된 계산 경로를 검증하는 boundary test이지 feature utility의 증거가 아니다. FP01부터 누적된 “centroid family만 바꾸면 추가 feature 정보가 살아난다”는 가설은 여전히 지지되지 않는다. 이번 none gate의 일부 성공을 missing-feature PFN 성공으로 이전하지 않는다.

## 4. ONE 다음 가설: 실제 inherited full/edge를 두 anchor로 가진 동일 conditional operator

**다음 단계가 있다면15개 a/b grid를 확장하지 말고, 이미 존재하는 frozen learned full/edge의 실제 conditional operators만을 두 anchor로 놓아 context calibration의 가치를 검사한다.** 질문은 다음 하나다.

> 실제로 유용했던 두 parent predictor를 정확히 포함할 때, 관측 context의 balanced predictive score가 동일한1/2 prior mixture보다 유용한 prediction을 만드는가?

후보 정의는 graph의 label-independent frozen homophilic potential `J_full`과 `J_edge=A*J_full`이다. 동일 BP24/damping.5·literal C_global/GH41 feature factors·context clamps를 두 endpoint 모두에 적용한다. Full은 실제 nonedge BP를 유지한다. FP03 sparse MF로 근사한 것을 full anchor라고 부르면 안 된다. Existing none/95 reference probabilities와 endpoint parity를 source lock 이전 correctness gate로 둔다. Label-dependent states는 fold마다 새로 시작하고, graph-only frozen potential만 공유한다. Full-context role weights를 fold에서 재사용하는 지름길도 피한다. 이 가설은 fixed homophilic branch에 대한 것이며 새 role selector를 함께 추가하지 않는다.

Weighting은 현재와 같은 balanced classwise folds와 fixed clipped macro mean score를 그대로 쓴다. 두 개의 weights를 context에서 계산해 최종 두 probability arrays를 평균한다. Prior는1/2–1/2로 사전 고정한다. 핵심 controls는 actual full, actual edge, 두-anchor prior, 두-anchor calibrated mixture이며 기존 strong LP와 PFN references도 그대로 남긴다. 두 parent 중 query accuracy가 높은 것을 graph별로 골라 primary로 제시하지 않는다.

이것이 **정당한 새 가설일 수 있는 이유**는 기존 family가 old_full을 포함하지 않는 확인된 설계 차이를 고치고, weighting과 anchor reconstruction을 분리할 수 있기 때문이다. 단지 관측한 best index 근처에 a/b를 더 촘촘하게 놓는 grid rescue와 다르다. 반대로15개 grid에 유리한 parent들을 계속 붙이거나, lambda/temperature를 같은10개 query 성능으로 조절하면 rescue가 된다. 가장 작은 두-anchor 실험으로 제한해야 한다. 이 문서에서는 intermediate coefficient나 추가 predictor를 제안하지 않는다.

이 새 실험도 이미 본 결과에서 고른 **development-informed** 가설이다. 같은5개 graph에서 잘돼도 independent real generalization이나 novelty가 되지 않는다. FP03 실패는 고치지 않고 새 cycle/plan/세 리뷰/lock에서 비용·endpoint parity·성공/중단 기준을 먼저 정해야 한다. 이전 gate 미통과로 중단하기로 한 fresh synthetic final bank와 neural training을 자동 실행할 근거도 없다. 새 가설의 개발 가능성을 확인한 뒤 별도 synthetic evaluation을 사전 설계해야 한다.

Dense full BP는 현재 sparse MF보다 비쌀 수 있다. 비용을 줄이기 위해 anchor를 다른 approximation으로 바꾸면 family-containment 질문이 다시 불분명해진다. 먼저 작은 batch와 고정 worker에서 자원을 명세하고, parity를 만족하는 실제 operator의 비용을 함께 보고해야 한다. 비용 문제 자체가 score/weight scientific setting 변경의 허가가 되지는 않는다.

## 5. 주장 범위와 문헌

Conditional Potts/BP, held-out score weighting과 predictive model averaging은 기존 원리다. [Yao et al.](https://arxiv.org/abs/1704.02030)은 stacking과 pseudo-BMA 등의 차이를 다룬다. 현재 mean-score weighting이나 위 두-anchor 검사가 그 문헌의 검증된 최적 알고리즘이라는 뜻은 아니다. 새 graph에서 neural gradient update가0이어도 context-derived weight adaptation은 발생한다.

기존 개발 graph를 반복해서 보며 family를 바꾸는 것은 [Cawley–Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)이 지적한 finite-sample model-selection 위험과도 연결된다. 그래서 이번 gate와 모든 candidate/control은 그대로 보존하고, 후속 개발 성공을 최종 일반화로 쓰지 않아야 한다.

현재 성과는 **classical context calibration이 이 사전 고정 family의 약한 uniform prior를 크게 개선한 것**, 그리고 score/weights/predictions를 재구성 가능한 상태로 남긴 것이다. Main missing-feature method의 실증 지지는1/5, weighting의 제한된 개발 효과는3/5로 평가한다. 다음 두-anchor 가설도 신경 모델의 새 학습 원리가 아니라 기존 predictor의 context calibration 연구다. Feature utility와 강한 classical comparator 대비 추가 가치가 없으면 PFN novelty를 주장하지 않는다.

## 최종 판정

FP03는5/10 gate 통과·전체FAIL로 동결한다. Adaptive>prior의25.641%p를 보존하되 fixed-edge/old-edge/minority 및 feature-utility 실패를 함께 남긴다. 다음 한 가설은 실제 inherited full/edge를 동일한 정확한 conditional inference family에 두 anchor로 포함하는 context calibration이며, 지금은 실행하지 않은 별도 개발 제안이다.
