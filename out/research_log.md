# GraphPFN 연구 기록

실패·중단, 보존한 효과, 교훈, 미실행을 구분한 누적 기록이다. 숫자는 보존한 원문 보고서에서 옮겼고, 원인 미상인 결과에 사후 인과 설명을 붙이지 않는다. 새 논문과의 연결은 revisit.md의 재검토 단서에서 확인한다.

총 18개 프로젝트 · 75개 기록 (실패·중단 29개, 보존 16개, 교훈 22개, 미실행 8개)

## Common-context witness 고정 곱셈

실험 조건: 실제 개발 graph, feature-free LP, 새 학습0회, 고정 same-context product.

### BIND-F01 [실패·중단] Class pooling 전 context identity를 보존해 곱하면 실제 graph에서 좋아진다.

k20 pooling-before-product: ACM76.390% vs uniform LP85.555%, IMDB28.311% vs35.413%. IMDB k20 test row 약96.1%는 공통 seed support가0이었다.

- 판정: 고정 common-witness product 중단
- 원인·병목 [진단이 지지]: 동일 witness를 강제하는 곱셈이 현재 sparse-support graph에서 대부분의 evidence를 제거했다. 표현력 collision toy가 실제 오류 원인이라는 증거는 없었다.
- 닫은 범위: 현재 모든 relation이 같은 context witness를 가져야 하는 고정 곱셈
- 닫지 않은 것: Task별 variable binding·논리 연산을 추론하는 일반 executor는 별개다.
- 재검토 조건: 공유 witness가 필요한 경우와 별개 witness를 허용할 경우를 관측 context로 구분하는 연산이 나올 때.
- 기준선: Uniform LP, class-pooling 후 product, 같은 graph의 강한 PFN
- 최소 재검증: 먼저 실제 common-support 빈도와 오류의 관계를 확인하고, 고정 곱셈이 아닌 task-adaptive binding의 independent gain을 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/binding.md](../findings/sources/graphpfn-2026-11-01/binding.md)

## Frozen counterfactual PFN residual

실험 조건: λ=1 고정, LP × PFN-full/PFN-neutral; 실제 개발 graph, 새 학습 없음.

### RATIO-F01 [실패·중단] Full−Neutral log probability가 LP에 더할 새로운 feature evidence다.

FISF ratio ACM82.35%, IMDB34.45%; LP85.55/35.41%, Full PFN74.36/44.55%. Controlled ratio ACM82.22/IMDB34.50%도 개선하지 못했다.

- 판정: Frozen 비율 보정 중단
- 원인·병목 [진단이 지지]: Neutral이 정확한 feature marginalization이 아니므로 비율을 likelihood ratio로 해석할 근거가 없다. IMDB LP log margin은0.134~0.151로 작고, residual argmax 변경률34.68%였다. 원인별 기여는 분리하지 않았다.
- 닫은 범위: 현재 frozen full/neutral 확률 비율의 λ=1 보정
- 닫지 않은 것: LP-conditioned 학습과 일관된 생성모형의 likelihood ratio 일반은 별개다.
- 재검토 조건: 통제 입력의 의미와 feature marginalization을 정의하고 확률 비율이 유효한 증거가 되는 조건을 제시하는 방법이 나올 때.
- 기준선: LP/PFN alone, 50:50, controlled neutral
- 최소 재검증: 먼저 feature intervention으로 순효용을 확인하고, 동일 구조·주입 아래 ratio가 정답 방향 보정인지 새 graph에서 평가한다.
- 원문: [findings/sources/graphpfn-2026-11-01/ratio.md](../findings/sources/graphpfn-2026-11-01/ratio.md); [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

## 실제 관측 feature 기여의 개입 분석

실험 조건: ACM/IMDB k20/class, 99.5% missing의 Full/No-observed/Shuffle 및 독립 FISF injection 통제.

### FEAT-F01 [실패·중단] IMDB의 PFN 우위는 남은 실제 관측 feature의 평균 추가 효용이다.

IMDB FISF Full44.55%→No-observed44.51%; controlled Full−Zero0.03%p. 제거 전후 argmax는 평균30.47% 바뀌었으므로 feature가 예측에 무영향이라는 뜻은 아니다.

- 판정: 해당 feature-reliability 동기 약화
- 원인·병목 [진단이 지지]: 평균상 실제 값 효과가 거의 상쇄됐다. 구조 입력 PFN과 synthetic-input 경로의 차이를 feature-only evidence로 오인할 수 없다. Node별 양·음의 효과가 왜 상쇄되는지는 미상이다.
- 닫은 범위: 현재 IMDB/99.5% 결과의 PFN−LP 격차를 실제 feature 복원 이득으로 설명
- 닫지 않은 것: 다른 결측률·graph와 query별 feature utility는 기각하지 않는다.
- 재검토 조건: 실제 관측 feature의 양·음 utility를 누수 없이 분리하거나, 유효 feature가 남는 조건을 특정한 방법이 나올 때.
- 기준선: Structure-only PFN, no-observed end-to-end, independent-injection controlled Full/Zero
- 최소 재검증: Mask·context·query·graph를 맞춘 제거/셔플 개입에서 실제 값 경로와 synthetic injection 변화를 분리한다.
- 원문: [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

## Feature↔Label 양방향 추론

실험 조건: 10K 규모 quadratic prototype, 합성1536 tasks, 2 seeds, 384 updates; 공식 GraphPFN 아님.

### COUP-F01 [실패·중단] Label→feature 반복 피드백이 일방향 추론보다 유용하다.

Matched 일방향72.72%, 양방향72.66%; 차이−0.06%p CI[−0.50,+0.39]. Shape shift 차이−0.01%p CI[−0.52,+0.55].

- 판정: 피드백 증분 미확인
- 원인·병목 [진단이 지지]: 피드백의 추가 정보를 확보하지 못했다. 관측 feature가 전혀 없으면 latent H가0에 머물고 피드백 제거와 출력이 같았다. 이 구조적 퇴화가 모든 조건의 미개선을 설명하지는 않는다.
- 닫은 범위: 현재 quadratic coupling을 더 반복·확대하는 방향
- 닫지 않은 것: 비선형 label-conditioned feature inference와 다른 유효 prior는 평가하지 않았다.
- 재검토 조건: 라벨 피드백이 실제로 새로운 feature 상태를 만들고 그 정보가 일방향 모델에 없음을 보이는 추론 연산이 나올 때.
- 기준선: 동일 예산 일방향 모델, 피드백 제거, 강한 context ridge baseline
- 최소 재검증: Zero-observation 퇴화와 nonzero 조건을 나눠 같은 초기화·task·예산으로 실제 예측 증분을 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

## Native heterogeneous 입력·기존 relational FM 대조

실험 조건: DBLP native block missing 및 ACM/IMDB/Freebase 개발용 비교. Adapted implementation과 공식 모델 성과를 구분.

### HET-F01 [실패·중단] Adapted HetGFD가 cross-type path 입력보다 큰 증분을 준다.

DBLP k=5 full: cross-type 89.96%, adapted HetGFD 89.92%; drop_A 90.18/90.19%. k=20 full은 둘 다93.55%.

- 판정: 새 메소드 headroom 부족
- 원인·병목 [진단이 지지]: 관계 경로와 남은 native source가 이미 강한 신호를 전달했다. 격차가 작다는 관찰은 확인했지만 redundancy만이 원인이라는 인과 분해는 하지 않았다.
- 닫은 범위: 현재 DBLP adapted diffusion의 작은 차이를 범용 missing hetero-PFN 기여로 확대
- 닫지 않은 것: 원래 HetGFD/HGT 재현이나 다른 schema에서의 결론이 아니다. 다중 block 소실 조건은 별도 여지가 있다.
- 재검토 조건: 타입별 feature 의미와 관측 pattern을 보존하며 기존 path redundancy를 넘어서는 장치가 나올 때.
- 기준선: Native cross-type paths + frozen GraphPFN, 강한 attribute-free metapath LP
- 최소 재검증: 파생 helper feature 누수를 제거하고, 실제 native block graph에서 missing/full 대비와 최강 구조 기준선을 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/native_blocks.md](../findings/sources/graphpfn-2026-11-01/native_blocks.md)

### HET-P01 [보존] Typed path 정보를 column으로 전달해도 기존 PFN이 이미 활용한다.

Gate1-P의 B5g는 k=3의 세 decision graph에서 사전 val-oracle−1%p 기준을 충족했다. 새 learned path adapter를 그 격차만으로 정당화하지 않았다.

- 확대 금지: Val oracle은 배포 모델이 아니며, 새로운 학습된 heterogeneous PFN의 성과도 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/hetero_gate1p.md](../findings/sources/graphpfn-2026-11-01/hetero_gate1p.md); [findings/sources/graphpfn-2026-11-01/hetero_gate0.md](../findings/sources/graphpfn-2026-11-01/hetero_gate0.md)

### HET-L01 [교훈] 기존 relational FM도 현재 node-classification adapter에서는 자동으로 missing에 강하지 않았다.

RDB-PFN의 ACM k20 mean_count_depth5 full81.53%/missing66.55%, official_depth2 full74.87%/missing35.18%. RT budget 표는 64-query·1 seed 대조로 제한된다.

- 원문: [findings/sources/graphpfn-2026-11-01/rdb.csv](../findings/sources/graphpfn-2026-11-01/rdb.csv); [findings/sources/graphpfn-2026-11-01/rt.csv](../findings/sources/graphpfn-2026-11-01/rt.csv); [findings/sources/graphpfn-2026-11-01/rt_coverage.csv](../findings/sources/graphpfn-2026-11-01/rt_coverage.csv); [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

### HET-L02 [교훈] 미리 집계된 helper feature를 독립 native 관측값으로 취급하지 않는다.

ACM/IMDB helper feature는 target feature에서 파생된 집계일 수 있음을 확인했다. DBLP는 서로 다른 native source block으로 별도 처리했다.

- 원문: [findings/sources/graphpfn-2026-11-01/native_blocks.md](../findings/sources/graphpfn-2026-11-01/native_blocks.md)

## PCFI/FISF 입력 확장

실험 조건: Cora/CiteSeer 개발 그래프와 PubMed secondary, frozen GraphPFN, context/mask 일치.

### VIEW-F01 [실패·중단] Support metadata·다중 전파 view·PCFI/FISF 입력 결합이 범용 증분을 준다.

8조건 평균 PCFI 69.65%, PCFI/FISF views 69.73%, multiscale 69.61%. 사전 +1%p 양쪽 parent 초과 기준을 만족한 후보는 없었다.

- 판정: 전체 조건 진입 기준 미통과
- 원인·병목 [원인 미상]: 출력에 보인 증분이 작고 조건별 부호가 달랐다. 표현 정보 중복, projection 손실, frozen backbone의 활용 한계 중 무엇이 주원인인지는 분리하지 못했다.
- 닫은 범위: 현재 frozen 저차원 support/view 추가를 범용 메소드로 확대
- 닫지 않은 것: Native cell-wise provenance encoder를 학습하는 방법과 입력 자체를 바꾸는 학습은 평가하지 않았다.
- 재검토 조건: 관측 provenance를 정보 손실 없이 표현·학습하고 equal-width 값 대조를 넘는 장치가 나올 때.
- 기준선: PCFI/FISF parents, equal-width value controls, 같은 비용의 single predictor
- 최소 재검증: 같은 차원·추론 비용으로 새 실제 graph에서 parent 둘과 metadata 없는 값 대조를 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md); [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md)

### VIEW-P01 [보존] PubMed whole-node secondary에서 input view의 작은 긍정 신호

평균 PCFI 70.05%, FISF 71.50%, input views 73.00%, prediction average 71.97%.

- 확대 금지: 한 graph의 secondary이며 전체 조건 우위·novelty·독립 다중 domain 검증으로 확대하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md); [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md)

## 초기 JR-PFN과 decoder 표현력

실험 조건: Feature-free binary SBM, target8+helper4, 총 context1/2/4, Exact Bayes 소형 열거.

### JR-P01 [보존] Exclusive의 joint-role 제약 보존은 강한 Plain/factorized를 넘었다.

Exclusive JR−Large Plain+3.20%p, JR−Factorized+1.76%p. Learned posterior−known prior는+0.14%p에 그쳤다.

- 확대 금지: 모든 prior에서 우월하거나 context에서 미지 joint dependence를 새로 식별했다는 성과가 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/jr.md](../findings/sources/graphpfn-2026-11-01/jr.md)

### JR-P02 [보존] Context posterior에는 실제 predictive value가 있었다.

Exclusive 총 context4에서 역할 정보0.12447 nats, exact role update 기대 NLL 이득0.02278. Decoder/predictive 병목도 함께 있었다.

- 확대 금지: 오라클 posterior의 이득이 곧 학습 모델의 성과나 효율을 보장하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/posterior_shift.md](../findings/sources/graphpfn-2026-11-01/posterior_shift.md)

### JR-L01 [교훈] Positive-only diffusion의 구조적 불가능성은 loss로 고칠 수 없었다.

Context label이 모두 A일 때 당시 decoder는 모든 query에서 p(A)≥0.5. Exact teacher는 nonedge로 반대 class를 지지할 수 있었다. 이후 contrastive/BP decoder가 이 제한을 넘었다.

- 원문: [findings/sources/graphpfn-2026-11-01/distillation.md](../findings/sources/graphpfn-2026-11-01/distillation.md); [findings/sources/graphpfn-2026-11-01/contrastive.md](../findings/sources/graphpfn-2026-11-01/contrastive.md)

### JR-L02 [교훈] 이미 해결한 중간 병목을 현재 미해결 문제로 다시 제안하지 않는다.

양의 diffusion 표현력 문제는 contrastive decoder로 전환했다. 이후 EAJ/R1의 경쟁력 실패는 별개의 문제이며 기존 LLBP 항목에 보존한다.

- 원문: [findings/sources/graphpfn-2026-11-01/contrastive.md](../findings/sources/graphpfn-2026-11-01/contrastive.md); [findings/sources/graphpfn-2026-11-01/llbp.md](../findings/sources/graphpfn-2026-11-01/llbp.md)

## Learned-Likelihood BP / EAJ

실험 조건: known-family binary SBM, feature-free 관계 그래프, target 8개 + helper 4개, 두 관계와 네 role 가설, 총 context 1/2/4 (클래스당 아님). 작은 latent state는 exact enumeration, conditional BP는 고정 반복.

### LLBP-F01 [실패·중단] Role-conditional 감독을 더하면 최종 예측이 개선된다.

C_conditional − B_predictive에서 conditional KL은 세 prior 모두 줄었지만, predictive KL 증분의 paired 95% CI는 세 prior 모두 0을 포함했다 (Independent +0.000188, Exclusive −0.000181, Coupled −0.000582).

- 판정: 추가 이득 미확인
- 원인·병목 [진단이 지지]: Component conditional fit 개선이 최종 predictive 개선으로 이어지지 않았다. Positive diffusion의 표현력 제한도 뒤에 확인했지만 전체 미개선의 유일 원인으로 분해하지 않았다.
- 닫은 범위: 기존 staged decoder에 conditional loss를 더해 메인 메소드로 확장하는 갈래
- 닫지 않은 것: conditional distillation 일반이 불가능하다는 결론은 아니다. 추가 감독은 의도한 conditional 오차를 실제로 줄였다.
- 재검토 조건: 표현 가능한 decoder에서 conditional 감독의 독립 predictive 이득을 보이는 방법이 나올 때.
- 기준선: Predictive-only JR, Plain predictive distillation
- 최소 재검증: Decoder와 teacher를 고정하고 auxiliary의 독립 증분을 같은 task·예산으로 비교한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F02 [실패·중단] Posterior까지 정확하게 학습하면 최종 예측도 좋아진다.

D_all − C_conditional에서 posterior KL은 줄었지만 predictive KL은 세 prior 모두 늘었고 CI도 모두 0을 제외했다 (예: Independent +0.004039, CI [+0.002707, +0.005418]).

- 판정: 유의 악화
- 원인·병목 [원인 미상]: Posterior fit 감소와 predictive KL 악화가 함께 관측됐다. 목적의 불일치나 decoder 오차 보상 중 원인을 하나로 확정하지 못했다.
- 닫은 범위: 현재 architecture·loss weight·예산에서 posterior loss를 추가하는 것
- 닫지 않은 것: posterior 감독 일반의 기각이 아니라, component 오차 감소가 최종 품질을 보장하지 않는다는 결과다.
- 재검토 조건: Posterior 정확성과 deployment prediction을 동시에 보존하는 감독·추론 원리가 나올 때.
- 기준선: Predictive-only, conditional-only, exact posterior substitution
- 최소 재검증: 같은 decoder에서 posterior KL과 deployment predictive KL을 따로 평가한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F03 [실패·중단] Masked-edge likelihood auxiliary가 learned BP의 핵심 학습 기여다.

BP + auxiliary − BP predictive-only가 Independent +0.000374 [+0.000187, +0.000587], Exclusive +0.000261 [+0.000144, +0.000364]로 유의하게 나빴고, Coupled는 CI가 0을 포함했다.

- 판정: 유의 악화 (새 loss 기여 주장 중단)
- 원인·병목 [원인 미상]: 추가 likelihood auxiliary는 predictive-only보다 악화했다. Local likelihood fit이 label risk와 정렬되지 않을 가능성은 가설이며 정확한 원인은 미상이다.
- 닫은 범위: masked-edge likelihood loss를 새 학습 원리·핵심 성능 증분으로 주장하는 것
- 닫지 않은 것: auxiliary 없이도 potential을 학습하고 BP로 좋은 예측을 만들 수 있었다. 실패한 것은 학습 가능성이 아니라 추가 loss의 우수성 주장이다.
- 재검토 조건: Local factor 감독이 query risk에 도움이 되는 조건을 증명·검증한 방법이 나올 때.
- 기준선: Learned BP predictive-only, fitted constant
- 최소 재검증: 같은 potential·BP·role update에서 loss만 바꿔 independent gain을 본다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F04 [실패·중단] Learned potential + 기존 frozen role q만으로 넓은 조건에서 adaptive OOD 추론이 된다.

Matched에서는 좋아졌지만, Coupled에서는 auxiliary 포함 learned BP가 constant보다 나빴다. Density 0.12 외삽에서는 일부 Plain/signed 대조보다도 나빴고, density/signal 이동에서 oracle potential과의 차이가 커졌다.

- 판정: 범용성 미확보
- 원인·병목 [진단이 지지]: Potential과 frozen role posterior의 OOD 불일치 및 conditional 근사 오차가 후보였다. 뒤의 coherent-update 대조가 일부 병목을 지지하지만 모든 shift의 원인을 확정하지 않는다.
- 닫은 범위: 관측 통계 encoder로 potential을 학습해 고정 role head와 결합하는 것만으로 범용 추론기를 주장하는 것
- 닫지 않은 것: 원인은 확정하지 못했다: potential point estimate, frozen posterior의 분포 이동, conditional BP 근사, 이들의 상호작용.
- 재검토 조건: 새 graph 통계로 일관된 role/parameter evidence를 추론하고 단순 EB/grid를 넘는 방법이 나올 때.
- 기준선: Constant+update, density estimator, Grid6/81
- 최소 재검증: 같은 conditional branches에서 posterior·potential 교차 대조와 새 density/signal bank를 평가한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F05 [실패·중단] Parameter 후보를 혼합하되 기존 role q를 유지해도 충분하다.

Coupled/high density에서 true-parameter Bayes KL이 기존 MRP 0.112820, full-grid parameter mixture + frozen role q 0.078075, 같은 BP branches + coherent joint weights 0.017615였다. 개선의 대부분은 role marginal 갱신에서 왔다 (dependence만 분리한 same-marginal 차이는 −0.001434 [−0.001946, −0.000872]).

- 판정: 기대 기각 + 해석 기각
- 원인·병목 [직접 확인]: 같은 BP branches에서 coherent joint weights가 frozen role q보다 좋았다. 큰 개선을 같은 주변분포의 dependence 효과 전체로 돌릴 수는 없다.
- 닫은 범위: frozen-hybrid로 충분하다는 기대, 그리고 전체 개선을 H–parameter dependence 보존만의 효과로 보는 해석
- 닫지 않은 것: frozen hybrid도 정규화된 분포이며 일부 조건부 dependence를 보존한다. 단순 독립 head의 곱이라고 부르지 않는다.
- 재검토 조건: 일관된 evidence 갱신을 미지 family에서도 정확하고 저렴하게 구현하는 장치가 나올 때.
- 기준선: Joint grid, same-marginal product, frozen-grid
- 최소 재검증: Conditional branches와 marginals를 맞춰 role update와 dependence 증분을 분리한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F06 [실패·중단] EAJ(Evidence-Amortized Joint PFN)가 exact-grid를 비슷한 품질·훨씬 낮은 전체 비용으로 대체한다.

Independent heads, analytic initializer, own-product보다는 좋았지만 teacher exact marginals의 곱보다 여섯 조건 모두 나빴다. Signal-high에서는 Constant+update가 세 prior 모두 EAJ보다 좋았다 (EAJ−Constant CI 모두 0 초과). Posterior-only는 exact evidence보다 약 1.9–2.9배 빨랐지만 공통 81×4 BP 때문에 end-to-end 감소는 약 6–12%였다.

- 판정: 경쟁력 미달 (대체·확장 중단)
- 원인·병목 [진단이 지지]: 학습 evidence의 정확도와 공통 BP 비용이 병목이었다. Posterior-only 속도 증분이 end-to-end에 크게 전달되지 않았다.
- 닫은 범위: EAJ에 새 surrogate loss를 더하거나 모델·학습량을 늘려 경쟁력을 추격하는 갈래
- 닫지 않은 것: joint 표현 자체의 장점(약한 head 대비)은 있었다.
- 재검토 조건: 강한 classical update의 품질을 유지하며 전체 branch/evidence 비용을 줄이는 연산이 나올 때.
- 기준선: Constant+role-update, exact-product, Joint Grid6/81
- 최소 재검증: Posterior 추정만의 시간과 shared BP를 포함한 end-to-end 품질·비용을 분리한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F07 [실패·중단] EAJ를 직접 predictive 학습하면 실용성과 posterior 해석을 함께 회복한다.

B는 A보다 unknown-parameter exact-grid predictive KL을 여섯 조건 모두 개선했다 (상대 약 15.7–31.9%, CI 0 제외). 그러나 posterior KL은 여섯 조건 모두 악화했고, signal-high 세 prior 모두 Constant+role-update보다 NLL·Brier가 유의하게 나빴으며, accuracy 개선은 미확인이었다. Convex oracle 기준 A→oracle gap 중 B가 회수한 비율은 약 16–32%였다.

- 판정: 부분 성공·메소드 구제 실패 (predictive 개선을 posterior 복원 개선으로 보는 해석 기각)
- 원인·병목 [진단이 지지]: 실제 decoder에 맞춘 predictive weights와 정확한 Bayesian posterior 복원은 다른 목적이었다. Direct predictive 개선은 있었지만 classical 기준선을 넘지 못했다.
- 닫은 범위: EAJ에 새 surrogate loss를 추가하는 갈래
- 닫지 않은 것: 직접 predictive 학습이 작동하지 않았다는 결론은 틀리다. B의 출력은 decoder에 맞춘 aggregation weights로 남긴다.
- 재검토 조건: Decoder 오차를 다루면서 posterior 해석과 predictive quality를 함께 보장하는 방법이 나올 때.
- 기준선: Direct predictive EAJ, Constant+update, exact-grid
- 최소 재검증: 두 KL과 실제 NLL/Brier를 함께 보고 calibrated posterior라는 주장을 별도 검증한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F08 [실패·중단] 같은 potential에서 role weights를 다시 계산하는 자기-update가 OOD 문제를 전반적으로 해결한다.

Exclusive high density에서는 자기 exact update 0.04029가 Constant+update 0.05073보다 좋았다 (세 prior 모두 CI 0 제외). 하지만 low density에서는 자기 exact update 0.01159가 frozen q 0.00915, Constant+update 0.00917보다 나빴다. Bethe를 exact로 바꿔도 해결되지 않았다.

- 판정: 보편적 OOD 해결 주장 기각
- 원인·병목 [진단이 지지]: Low density에서는 exact role update도 실패해 Bethe partition 근사만으로 설명할 수 없었다. Potential·conditional·parameter uncertainty 상호작용은 미해결이다.
- 닫은 범위: potential과 weights를 맞추면 분포 이동도 전반적으로 해결된다는 주장
- 닫지 않은 것: 자기-update는 특정 조건에서 유효한 추론 개선이다.
- 재검토 조건: Sparse/OOD potential uncertainty와 conditional inference를 함께 해결하는 연산이 나올 때.
- 기준선: Constant+update, R1, Grid6/81
- 최소 재검증: Low density를 포함한 새 bank에서 exact/Bethe·conditional 교차를 고정해 본다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F09 [실패·중단] 자기-update 경로까지 학습하는 T1이 R1보다 우수하다.

T1−R1 KL −0.000192, 95% CI [−0.000576, +0.000164], accuracy 약 −0.093%p. Low-density 세 prior에서는 R1보다 좋았지만 별도 16조건에서 KL 점추정이 악화했고, low density에서도 R0에는 못 미쳤다.

- 판정: 추가 이득 미확인
- 원인·병목 [원인 미상]: Self-update를 미분해 학습하는 경로는 작동했지만 R1 대비 이득이 없었다. Optimization·model bias 중 고유 원인은 미상이다.
- 닫은 범위: T1의 우수성을 확보하기 위해 loss·예산을 계속 늘리는 것
- 닫지 않은 것: head 없는 학습 경로와 gradient 흐름은 확인됐다. 더 큰 예산에서 절대 개선되지 않는다는 증명은 아니다.
- 재검토 조건: 동일 계산 budget에서 inference-aware 학습의 독립 이득을 보이는 원리가 나올 때.
- 기준선: Frozen R1, R0, Constant+update, Grid6
- 최소 재검증: 추론 시 head 제거 효과와 새 학습 효과를 분리하고 initial state·schedule·예산을 맞춘다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F10 [실패·중단] Low-density 실패는 partition 근사 또는 role weights만의 문제다.

Constant/learned weights × constant/learned conditional 네 조합 교차에서 learned conditional 교체는 세 prior 모두 손해였다. Learned weights 교체는 Independent·Exclusive에서 손해, Coupled에서 도움이었고, 비가법적 상호작용이 있었다. Exact role update에서도 실패했다.

- 판정: 단일 병목 설명 기각
- 원인·병목 [진단이 지지]: 교차 대조에서 learned conditional 손해와 비가법 interaction을 확인했다. Role weights만 고치면 된다는 단일 원인 설명은 지지되지 않는다.
- 닫은 범위: role posterior만 고치면 된다는 설명, exact partition이면 해결된다는 설명
- 닫지 않은 것: 기각하지 못한 가능성: potential 추정 편향, parameter uncertainty 부족, conditional BP 오차, 과확신, 이들의 상호작용.
- 재검토 조건: Potential·partition·conditional 오차를 일관되게 제어하는 추론 연산이 나올 때.
- 기준선: Constant/learned weights×conditional 교차, exact update
- 최소 재검증: 네 교차 조합과 exact-role/BP conditional을 함께 비교해 단일 병목 주장을 검증한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F11 [실패·중단] R1은 강한 model-based 기준선보다 좋은 메인 메소드다.

Bayes predictive KL / NLL이 R1 0.020488 / 0.615840, Joint Grid81 0.014249 / 0.610493, Joint Grid6 0.011939 / 0.608083이었다. Grid6이 KL·NLL·Brier 모두 우수했고 paired CI가 0을 제외했다. R1−Constant는 −0.001744 [−0.002072, −0.001425]로 유의하게 좋았다.

- 판정: 경쟁력 미달
- 원인·병목 [직접 확인]: 현재 known-family에서 작은 training-support joint grid가 품질·비용 기준을 더 잘 충족했다. Learned 모델 전체가 불필요하다는 보편 원인은 아니다.
- 닫은 범위: R1을 강한 기준선보다 우수한 메인 메소드로 승격하는 것
- 닫지 않은 것: R1−R0, R1−Constant의 유의한 개선은 그대로 남는다. Grid6과 Grid81은 격자 수뿐 아니라 prior 자체가 다르므로 '작은 grid가 일반적으로 낫다'로 해석하지 않는다.
- 재검토 조건: 작은 명시적 grid가 처리하지 못하는 실제 family/schema에서 우위를 보이는 범용 추론법이 나올 때.
- 기준선: 사전 prior 고정 Grid6/81, Constant+update, R1
- 최소 재검증: 알려진 family·prior 접근 조건과 end-to-end 비용을 맞춘 새 bank에서 비교한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F12 [실패·중단] R1은 품질이 낮아도 큰 속도 이점으로 정당화된다.

RTX 3090에서 단일 graph latency R1 5.17 ms vs Grid6 4.66 ms, Batch16 6.41 ms vs 6.81 ms, Batch16 추가 작업 메모리 1.44 MiB vs 64.04 MiB. Latency 반복 측정의 분산·CI는 없다.

- 판정: 큰 속도 우위 미확보
- 원인·병목 [직접 확인]: 병렬 grid와의 실제 latency가 비슷했고 posterior/branch 수 감소가 큰 전체 속도 이점으로 이어지지 않았다. 작업 메모리 절감은 별도로 보존한다.
- 닫은 범위: R1의 존재 이유를 추론 가속으로 바꾸는 것
- 닫지 않은 것: 추가 작업 메모리 절감은 사실이다. 다만 더 큰 graph 처리나 배포상 이익은 검증하지 않았고, 이를 사후 메인 주장으로 바꾸지 않기로 했다.
- 재검토 조건: 실제 처리 규모나 memory constraint에서 quality를 유지하는 비용 우위를 입증하는 연산이 나올 때.
- 기준선: 병렬 Grid6/81, R1
- 최소 재검증: 동일 hardware·batch·quality 범위에서 latency 반복 변동과 total/working memory 및 최대 처리 graph를 측정한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P1 [보존] Learned BP checkpoint: 원래 positive/signed decoder보다 좋은 matched 확률 예측

Matched Bayes predictive KL이 Frozen positive D 0.063955 / 0.052586 / 0.085548에서 Learned BP predictive-only 0.015946 / 0.011013 / 0.028391로 줄었다 (Independent / Exclusive / Coupled). 단순 signed diffusion만으로는 이 개선을 설명할 수 없었다.

- 확대 금지: 모든 생성 family에서 learned likelihood가 필요하다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P2 [보존] R1 자기-update: 같은 potential의 활용 방식만 바꿔 개선했고, 새 graph bank에서 재현됐다.

R1−R0 KL −0.012995 (약 38.8% 감소), accuracy +0.376%p.

- 확대 금지: 학습 단계도 head가 필요 없었다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P3 [보존] R1−Constant paired 결과

최신 비교에서 KL 개선 CI가 0을 제외했다 (−0.001744 [−0.002072, −0.001425]).

- 확대 금지: 강한 모든 baseline보다 우수하다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P4 [보존] T1: head 없는 학습 경로와 gradient 흐름 확인

Role-update 경로까지 gradient가 흐르는 것을 검증했다.

- 확대 금지: R1보다 우수한 학습법이다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P5 [보존] EAJ A→B: 직접 predictive 학습의 효과

여섯 조건 모두 grid predictive KL 개선 CI가 0을 제외했다.

- 확대 금지: posterior 복원도 개선됐다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P6 [보존] Convex oracle: 보고된 certificate와 큰 미회수 gap

Graph-shared convex oracle 12,096개의 최적성 certificate를 확인했다고 보고됐다. 정확한 bound는 메시지 손상으로 미상이다.

- 확대 금지: 현재 모델이 그 gap을 쉽게 회수한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P7 [보존] Nonedge/mask 대조: 관측 상태에 맞는 factor를 쓰는 것이 중요하다.

Observed nonedge를 버리거나 unknown pair를 nonedge로 처리하면 크게 악화했다. 예: Independent에서 edge-only BP 0.082736 vs full edge/nonedge BP 0.016319, 관측률 0.2에서 올바른 mask 0.000226 vs unknown→nonedge 0.044486.

- 확대 금지: 모든 현실 데이터에서 해당 관측 구분이 주어진다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P8 [보존] 작업 메모리: 지정 Batch16 측정에서 큰 절감

추가 작업 메모리 1.44 MiB vs 64.04 MiB (총 GPU 메모리 차이가 아님).

- 확대 금지: 실제 처리 가능한 graph 규모가 더 크다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P9 [보존] Joint Grid6/81과 Constant+update는 이후 비교 자산으로 보존한다.

알려진 family와 선언된 prior를 쓰는 model-based 기준선이 learned 방법이 넘어야 할 기준이 됐다.

- 확대 금지: 어떤 하나의 grid prior가 모든 조건을 지배한다 (Grid81은 matched Exclusive·sparse observation에서, Grid6은 low density에서 상대적으로 나쁠 수 있었다).
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L0 [교훈] Teacher를 정확하게 주는 것과 decoder가 그 target을 표현할 수 있는 것은 다르다.

양의 relation weight와 normalization을 쓰는 label diffusion은 context가 전부 클래스 a이면 모든 query에서 p(a) ≥ 1/2를 유지한다. Nonedge로 반대 클래스를 지지하는 exact SBM teacher를 표현할 수 없었고, 이후 decoder 자체를 바꾸는 방향으로 이동했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L1 [교훈] Component fitting이 좋아져도 전체 predictor가 좋아진다는 보장은 없다.

초기 conditional/posterior loss, EAJ predictive 학습의 posterior 악화, low-density 교차 분석에서 반복해서 분리됐다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L2 [교훈] 더 직접적인 loss가 메소드 경쟁력을 해결해주지 않는다. Loss 변경의 효과와 메소드 경쟁력은 다른 질문이다.

EAJ에서는 직접 predictive 학습이 지표를 개선했지만 Constant+update를 넘지 못했고, T1은 R1 대비 추가 이득이 없었다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L3 [교훈] 일관된 추론이라고 OOD가 해결되지 않는다. 같은 potential 기반 추론은 잘못 맞는 potential을 자동으로 교정하지 않는다.

Self-update는 효과가 있었지만 low density를 해결하지 못했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L4 [교훈] Neural amortization이나 branch 수 감소가 곧 wall-clock 우위는 아니다.

EAJ에서는 공통 BP 계산이 비용을 지배했고, R1은 병렬 Grid6과 latency가 비슷했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L5 [교훈] 기준선 하나를 이겼다고 메인 메소드가 되지 않는다. 약한 비교의 유의성과 강한 비교의 경쟁력은 다른 기준이다.

R1−R0, R1−Constant는 성공이었지만 Grid6과의 비교에서 승격 근거가 사라졌다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L6 [교훈] 강한 기준선(이미 존재하는 명시적 grid 등)과의 품질·비용 비교를 재학습·확장 실험보다 먼저 한다.

Joint-grid가 이미 강했는데도 R1 개선 후 T1 재학습을 먼저 했다. R1 대 병렬 Grid6/81 비교가 T1보다 앞에 있어야 했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L7 [교훈] '추론 변경 효과가 확인된 후보'를 '메인 방법'으로 부르지 않는다. 반대로 강한 기준선에 밀렸다고 국소 개선의 실재까지 부정하지 않는다.

R1을 성급하게 '메인 방법으로 유지'라고 표현했다가 Grid6 비교 후 정정했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L8 [교훈] Posterior KL, predictive KL, accuracy, 비용, 일반화는 각각 다른 검증이 필요하다. 한 단계의 성공을 다음 단계의 성공처럼 이어 붙이지 않는다.

여러 목표의 성공을 하나의 강한 주장으로 연결할 위험이 있었다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L9 [교훈] Oracle까지 gap이 크다는 것은 다음 투자의 자동 근거가 아니다.

Gap을 관측 정보와 현재 모델로 회수할 수 있는지, 회수해도 강한 baseline을 넘는지는 별개였다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L10 [교훈] 원래 목표가 무너진 뒤 남은 장점 하나(메모리 등)에 맞춰 목표를 사후 변경하지 않는다. 실패 조건에 맞춘 gate·temperature 추가도 기본 연장으로 채택하지 않는다.

메모리 이점은 사실이지만 검증 없이 메인 주장으로 바꾸지 않았다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U1 [미실행] 관계의 관측과정 자체를 학습하는 BP

제안만 있었고 완료된 실험이 없다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U2 [미실행] Relation missingness가 label/role에 의존하는 설정

실험하지 않았다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U3 [미실행] Unseen semantic schema, 실제 heterograph, 큰 graph scaling

실험하지 않았다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U4 [미실행] Temperature/shrinkage/density gate, 더 큰 모델, 추가 surrogate loss, 더 긴 T1 학습

추가 실행을 제안하지 않았거나 중단한 것이지, 실제로 학습해서 실패를 증명한 것은 아니다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

## LP-conditioned structural residual

실험 조건: 소형 일방향 prototype에 LP offset/내부 conditioning 추가, 2 seeds 및 새 paired factorial; 새 schema 아님.

### LPC-F01 [실패·중단] LP internal conditioning이 output offset과 Plain을 일관되게 넘는다.

Matched conditioned−Plain−0.35%p, heldout_shape+0.90%p; offset 대비−0.03/+0.35%p. 새 factorial 내부 효과+0.36~0.52%p이나 relation 추가효과+0.10 CI[−0.21,+0.42], feature 효과+0.06 CI[−0.11,+0.23]. NLL/Brier는 offset-only가 더 좋았다.

- 판정: LP residual tuning·대규모 확장 중단
- 원인·병목 [원인 미상]: 보인 이득 대부분과 output offset 효과를 구분해야 했고, schema adaptation 특유의 이득이라는 증거가 약했다. 조기 relation aggregation이 원인이라는 설명은 가설이지 확정 원인이 아니다.
- 닫은 범위: 현재 LP offset/residual conditioning 확대
- 닫지 않은 것: Relation 정보를 합치기 전 보존하는 구조와 별도 semantic schema 전이는 별개다.
- 재검토 조건: 압축된 LP 확률보다 풍부한 context 정보를 활용하며 output offset만의 효과를 넘어서는 연산이 나올 때.
- 기준선: Plain, offset-only, 같은 파라미터·초기화·학습 budget
- 최소 재검증: Relation 수·feature 폭·missing을 분리한 factorial과 새로운 semantic schema에서 내부 conditioning의 독립 증분을 확인한다.
- 원문: [findings/sources/graphpfn-2026-11-01/lpc.md](../findings/sources/graphpfn-2026-11-01/lpc.md); [findings/sources/graphpfn-2026-11-01/lpc_factorial.md](../findings/sources/graphpfn-2026-11-01/lpc_factorial.md)

## 초기 Missingness-aware GraphPFN

실험 조건: 공식 prior의 소형 binary tasks, missingness mechanism mixture, 단일 학습 seed. NaN 학습과 official imputation 경로를 구분.

### MISS-F01 [실패·중단] Missingness prior 추가가 공식 입력 경로와 미학습 MNAR에서 유리하다.

NaN-input Ours−B4는 label_dep에서 +0.75/+1.43/+0.78%p였지만 degree_dep/self_mask는 약 ±0.1%p. Official imputation의 released GraphPFN이 NaN-trained Ours보다 label_dep에서 0.54/0.63/0.12%p 높았다.

- 판정: 범용 우위 미확보·해당 라인 중단
- 원인·병목 [진단이 지지]: Raw-NaN 기준선이 공식 imputation보다 약했고, official fill만으로도 mask가 잘 드러났다(AUROC 0.97). Context-query mask 관계 불일치에서 mixed-prior 이득이 유지되지 않았다. 이것만으로 모든 원인을 확정하지는 않는다.
- 닫은 범위: NaN 학습의 작은 이득을 official-input robustness나 unseen-MNAR 일반화로 승격
- 닫지 않은 것: Imputed-input에 맞춘 prior의 효과는 미평가다. 더 넓은 MNAR 식별 문제도 기각하지 않는다.
- 재검토 조건: 관측 메커니즘의 context-query shift를 다루면서 official preprocessing 대비 추가 가치를 보이는 학습 원리가 나올 때.
- 기준선: 공개 GraphPFN + official imputation, 동일 예산 B4
- 최소 재검증: 동일 입력 경로·초기화·예산의 새 bank에서 context-query mismatch와 미학습 mask를 paired 평가한다.
- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md); [findings/sources/graphpfn-2026-11-01/missing_pilot.md](../findings/sources/graphpfn-2026-11-01/missing_pilot.md); [findings/sources/graphpfn-2026-11-01/missing_context.md](../findings/sources/graphpfn-2026-11-01/missing_context.md)

### MISS-U01 [미실행] Official imputed-input에 맞춘 Ours-Imp 최종 비교

B4imp는 255/500 update에서 중단, 저장 checkpoint는 update 200. Ours-Imp와 seed 1 및 imp evaluation은 실행하지 않았다.

- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md)

### MISS-L01 [교훈] 공식 전처리와 신호 대비 headroom을 먼저 확인한다.

No-missing accuracy 약72.9%, majority 약62%, MCAR 80% 손실 약3.5%p. 약한 입력 경로와의 비교만으로 새 prior의 가치를 주장하지 않는다.

- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md)

## Observation-aware subgraph policy

실험 조건: Frozen 공개 GraphPFN, 새64/32 graphs, 2 policies×3 seeds, node budget64 학습·32/64/96평가, query8/graph.

### OBS-F01 [실패·중단] Mask-aware 학습 정책이 강한 경로 규칙을 넘는다.

Budget64 Observation/Paths NLL matched1.07193/1.05353, cluster1.08551/1.07685. 차이+0.01839 CI[−0.06529,+0.10213], +0.00866 CI[−0.11473,+0.11820].

- 판정: 주 성공 기준 미통과·확장 보류
- 원인·병목 [원인 미상]: Mask를 보는 것은 유용했지만 비슷한 효과를 고정 shortest-path closure도 얻었다. Joint subset 표현력·stochastic train/deterministic inference·학습 budget 중 정확한 원인은 분리하지 않았다.
- 닫은 범위: 현재 mask/geometry shared-node MLP를 더 학습해 메인 방법으로 확대
- 닫지 않은 것: 모든 learned retrieval을 기각하지 않는다. Budget32 cluster 차이−0.19284 CI[−0.39771,+0.00284]는 보조 개발 신호이며 승리가 아니다.
- 재검토 조건: 동일 resident budget에서 mask-aware 공동 부분그래프 구성이 강한 경로 규칙보다 나은 utility를 보이는 장치가 나올 때.
- 기준선: Evidence-paths, PPR, equal-capacity structure policy, random, context prior
- 최소 재검증: Fresh graph에서 사전 budget·mask gate를 고정하고 preprocessing/search를 포함한 비용과 label 예측을 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

### OBS-F02 [실패·중단] 부분그래프 선택 개선으로 경쟁력 있는 확률 예측을 확보했다.

사후 Laplace context-prior NLL0.85241/Brier0.55736. Budget64 Observation의 matched NLL은+0.21952 CI[+0.12555,+0.31655], cluster+0.23310 CI[+0.14044,+0.32727] 더 나빴다.

- 판정: 단순 확률 기준선에도 경쟁력 부족
- 원인·병목 [원인 미상]: Frozen PFN끼리의 상대 개선이 단순 class prior 대비 utility를 보장하지 않았다. Calibration·적은 context·subgraph mismatch·전처리 중 원인은 미상이다.
- 닫은 범위: PPR/구조 정책 대비 개선만으로 좋은 missing-feature predictor라고 주장
- 닫지 않은 것: 확률 점수 비교를 accuracy의 같은 결론으로 자동 확대하지 않는다. Prior 대조는 사후 sanity check다.
- 재검토 조건: 극소 context·불완전 입력에서 단순 prior보다 좋은 확률 정보를 실제로 회수하는 추론 원리가 나올 때.
- 기준선: Uniform class probability, Laplace context prior, full graph, strongest fixed retrieval
- 최소 재검증: 학습 전에 단순 확률·majority 대조를 고정하고 NLL/Brier/accuracy를 독립 graph에서 함께 평가한다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

### OBS-P01 [보존] 관측 상태를 선택 정책에 주는 독립 NLL 효과

Budget64 Observation−Structure NLL matched−0.16128 CI[−0.27147,−0.05871], cluster−0.24695 CI[−0.38936,−0.11246]. PPR 대비도 두 조건 NLL CI가0 미만이었다.

- 확대 금지: Feature 값의 인과적 효용, strong rule 초과, 확실한 accuracy 개선이나 실제 heterograph 전이로 확대하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

### OBS-L01 [교훈] 학습 정책의 선택 단계 속도를 end-to-end 우위로 승격하지 않는다.

Matched budget64 selection1.76ms vs Paths13.73ms지만 PCFI+forward 포함 약146.82/155.90ms. Global statistics와 정책학습 비용은 이 합계 밖이다.

- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

## Context path 선택과 고정 LP/PFN 혼합

실험 조건: 기존 실제 개발 graph, k5/20/class, 같은 context와 public test. 새 dataset holdout 아님.

### XFER-F01 [실패·중단] DBLP에서 통했던 context-LOO path 선택을 다른 graph로 옮기면 강하다.

Expanded LOO는 ACM k20 75.78% vs uniform LP85.55%, IMDB k5 27.53% vs uniform32.04%. Context accuracy로 목적을 바꿔도 IMDB k20 33.55%였다.

- 판정: 고정 선택 규칙의 전이 실패
- 원인·병목 [진단이 지지]: 소수 context의 ranking이 query utility와 맞지 않았다. 후보 pool 확대와 accuracy 목적 변경만으로는 해결되지 않았다. 추정 분산·risk mismatch의 상대 기여는 미상이다.
- 닫은 범위: 현재 LOO scalar score로 하나의 metapath를 고르는 규칙
- 닫지 않은 것: Context가 유용한 관계를 알려주는 다른 설정이나 relation 의미 추론 일반은 기각하지 않는다.
- 재검토 조건: 극소 context의 추정 불확실성과 실제 query risk를 다루며 task 내 utility를 식별하는 방법이 나올 때.
- 기준선: Uniform/union LP, individual paths, fixed GraphPFN path inputs
- 최소 재검증: 독립 graph에서 동일 context·후보 pool·선택 예산으로 사전 규칙을 고정해 검증한다.
- 원문: [findings/sources/graphpfn-2026-11-01/transfer.md](../findings/sources/graphpfn-2026-11-01/transfer.md); [findings/sources/graphpfn-2026-11-01/expanded.md](../findings/sources/graphpfn-2026-11-01/expanded.md); [findings/sources/graphpfn-2026-11-01/accuracy_selector.md](../findings/sources/graphpfn-2026-11-01/accuracy_selector.md)

### XFER-F02 [실패·중단] LP와 PFN 확률의 고정 혼합이 각자의 강점을 보존한다.

k20 FISF/LP 50:50: ACM82.20% vs LP85.55%, IMDB42.54% vs FISF44.55%. 같은 query에서 상보성은 있지만 혼합은 최강 단일 예측기를 못 넘었다.

- 판정: 고정 혼합 메인 방향 중단
- 원인·병목 [원인 미상]: 서로 맞히는 query가 다르다는 사실이 고정 확률 평균의 utility를 보장하지 않았다. 중복 증거가 원인일 수 있으나 직접 분리한 결과는 아니다.
- 닫은 범위: 현재 50:50/equal-thirds 사후 확률 혼합
- 닫지 않은 것: 모든 ensemble이 불가능한 것도, 각 branch가 feature-only/structure-only인 것도 아니다.
- 재검토 조건: 중복 evidence와 decision risk를 실제 관측량으로 구분하며 최강 단일 predictor를 보존하는 원리가 나올 때.
- 기준선: LP alone, PCFI/FISF+PFN alone, compute-matched fixed mixtures
- 최소 재검증: 같은 query·context에서 within-task 오류 보완성과 독립 개발/확인 graph의 평균·worst-case utility를 함께 본다.
- 원문: [findings/sources/graphpfn-2026-11-01/fusion.md](../findings/sources/graphpfn-2026-11-01/fusion.md)

## Plain relation adaptation과 CRR 제안 철회

실험 조건: 기존 Plain, exact-degree bipartite 재배선, 역할 교환·mechanism freeze·label shuffle 진단. CRR 추가 학습은 하지 않음.

### ROLE-L01 [교훈] Plain이 전역 relation utility 적응을 못한다는 전제로 새 router를 정당화하지 않는다.

공유 relation MLP에 relation-ID embedding이 없고, context 통계를 따라 역할 교환에 적응했다. Conflict end-to-end는 약2.09%p 하락해 완전 robustness는 아니다.

- 원문: [findings/sources/graphpfn-2026-11-01/role_reversal.md](../findings/sources/graphpfn-2026-11-01/role_reversal.md)

### ROLE-L02 [교훈] k=1의 centered label-alignment 통계에는 식별 한계가 있다.

클래스당 context 하나면 서로 다른 class centered one-hot 내적은 모두 −1/C다. 이 통계만 복잡하게 인코딩해도 새 정보가 생기지 않는다. 재배선 경로 품질 차이도 함께 고려해야 한다.

- 원문: [findings/sources/graphpfn-2026-11-01/role_reversal.md](../findings/sources/graphpfn-2026-11-01/role_reversal.md)

### ROLE-U01 [미실행] 새 CRR router·query-conditioned relation inference의 독립 학습

CRR 제안은 Plain과 기능이 겹쳐 철회했다. 기존 모델 진단을 새 router를 실제 학습해 실패한 결과로 세지 않는다.

- 원문: [findings/sources/graphpfn-2026-11-01/role_reversal.md](../findings/sources/graphpfn-2026-11-01/role_reversal.md)

## Reference-regret prior curriculum

실험 조건: 공개 GraphPFN continued training, 새64/32 graphs, 3 schedule seeds, 15 runs. Finite bank sampling만 변경.

### REG-F01 [실패·중단] Reference-regret priority가 hard-loss sampling 및 같은 비용의 Random보다 유리하다.

Matched NLL Regret0.76377, Loss0.76467, RandomBudget0.76266. Regret−Loss−0.00090 CI[−0.00543,+0.00404], Regret−RandomBudget+0.00111 CI[−0.00145,+0.00387].

- 판정: 메인 방법 확장 보류
- 원인·병목 [원인 미상]: Reference의 독립 utility와 채점 비용을 정당화하지 못했다. Stale priority·짧은 학습·input loss 중 원인별 영향은 분리하지 않았다. Loss와 regret의 초기 Spearman0.4264는 다른 task를 강조함을 보일 뿐 효용 증명이 아니다.
- 닫은 범위: 현재 cached finite-bank reference-regret sampler 확대
- 닫지 않은 것: 학습된 생성기·library·무한 stream prior learning을 구현하거나 실패시킨 것은 아니다.
- 재검토 조건: Reference의 추가 감독 비용을 회수하면서 실제로 회수 가능한 task difficulty를 식별하는 학습·생성 원리가 나올 때.
- 기준선: Random, Manual, Loss priority, cold-score-cost-matched RandomBudget, frozen FISF
- 최소 재검증: 새 graph bank와 고정 비용으로 Loss/RandomBudget을 함께 넘는지 확인하고 task weighting과 generator 학습을 분리한다.
- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md); [findings/sources/graphpfn-2026-11-01/regret_scope.md](../findings/sources/graphpfn-2026-11-01/regret_scope.md)

### REG-P01 [보존] 같은 update 수의 Random 대비 작은 Matched NLL 개선은 있었다.

Regret−Random−0.00185 CI[−0.00353,−0.00034]. Accuracy58.78→58.79%로 정확도 개선은 미확인.

- 확대 금지: Loss priority·같은 비용 대조·OOD를 넘었다고 확대하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md)

### REG-L01 [교훈] Singleton observed feature centering은 입력 정보를 없앨 수 있다.

Test99.5% whole-node에서 PCFI input 전체가0인 episode는 uniform20/32, clustered32/32. 관측 행0~1개에서 centering 손실을 직접 확인했다.

- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md); [findings/sources/graphpfn-2026-11-01/regret_scope.md](../findings/sources/graphpfn-2026-11-01/regret_scope.md)

### REG-U01 [미실행] 학습된 prior generator·새 mechanism library

실행한 것은 같은 official generator의 finite task bank를 재가중하는 pilot이다.

- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md)

## Relation-wise Evidence Head

실험 조건: 소형 shared relation backbone, 3 seeds, consistent/noisy/conflict, k1/3/5/10/20/class, 99.5% missing.

### EVID-F01 [실패·중단] Relation 출처를 보존하는 head가 강한 Plain의 class decision을 개선한다.

Conflict Separate86.09%, Mean85.34%, Plain88.96%. Separate−Plain−2.88%p CI[−3.66,−2.09], Separate−Mean+0.74%p CI[+0.49,+1.03].

- 판정: 현재 Evidence Head No-Go
- 원인·병목 [원인 미상]: Plain 정답을 잃는 mass5.36%가 구제2.48%보다 컸다. 정보 중복과 representation 간섭 중 원인은 식별하지 못했다. Confidence를 낮춰서 NLL이 좋아졌다는 설명은 후속 분석에서 기각됐다.
- 닫은 범위: 현재 histogram/evidence head의 확대와 calibration/margin loss로 구제
- 닫지 않은 것: Relation source 정보가 불필요하다는 결론이나 다른 내부 추론 연산의 기각은 아니다.
- 재검토 조건: Plain의 강한 결정 능력을 유지하면서 출처별 정보의 독립 가치를 회수하는 장치가 나올 때.
- 기준선: Plain, Mean/Separate 동일 용량 control, fixed context-weighted evidence
- 최소 재검증: 동일 task에서 harmed/rescued partition과 NLL/Brier/accuracy를 함께 비교하고 conflict 강도별 증분을 확인한다.
- 원문: [findings/sources/graphpfn-2026-11-01/evidence.md](../findings/sources/graphpfn-2026-11-01/evidence.md); [findings/sources/graphpfn-2026-11-01/evidence_utility.md](../findings/sources/graphpfn-2026-11-01/evidence_utility.md)

### EVID-L01 [교훈] 확률 점수 개선을 relation 출처의 새 정보나 accuracy 개선으로 치환하지 않는다.

Conflict NLL0.5320→0.3585, Brier0.2905→0.2044; NLL 개선의 산술90.5%는 Mean에서도 나타났다. Confidence63.99→84.31%였다.

- 원문: [findings/sources/graphpfn-2026-11-01/evidence_utility.md](../findings/sources/graphpfn-2026-11-01/evidence_utility.md)

## Compiler–Executor GraphPFN pilot

실험 조건: Fully observed Boolean graph, 문법222개(학습186/미학습36), train/test512/64 graphs, 3 seeds, context총4/8/16.

### PROG-F01 [실패·중단] 고정 executor 위 compiler가 조합 일반화와 품질·비용에서 강한 대조를 넘는다.

미학습 조합: compiler61.59%/NLL0.68349, Plain56.15%/0.68425, relational features+TabPFN85.18%/0.33352, ProgramGrid87.73%/0.23073. Grid latency도 더 짧았다.

- 판정: 현재 compiler 확장 보류
- 원인·병목 [원인 미상]: 관측 정보는 Grid/TabPFN이 회수했지만 compiler는 context 증가에도 약61.6%에 머물렀다. 표현·search guidance·학습 최적화 중 정확한 원인은 식별하지 못했다.
- 닫은 범위: 현재 작은 compiler를 missing·대형 heterograph로 확대
- 닫지 않은 것: 프로그램 합성·library learning 일반은 기각하지 않는다. 후보 문법 자체는 학습 중 모두 제공했고 missing은 실험하지 않았다.
- 재검토 조건: Context에서 계산 구성을 더 효율적으로 식별하고 bounded enumeration/강한 PFN feature 기준선을 넘어서는 장치가 나올 때.
- 기준선: Cached ProgramGrid grammar prior, relational execution features+TabPFN, matched Plain
- 최소 재검증: 같은 grammar와 program prior, 후보 구조 접근, fresh graph와 heldout composition에서 품질과 end-to-end 비용을 맞춰 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/program.md](../findings/sources/graphpfn-2026-11-01/program.md)

### PROG-U01 [미실행] Primitive library를 실제로 학습하거나 missing hidden fact를 공동 주변화하는 확장

실행된 pilot은 고정 grammar/executor의 fully observed 비교다. BDD/weighted model counting, learned library, 무한 grammar 탐색 성과는 없다.

- 원문: [findings/sources/graphpfn-2026-11-01/program.md](../findings/sources/graphpfn-2026-11-01/program.md)

## Predictor reliability Gate A/B

실험 조건: ACM/IMDB 99.5% missing, k5/20/class, 세 context/mask draw의 개발용 진단.

### SEL-F01 [실패·중단] Context Brier·local reliability로 PFN/LP의 정답을 선택할 수 있다.

IMDB k20 FISF: PFN44.55%, LP35.41%, Global41.51%, Local41.92%, Oracle59.16%. Local은 최강 predictor보다2.63%p 낮았다.

- 판정: Gate B 실패, Gate C 보류
- 원인·병목 [진단이 지지]: Brier와 accuracy 목적이 불일치하는 사례를 직접 관측했다. IMDB seed11에서 LP의 test Brier0.6624가 PFN0.6672보다 좋지만 accuracy는35.54% vs44.65%였다. Context noise만으로 설명할 수 없다.
- 닫은 범위: 현재 crossfit Brier·local propagation 기반 predictor 선택 규칙
- 닫지 않은 것: Oracle headroom을 실제로 회수하는 다른 risk estimator는 기각하지 않는다. 학습 selector는 이 실험에서 미실행이다.
- 재검토 조건: 극소 context에서 정확도 효용과 uncertainty를 함께 식별하는 추정법이 나올 때.
- 기준선: 최강 단일 PFN/LP, global/local selector, fixed mixture
- 최소 재검증: Query 정답 없는 입력만으로 실제 선택 utility를 독립 graph에서 평가하고 proper score와 accuracy를 분리한다.
- 원문: [findings/sources/graphpfn-2026-11-01/reliability.md](../findings/sources/graphpfn-2026-11-01/reliability.md)

### SEL-P01 [보존] Within-query complementarity는 실제로 있다.

ACM k20 FISF oracle90.43%, IMDB59.16%. Oracle은 query 정답으로 만든 상한이다.

- 확대 금지: 큰 oracle gap만으로 selector 학습의 성공 가능성을 입증하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/reliability.md](../findings/sources/graphpfn-2026-11-01/reliability.md)

## 출처 중첩·공유 보간 오차 attention

실험 조건: 1 seed, 64 updates×2 episodes, 16 heldout synthetic graphs; stochastic feature 위치를 맞춘 paired 결과.

### SRC-F01 [실패·중단] Source covariance를 보정하면 분류가 개선된다.

Vanilla accuracy 57.01%, covariance 57.04%. Covariance−vanilla +0.032%p, graph CI [−0.092,+0.157]. Source correspondence 파괴 시 working error score는 0.1395 증가했지만 분류 변화는 작았다.

- 판정: 독립적인 task-level 이득 미확인
- 원인·병목 [원인 미상]: 오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.
- 닫은 범위: 현재 source-overlap penalty/log 보정을 메인 방법으로 확대
- 닫지 않은 것: 출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.
- 재검토 조건: 공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.
- 기준선: Vanilla continuation, matched/log/diagonal, covariance-shuffled controls
- 최소 재검증: 분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

### SRC-L01 [교훈] 추가 사전학습의 효과를 오차 보정 모듈 효과로 치환하지 않는다.

Initial 대비 accuracy는 올랐지만 matched continuation 대비 증분은 약0.03%p였다. Balanced accuracy와 macro-F1은 initial 대비 하락했다.

- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)
