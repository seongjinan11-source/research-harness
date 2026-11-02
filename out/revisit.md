# 실패 병목과 논문 강점: 재검토 단서

자동 매칭은 논문 metadata의 강점·태그·키워드 검색이다. 해결·novelty·재개 승인이 아니다. 새 논문 원문에서 필요한 장치와 가정을 확인하고, 아래 최소 재검증을 통과해야 한다. 기존 논문도 포함하므로 목록에 나타났다는 사실이 새 해결책의 출현을 뜻하지 않는다.

원인 수준: 직접 확인 / 진단이 지지 / 미검증 가설 / 원인 미상. 직접 확인은 적힌 병목·대조를 확인했다는 뜻이며 모든 인과 원인을 밝혔다는 뜻은 아니다.

## BIND-F01 — Class pooling 전 context identity를 보존해 곱하면 실제 graph에서 좋아진다.

- 판정: 고정 common-witness product 중단
- 원인·병목 [진단이 지지]: 동일 witness를 강제하는 곱셈이 현재 sparse-support graph에서 대부분의 evidence를 제거했다. 표현력 collision toy가 실제 오류 원인이라는 증거는 없었다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 공유 witness가 필요한 경우와 별개 witness를 허용할 경우를 관측 context로 구분하는 연산이 나올 때.
- 먼저 이겨야 할 기준선: Uniform LP, class-pooling 후 product, 같은 graph의 강한 PFN
- 최소 재검증: 먼저 실제 common-support 빈도와 오류의 관계를 확인하고, 고정 곱셈이 아닌 task-adaptive binding의 independent gain을 비교한다.
- 닫지 않은 것: Task별 variable binding·논리 연산을 추론하는 일반 executor는 별개다.
- 원문: [findings/sources/graphpfn-2026-11-01/binding.md](../findings/sources/graphpfn-2026-11-01/binding.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## RATIO-F01 — Full−Neutral log probability가 LP에 더할 새로운 feature evidence다.

- 판정: Frozen 비율 보정 중단
- 원인·병목 [진단이 지지]: Neutral이 정확한 feature marginalization이 아니므로 비율을 likelihood ratio로 해석할 근거가 없다. IMDB LP log margin은0.134~0.151로 작고, residual argmax 변경률34.68%였다. 원인별 기여는 분리하지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 통제 입력의 의미와 feature marginalization을 정의하고 확률 비율이 유효한 증거가 되는 조건을 제시하는 방법이 나올 때.
- 먼저 이겨야 할 기준선: LP/PFN alone, 50:50, controlled neutral
- 최소 재검증: 먼저 feature intervention으로 순효용을 확인하고, 동일 구조·주입 아래 ratio가 정답 방향 보정인지 새 graph에서 평가한다.
- 닫지 않은 것: LP-conditioned 학습과 일관된 생성모형의 likelihood ratio 일반은 별개다.
- 원문: [findings/sources/graphpfn-2026-11-01/ratio.md](../findings/sources/graphpfn-2026-11-01/ratio.md); [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## DMARG-F01 — 현재 conditional tree circuit이 알려진 family의 강한 classical inference와 경쟁하는 메인 메소드다.

- 판정: Neural-baseline gate는 성공했지만 classical 품질·비용 gate는 실패. 현재 구성의 메인 메소드 확대는 보류.
- 원인·병목 [원인 미상]: 유효한 주변화에도 예측 fit의 격차가 남았다. Tree family의 표현력, G만 보는 potential encoder의 근사, 학습 예산의 기여를 분리하지 않았다. 정확한 고유 원인은 미상이다. 작은 known-family baseline의 우위를 모델 학습 전체의 불필요함으로 확대하지 않는다.
- 반증·배제한 설명: 모델 내부의 정규화·label 합 규칙·연속 feature 적분·conditional ratio는 열거와 대조했다.; Grid6-BP 비용에서 불필요한 exact query marginal 출력을 제외했고 원 reference와 예측 차이0을 검사했다.
- 재검토 조건: 강한 known-family inference의 예측 품질을 유지하거나, 작은 고정 grid가 다루지 못하는 family에서 실제 관측값과 구조의 결합 이득을 회수하는 연산이 나올 때.
- 먼저 이겨야 할 기준선: Gaussian-feature orientation을 적분하는 Grid6 exact/BP, known-feature-only, constant tree, 실제 public Mask PFN 및 같은 예산 direct graph. 더 작은 network 대비 승리만으로 재개하지 않는다.
- 최소 재검증: 같은 관측 정보·학습 task·전체 비용으로 fixed Gaussian 통계를 넘는 graph-specific 이득부터 확인한다. 유효한 circuit의 범위와 원 데이터 posterior를 분리하고 fresh bank·heldout family를 사용한다.
- 닫지 않은 것: 다른 유효 circuit 구조, family/schema 일반화, exact variable elimination의 다른 근사·공유 연산, MaM 자체를 실행하거나 실패시킨 결과가 아니다. DMARG-P01/P02의 실제 개선과 feature 이용은 보존한다.
- 원문: [findings/sources/direct-marginal-2026-11-02/results/paired.json](../findings/sources/direct-marginal-2026-11-02/results/paired.json); [findings/sources/direct-marginal-2026-11-02/results/benchmark.json](../findings/sources/direct-marginal-2026-11-02/results/benchmark.json); [findings/sources/direct-marginal-2026-11-02/results/decision.json](../findings/sources/direct-marginal-2026-11-02/results/decision.json)

- [NodePFN](https://arxiv.org/abs/2604.19028): 복수 태그 일치 (약한 단서); tags=graph-structure-use, prior-mismatch; keywords=-
  - 기록된 강점: cSBM의 homophily를 넓게 샘플링해 heterophily 쪽도 prior에 넣었다. 그 결과 heterophilous 그룹 평균에서 GCN보다 6%p 이상 높았다.

## DFACT-F01 — 현재 learned node/feature factor PFN이 강한 model-based inference와 경쟁하는 독립적인 메인 방법이다.

- 판정: 정확도 개선과 입력 민감도는 보존하지만 현재 known-family 구성의 메인 메소드 확대는 보류한다.
- 원인·병목 [원인 미상]: 유의한 neural accuracy 개선만으로 learned potential의 필요성이나 classical 대비 품질·비용 우위를 확보하지 못했다. Potential fit, binary model-family 제약, finite BP/Bethe 근사, 학습량의 기여를 분리하지 않았다. 고유 원인은 미상이다.
- 반증·배제한 설명: Source와 bank lock, all3200 bank hashes, frozen public weights, 실제 성공update·schedule, finite checkpoints와11952 probability 재채점을 확인했다.; Tree에서 BP/Bethe 계산을 독립 열거와 대조했고, class/feature/relation/node 순열과 hidden-value poison 검사를 통과했다. 이것이 cyclic production graph의 정확성 보장은 아니다.; BP-only reference에서 불필요한 exact query moments를 제외했고 원래 reference와 예측 차이0을 확인했다.
- 재검토 조건: 작은 알려진 Gaussian family의 고정 factor/parameter grid로 충분하지 않은 과제에서, 실제 관측값과 context를 결합하는 학습 원리가 독립적 품질·비용 이득을 제공할 근거가 생길 때.
- 먼저 이겨야 할 기준선: 동일 입력·예산의 실제 Mask/PCFI/FISF-trained PFN, fixed-factor BP, known-family Grid6 exact/BP, feature-only, LP/class prior. 실제 graph/schema 확장에서는 원래 Ephris/Wander와 해당 model-based 추론도 검토한다.
- 최소 재검증: 실패 bank에서 새 temperature/head/loss로 구제하지 않는다. 어떤 likelihood/function-family 또는 inference operation을 바꾸는지 먼저 명시하고 새 독립 bank에서 accuracy와 probabilistic quality·cost를 분리해 검증한다. Context total과 per-class, primary와 secondary를 구분한다.
- 닫지 않은 것: DFACT-P01/P02/P03의 실제 성과를 지우지 않는다. 다른 feature law, 추론 구조, variable schema, 큰 graph에서 학습한 potential의 효용을 기각하지 않는다. 미실행 후속을 실패로 세지 않는다.
- 원문: [findings/sources/disjoint-factor-2026-11-02/results/decision.json](../findings/sources/disjoint-factor-2026-11-02/results/decision.json); [findings/sources/disjoint-factor-2026-11-02/results/paired.json](../findings/sources/disjoint-factor-2026-11-02/results/paired.json); [findings/sources/disjoint-factor-2026-11-02/results/benchmark.json](../findings/sources/disjoint-factor-2026-11-02/results/benchmark.json); [findings/sources/disjoint-factor-2026-11-02/results/final_integrity.json](../findings/sources/disjoint-factor-2026-11-02/results/final_integrity.json)

- [NodePFN](https://arxiv.org/abs/2604.19028): 복수 태그 일치 (약한 단서); tags=graph-structure-use, prior-mismatch; keywords=-
  - 기록된 강점: cSBM의 homophily를 넓게 샘플링해 heterophily 쪽도 prior에 넣었다. 그 결과 heterophilous 그룹 평균에서 GCN보다 6%p 이상 높았다.

## FEAT-F01 — IMDB의 PFN 우위는 남은 실제 관측 feature의 평균 추가 효용이다.

- 판정: 해당 feature-reliability 동기 약화
- 원인·병목 [진단이 지지]: 평균상 실제 값 효과가 거의 상쇄됐다. 구조 입력 PFN과 synthetic-input 경로의 차이를 feature-only evidence로 오인할 수 없다. Node별 양·음의 효과가 왜 상쇄되는지는 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 실제 관측 feature의 양·음 utility를 누수 없이 분리하거나, 유효 feature가 남는 조건을 특정한 방법이 나올 때.
- 먼저 이겨야 할 기준선: Structure-only PFN, no-observed end-to-end, independent-injection controlled Full/Zero
- 최소 재검증: Mask·context·query·graph를 맞춘 제거/셔플 개입에서 실제 값 경로와 synthetic injection 변화를 분리한다.
- 닫지 않은 것: 다른 결측률·graph와 query별 feature utility는 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [GSPN](https://arxiv.org/abs/2305.10544): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [MCSSL](https://arxiv.org/abs/2605.11846): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 정보 증가에 따른 예측 갱신을 허용하면서 평균의 일관성을 제약한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## COUP-F01 — Label→feature 반복 피드백이 일방향 추론보다 유용하다.

- 판정: 피드백 증분 미확인
- 원인·병목 [진단이 지지]: 피드백의 추가 정보를 확보하지 못했다. 관측 feature가 전혀 없으면 latent H가0에 머물고 피드백 제거와 출력이 같았다. 이 구조적 퇴화가 모든 조건의 미개선을 설명하지는 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 라벨 피드백이 실제로 새로운 feature 상태를 만들고 그 정보가 일방향 모델에 없음을 보이는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 예산 일방향 모델, 피드백 제거, 강한 context ridge baseline
- 최소 재검증: Zero-observation 퇴화와 nonzero 조건을 나눠 같은 초기화·task·예산으로 실제 예측 증분을 비교한다.
- 닫지 않은 것: 비선형 label-conditioned feature inference와 다른 유효 prior는 평가하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
- [CSBMInference](https://arxiv.org/abs/2306.07948): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=latent feature
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [Wander](https://arxiv.org/abs/2610.06694): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=bidirectional
  - 기록된 강점: 구조적 수용 범위를 추론 때 조절할 수 있다. Message passing 기반 GFM은 이 범위가 network 깊이에 고정된다. 서로 떨어진 25×25 grid 5개에 grid마다 label을 2개만 준 GRIDS에서, walk 길이를 4에서 64로 늘리자 정확도가 56.6%에서 96.6%로 올랐다. Labeled 노드를 지나는 walk만 남기면 100%였다. 같은 과제에서 GraphPFN은 23.3%, NodePFN은 26.6%였다. Feature 없는 실제 그래프 4개에서도 Wander는 76.5%로 GraphPFN(60.1%)을 크게 앞섰다.

## FLAW-F01 — 관측 cell의 부호 제약을 제거하고 새 graph의 column 통계에 조건부인 factor를 학습하면 static과 기존 Factor보다 자기 피처가 없는 query 분류가 좋아진다.

- 판정: 사용자가 정한 primary accuracy 기준을 충족하지 못했으므로, 현재 cell 함수 교체만으로 독립 메인 메소드가 된다는 가설은 닫는다. 추가 head/loss/temperature/updates로 이 bank에서 구제하지 않는다.
- 원인·병목 [원인 미상]: 부호 제약 제거와 관측 column 통계 conditioning만으로 이 예산에서 분류 이득을 확인하지 못했다. 관측 정보량·통계 요약·함수족·비선형 최적화·finite BP/Bethe·학습량의 기여를 분리하지 않아 고유 원인은 미상이다. CI에0을 포함한 것을 효과0 또는 동등성의 증명으로 쓰지 않는다.
- 반증·배제한 설명: 세 Factor와 두 fixed-law controls는 사전 선택한 동일 CE seed11 graph encoder/factor를 고정한다. Static1547/adaptive1575 trainable params, 동일 초기 predictive scores, 동일 seed별2048 tasks와256 successful updates다. Frozen graph 재학습 효과가 아니다.; 새15개 모델을 CE만으로 학습했다. Public Mask/PCFI는 새 prior로 실제 학습한 별도 adapter architecture다. Law·actual parameter·orientation·query truth를 predictor 입력에 넣지 않았다.; Dev12 graphs의 수치·finite-gradient smoke weights3개를 폐기하고 설계를 잠근 뒤 final192 graphs를 생성했다. 최종 bank 생성 후 optimizer 전 발생한 keyword-only API 오류만 수정해 같은 bank와 초기화로 재시작했고 원 lock/로그를 보존했다. 분석 entrypoint는 같은 이름의 역사적 모듈과 import 충돌을 분리했다.; 전체 source/config/bank/reference/checkpoint hashes, freeze/schedule, 모든 mask와45개 prediction replay를 확인했다. 작은 tree의 normalized law/orientation 완전 열거와 fixed BP, hidden-value poison, class/relation/column permutation 검사도 통과했다.
- 재검토 조건: 관측 통계로 pointwise rho만 조절하는 것과 다른, 현재 함수족이 놓치는 feature–relation 의존성을 학습·전이하는 구체적인 연산이 등장하고 강한 명시적 법칙 주변화보다 유리할 근거가 생길 때.
- 먼저 이겨야 할 기준선: Original sign-constrained Factor, parameter-matched Flexible-static/adaptive, 동일 frozen graph의 full normalized law/parameter/orientation fixed-exact 및 same24-step fixed-BP, 실제 새 prior로 학습한 Mask/PCFI public PFN. Accuracy/NLL/Brier/reference KL과 end-to-end cost를 분리한다.
- 최소 재검증: 이 final bank를 추가 tuning에 쓰지 않는다. 새 연산의 구체적인 차이를 먼저 고정하고 독립 base graphs에서 자기 피처 없는 query accuracy를 primary로 비교하며 Gaussian 보존과 고정 추론 품질–비용을 함께 확인한다.
- 닫지 않은 것: 기존 disjoint feature/label 연결 연산·signed likelihood·자기 role-update와 이전 Factor/teacher의 국소 개선은 보존한다. 모든 학습된 feature factor, unseen feature laws, 가변 feature/schema, 더 큰 실제 graph,99.5% missing의 효과를 기각한 결과는 아니다. 이번에도 law 이름은 이미 보았다.
- 원문: [findings/sources/feature-law-2026-11-02/REPORT_KO.md](../findings/sources/feature-law-2026-11-02/REPORT_KO.md); [findings/sources/feature-law-2026-11-02/METHOD.md](../findings/sources/feature-law-2026-11-02/METHOD.md); [findings/sources/feature-law-2026-11-02/protocol.json](../findings/sources/feature-law-2026-11-02/protocol.json); [findings/sources/feature-law-2026-11-02/results/statistics.json](../findings/sources/feature-law-2026-11-02/results/statistics.json); [findings/sources/feature-law-2026-11-02/results/cost.json](../findings/sources/feature-law-2026-11-02/results/cost.json); [findings/sources/feature-law-2026-11-02/results/audit.json](../findings/sources/feature-law-2026-11-02/results/audit.json); [findings/sources/feature-law-2026-11-02/results/preflight.json](../findings/sources/feature-law-2026-11-02/results/preflight.json); [findings/sources/feature-law-2026-11-02/results/runtime_amendment.json](../findings/sources/feature-law-2026-11-02/results/runtime_amendment.json)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## FMARG-F01 — 단일 대치 대신 graph-correlated feature completion의 예측을 평균하면 두 주 결측 조건에서 point/diagonal/zero/PCFI 대조를 모두 넘는다.

- 판정: 현재 frozen surrogate completion operator의 사전 primary gate 미통과·확대 중단
- 원인·병목 [원인 미상]: 명시한 Gaussian-process surrogate와 4-sample frozen PFN 연산에서 필요한 동시 우위가 없었다. Feature-law misspecification, zero-fill 학습 PFN의 입력 활용 한계, 유한 Monte Carlo, 실제 feature 정보 부족의 원인은 분리하지 못했다.
- 반증·배제한 설명: Observed clamp·label 분리·covariance sampler·완전 관측 일치·weights 보존을 확인했다. 이것이 surrogate의 통계적 적합성을 보장하지는 않는다.; 모든 arm이 동일 checkpoint와 4개 feature-position view를 사용했다. 별도의 대조군별 재학습 예산 차이는 없다.
- 재검토 조건: 관측된 feature evidence와 graph의 joint missing-feature law를 검증 가능하게 연결하고, point fill 및 같은 예산의 covariance 제거·구조 random-feature 대조를 안정적으로 넘는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 PFN의 zero/PCFI/mean/native-NaN·independent sampling·context prior. 후속 positive 주장은 별도의 PCFI/FISF-trained 대조, 비조건부 구조 random features, GCNmf/GSPN/PEMix와의 차별화 필요.
- 최소 재검증: Source·data·예산을 먼저 고정한 새 graph에서 feature conditional의 품질과 predictive marginalization의 증분을 분리한다. Frozen replacement와 distributional-input 재학습을 같은 성과로 세지 않는다.
- 닫지 않은 것: Context labels에 조건화한 feature law, joint predictive training, cross-feature dependency, 정확한 SCM conditional, 실제 graph·unseen schema는 평가하지 않았다. GP likelihood는 공식 SCM posterior가 아니다.
- 원문: [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md); [findings/sources/feature-marginal-2026-11-02/paired.json](../findings/sources/feature-marginal-2026-11-02/paired.json)

- [PEMix](https://arxiv.org/abs/2610.18113): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use, observation-mask, uncertainty; keywords=feature distribution
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [GSPN](https://arxiv.org/abs/2305.10544): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [MCSSL](https://arxiv.org/abs/2605.11846): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 정보 증가에 따른 예측 갱신을 허용하면서 평균의 일관성을 제약한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## FMEM-F01 — Label context와 관측 feature memory를 분리하면 기존 context-only 및 all-node feature attention보다 accuracy가 높다.

- 판정: 현재 feature-group routing의 사전 accuracy gate 미통과. NLL은 기존 context 및 PCFI보다 악화.
- 원인·병목 [원인 미상]: 관측 전용 memory에서 추가 예측 가치를 회수하지 못했다. Frozen sample-attention projection, native feature-group 표현, sparse 입력 정보량,512-update prior fit 가운데 고유 원인을 분리하지 않았다. Context-only 제한이 이전 실패의 원인이라는 가설도 확인되지 않았다.
- 반증·배제한 설명: 초기 가중치·4,012,259 trainable parameters·logical schedule·성공 update 수는 네 arm에서 같았다.; Context arm과 기존 public Mask Plain의 초기 출력이 정확히 일치했고 empty-memory/padding 및 finite gradient를 검사했다.
- 재검토 조건: Label이 없는 노드의 실제 관측값을 읽는 연산이 strong all-node memory와 PCFI/FISF-trained PFN보다 같은 예산에서 유용하다는 근거가 생길 때.
- 먼저 이겨야 할 기준선: 동일 초기화·성공 update·prior의 C-only, all-node memory, PCFI/FISF-trained PFN, LP, class prior. Observed-group과 실제 sparse cell 연산을 구분.
- 최소 재검증: 새 graph bank에서 accuracy primary를 먼저 고정하고 no-observed/shuffled-observed와 전체 비용을 함께 비교한다. 단순 KV mask 변경이나 추가 학습만의 효과로 재개하지 않는다.
- 닫지 않은 것: 완전한 sparse cell-token 모델, attention projection까지 새로 학습하는 구조, 다른 prior·규모·실제 graph는 평가하지 않았다. MCP-P02의 +3.43%p는 이 실험으로 재현하거나 기각한 것이 아니다.
- 원문: [findings/sources/feature-memory-2026-11-02/REPORT_KO.md](../findings/sources/feature-memory-2026-11-02/REPORT_KO.md); [findings/sources/feature-memory-2026-11-02/results/paired.json](../findings/sources/feature-memory-2026-11-02/results/paired.json); [findings/sources/feature-memory-2026-11-02/results/interventions.json](../findings/sources/feature-memory-2026-11-02/results/interventions.json)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask, tokenization; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
  - 기록된 강점: Fully missing node에서도 PE-conditioned mixture responsibility가 남아 노드별 대치를 구분할 수 있다.
- [Ephris](https://arxiv.org/abs/2609.37057): 복수 태그 일치 (약한 단서); tags=graph-structure-use, tokenization; keywords=-
  - 기록된 강점: Section 3.1과 Appendix B의 feature별 Gather는 context와 query 노드를 모두 사용한다. Context-label embedding도 압축 전 scalar token에 제공하므로, feature memory를 label-context 노드에만 한정하지 않는 직접적인 선행 대조다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=observation-mask, tokenization; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
  - 기록된 강점: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.

## G6T-F01 — 기존 생성 prior의 exact Grid6 predictive를 가르치면 현재 공개 Mask GraphPFN이 teacher 추론을 더 잘 압축한다.

- 판정: 현재 공개 Mask 구성에 exact teacher 감독만 교체하는 갈래의 메인 방법 확대는 보류한다. Factor secondary의 별도 성과는 보존한다.
- 원인·병목 [원인 미상]: 조밀한 exact 감독만으로 현재 학생의 predictive gap을 줄이는 확실한 이득을 확인하지 못했다. Input 경로·표현력·최적화·frozen backbone·학습량의 기여를 분리하지 않아 고유 원인은 미상이다.
- 반증·배제한 설명: 모든 새 학생은 원래 CE와 initial/frozen weights·task schedule·optimizer·학습률·trainable parameter 수·2048 exposures/256 successful updates가 일치한다. 기존 학습 checkpoint를 추가 학습한 비교가 아니다.; Teacher는 graph/관측feature/mask/context/family만 사용한다. 실제 query/helper label이나 실제 density/signal/role/feature orientation을 입력받지 않는다. Exact teacher의 normalization·class flip·hidden-value poison과 query-only KL/CE gradient 검사를 통과했다.; Source/bank/teacher/reference hashes, checkpoint 유한성, 저장예측 전체 재채점과45개 예측 재실행을 확인했다. CI에0을 포함한다는 사실을 효과0의 증명으로 해석하지 않는다.
- 재검토 조건: 학생이 teacher의 관계·관측 정보 추론을 표현하고 회수할 수 있도록 바꾸는 구체적 학습 또는 추론 연산이, 강한 기준선 대비 효율적 예측을 제공할 근거가 생길 때.
- 먼저 이겨야 할 기준선: 동일 architecture/initialization/tasks/updates의 sampled CE, 실제 PCFI 입력으로 학습한 PFN, Factor CE/distill, fixed-mode factor BP, exact Grid6/Grid6-BP. Teacher 생성과 학생 학습 비용을 포함해 보고한다.
- 최소 재검증: 실패 bank에서 temperature·새 head·loss 혼합·학습량을 조정해 구제하지 않는다. 무엇을 바꾸는지 먼저 고정하고 새 graph/schema에서 primary KL·accuracy·NLL/Brier와 비용을 분리한다. Student architecture 변경과 teacher 변경을 동시에 하지 않는다.
- 닫지 않은 것: G6T-P01/P02의 Factor 효과와 이전 disjoint-factor 정확도 개선은 보존한다. Distillation 전체, fully trained 다른 구조, 다른 feature law·real graph·unseen schema·99.5% missing·큰 graph의 teacher 학습은 기각하지 않는다. 기존 feature-free JR distillation과 다른 이번 범위를 구분한다.
- 원문: [findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md](../findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md); [findings/sources/grid6-teacher-2026-11-02/protocol.json](../findings/sources/grid6-teacher-2026-11-02/protocol.json); [findings/sources/grid6-teacher-2026-11-02/results/paired.json](../findings/sources/grid6-teacher-2026-11-02/results/paired.json); [findings/sources/grid6-teacher-2026-11-02/results/decision.json](../findings/sources/grid6-teacher-2026-11-02/results/decision.json); [findings/sources/grid6-teacher-2026-11-02/results/benchmark.json](../findings/sources/grid6-teacher-2026-11-02/results/benchmark.json); [findings/sources/grid6-teacher-2026-11-02/results/final_integrity.json](../findings/sources/grid6-teacher-2026-11-02/results/final_integrity.json)

- [GSPN](https://arxiv.org/abs/2305.10544): 복수 태그 일치 (약한 단서); tags=differentiable-reasoning, observation-mask; keywords=-
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.

## HET-F01 — Adapted HetGFD가 cross-type path 입력보다 큰 증분을 준다.

- 판정: 새 메소드 headroom 부족
- 원인·병목 [진단이 지지]: 관계 경로와 남은 native source가 이미 강한 신호를 전달했다. 격차가 작다는 관찰은 확인했지만 redundancy만이 원인이라는 인과 분해는 하지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 타입별 feature 의미와 관측 pattern을 보존하며 기존 path redundancy를 넘어서는 장치가 나올 때.
- 먼저 이겨야 할 기준선: Native cross-type paths + frozen GraphPFN, 강한 attribute-free metapath LP
- 최소 재검증: 파생 helper feature 누수를 제거하고, 실제 native block graph에서 missing/full 대비와 최강 구조 기준선을 함께 비교한다.
- 닫지 않은 것: 원래 HetGFD/HGT 재현이나 다른 schema에서의 결론이 아니다. 다중 block 소실 조건은 별도 여지가 있다.
- 원문: [findings/sources/graphpfn-2026-11-01/native_blocks.md](../findings/sources/graphpfn-2026-11-01/native_blocks.md)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.

## VIEW-F01 — Support metadata·다중 전파 view·PCFI/FISF 입력 결합이 범용 증분을 준다.

- 판정: 전체 조건 진입 기준 미통과
- 원인·병목 [원인 미상]: 출력에 보인 증분이 작고 조건별 부호가 달랐다. 표현 정보 중복, projection 손실, frozen backbone의 활용 한계 중 무엇이 주원인인지는 분리하지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 관측 provenance를 정보 손실 없이 표현·학습하고 equal-width 값 대조를 넘는 장치가 나올 때.
- 먼저 이겨야 할 기준선: PCFI/FISF parents, equal-width value controls, 같은 비용의 single predictor
- 최소 재검증: 같은 차원·추론 비용으로 새 실제 graph에서 parent 둘과 metadata 없는 값 대조를 함께 비교한다.
- 닫지 않은 것: Native cell-wise provenance encoder를 학습하는 방법과 입력 자체를 바꾸는 학습은 평가하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md); [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md)

- [GraphPFN](https://arxiv.org/abs/2509.21489): 구체 키워드+태그 일치 (미검증); tags=feature-heterogeneity, prior-mismatch; keywords=feature token
  - 기록된 강점: LimiX의 per-feature tokenization을 그대로 물려받아, 데이터셋마다 feature 개수가 달라도 차원 축소 없이 하나의 모델로 처리한다. 같은 모델이 node classification과 node regression을 모두 다룬다.
  - 기록된 강점: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.
- [GSPN](https://arxiv.org/abs/2305.10544): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missing
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missing
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
  - 기록된 강점: Fully missing node에서도 PE-conditioned mixture responsibility가 남아 노드별 대치를 구분할 수 있다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missing
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missing
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
  - 기록된 강점: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.

## IPCFI-F01 — 실제 PCFI 전파를 공개 PFN 내부에 배치하고 context/PFN 상태로 전파 가중치를 조절하면 외부 PCFI·내부 고정 PCFI·label-blind router보다 분류가 좋아진다.

- 판정: 현재100-step literal PCFI/25-step interleaving/513-parameter router/256-update 구현의 메인 방법 확대를 보류한다. 이 final bank에서 head/loss/temperature/전파 횟수/학습량을 바꿔 구제하지 않는다.
- 원인·병목 [원인 미상]: 사전 비교를 모두 만족하는 분류 이득을 확인하지 못했다. 전파가 이미 충분한지, context 요약·bounded router·frozen backbone/input interface·prior·최적화·학습량 중 무엇 때문인지는 분리하지 않아 고유 원인은 미상이다. 상수 신호를 row-stochastic 전파로 바꿀 수 없다는 수학적 제약을 전체 실패의 단일 원인으로 쓰지 않는다.
- 반증·배제한 설명: 이전 generic MLP refiner가 아니라 원래 PCFI의 거리 confidence/관측값 clamp/100회 전파/1회 correlation을 구현했고 독립 SciPy 기준과 비교했다.; 네 모델은 같은 공개 초기화·train tasks·seed별 schedule·256 successful CE updates·100 propagation steps·1 correlation·12PFN layers다. Context/label-blind의513개 router 파라미터와 초기 예측도 같다. 고정 대조는 이513개가 없으므로 모든 arm의 정확한 동일 파라미터 비교라고 주장하지 않는다.; Label-blind router는 labels가 섞인 PFN hidden까지 읽지 않는다. 실제 optimizer smoke, router/hidden gradient, 저장/복원, poison/clamp와 source/bank/prediction hash/replay를 확인했다.
- 재검토 조건: 현재 동일한 numeric propagation을 다시 배치하거나 bounded weight만 조절하는 것을 넘어, 새 그래프의 context가 전달해야 할 정보를 바꾸는 구체적인 연산이 강한 외부 PCFI/PFN보다 유리할 근거가 생길 때.
- 먼저 이겨야 할 기준선: 동일 actual PCFI/100steps/1correlation/12PFN layers의 외부 PCFI, 내부 고정, context와 strict label-blind router, 같은 train tasks/CE/initialization/updates. Context-frequency/uniform prior와 NLL/Brier/CPU→CPU end-to-end 비용도 보고한다.
- 최소 재검증: 본 final bank는 tuning하지 않는다. 변경할 정보·연산과 사전 primary를 먼저 정하고 독립 graph/schema에서 node-level 대신 base-graph paired 분석으로 한 번 검증한다. Labels-only 효과와 PFN-feedback 전체 효과를 구분한다.
- 닫지 않은 것: JIMP-F01의 generic refiner와 별개 실행이다. 외부 PCFI의 실제 그래프 성과, 모든 내부 PCFI 방식, 다른 전파/표현/학습 원리, native NaN interface, 실제 heterograph·unseen schema 일반을 기각하지 않는다.
- 원문: [findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md](../findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md); [findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md](../findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md); [findings/sources/internal-pcfi-2026-11-02/protocol.json](../findings/sources/internal-pcfi-2026-11-02/protocol.json); [findings/sources/internal-pcfi-2026-11-02/results/statistics.json](../findings/sources/internal-pcfi-2026-11-02/results/statistics.json); [findings/sources/internal-pcfi-2026-11-02/results/cost.json](../findings/sources/internal-pcfi-2026-11-02/results/cost.json); [findings/sources/internal-pcfi-2026-11-02/results/audit.json](../findings/sources/internal-pcfi-2026-11-02/results/audit.json); [findings/sources/internal-pcfi-2026-11-02/results/preflight.json](../findings/sources/internal-pcfi-2026-11-02/results/preflight.json); [findings/sources/internal-pcfi-2026-11-02/results/development_actual_pcfi_parity.json](../findings/sources/internal-pcfi-2026-11-02/results/development_actual_pcfi_parity.json)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask, tokenization; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
  - 기록된 강점: Fully missing node에서도 PE-conditioned mixture responsibility가 남아 노드별 대치를 구분할 수 있다.
- [Ephris](https://arxiv.org/abs/2609.37057): 복수 태그 일치 (약한 단서); tags=graph-structure-use, tokenization; keywords=-
  - 기록된 강점: Section 3.1과 Appendix B의 feature별 Gather는 context와 query 노드를 모두 사용한다. Context-label embedding도 압축 전 scalar token에 제공하므로, feature memory를 label-context 노드에만 한정하지 않는 직접적인 선행 대조다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=observation-mask, tokenization; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
  - 기록된 강점: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.

## JIMP-F01 — PFN 내부의 반복 대치가 동일한 대치기를 PFN 앞에서 실행하는 것과 강한 동일 예산 Plain을 넘는다.

- 판정: 현재 구현 확대 보류·내부 대치 증분 미입증
- 원인·병목 [원인 미상]: 실제 공개 PFN과 대치 상태를 연결해 gradient를 학습했지만, 위치를 바꾼 독립 예측 이득은 확인하지 못했다. 정보 중복·현재 numeric 함수족·prior·학습 예산 중 원인은 분리하지 못했다.
- 반증·배제한 설명: 대치 모듈이 label readout에 연결되지 않은 dead module이라는 설명은 nonzero gradient와 동작 대조로 지지되지 않는다. 이것이 유용한 정보의 학습을 뜻하지는 않는다.; Frontend와 Interleaved의 초기 가중치·학습 과제 순서·파라미터 수·성공 update 수 차이로 설명할 수 없다. 실제 학습 시간과 수치 재시도 비용은 같다고 주장하지 않는다.
- 재검토 조건: 대치 상태를 PFN 추론 중 갱신할 때 사전 대치에 없던 예측 정보를 보존·전달하고, 동일 예산 Plain/PCFI 대조를 넘는 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 synthetic missing prior로 추가 학습한 PCFI Plain·Mask Plain·Frontend; native NaN GraphPFN의 같은 예산 학습도 향후 필요
- 최소 재검증: 공통 전처리·초기화·학습 과제·성공 update 수를 맞춘 새 bank에서 반복 위치 효과를 비교한다. 같은 checkpoint의 피드백 제거와 재학습 대조를 구분하고 NLL/Brier/accuracy를 따로 보고한다.
- 닫지 않은 것: 다른 대치 함수족, pretrained feature decoder, latent belief representation, native NaN-input 대조와 실제 그래프 전이는 별개다. 이 실험은 이전 10K quadratic prototype과 다른 공개 GraphPFN 구현이다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/paired.json](../findings/sources/joint-imputation-2026-11-01/paired.json)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask, tokenization; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
  - 기록된 강점: Fully missing node에서도 PE-conditioned mixture responsibility가 남아 노드별 대치를 구분할 수 있다.
- [Ephris](https://arxiv.org/abs/2609.37057): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use, tokenization; keywords=feature token
  - 기록된 강점: Section 3.1과 Appendix B의 feature별 Gather는 context와 query 노드를 모두 사용한다. Context-label embedding도 압축 전 scalar token에 제공하므로, feature memory를 label-context 노드에만 한정하지 않는 직접적인 선행 대조다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=observation-mask, tokenization; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
  - 기록된 강점: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.
- [GraphPFN](https://arxiv.org/abs/2509.21489): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=feature token
  - 기록된 강점: Hand-crafted graph feature(Laplacian PE, PageRank, 이웃 평균 등)에 기대지 않고, 모델 안에서 adjacency-masked attention으로 message passing을 직접 학습한다. 그 결과 ICL만으로도 G2T-FM·TAG와 튜닝된 GNN보다 평균 rank가 높고, 무거운 전처리도 필요 없다.

## JIMP-F02 — PFN의 query 분류 목적에 맞춘 사전 대치기로 PCFI를 교체하면 주 결측 조건에서 독립 개선을 얻는다.

- 판정: 현재 task-trained frontend의 주 조건 우위 미입증
- 원인·병목 [원인 미상]: Query CE로 학습된 네 단계 대치가 고정 PCFI나 일반 mask/token adapter보다 유용하다는 증거가 부족했다. 정확한 원인은 미상이며 복원 목적이 아닌 분류 목적을 썼다는 사실만으로 개선되지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: PFN에 필요한 정보와 단순 복원값을 구분하는 대치·입력 연산이 강한 동일 학습 대조보다 안정적인 예측 이득을 보일 때.
- 먼저 이겨야 할 기준선: PCFI Plain, Mask Plain, native NaN input, 동일 예산·초기화·query CE
- 최소 재검증: 대치 연산의 개선과 backbone 추가 학습을 분리하고, 먼저 새 그래프 bank에서 prior-only 확률 대조까지 함께 비교한다.
- 닫지 않은 것: 모든 task-aware imputation과 PFN용 전처리가 실패했다는 뜻은 아니다. 피처 복원 MSE를 최소화한 모델과 비교한 결과도 아니다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/frontend_comparisons.json](../findings/sources/joint-imputation-2026-11-01/frontend_comparisons.json)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=channel-dependency, observation-mask, tokenization; keywords=imputation
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
  - 기록된 강점: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=observation-mask, tokenization; keywords=-
  - 기록된 강점: Fully missing node에서도 PE-conditioned mixture responsibility가 남아 노드별 대치를 구분할 수 있다.
- [MOMENT](https://arxiv.org/abs/2402.03885): 구체 키워드+태그 일치 (미검증); tags=tokenization; keywords=imputation
  - 기록된 강점: patching, RevIN, 좌측 zero-padding, channel 단위 처리라는 단순한 조합으로 길이·스케일·채널 수가 다른 데이터셋을 한 모델에 넣는다. 데이터셋마다 구조를 바꿀 필요가 없다.
- [TSFM-Interventions](https://arxiv.org/abs/2409.12915): 구체 키워드+태그 일치 (미검증); tags=tokenization; keywords=imputation
  - 기록된 강점: concept이 어느 layer와 어느 patch에서 구분되는지 위치를 찾았다. 일정한 시계열과 sinusoid의 구분은 시계열 전체를 봐야 생기고, 증가·감소 trend는 처음과 끝 patch에서 구분되었다. amplitude 차이는 어디서도 뚜렷하게 구분되지 않았는데, 저자들은 입력 정규화가 원인일 수 있다고 해석했다.

## LLBP-F01 — Role-conditional 감독을 더하면 최종 예측이 개선된다.

- 판정: 추가 이득 미확인
- 원인·병목 [진단이 지지]: Component conditional fit 개선이 최종 predictive 개선으로 이어지지 않았다. Positive diffusion의 표현력 제한도 뒤에 확인했지만 전체 미개선의 유일 원인으로 분해하지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 표현 가능한 decoder에서 conditional 감독의 독립 predictive 이득을 보이는 방법이 나올 때.
- 먼저 이겨야 할 기준선: Predictive-only JR, Plain predictive distillation
- 최소 재검증: Decoder와 teacher를 고정하고 auxiliary의 독립 증분을 같은 task·예산으로 비교한다.
- 닫지 않은 것: conditional distillation 일반이 불가능하다는 결론은 아니다. 추가 감독은 의도한 conditional 오차를 실제로 줄였다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## LLBP-F02 — Posterior까지 정확하게 학습하면 최종 예측도 좋아진다.

- 판정: 유의 악화
- 원인·병목 [원인 미상]: Posterior fit 감소와 predictive KL 악화가 함께 관측됐다. 목적의 불일치나 decoder 오차 보상 중 원인을 하나로 확정하지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Posterior 정확성과 deployment prediction을 동시에 보존하는 감독·추론 원리가 나올 때.
- 먼저 이겨야 할 기준선: Predictive-only, conditional-only, exact posterior substitution
- 최소 재검증: 같은 decoder에서 posterior KL과 deployment predictive KL을 따로 평가한다.
- 닫지 않은 것: posterior 감독 일반의 기각이 아니라, component 오차 감소가 최종 품질을 보장하지 않는다는 결과다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## LLBP-F03 — Masked-edge likelihood auxiliary가 learned BP의 핵심 학습 기여다.

- 판정: 유의 악화 (새 loss 기여 주장 중단)
- 원인·병목 [원인 미상]: 추가 likelihood auxiliary는 predictive-only보다 악화했다. Local likelihood fit이 label risk와 정렬되지 않을 가능성은 가설이며 정확한 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Local factor 감독이 query risk에 도움이 되는 조건을 증명·검증한 방법이 나올 때.
- 먼저 이겨야 할 기준선: Learned BP predictive-only, fitted constant
- 최소 재검증: 같은 potential·BP·role update에서 loss만 바꿔 independent gain을 본다.
- 닫지 않은 것: auxiliary 없이도 potential을 학습하고 BP로 좋은 예측을 만들 수 있었다. 실패한 것은 학습 가능성이 아니라 추가 loss의 우수성 주장이다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## LLBP-F04 — Learned potential + 기존 frozen role q만으로 넓은 조건에서 adaptive OOD 추론이 된다.

- 판정: 범용성 미확보
- 원인·병목 [진단이 지지]: Potential과 frozen role posterior의 OOD 불일치 및 conditional 근사 오차가 후보였다. 뒤의 coherent-update 대조가 일부 병목을 지지하지만 모든 shift의 원인을 확정하지 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 새 graph 통계로 일관된 role/parameter evidence를 추론하고 단순 EB/grid를 넘는 방법이 나올 때.
- 먼저 이겨야 할 기준선: Constant+update, density estimator, Grid6/81
- 최소 재검증: 같은 conditional branches에서 posterior·potential 교차 대조와 새 density/signal bank를 평가한다.
- 닫지 않은 것: 원인은 확정하지 못했다: potential point estimate, frozen posterior의 분포 이동, conditional BP 근사, 이들의 상호작용.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## LLBP-F05 — Parameter 후보를 혼합하되 기존 role q를 유지해도 충분하다.

- 판정: 기대 기각 + 해석 기각
- 원인·병목 [직접 확인]: 같은 BP branches에서 coherent joint weights가 frozen role q보다 좋았다. 큰 개선을 같은 주변분포의 dependence 효과 전체로 돌릴 수는 없다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 일관된 evidence 갱신을 미지 family에서도 정확하고 저렴하게 구현하는 장치가 나올 때.
- 먼저 이겨야 할 기준선: Joint grid, same-marginal product, frozen-grid
- 최소 재검증: Conditional branches와 marginals를 맞춰 role update와 dependence 증분을 분리한다.
- 닫지 않은 것: frozen hybrid도 정규화된 분포이며 일부 조건부 dependence를 보존한다. 단순 독립 head의 곱이라고 부르지 않는다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## LLBP-F06 — EAJ(Evidence-Amortized Joint PFN)가 exact-grid를 비슷한 품질·훨씬 낮은 전체 비용으로 대체한다.

- 판정: 경쟁력 미달 (대체·확장 중단)
- 원인·병목 [진단이 지지]: 학습 evidence의 정확도와 공통 BP 비용이 병목이었다. Posterior-only 속도 증분이 end-to-end에 크게 전달되지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 강한 classical update의 품질을 유지하며 전체 branch/evidence 비용을 줄이는 연산이 나올 때.
- 먼저 이겨야 할 기준선: Constant+role-update, exact-product, Joint Grid6/81
- 최소 재검증: Posterior 추정만의 시간과 shared BP를 포함한 end-to-end 품질·비용을 분리한다.
- 닫지 않은 것: joint 표현 자체의 장점(약한 head 대비)은 있었다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [MaM](https://proceedings.mlr.press/v235/liu24az.html): 복수 태그 일치 (약한 단서); tags=amortized-inference, inference-cost; keywords=-
  - 기록된 강점: 관측 subset의 marginal probability를 직접 근사해 반복 conditional 평가의 비용을 줄인다.

## LLBP-F07 — EAJ를 직접 predictive 학습하면 실용성과 posterior 해석을 함께 회복한다.

- 판정: 부분 성공·메소드 구제 실패 (predictive 개선을 posterior 복원 개선으로 보는 해석 기각)
- 원인·병목 [진단이 지지]: 실제 decoder에 맞춘 predictive weights와 정확한 Bayesian posterior 복원은 다른 목적이었다. Direct predictive 개선은 있었지만 classical 기준선을 넘지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Decoder 오차를 다루면서 posterior 해석과 predictive quality를 함께 보장하는 방법이 나올 때.
- 먼저 이겨야 할 기준선: Direct predictive EAJ, Constant+update, exact-grid
- 최소 재검증: 두 KL과 실제 NLL/Brier를 함께 보고 calibrated posterior라는 주장을 별도 검증한다.
- 닫지 않은 것: 직접 predictive 학습이 작동하지 않았다는 결론은 틀리다. B의 출력은 decoder에 맞춘 aggregation weights로 남긴다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, uncertainty; keywords=posterior, predictive
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## LLBP-F08 — 같은 potential에서 role weights를 다시 계산하는 자기-update가 OOD 문제를 전반적으로 해결한다.

- 판정: 보편적 OOD 해결 주장 기각
- 원인·병목 [진단이 지지]: Low density에서는 exact role update도 실패해 Bethe partition 근사만으로 설명할 수 없었다. Potential·conditional·parameter uncertainty 상호작용은 미해결이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Sparse/OOD potential uncertainty와 conditional inference를 함께 해결하는 연산이 나올 때.
- 먼저 이겨야 할 기준선: Constant+update, R1, Grid6/81
- 최소 재검증: Low density를 포함한 새 bank에서 exact/Bethe·conditional 교차를 고정해 본다.
- 닫지 않은 것: 자기-update는 특정 조건에서 유효한 추론 개선이다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [CSBMInference](https://arxiv.org/abs/2306.07948): 구체 키워드+태그 일치 (미검증); tags=uncertainty; keywords=sparse
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [Moirai-MoE](https://arxiv.org/abs/2410.10469): 구체 키워드+태그 일치 (미검증); tags=uncertainty; keywords=sparse
  - 기록된 강점: 학습에 쓰지 않은 10개 데이터셋 zero-shot 평가에서 Base가 CRPS와 MASE 평균 모두 최고였고, 평가 데이터 일부가 사전학습에 들어간 TimesFM, Chronos, Time-MoE보다도 나았다. Small도 모든 크기의 Moirai보다 CRPS가 3~14%, MASE가 8~16% 좋았다. point 예측 지표 외에 PICP, QICE 같은 구간 coverage 지표까지 함께 보고했다.

## LLBP-F09 — 자기-update 경로까지 학습하는 T1이 R1보다 우수하다.

- 판정: 추가 이득 미확인
- 원인·병목 [원인 미상]: Self-update를 미분해 학습하는 경로는 작동했지만 R1 대비 이득이 없었다. Optimization·model bias 중 고유 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 동일 계산 budget에서 inference-aware 학습의 독립 이득을 보이는 원리가 나올 때.
- 먼저 이겨야 할 기준선: Frozen R1, R0, Constant+update, Grid6
- 최소 재검증: 추론 시 head 제거 효과와 새 학습 효과를 분리하고 initial state·schedule·예산을 맞춘다.
- 닫지 않은 것: head 없는 학습 경로와 gradient 흐름은 확인됐다. 더 큰 예산에서 절대 개선되지 않는다는 증명은 아니다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [LoGIC](https://arxiv.org/abs/2609.05955): 구체 키워드+태그 일치 (미검증); tags=adaptation; keywords=training
  - 기록된 강점: Backbone을 재학습하지 않는 추론 시점 인터페이스라서, pure-tabular backbone(G2T featurization + TabPFNv2·LimiX)과 adapter backbone(GraphPFN)에 같은 방식으로 붙는다. 정식화 자체가 full-context tabular 추론, per-query retrieval, full-graph 추론을 특수한 경우로 포함하므로 기존 protocol과의 관계도 분명하다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=amortized-inference; keywords=training
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## LLBP-F10 — Low-density 실패는 partition 근사 또는 role weights만의 문제다.

- 판정: 단일 병목 설명 기각
- 원인·병목 [진단이 지지]: 교차 대조에서 learned conditional 손해와 비가법 interaction을 확인했다. Role weights만 고치면 된다는 단일 원인 설명은 지지되지 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Potential·partition·conditional 오차를 일관되게 제어하는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: Constant/learned weights×conditional 교차, exact update
- 최소 재검증: 네 교차 조합과 exact-role/BP conditional을 함께 비교해 단일 병목 주장을 검증한다.
- 닫지 않은 것: 기각하지 못한 가능성: potential 추정 편향, parameter uncertainty 부족, conditional BP 오차, 과확신, 이들의 상호작용.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [CSBMInference](https://arxiv.org/abs/2306.07948): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=BP
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [GraphPFN](https://arxiv.org/abs/2509.21489): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=BP
  - 기록된 강점: Hand-crafted graph feature(Laplacian PE, PageRank, 이웃 평균 등)에 기대지 않고, 모델 안에서 adjacency-masked attention으로 message passing을 직접 학습한다. 그 결과 ICL만으로도 G2T-FM·TAG와 튜닝된 GNN보다 평균 rank가 높고, 무거운 전처리도 필요 없다.
- [LoGIC](https://arxiv.org/abs/2609.05955): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=BP
  - 기록된 강점: Full context가 항상 최선은 아니라는 점을 보였다. TabPFNv2로 twitch-views에서 LoGIC은 73.25 R²로 published full-context 70.21보다 3점 높았고, LimiX에서도 74.15 대 71.31이었다. 어떤 labeled 노드를 넣느냐가 그 자체로 성능을 움직이는 설계 변수라는 뜻이다.
  - 기록된 강점: Label 예산과 구조 예산을 분리한 halo 설계가 adapter backbone에서 효과를 보였다. tolokers-2에서 PPR context(k=4096)만 넣으면 54.20 AP였지만, label을 가린 1-hop halo를 더하자 61.09 AP로 올라 full-graph 기준치(61.29)에 거의 도달했다.
- [NodePFN](https://arxiv.org/abs/2604.19028): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=BP
  - 기록된 강점: cSBM의 homophily를 넓게 샘플링해 heterophily 쪽도 prior에 넣었다. 그 결과 heterophilous 그룹 평균에서 GCN보다 6%p 이상 높았다.
  - 기록된 강점: attention branch와 GCN branch를 층마다 병렬로 두는 단순한 dual-branch 구조로 그래프 구조를 PFN에 넣었다. MPNN branch를 빼면 Cora와 Wisconsin에서 accuracy가 크게 떨어져, 구조 branch의 기여가 ablation으로 확인된다.
- [Wander](https://arxiv.org/abs/2610.06694): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=BP
  - 기록된 강점: 구조적 수용 범위를 추론 때 조절할 수 있다. Message passing 기반 GFM은 이 범위가 network 깊이에 고정된다. 서로 떨어진 25×25 grid 5개에 grid마다 label을 2개만 준 GRIDS에서, walk 길이를 4에서 64로 늘리자 정확도가 56.6%에서 96.6%로 올랐다. Labeled 노드를 지나는 walk만 남기면 100%였다. 같은 과제에서 GraphPFN은 23.3%, NodePFN은 26.6%였다. Feature 없는 실제 그래프 4개에서도 Wander는 76.5%로 GraphPFN(60.1%)을 크게 앞섰다.

## LLBP-F11 — R1은 강한 model-based 기준선보다 좋은 메인 메소드다.

- 판정: 경쟁력 미달
- 원인·병목 [직접 확인]: 현재 known-family에서 작은 training-support joint grid가 품질·비용 기준을 더 잘 충족했다. Learned 모델 전체가 불필요하다는 보편 원인은 아니다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 작은 명시적 grid가 처리하지 못하는 실제 family/schema에서 우위를 보이는 범용 추론법이 나올 때.
- 먼저 이겨야 할 기준선: 사전 prior 고정 Grid6/81, Constant+update, R1
- 최소 재검증: 알려진 family·prior 접근 조건과 end-to-end 비용을 맞춘 새 bank에서 비교한다.
- 닫지 않은 것: R1−R0, R1−Constant의 유의한 개선은 그대로 남는다. Grid6과 Grid81은 격자 수뿐 아니라 prior 자체가 다르므로 '작은 grid가 일반적으로 낫다'로 해석하지 않는다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## LLBP-F12 — R1은 품질이 낮아도 큰 속도 이점으로 정당화된다.

- 판정: 큰 속도 우위 미확보
- 원인·병목 [직접 확인]: 병렬 grid와의 실제 latency가 비슷했고 posterior/branch 수 감소가 큰 전체 속도 이점으로 이어지지 않았다. 작업 메모리 절감은 별도로 보존한다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 실제 처리 규모나 memory constraint에서 quality를 유지하는 비용 우위를 입증하는 연산이 나올 때.
- 먼저 이겨야 할 기준선: 병렬 Grid6/81, R1
- 최소 재검증: 동일 hardware·batch·quality 범위에서 latency 반복 변동과 total/working memory 및 최대 처리 graph를 측정한다.
- 닫지 않은 것: 추가 작업 메모리 절감은 사실이다. 다만 더 큰 graph 처리나 배포상 이익은 검증하지 않았고, 이를 사후 메인 주장으로 바꾸지 않기로 했다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [MaM](https://proceedings.mlr.press/v235/liu24az.html): 복수 태그 일치 (약한 단서); tags=amortized-inference, inference-cost; keywords=-
  - 기록된 강점: 관측 subset의 marginal probability를 직접 근사해 반복 conditional 평가의 비용을 줄인다.
- [Time-MoE](https://arxiv.org/abs/2409.16040): 구체 키워드+태그 일치 (미검증); tags=inference-cost; keywords=shared
  - 기록된 강점: sparse MoE로 시계열 foundation model을 처음 2.4B 파라미터까지 키웠다. 같은 활성 파라미터의 dense 모델과 비교하면 학습 비용은 평균 78%, 추론 비용은 39% 줄었다. ultra 모델도 8GB 미만 VRAM의 소비자용 GPU에서 돌아간다.
  - 기록된 강점: 여러 해상도의 출력 head를 함께 학습해서 긴 horizon을 적은 decoding step으로 만든다. ablation에서 horizon 1짜리 head 하나로만 autoregressive하게 예측하면 MSE가 1.382까지 무너졌지만, 네 head를 쓰면 0.262였고 추론도 가장 빨랐다.

## LPC-F01 — LP internal conditioning이 output offset과 Plain을 일관되게 넘는다.

- 판정: LP residual tuning·대규모 확장 중단
- 원인·병목 [원인 미상]: 보인 이득 대부분과 output offset 효과를 구분해야 했고, schema adaptation 특유의 이득이라는 증거가 약했다. 조기 relation aggregation이 원인이라는 설명은 가설이지 확정 원인이 아니다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 압축된 LP 확률보다 풍부한 context 정보를 활용하며 output offset만의 효과를 넘어서는 연산이 나올 때.
- 먼저 이겨야 할 기준선: Plain, offset-only, 같은 파라미터·초기화·학습 budget
- 최소 재검증: Relation 수·feature 폭·missing을 분리한 factorial과 새로운 semantic schema에서 내부 conditioning의 독립 증분을 확인한다.
- 닫지 않은 것: Relation 정보를 합치기 전 보존하는 구조와 별도 semantic schema 전이는 별개다.
- 원문: [findings/sources/graphpfn-2026-11-01/lpc.md](../findings/sources/graphpfn-2026-11-01/lpc.md); [findings/sources/graphpfn-2026-11-01/lpc_factorial.md](../findings/sources/graphpfn-2026-11-01/lpc_factorial.md)

- [NodePFN](https://arxiv.org/abs/2604.19028): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=residual
  - 기록된 강점: cSBM의 homophily를 넓게 샘플링해 heterophily 쪽도 prior에 넣었다. 그 결과 heterophilous 그룹 평균에서 GCN보다 6%p 이상 높았다.
  - 기록된 강점: attention branch와 GCN branch를 층마다 병렬로 두는 단순한 dual-branch 구조로 그래프 구조를 PFN에 넣었다. MPNN branch를 빼면 Cora와 Wisconsin에서 accuracy가 크게 떨어져, 구조 branch의 기여가 ablation으로 확인된다.

## MCP-F01 — 동일 주변분포의 feature-style/label-rule crossing이 강한 입력 기준선을 넘어 공식-prior 결측 과제로 전이된다.

- 판정: 두 gate 동시 통과 실패; 현재 controlled prior를 메인 방법으로 확대하지 않음
- 원인·병목 [원인 미상]: 정해진 초기화·256 성공 updates·함수족에서 요구한 predictive 우위를 확보하지 못했다. Prior의 한계, 학습량, preprocessing, context 식별 난도 중 원인을 분리한 실험은 아니다.
- 반증·배제한 설명: 서로 다른 trainable parameter 수나 seed별 update schedule 차이로 대조를 설명할 수 없도록 검사했다.
- 재검토 조건: 어떤 관측 정보와 예측 rule의 조합이 전이에 필요한지를 구체적으로 바꾸는 generator/inference 원리가 나오고 동일 비용의 강한 대조를 넘을 근거가 생길 때.
- 먼저 이겨야 할 기준선: 동일 구조·초기화·update의 Official/Coupled Mask, 입력으로 학습한 Coupled PCFI, context/uniform prior, LP, 4-view random-fill; 실제 graph에서는 PCFI/FISF-trained PFN·Ephris도 필요.
- 최소 재검증: 새 독립 bank에서 full/node/cell 조건과 actual-feature 제거·셔플을 고정해 검증한다. 같은 bank의 loss/epochs 튜닝으로 구제하지 않는다. Controlled/official family 변경과 coupling만의 효과를 분리한다.
- 닫지 않은 것: 다른 그래프 prior·관측 정보가 충분한 task family·실제 graph·동적 learned generator 일반을 기각하지 않는다. 기존 공식 prior의 induced coupling 자체가 실패 원인이라고 입증하지 않았다.
- 원문: [findings/sources/mechanism-crossed-2026-11-02/REPORT.md](../findings/sources/mechanism-crossed-2026-11-02/REPORT.md); [findings/sources/mechanism-crossed-2026-11-02/paired.json](../findings/sources/mechanism-crossed-2026-11-02/paired.json); [findings/sources/mechanism-crossed-2026-11-02/protocol.json](../findings/sources/mechanism-crossed-2026-11-02/protocol.json)

- [Chronos-2](https://arxiv.org/abs/2510.15821): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: 실제 다변량 데이터 없이 synthetic multivariatizer만으로 다변량·covariate 능력을 얻었다. synthetic 데이터만으로 학습한 변형도 GIFT-Eval과 Chronos Benchmark II에서 본 모델에 거의 근접해, synthetic prior 설계의 효과를 보여준다.
- [GraphPFN](https://arxiv.org/abs/2509.21489): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.
- [NodePFN](https://arxiv.org/abs/2604.19028): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: 합성 그래프 수를 늘릴수록 성능이 오르는 prior-scale 분석과, ER·cSBM·BA 구성을 바꾼 prior ablation을 함께 제시해 어떤 prior 성분이 도움이 되는지 일부 보여준다.
- [PFNSeparability](https://arxiv.org/abs/2610.14989): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Feature law와 conditional label law의 의존성을 prior 설계의 명시적 선택으로 다룬다.
- [TimesFM](https://arxiv.org/abs/2310.10688): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: synthetic 데이터를 빼면 실제 데이터에 드문 granularity(quarterly, yearly, 10분 등)와 15분 ETT에서 성능이 떨어진다는 ablation을 보여, 실제 corpus의 빈 곳을 synthetic prior로 메우는 효과를 실증했다.
- [Wander](https://arxiv.org/abs/2610.06694): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Synthetic prior가 NodePFN식과 GraphPFN식 생성기에 Watts–Strogatz, random geometric, graphon 기반 구조를 더해 섞는다. 그래서 GraphPFN prior가 덮지 못한다고 인정한 공간·기하 구조 그래프를 사전학습 분포에 명시적으로 포함한다.

## MISS-F01 — Missingness prior 추가가 공식 입력 경로와 미학습 MNAR에서 유리하다.

- 판정: 범용 우위 미확보·해당 라인 중단
- 원인·병목 [진단이 지지]: Raw-NaN 기준선이 공식 imputation보다 약했고, official fill만으로도 mask가 잘 드러났다(AUROC 0.97). Context-query mask 관계 불일치에서 mixed-prior 이득이 유지되지 않았다. 이것만으로 모든 원인을 확정하지는 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 관측 메커니즘의 context-query shift를 다루면서 official preprocessing 대비 추가 가치를 보이는 학습 원리가 나올 때.
- 먼저 이겨야 할 기준선: 공개 GraphPFN + official imputation, 동일 예산 B4
- 최소 재검증: 동일 입력 경로·초기화·예산의 새 bank에서 context-query mismatch와 미학습 mask를 paired 평가한다.
- 닫지 않은 것: Imputed-input에 맞춘 prior의 효과는 미평가다. 더 넓은 MNAR 식별 문제도 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md); [findings/sources/graphpfn-2026-11-01/missing_pilot.md](../findings/sources/graphpfn-2026-11-01/missing_pilot.md); [findings/sources/graphpfn-2026-11-01/missing_context.md](../findings/sources/graphpfn-2026-11-01/missing_context.md)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=imputation, missingness
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
  - 기록된 강점: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.
- [PEMix](https://arxiv.org/abs/2610.18113): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missingness
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
  - 기록된 강점: Fully missing node에서도 PE-conditioned mixture responsibility가 남아 노드별 대치를 구분할 수 있다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missingness
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## OBS-F01 — Mask-aware 학습 정책이 강한 경로 규칙을 넘는다.

- 판정: 주 성공 기준 미통과·확장 보류
- 원인·병목 [원인 미상]: Mask를 보는 것은 유용했지만 비슷한 효과를 고정 shortest-path closure도 얻었다. Joint subset 표현력·stochastic train/deterministic inference·학습 budget 중 정확한 원인은 분리하지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 동일 resident budget에서 mask-aware 공동 부분그래프 구성이 강한 경로 규칙보다 나은 utility를 보이는 장치가 나올 때.
- 먼저 이겨야 할 기준선: Evidence-paths, PPR, equal-capacity structure policy, random, context prior
- 최소 재검증: Fresh graph에서 사전 budget·mask gate를 고정하고 preprocessing/search를 포함한 비용과 label 예측을 함께 비교한다.
- 닫지 않은 것: 모든 learned retrieval을 기각하지 않는다. Budget32 cluster 차이−0.19284 CI[−0.39771,+0.00284]는 보조 개발 신호이며 승리가 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

- [LoGIC](https://arxiv.org/abs/2609.05955): 구체 키워드+태그 일치 (미검증); tags=context-scale, graph-structure-use; keywords=halo, retrieval
  - 기록된 강점: 예산을 정한 context만으로 full-context 정확도를 유지하면서 peak 메모리를 크게 줄인다. GraphPFN 기준 artnet-exp에서 full graph 추론은 51.86 AP에 19.59GB였는데, LoGIC은 52.06 AP에 8.24GB였다. TabPFNv2 기준 tolokers-2에서도 4.10GB가 2.20GB로 줄었고 AP는 오히려 조금 올랐다.
  - 기록된 강점: Full context가 항상 최선은 아니라는 점을 보였다. TabPFNv2로 twitch-views에서 LoGIC은 73.25 R²로 published full-context 70.21보다 3점 높았고, LimiX에서도 74.15 대 71.31이었다. 어떤 labeled 노드를 넣느냐가 그 자체로 성능을 움직이는 설계 변수라는 뜻이다.
  - 기록된 강점: Label 예산과 구조 예산을 분리한 halo 설계가 adapter backbone에서 효과를 보였다. tolokers-2에서 PPR context(k=4096)만 넣으면 54.20 AP였지만, label을 가린 1-hop halo를 더하자 61.09 AP로 올라 full-graph 기준치(61.29)에 거의 도달했다.
- [Ephris](https://arxiv.org/abs/2609.37057): 복수 태그 일치 (약한 단서); tags=context-scale, graph-structure-use; keywords=-
  - 기록된 강점: Section 3.1과 Appendix B의 feature별 Gather는 context와 query 노드를 모두 사용한다. Context-label embedding도 압축 전 scalar token에 제공하므로, feature memory를 label-context 노드에만 한정하지 않는 직접적인 선행 대조다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.

## OBS-F02 — 부분그래프 선택 개선으로 경쟁력 있는 확률 예측을 확보했다.

- 판정: 단순 확률 기준선에도 경쟁력 부족
- 원인·병목 [원인 미상]: Frozen PFN끼리의 상대 개선이 단순 class prior 대비 utility를 보장하지 않았다. Calibration·적은 context·subgraph mismatch·전처리 중 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 극소 context·불완전 입력에서 단순 prior보다 좋은 확률 정보를 실제로 회수하는 추론 원리가 나올 때.
- 먼저 이겨야 할 기준선: Uniform class probability, Laplace context prior, full graph, strongest fixed retrieval
- 최소 재검증: 학습 전에 단순 확률·majority 대조를 고정하고 NLL/Brier/accuracy를 독립 graph에서 함께 평가한다.
- 닫지 않은 것: 확률 점수 비교를 accuracy의 같은 결론으로 자동 확대하지 않는다. Prior 대조는 사후 sanity check다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.


## XFER-F01 — DBLP에서 통했던 context-LOO path 선택을 다른 graph로 옮기면 강하다.

- 판정: 고정 선택 규칙의 전이 실패
- 원인·병목 [진단이 지지]: 소수 context의 ranking이 query utility와 맞지 않았다. 후보 pool 확대와 accuracy 목적 변경만으로는 해결되지 않았다. 추정 분산·risk mismatch의 상대 기여는 미상이다.
- 반증·배제한 설명: Uniform/union 후보가 없어서만 실패했다는 설명은 확대 pool 대조로 충분하지 않았다.
- 재검토 조건: 극소 context의 추정 불확실성과 실제 query risk를 다루며 task 내 utility를 식별하는 방법이 나올 때.
- 먼저 이겨야 할 기준선: Uniform/union LP, individual paths, fixed GraphPFN path inputs
- 최소 재검증: 독립 graph에서 동일 context·후보 pool·선택 예산으로 사전 규칙을 고정해 검증한다.
- 닫지 않은 것: Context가 유용한 관계를 알려주는 다른 설정이나 relation 의미 추론 일반은 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/transfer.md](../findings/sources/graphpfn-2026-11-01/transfer.md); [findings/sources/graphpfn-2026-11-01/expanded.md](../findings/sources/graphpfn-2026-11-01/expanded.md); [findings/sources/graphpfn-2026-11-01/accuracy_selector.md](../findings/sources/graphpfn-2026-11-01/accuracy_selector.md)

- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.

## XFER-F02 — LP와 PFN 확률의 고정 혼합이 각자의 강점을 보존한다.

- 판정: 고정 혼합 메인 방향 중단
- 원인·병목 [원인 미상]: 서로 맞히는 query가 다르다는 사실이 고정 확률 평균의 utility를 보장하지 않았다. 중복 증거가 원인일 수 있으나 직접 분리한 결과는 아니다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 중복 evidence와 decision risk를 실제 관측량으로 구분하며 최강 단일 predictor를 보존하는 원리가 나올 때.
- 먼저 이겨야 할 기준선: LP alone, PCFI/FISF+PFN alone, compute-matched fixed mixtures
- 최소 재검증: 같은 query·context에서 within-task 오류 보완성과 독립 개발/확인 graph의 평균·worst-case utility를 함께 본다.
- 닫지 않은 것: 모든 ensemble이 불가능한 것도, 각 branch가 feature-only/structure-only인 것도 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/fusion.md](../findings/sources/graphpfn-2026-11-01/fusion.md)

- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [GraphPFN](https://arxiv.org/abs/2509.21489): 구체 키워드+태그 일치 (미검증); tags=graph-structure-use; keywords=ensemble
  - 기록된 강점: Hand-crafted graph feature(Laplacian PE, PageRank, 이웃 평균 등)에 기대지 않고, 모델 안에서 adjacency-masked attention으로 message passing을 직접 학습한다. 그 결과 ICL만으로도 G2T-FM·TAG와 튜닝된 GNN보다 평균 rank가 높고, 무거운 전처리도 필요 없다.

## RFILL-F01 — 현재 frozen conditional Gaussian completion이 분산을 맞춘 random-fill/graph-prior null보다 안정적으로 좋은 예측을 제공한다.

- 판정: 사전 random-control gate 미통과; 현재 frozen conditional의 독립 가치 미입증
- 원인·병목 [원인 미상]: 동일 관측 입력·checkpoint·분산/forward-budget 대조에서 요구한 conditional law 우위가 없었다. 관측값에 조건화한 복원, 구조 random feature, 확률 smoothing의 원인별 기여를 완전히 식별한 결과는 아니다.
- 반증·배제한 설명: Null은 분산·동일 PFN 호출 수를 맞췄다. 이것이 정확한 true posterior 또는 모든 feature law에 대한 대조를 뜻하지는 않는다.
- 재검토 조건: Context와 관측 피처가 joint feature law에 제공하는 추가 정보를 proper feature score와 실제 query improvement로 입증하고, 같은 예산의 random/구조 입력보다 좋은 연산이 나올 때.
- 먼저 이겨야 할 기준선: Variance-matched iid/graph prior, iid unit, zero/PCFI/FISF 입력 및 같은 예산·별도 학습의 강한 PFN. 실제 관측 입력 직접 경로와 proposal 경로를 구분.
- 최소 재검증: 새 학습 가설을 source/data/budget 고정 후 별도 bank에서 비교한다. Label-blind law 대조와 true-label-context conditional-law 증분을 분리하며 query labels를 forward에 사용하지 않는다.
- 닫지 않은 것: 실제 관측 피처의 직접 효용·context-conditioned feature-law 학습·shared suffix·coherence 일반은 평가하지 않았다. 분산 맞춤 null에는 second-moment 정보가 남고 unit null에는 실제 값 clamp가 남는다.
- 원문: [findings/sources/random-fill-controls-2026-11-02/REPORT.md](../findings/sources/random-fill-controls-2026-11-02/REPORT.md); [findings/sources/random-fill-controls-2026-11-02/paired.json](../findings/sources/random-fill-controls-2026-11-02/paired.json)

- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, observation-mask, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [GSPN](https://arxiv.org/abs/2305.10544): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [MCSSL](https://arxiv.org/abs/2605.11846): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 정보 증가에 따른 예측 갱신을 허용하면서 평균의 일관성을 제약한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## REG-F01 — Reference-regret priority가 hard-loss sampling 및 같은 비용의 Random보다 유리하다.

- 판정: 메인 방법 확장 보류
- 원인·병목 [원인 미상]: Reference의 독립 utility와 채점 비용을 정당화하지 못했다. Stale priority·짧은 학습·input loss 중 원인별 영향은 분리하지 않았다. Loss와 regret의 초기 Spearman0.4264는 다른 task를 강조함을 보일 뿐 효용 증명이 아니다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Reference의 추가 감독 비용을 회수하면서 실제로 회수 가능한 task difficulty를 식별하는 학습·생성 원리가 나올 때.
- 먼저 이겨야 할 기준선: Random, Manual, Loss priority, cold-score-cost-matched RandomBudget, frozen FISF
- 최소 재검증: 새 graph bank와 고정 비용으로 Loss/RandomBudget을 함께 넘는지 확인하고 task weighting과 generator 학습을 분리한다.
- 닫지 않은 것: 학습된 생성기·library·무한 stream prior learning을 구현하거나 실패시킨 것은 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md); [findings/sources/graphpfn-2026-11-01/regret_scope.md](../findings/sources/graphpfn-2026-11-01/regret_scope.md)

- [Chronos-2](https://arxiv.org/abs/2510.15821): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: 실제 다변량 데이터 없이 synthetic multivariatizer만으로 다변량·covariate 능력을 얻었다. synthetic 데이터만으로 학습한 변형도 GIFT-Eval과 Chronos Benchmark II에서 본 모델에 거의 근접해, synthetic prior 설계의 효과를 보여준다.
- [GraphPFN](https://arxiv.org/abs/2509.21489): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.
- [NodePFN](https://arxiv.org/abs/2604.19028): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: 합성 그래프 수를 늘릴수록 성능이 오르는 prior-scale 분석과, ER·cSBM·BA 구성을 바꾼 prior ablation을 함께 제시해 어떤 prior 성분이 도움이 되는지 일부 보여준다.
- [PFNSeparability](https://arxiv.org/abs/2610.14989): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Feature law와 conditional label law의 의존성을 prior 설계의 명시적 선택으로 다룬다.
- [TimesFM](https://arxiv.org/abs/2310.10688): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: synthetic 데이터를 빼면 실제 데이터에 드문 granularity(quarterly, yearly, 10분 등)와 15분 ETT에서 성능이 떨어진다는 ablation을 보여, 실제 corpus의 빈 곳을 synthetic prior로 메우는 효과를 실증했다.
- [Wander](https://arxiv.org/abs/2610.06694): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Synthetic prior가 NodePFN식과 GraphPFN식 생성기에 Watts–Strogatz, random geometric, graphon 기반 구조를 더해 섞는다. 그래서 GraphPFN prior가 덮지 못한다고 인정한 공간·기하 구조 그래프를 사전학습 분포에 명시적으로 포함한다.

## EVID-F01 — Relation 출처를 보존하는 head가 강한 Plain의 class decision을 개선한다.

- 판정: 현재 Evidence Head No-Go
- 원인·병목 [원인 미상]: Plain 정답을 잃는 mass5.36%가 구제2.48%보다 컸다. 정보 중복과 representation 간섭 중 원인은 식별하지 못했다. Confidence를 낮춰서 NLL이 좋아졌다는 설명은 후속 분석에서 기각됐다.
- 반증·배제한 설명: 과신을 줄여 확률 점수가 좋아졌다는 설명: Plain은 confidence63.99% vs accuracy88.96%로 과소확신이었다.
- 재검토 조건: Plain의 강한 결정 능력을 유지하면서 출처별 정보의 독립 가치를 회수하는 장치가 나올 때.
- 먼저 이겨야 할 기준선: Plain, Mean/Separate 동일 용량 control, fixed context-weighted evidence
- 최소 재검증: 동일 task에서 harmed/rescued partition과 NLL/Brier/accuracy를 함께 비교하고 conflict 강도별 증분을 확인한다.
- 닫지 않은 것: Relation source 정보가 불필요하다는 결론이나 다른 내부 추론 연산의 기각은 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/evidence.md](../findings/sources/graphpfn-2026-11-01/evidence.md); [findings/sources/graphpfn-2026-11-01/evidence_utility.md](../findings/sources/graphpfn-2026-11-01/evidence_utility.md)

- [CSBMInference](https://arxiv.org/abs/2306.07948): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Node labels와 latent feature centroids를 같은 추론 계산에서 갱신하는 직접적인 선행연구다. Feature–label 대응을 graph로 연결한다는 발상 자체의 novelty를 제한한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.

## PROG-F01 — 고정 executor 위 compiler가 조합 일반화와 품질·비용에서 강한 대조를 넘는다.

- 판정: 현재 compiler 확장 보류
- 원인·병목 [원인 미상]: 관측 정보는 Grid/TabPFN이 회수했지만 compiler는 context 증가에도 약61.6%에 머물렀다. 표현·search guidance·학습 최적화 중 정확한 원인은 식별하지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Context에서 계산 구성을 더 효율적으로 식별하고 bounded enumeration/강한 PFN feature 기준선을 넘어서는 장치가 나올 때.
- 먼저 이겨야 할 기준선: Cached ProgramGrid grammar prior, relational execution features+TabPFN, matched Plain
- 최소 재검증: 같은 grammar와 program prior, 후보 구조 접근, fresh graph와 heldout composition에서 품질과 end-to-end 비용을 맞춰 비교한다.
- 닫지 않은 것: 프로그램 합성·library learning 일반은 기각하지 않는다. 후보 문법 자체는 학습 중 모두 제공했고 missing은 실험하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/program.md](../findings/sources/graphpfn-2026-11-01/program.md)

- [DreamCoder](https://arxiv.org/abs/2006.08381): 구체 키워드+태그 일치 (미검증); tags=amortized-inference, compositional-generalization, program-synthesis; keywords=library, program
  - 기록된 강점: 해의 공통 코드를 계층적 library로 압축해 더 복잡한 프로그램을 짧게 표현한다.
  - 기록된 강점: 학습한 library의 dream task와 실제 해의 replay로 neural search를 개선한다.
  - 기록된 강점: List 처리·그림·장면 구성 등에서 학습한 구성요소를 미지 태스크에 재사용한다.
- [DeepProbLog](https://arxiv.org/abs/1805.10872): 구체 키워드+태그 일치 (미검증); tags=compositional-generalization; keywords=program
  - 기록된 강점: 재귀 규칙을 이용해 한 자리 숫자로 훈련한 인식기를 세 자리 숫자의 덧셈에 재사용한다.
- [Scallop](https://arxiv.org/abs/2304.04812): 구체 키워드+태그 일치 (미검증); tags=compositional-generalization; keywords=program
  - 기록된 강점: Recursion·negation·aggregation으로 이미지·언어·그래프 등의 8개 응용을 표현한다.

## SEL-F01 — Context Brier·local reliability로 PFN/LP의 정답을 선택할 수 있다.

- 판정: Gate B 실패, Gate C 보류
- 원인·병목 [진단이 지지]: Brier와 accuracy 목적이 불일치하는 사례를 직접 관측했다. IMDB seed11에서 LP의 test Brier0.6624가 PFN0.6672보다 좋지만 accuracy는35.54% vs44.65%였다. Context noise만으로 설명할 수 없다.
- 반증·배제한 설명: Context Brier 추정 noise만의 문제라는 설명은 실제 test Brier/accuracy 역전으로 충분하지 않다.
- 재검토 조건: 극소 context에서 정확도 효용과 uncertainty를 함께 식별하는 추정법이 나올 때.
- 먼저 이겨야 할 기준선: 최강 단일 PFN/LP, global/local selector, fixed mixture
- 최소 재검증: Query 정답 없는 입력만으로 실제 선택 utility를 독립 graph에서 평가하고 proper score와 accuracy를 분리한다.
- 닫지 않은 것: Oracle headroom을 실제로 회수하는 다른 risk estimator는 기각하지 않는다. 학습 selector는 이 실험에서 미실행이다.
- 원문: [findings/sources/graphpfn-2026-11-01/reliability.md](../findings/sources/graphpfn-2026-11-01/reliability.md)

- [GSPN](https://arxiv.org/abs/2305.10544): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [MCSSL](https://arxiv.org/abs/2605.11846): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 정보 증가에 따른 예측 갱신을 허용하면서 평균의 일관성을 제약한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## SRC-F01 — Source covariance를 보정하면 분류가 개선된다.

- 판정: 독립적인 task-level 이득 미확인
- 원인·병목 [원인 미상]: 오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.
- 먼저 이겨야 할 기준선: Vanilla continuation, matched/log/diagonal, covariance-shuffled controls
- 최소 재검증: 분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.
- 닫지 않은 것: 출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.
- [GSPN](https://arxiv.org/abs/2305.10544): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 숫자 대치나 여러 completion forward 없이 모델 내 partial-evidence 추론을 제공한다.
- [MCSSL](https://arxiv.org/abs/2605.11846): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: 정보 증가에 따른 예측 갱신을 허용하면서 평균의 일관성을 제약한다.
- [PEMix](https://arxiv.org/abs/2610.18113): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Clustered missingness의 전파 경계 병목을 분석하고, 관측값 전파에 의존하지 않는 구조 조건부 feature distribution을 제시한다.
- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
