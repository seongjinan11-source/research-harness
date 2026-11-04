# FP04 method result review

2026-11-04. 검토 자료: results의 REPORT_KO/summary/audit/budget/paired/lock, checks/preflight receipt, 저장된 final evaluation metric과 train log, train manifest의 support metadata. 새 예측·학습·GPU 작업은 실행하지 않았고 source/weights/bank/원결과를 수정하지 않았다. 본 reviewer는 앞서 model 구현을 담당했으므로 모델 구현과 완전히 독립된 reviewer는 아니다. 여기서는 root의 결과 서술과 별도로 저장된 수치·실패 범위를 재검토했다. 8,181개 NPZ를 모두 재채점한 것은 아니며, 그 작업의 PASS 기록과 receipt/hash를 확인했다.

**판정: 사전 정의한 네 조건 모두 FAIL이며, 현재 후보를 성공으로 유지할 근거가 없다.** Observed routing의 작은 양의 primary point estimate는 남지만 확실한 우위가 아니고, 관측값의 실제 이용을 지지하는 none/shuffle utility는 확인되지 않았다. 이 결과를 새로운 구현 버그나 모든 observed-cell PFN의 실패로 바꾸어 부르면 안 된다.

점수(1–5): 결과 추적 가능성 4, 사전 성공조건 충족 1, 학습된 feature utility 증거 1, 현 결과로 뒷받침되는 방법론적 novelty 1. 점수는 연구 일반의 가능성 평가가 아니다.

## 확인한 실제 결과와 범위

| 비교 | Accuracy 차이 %p | 사전 97.5% CI %p | 단위 | 판정 |
|---|---:|---:|---|---|
| observed − mask | +0.423177 | [−0.292969, +1.134033] | 96 base graphs | FAIL |
| observed − pcfi | +0.423177 | [−0.271267, +1.117622] | 96 base graphs | FAIL |
| observed − none | +0.043403 | [−0.629340, +0.629340] | 48 eligible disjoint graphs / 68 cases | FAIL |
| observed − shuffle | −0.325521 | [−0.824653, +0.086806] | 같은 conditional utility subset | FAIL |

Primary accuracy는 observed 48.253038%, mask/PCFI 각각 47.829861%다. Observed NLL 1.021423과 Brier .613500은 두 대조의 약 1.02359/.61510보다 조금 낮지만, secondary metric으로 gate를 번복할 수 없다. CI가 0을 포함한다는 것은 동등성 검정의 성공도 아니다. 작은 양의 효과는 허용되지만, 사전 요구한 확실한 양의 효과는 입증하지 못했다.

Primary는 source0/1을 포함하고, 고정 세 training seeds와 두 whole-node rates를 graph 안에서 평균한 뒤 12 strata 안에서 graph bootstrap을 했다. Utility는 일부 cases를 제외한 다른 conditional estimand다. 모든 48 disjoint graphs가 최소 한 eligible rate를 가져 68 cases가 남았으므로, 이번 FAIL은 empty subset이나 missing-strata 처리 때문이 아니다.

실제 9 fits, 6,912 optimizer updates, 27,648 task exposures가 기록됐다. 각 fit은 768 updates/3,072 exposures이며 모든 19,748 parameters가 trainable하다. Same-seed schedule/initialization hash가 세 arm 사이에서 일치한다. Budget의 train wall time 합은 448.413초다. 이는 큰 사전학습 예산의 결과가 아니다. 같은 parameter/update/exposure가 같은 FLOPs나 wall time을 뜻하지 않는다.

Audit는 final 7,776 + real 405 = 8,181 prediction records를 보고하고, 별도 external 225 records를 재채점했다. 최대 metric 재채점 오차는 3.33e−16이다. 현재 scientific source hash를 lock의 모든 항목과 직접 대조했으며 불일치는 없었다. Checks/preflight receipt의 실제 hash도 lock과 일치했다. Lock SHA256은 `ac86695ede9281d97208bb00ccf2846c439860c8fdf8ef27d578efdbdf2ee49c`, model SHA256은 `112cfdbd73c115c9bccbb64fe99ee73897dd8c266a0c0b1ff5e786c3f58854ac`다. 이 증거에서 새 provenance 결함을 발견하지 않았다.

## 핵심 해석 — 최대 5개

### 1. 구현된 정보 경로와 학습된 정보 이용은 다른 주장이다

Independent checks는 loop contraction/gradient, sparse-dense, chunk/permutation, poison, CPU/GPU 일치를 검증했다. Balanced two-component bridge의 class-contrast derivative는 float64에서 column enabled 약 2.54e−7, disabled 정확히 0이고, context→source edge를 끊으면 약 9.70e−22였다. CE gradient도 value/P/column/V/read 모듈에 존재했다. 따라서 의도한 경로가 계산 그래프에 있고 미분 가능하다는 근거는 있다.

이 검사는 학습 후 해당 경로가 유용한 class evidence를 운반한다는 증거가 아니다. 초기/작은 예제의 derivative 크기를 실제 모델의 causal feature contribution이나 vanishing-gradient 실패의 증거로 읽을 수도 없다. 이번 utility gate는 바로 그 간극을 검사했고 통과하지 못했다. “열 경로가 구현됐으니 missing inference가 된다”는 서술을 버려야 한다.

### 2. Observed의 arm 우위가 있어도 값의 semantic evidence 때문이라고 할 수 없다

Source0 case-weighted descriptive accuracy도 observed 48.0114%, mask 47.4432%, PCFI 47.5379%로 다르다. 관측값이 전혀 없는 조건에서도 모델별 학습된 graph/seed prior와 routing 구조가 다르다는 뜻이다. 이 subset 값은 primary의 graph-weighted 추정량과 같지 않으므로 직접 같은 CI로 취급하면 안 되지만, observed−mask를 곧바로 feature-value 효과로 부를 수 없다는 분명한 경계다.

Family primary에서 observed는 centroid 58.6806%, field 37.8255%이고 대조보다 각각 약 .56/.28%p 높다. Family별 학습 가능성·class imbalance·graph-only 예측력은 다르므로 raw 수준만으로 원인을 정할 수 없다. 현재 결과는 shared value basis/mean compression, graph-only shortcut, 약한 conditional signal, 제한된 최적화 중 무엇이 병목인지 식별하지 않는다. “실제 관측값이 무정보다” 또는 “오직 학습 시간이 부족하다”도 아직 결론이 아니다.

### 3. Real의 비교적 좋은 observed accuracy는 feature 이용 성공을 보이지 않는다

| Graph, 95% missing | Observed % | Mask % | PCFI % | Observed − none %p | Observed − shuffle95 %p | Restart LP % |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 52.8759 | 47.8193 | 47.9245 | −0.0476 | −0.0466 | 59.0562 |
| CiteSeer | 29.9769 | 25.8398 | 25.8282 | −0.1419 | −0.0242 | 38.4103 |
| PubMed | 64.0018 | 55.3311 | 55.0969 | −0.2626 | −0.0214 | 73.6900 |

Observed가 동일 모델 대조보다 높은 것은 보존할 사실이다. 그러나 같은 observed checkpoint의 none/shuffle이 거의 같거나 더 높아 값의 class 의미를 활용했다는 설명이 성립하지 않는다. 세 graph에서 restart LP에도 미달한다. Plain public PFN보다 높다는 비교만으로 강한 missing-feature PFN이라 주장할 수 없다. PCFI+public PFN은 95%에서 Cora 54.7965%, CiteSeer 34.0780%로 observed보다 높고 PubMed에서는 낮다. 외부 모델은 물리적 graph/context/query/관측값 identity가 맞아도 preprocessing·학습 예산이 다르므로 세 matched-arm의 인과 대조와 분리해야 한다. 실제 세 그래프는 재사용한 development data이며 새 독립 graph 일반화 CI가 아니다.

### 4. 이번에 시도한 것은 완전 학습된 작은 separable encoder이며 full sparse Transformer가 아니다

이 주기는 frozen predictor 수정과 다르게 value/graph/column/head를 실제로 함께 학습했다. 따라서 “아직 observed-cell 경로를 end-to-end 학습한 적 없다”는 과거 상태는 갱신해야 한다. 반면 이번 구조는 width32, signed scalar basis4, shared round4와 column memory의 mean aggregation이다. Class/round와 독립된 value basis를 공유하며, 모든 observed token 간 학습된 query/key attention이나 자유로운 query-specific retrieval을 구현한 것이 아니다. 이를 **full sparse observed-cell Transformer의 실패**라고 부를 근거는 없다.

또한 train과 final의 모든 query16은 항상 feature-free다. `full`도 **nonquery-visible** 조건이며 fully observed query task가 아니다. 완전 관측 M=1 tiny parity는 수치 검증일 뿐 feature-label 법칙의 학습 성공 검사가 아니다. 저장된 secondary full−none accuracy는 observed에서 centroid +.3038%p, field −.2170%p였으나, 두 조건 모두 query가 feature-free라 이 사실을 “완전 피처로도 못 배운다”로 해석하면 틀린다.

훈련 manifest 2,688 episodes의 source0/1/2+는 522/131/2,035개, 즉 19.42%/4.87%/75.71%다. Source0의 384개는 explicit none이다. 따라서 source0와 source1을 모두 거의 학습하지 않았다고 단정할 수도 없다. Informative singleton 경계의 비중과 query-observation support의 부재는 가능한 설명이지 확정 원인이 아니다. 학습 loss가 감소했다는 사실만으로 conditional feature law의 학습을 보증하지 않는다.

### 5. Negative result의 novelty와 다음 결정을 절제해야 한다

관측 subset 인코딩과 synthetic PFN 학습은 이미 선행 연구에 있으며, graph PFN에서도 scalar/column/node compression과 message passing이 사용됐다. 앞선 method review에서 정리한 Ephris/TabImpute 등의 관계는 이번 작은 비유의 primary 이득으로 바뀌지 않는다. 현재 남는 성과는 route·invariance·동일-arm 대조를 구현하고, **이 모델/두 synthetic families/고정 예산에서 유용한 feature transfer를 입증하지 못했다**는 재현 가능한 결과다. 새로운 Bayesian inference 원리나 robust missing PFN의 성공으로 포장할 수 없다.

PCFI의 literal raw cache와 common ±10 representation clipping도 구분해야 한다. 실제 native raw maxabs가 10을 넘는 경우가 있었고 clip fraction은 기록상 최대 약 .340%였다. Raw cache 보존은 검증됐지만, magnitude의 일부가 표현에서 버려졌다. 그렇다고 이 관측만으로 PCFI와 mask의 거의 같은 결과를 clipping 탓으로 돌리거나, clipping을 풀면 성공한다고 주장할 수 없다.

## 다음에 허용할 질문 하나 — 제안이며 실행하지 않음

**같은 encoder가 query 관측이 있는 task에서는 task별 feature-label 법칙을 먼저 학습할 수 있고, query 관측 유무를 포함한 mixed-observation 훈련이 feature-free query로 그 이용을 전이시키는가?**

이 질문을 다음 후보의 중심으로 권한다. 새 head나 더 큰 rank sweep보다, 현재 빠진 **complete-query trainability → partial-observation transfer**를 먼저 분리한다. 같은 generator·encoder를 유지한 새 독립 split에서 observed-query의 none/shuffle utility를 사전 검증하고, query-availability support를 훈련 prior에 넣은 비교를 같은 총 exposure 예산과 동일 기준선으로 고정하는 방향이다. Fully observed query에서도 utility를 못 배우면 missing routing의 성능 실험으로 바로 넘어갈 이유가 없다. Observed-query에서 배우고 feature-free에서 못 옮기면 그때 비로소 transfer 경로/학습 prior를 더 좁혀 검토할 수 있다.

이는 현 실패의 확정 원인도, 성공이 예정된 새 방법도 아니다. 현재 final과 real 결과는 다음 설계에 이미 사용된 development evidence로 보존하고, 새 실험이라면 별도 protocol·fresh 평가가 필요하다. 이번 후보의 gate FAIL은 변경하지 않는다.
