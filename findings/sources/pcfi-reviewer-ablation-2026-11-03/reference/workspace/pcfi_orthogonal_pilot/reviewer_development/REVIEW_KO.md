# Confidence Sheaf PFN 메소드 검토와 ablation 계획

2026년 11월 3일. 현재 숫자 기반 Node-frame을 개발 후보로 유지한다. Confidence 가중 전파와 context로 구성한 좌표계를 결합하고, 고정 PFN의 예측 손실로 그 좌표계를 학습하는 메소드다. 부품마다 새로운 발명일 필요는 없지만, 이 결합이 실제로 제공하는 기능과 성능을 분리해 보여야 한다.

기존 실험의 소스와 체크포인트, 점수, 성공 기준은 유지한다. 현재 실제 heterophily 두 그래프의 PCFI 대비 평균 +0.371%p는 개발용 점추정이고, 새 합성 그래프에서의 차이는 −0.226%p였다. 이번 검토에서 추가한 것은 동치 구현과 수치 검증이며, 새 분류 성능으로 보고하지 않는다.

## 메소드의 계산 정의

관측 피처와 mask, graph, context labels로 노드와 피처 column별 각도 theta를 계산한다. 이 각도는 한 과제의 100회 공간 전파 동안 고정한다. 같은 column에서 다음 transport를 쓴다.

\[
T_{ij,f}=Q_{if}^{\top}Q_{jf},\qquad Q_{if}=R(\theta_{if}).
\]

전파 가중치는 기존 PCFI의 거리 pseudo-confidence를 유지한다.

\[
P_{ij,f}=\frac{A_{ij}c_{jf}}{\sum_k A_{ik}c_{kf}}.
\]

관측 cell은 매번 (x,0)으로 고정하고 결측 cell만 이웃 메시지로 갱신한다. 첫 좌표를 읽어 기존 채널 상관 보정을 한 번 적용한 뒤 같은 고정 PFN에 넣는다. Query CE의 gradient가 frontend로 흐르지만 PFN 가중치는 갱신하지 않는다. Confidence는 보정된 관측 확률이 아니며, 실제 그래프에서도 가중치 갱신은 없다.

## 좌표를 맞춘 공간에서의 동치 계산

현재 Node-frame은 node와 column마다 하나의 frame을 공유하므로 경로에 독립적인 transport다. 전파 상태를 u_i=Q_i z_i로 바꾸면 결측 cell의 갱신은 다음이 된다.

\[
u_i^{t+1}=\sum_j P_{ij}u_j^t.
\]

관측 경계도 함께 바꾸어 u_i=Q_i(x_i,0)으로 고정해야 한다. D를 같은 mask와 confidence를 사용하는 100회 scalar 공간 전파라고 하면, 채널 상관 보정 전의 출력은 다음과 같다.

\[
\hat x^{\mathrm{sp}}_{if}
=\cos\theta_{if}\,D_f(x_f\cos\theta_f)_i
+\sin\theta_{if}\,D_f(x_f\sin\theta_f)_i.
\]

따라서 frame을 한 번 적용한 두 scalar 채널을 전파하고, 마지막에 원 좌표로 되돌릴 수 있다. 매 edge에서 좌표 변환을 반복할 필요가 없다. 이후의 같은 채널 보정까지 적용하면 원래 Node frontend와 동치인 추정기를 얻는다. 이는 현재 코드에서 유도한 식이며 새 정리라고 주장하지 않는다. 경로 독립적인 직교 transport와 일반적인 sheaf의 차이는 [Neural Sheaf Diffusion](https://arxiv.org/html/2202.04579)의 관련 논의와 구분해 설명한다.

더 직접적으로는 source에서 각 query로 전달되는 scalar 전파 계수를 H라고 할 때 다음과 같이 쓸 수 있다.

\[
\hat x^{\mathrm{sp}}_{if}
=\sum_j H_{ij,f}\cos(\theta_{jf}-\theta_{if})x_{jf}.
\]

H는 같은 100회 전파와 관측 경계에서 정해지며 값이 음수가 아니다. 학습한 frame은 source와 query 사이의 기여를 줄이거나 부호를 바꿀 수 있다. 이 식은 채널 보정 전의 공간 전파에 대한 것이다. 각도가 입력에 의존하므로 전체 추정기가 입력에 대해 선형이라는 뜻은 아니다.

리뷰어에게는 현재 방법을 flat connection을 쓰는 confidence 기반 보간으로 설명할 수 있다. 임의의 경로 의존 geometry나 비자명한 cycle 효과가 필요하다고 주장할 근거는 없다. Node가 Edge보다 낫다는 점추정을 근거로 더 복잡한 edge map을 추가할 필요도 없다.

## 완료한 구현 개선과 검증

[aligned_operator.py](aligned_operator.py)에 두 동치 경로를 구현했다. Index-add 경로는 기존에 저장된 P를 그대로 쓰고, sparse 경로는 A와 confidence로 P를 다시 계산해 두 채널을 함께 전파한다. Sparse 경로에서는 기존 float32 P의 반올림과 재계산에 따른 미세한 수치 차이가 생길 수 있다.

[수치 기록](results/node_alignment.json)에는 다음 검사가 있다.

| 검사 | 범위 | 결과 |
|---|---|---|
| Forward 동치 | 3개 학습 seed의 기존 dev와 full, source0, cell mask, 분리된 성분, 큰 각도 stress 총 69건 | 최대 절대 차이 7.75e-7 |
| 파라미터 gradient | float64의 학습 각도와 큰 각도 stress 총 6건 | Index-add 최대 1.78e-15, sparse 최대 2.22e-9 |
| Source 계수의 독립 계산 | 작은 graph에서 dense H를 직접 계산한 18건 | 최대 절대 차이 2.22e-16 |
| 관측값과 source0 | 관측 cell 고정과 관측 source가 없는 입력 | 정확히 유지 |
| 기존 기록 보존 | 주요 소스와 점수, frontend 9개 및 PFN 3개 체크포인트를 포함한 22개 파일 | 실행 전후 hash 일치 |

새 optimizer update는 0회이고 PFN을 실행하지 않았다. Gradient 검사는 학습된 frontend의 출력 함수 자체에 대해 수행했다. 이 검증은 CPU 수치와 미분에 대한 것이며 전체 PFN의 GPU 예측 동등성을 검사한 결과는 아니다.

## 추정기만의 CPU 비용

같은 학습 Node-frame, 미리 계산한 동일 geometry, 2 CPU threads, warmup 3회와 반복 10회로 측정했다. 실행 순서를 번갈아 배치했다. 모든 경로에 angle 계산, 100회 전파, 채널 상관 보정을 포함했고 sparse 경로에는 sparse adjacency와 분모를 만드는 비용도 포함했다.

| 입력 | 원 Node ms | 좌표 정렬 Index-add ms | 좌표 정렬 Sparse ms |
|---|---|---|---|
| 기존 synthetic dev graph 0 | 26.66 | 19.85 | 7.54 |
| Cora | 130.89 | 84.45 | 75.11 |
| Texas | 24.75 | 18.44 | 6.31 |

이는 frontend만의 CPU 측정이다. Geometry 생성과 PFN 추론을 포함한 전체 지연 시간이나 GPU 속도 우위를 뜻하지 않는다. 기존 보고서의 전체 pipeline GPU 측정과 직접 나누어 비교하지 않는다. 새 구현은 우선 별도 파일로 유지하며 기존 checkpoint 평가 경로를 교체하지 않는다.

## 리뷰어의 질문과 필요한 ablation

| 의심할 주장 | 기존 비교가 답한 범위 | 다음 비교와 해석 |
|---|---|---|
| Context가 geometry를 유용하게 바꾸는가 | 기존 context shuffle은 frontend와 PFN에 들어가는 label을 함께 바꿨음 | PFN에는 원 label을 유지하고 geometry 생성에만 label assignment shuffle을 적용. 이어서 label 신호가 없는 geometry를 같은 예산으로 학습해야 context의 학습 효과를 판단할 수 있음 |
| Confidence와 좌표 변환의 결합이 필요한가 | 현재 모든 arm이 같은 confidence를 사용함 | 공간 전파의 confidence 유무와 Node-frame 유무를 2×2로 비교. Descriptor와 채널 보정은 고정해 우선 공간 가중치 효과만 분리 |
| 2D transport가 강한 scalar 연산보다 유리한가 | 현재 Scalar는 매 edge에서 sin을 제거하는 별도 함수족임 | 학습 가능한 bounded signed scalar 계수를 같은 관측 입력과 예산으로 비교. 동치인 두 채널 구현은 별도 경쟁 모델로 세지 않음 |
| 관측 피처의 실제 대응을 활용하는가 | 기존 seed11 개입에서 graph마다 반응이 달랐음 | 고정 query에서 feature 제거와 source 간 feature shuffle을 3개 seed로 확인. Feature 제거 시 향상되는 조건도 함께 남김 |
| 추가 모수가 주는 효과인가 | 동일 737개 parameter의 Scalar와 비교했으나 conditioning과 연산이 다름 | Node의 학습 각도와 같은 크기 분포의 고정 random 각도를 비교하고, geometry 입력을 제거해 학습한 전역 frame도 보조 대조로 사용 |
| 강한 전체 pipeline을 실제로 개선하는가 | RP16 PCFI와의 공간 교체 효과만 직접 통제됨 | Native-PCFI를 전체 pipeline 기준으로 유지하고, representation 차이를 별도 분석. Native와의 차이를 geometry 하나의 효과로 귀속하지 않음 |
| PFN과 공동 학습해야 하는가 | 현재 구성에서는 PFN update가 0회임 | 선택적 후속 2×2로 frozen와 trainable PFN을 비교. PCFI 쪽도 같은 시작점과 update 예산을 제공하며 공동 학습을 메소드 성립 조건으로 삼지 않음 |

클래스당 context가 하나이면 label assignment의 순열이 전체 class 이름의 순열과 같을 수 있다. 현재 geometry는 class 이름에 불변인 통계를 사용하므로 이때 shuffle에 반응하지 않는 것은 예상되는 대칭이다. Context 효과를 검증하려면 클래스당 여러 context가 있는 조건을 별도로 봐야 한다. Shuffle 후에도 성능이 같다는 이유만으로 context의 가치 전체를 부정하지 않는다.

동치 계산은 현재 Scalar 대조의 해석도 좁힌다. 매 edge에서 cos만 남기면 경로마다 cos를 곱하지만, Node의 원 연산에서는 두 좌표를 보존해 source와 query의 최종 각도 차이가 남는다. 따라서 현재 Scalar 대비 이득만으로 모든 scalar 방법보다 우월하다고 주장하지 않는다.

전체 theta에 같은 각도를 더해도 endpoint의 각도 차이는 바뀌지 않는다. 따라서 학습한 절대 각도 자체를 유일한 정답 좌표계로 해석하지 않고, 관측값 전달과 최종 예측에 미치는 효과를 평가한다.

## 개발과 최종 검증의 분리

현재 dev 14개 episode는 base graph 두 개에서 파생됐으며 둘 다 label_mode가 homo다. Feature law는 smooth와 rough가 하나씩 있다. 앞으로 heterophily 관련 선택에 사용할 개발 bank는 label homophily와 feature law를 독립적으로 교차해 구성해야 한다. Label heterophily가 곧 feature의 부호 반전이라는 가정은 두지 않는다.

다음 개발 순서는 geometry 전용 context 개입, 공간 confidence 2×2, 강한 scalar 대조다. 새로운 head나 loss를 추가하지 않고 한 번에 한 계산만 바꾼다. 학습 비교에는 같은 train task, 초기화 기준, 노출 수, optimizer와 update budget을 적용한다. 현재 Node를 선택한 것은 기존 결과를 본 개발 결정으로 남긴다.

개발 단계에서는 ablation이 예상과 달라도 원인을 단정하거나 매번 전체 연구를 종료하지 않는다. 예를 들어 context 제거가 같은 품질을 낸다면 context 적응이라는 주장을 좁히고, 더 단순한 추정기가 남는지 판단한다. 각 모듈이 모든 조건에서 유의하게 좋아야 한다는 기준도 두지 않는다.

최종 구조와 평가 기준을 잠근 뒤 새로운 base graph bank와 실제 holdout graph에서 확인한다. Primary는 자기 피처가 없는 query의 accuracy로 유지하고 NLL과 Brier는 함께 보고한다. 클래스별 실패와 source0를 따로 남긴다. CI는 base graph 단위 paired 집계로 계산하고 GPU 수치 변동도 별도로 측정한다. 현재 실제 네 graph는 개발용이므로 독립적인 최종 검증으로 다시 부르지 않는다.

## 현재 주장과 다음 판단

현재의 메소드 기여 후보는 불완전한 입력에서 과제별 confidence 기반 좌표 변환을 추론하고, synthetic task의 분류 목적에 맞게 학습해 실제 graph에 고정 가중치로 적용하는 것이다. Confidence와 Sheaf, PFN 각각의 독립 발명이나 현재 +0.371%p만으로 논문 성공을 주장하지 않는다.

이번에 확보한 발전은 현재 Node 추정기를 더 명확하게 정의하고 더 단순하게 구현한 것이다. 다음 ablation의 역할은 이 구조에서 실제로 필요한 부분을 남기는 데 있다. 최종 분류 성능은 강한 PCFI pipeline과의 비교로 판단한다.

[기존 메소드](../METHOD_KO.md)와 [기존 결과](../REPORT_KO.md)는 그대로 보존했다. 새 검증은 `verify_aligned.py`로 재실행할 수 있다.
