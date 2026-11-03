# Confidence Sheaf PFN 구성 요소 비교

2026년 11월 3일. 기존 숫자 기반 Node-frame의 세 가지 주장을 검증한다. Context label이 geometry에 독립적으로 기여하는지, 공간 confidence와 frame의 결합이 유용한지, 같은 예산의 직접 signed scalar 전파보다 유리한지를 구분한다. 기존 소스와 결과는 보존한다.

## 유지하는 추론 구조

관측 피처에서 출발해 결측 피처를 추정한 다음 동일한 PFN에 입력한다. 관측 cell은 매 반복 고정하고, 결측은 0으로 시작하며, 공간 전파 100회와 원래 채널 보정 1회를 유지한다. 각 seed의 기존 synthetic-trained PCFI-PFN을 모든 arm에서 고정한다. Query CE의 gradient만 frontend로 흐르며 PFN의 학습 중 update와 실제 graph에서의 update는 모두 0회다.

기존 Node-frame의 transport는 T_ij,f=Q_if^T Q_jf이고, frame은 입력의 관측값과 mask, graph, context 통계에서 계산한 뒤 100회 동안 고정한다. 이번 비교에서는 새로 검증한 좌표 정렬 sparse 구현으로 동시에 바꾸지 않고 기존 index-add 경로를 유지한다. 계산 방식 변경과 구성 요소 변경을 분리하기 위해서다.

## 비교하는 일곱 arm

| Arm | 공간 가중치 | Transport | Geometry의 label 신호 | 새 학습 |
|---|---|---|---|---|
| PCFI | 원 confidence | I | 사용하지 않음 | 없음 |
| Uniform | 균등 | I | 사용하지 않음 | 없음 |
| Node | 원 confidence | 기존 학습 Node-frame | 유지 | 기존 checkpoint |
| Node without labels | 원 confidence | Node-frame | label 유래 통계 제거 | 3 seed |
| Node uniform | 균등 | Node-frame | 유지 | 3 seed |
| Signed scalar | 원 confidence | 직접 학습한 대칭 scalar | 유지 | 3 seed |
| Geometry shuffle | 원 confidence | 기존 학습 Node-frame | 생성 때 assignment만 셔플 | 없음 |

Label 제거는 descriptor와 이웃 descriptor의 entropy, 최대 class support, context edge agreement 및 class-affinity 입력을 0으로 만드는 것이다. Context의 위치, graph, 관측 feature와 mask, confidence, degree는 유지한다. PFN에는 원래 context labels를 그대로 제공한다. 이 arm은 기존 Node에서 추론 때 입력만 바꾼 모델이 아니라, 같은 초기화부터 새로 학습한 모델이다.

Uniform은 공간 전파 가중치만 A_ij/degree_i로 바꾼다. Geometry descriptor의 confidence와 마지막 채널 보정의 confidence는 기존대로 유지한다. 따라서 이 2×2는 confidence 전체의 제거가 아닌 공간 confidence 가중치의 효과를 측정한다.

Signed scalar는 21→32→1 공유 MLP의 두 endpoint 순서 출력을 평균해 residual r_ij를 만들고, g_ij=clip(0.99+r_ij,−1,1)로 전파한다. Self-loop gain은 1이다. 원래 confidence P에 g를 곱하고 관측값을 고정해 100회 scalar 전파한다. 각 edge의 계수를 직접 학습하므로 cos(theta_j−theta_i)라는 제약이 없다. 새 parameter는 737개로 다른 학습 arm과 같다. 0.99는 결과를 보기 전에 고정한 초기 offset이며, Phi parameter는 모든 새 arm과 기존 Node의 초기 상태가 정확히 같다. 실제 초기 전파 함수가 같다는 뜻은 아니다. Clip 비율과 음수 gain 사용률을 보고하고, 함수족이 부호를 허용한다는 이유만으로 음의 evidence를 학습했다고 주장하지 않는다.

Geometry shuffle은 context label assignment를 바꾸어 geometry를 다시 계산하되, PFN에 전달하는 label과 관측값, query 집합은 원본을 유지한다. 클래스당 하나의 context가 있으면 shuffle이 class 이름만 바꾸는 대칭이 될 수 있다. k=1과 k=4 결과를 나누어 해석한다.

## 학습과 개발 입력

새 frontend는 3종×3 seed, 각 1,024 update로 총 9개와 9,216 update다. 기존 512개 synthetic train episode와 seed별 과제 순서를 그대로 재사용한다. 기존 Node와 초기 Phi state, PFN checkpoint, 과제 노출 수, AdamW 설정, 학습률 schedule, gradient clip, query CE 목적을 일치시킨다. 추가 loss나 teacher는 없다. 사전 검사의 9개 단일 optimizer step은 폐기한다.

새 개발 bank는 같은 label-SBM×feature-filter family의 32개 base graph다. Label homophily와 feature smooth/rough를 교차해 네 cell마다 8개 graph를 만든다. Graph를 원하는 점수가 나올 때까지 재생성하지 않는다. 각 graph에 context 1개와 4개/class를 모두 만들고, 네 결측 조건과 full 및 source2/8을 적용해 총 448 episode를 만든다. Context 수 사이에는 같은 관측 mask를 사용하고, 같은 graph와 context 수에서는 결측 조건 사이에 context를 유지한다. 참 법칙이나 query 정답은 추론 입력에 넣지 않는다.

실제 Cora, CiteSeer, Texas, Wisconsin의 기존 입력 60개는 개발 평가로 재사용한다. 이 네 graph를 독립 최종 검증으로 부르지 않는다. 새 구조 선택은 이번 bank를 보기 전에 고정했고, 결과에 따른 추가 loss, budget, 초기화 조정은 이 실행에서 하지 않는다.

## 평가와 해석

Primary는 자기 피처가 없는 query의 accuracy다. 3개 고정 학습 seed와 context 두 조건, 네 primary mask를 각 base graph 안에서 평균한 다음 base graph 단위 paired bootstrap 5,000회로 차이의 CI를 계산한다. NLL, Brier와 source0, source가 있는 조건, context 수, label와 feature 법칙별 결과를 함께 보고한다. 여러 CI는 다중비교 미보정이고 저장된 예측에 조건부다.

공간 confidence와 frame의 interaction은 다음 차이다.

\[
(\mathrm{Node}_{conf}-\mathrm{I}_{conf})
-(\mathrm{Node}_{uniform}-\mathrm{I}_{uniform}).
\]

양수이면 이 bank에서 공간 confidence가 Node의 증분 효과를 높였다는 뜻이다. 하나의 arm만 좋아진 것을 interaction으로 부르지 않는다. Context와 scalar 비교가 예상과 달라도 전체 연구를 자동 종료하지 않고, 해당 구성 요소의 필요성과 주장 범위를 조정한다.

Native-PCFI와 고정 signed LP는 같은 실제 입력의 기존 예측을 hash로 확인해 강한 전체 pipeline 기준으로 함께 남긴다. Native-PCFI는 원본 고차원 공간에서 먼저 전파하므로 동일표현의 geometry-only 대조와 구분한다. GPU 반복 변동과 전체 입력·체크포인트 보존을 최종 audit에 기록한다.
