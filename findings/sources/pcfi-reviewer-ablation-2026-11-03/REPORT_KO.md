# Confidence Sheaf PFN의 context와 confidence 및 scalar 비교

2026년 11월 3일. 기존 Node-frame을 개발 후보로 유지한 상태에서 구성 요소를 분리했다. 새 학습은 frontend 9개와 9,216 update이며, PFN과 실제 graph의 가중치 update는 모두 0회다. 결과에 따른 추가 loss나 budget 조정은 하지 않았다.

## 새 개발 graph의 primary

동일 known family의 새 base graph 32개다. Label homophily와 feature 법칙 네 cell에 각각 8개 graph를 배정하고, 두 context 수와 네 primary 결측 조건 및 고정 학습 seed 3개를 graph 안에서 평균했다. 자기 피처가 없는 query의 accuracy가 primary다.

| 방법 | Accuracy % | NLL | Brier |
|---|---|---|---|
| PCFI | 44.772 | 1.30606 | 0.64059 |
| Uniform I | 44.604 | 1.30738 | 0.64116 |
| Node | 44.852 | 1.30707 | 0.64125 |
| Node label 제거 | 44.583 | 1.30764 | 0.64183 |
| Node 공간 균등 | 44.314 | 1.31496 | 0.64418 |
| Signed scalar | 44.599 | 1.32133 | 0.64360 |
| Geometry label shuffle | 44.887 | 1.30742 | 0.64137 |

| 비교 | Accuracy 차이 %p | 95% CI |
|---|---|---|
| Node − PCFI | +0.080 | [-0.137, +0.303] |
| Node − label 제거 | +0.269 | [+0.042, +0.506] |
| Node − signed scalar | +0.253 | [-0.021, +0.531] |
| Node − geometry label shuffle | -0.035 | [-0.122, +0.048] |
| 공간 균등의 Node − I | -0.291 | [-0.600, -0.019] |
| Node의 confidence − 공간 균등 | +0.538 | [+0.118, +0.969] |
| I의 confidence − 공간 균등 | +0.167 | [-0.290, +0.628] |
| Signed scalar − PCFI | -0.173 | [-0.431, +0.101] |
| Label 제거 − PCFI | -0.189 | [-0.362, -0.021] |
| 공간 confidence × Node interaction | +0.371 | [+0.055, +0.697] |

CI는 base graph 단위 paired bootstrap 5,000회다. 여러 비교는 다중비교 미보정이며 저장된 예측과 세 고정 학습 seed에 조건부다. 구성 요소의 ablation이므로 모든 비교를 통과해야 한다는 종료 기준을 새로 붙이지 않는다.

## 세 비교의 해석

**Context의 독립 효과.** Label 신호를 제거해 같은 예산으로 학습한 geometry보다 원 Node가 높았다. 이 개발 bank에서는 geometry의 context-label 통계에 추가 가치가 있다.
Frozen geometry shuffle은 PFN의 label을 바꾸지 않은 별도 개입이다. Assignment 제거 후 재학습과 추론 때의 shuffle은 다른 질문이므로 함께 보고한다. 이번에는 geometry shuffle의 accuracy 차이 CI가 0을 포함했다. 따라서 재학습 제거 대조의 이득만으로 올바른 class 묶음의 의미를 읽어 geometry를 적응시켰다고 주장하지 않는다.

**공간 confidence와 frame의 결합.** Interaction은 (Node−PCFI)−(Node uniform−Uniform I)로 계산했다. 평균 +0.371%p이며 양의 interaction을 지지한다. Descriptor와 채널 보정의 confidence는 유지했으므로 confidence 전체의 유무로 확대하지 않는다.

이 양의 interaction에는 공간 가중치가 균등할 때 Node가 I보다 낮아진 효과도 포함된다. Weighted Node 자체의 PCFI 대비 차이 CI는 0을 포함하므로, confidence와 frame이 함께 강한 기준선을 넘는 새로운 예측 신호를 만들었다고 주장하지 않는다.

**직접 scalar와의 비교.** CI가 0을 포함해 두 함수족의 우위가 확인되지 않았다. 이 scalar는 edge 계수를 직접 학습하지만 clipping과 별도 초기 전파 함수를 사용한다. 가능한 모든 scalar 방법에 대한 최강 대조나 전체 함수족의 우열로 확대하지 않는다.

## Context 수와 대칭

| k per class | 방법 | Accuracy % |
|---|---|---|
| 1 | Node | 39.502 |
| 1 | Node label 제거 | 39.244 |
| 1 | Geometry label shuffle | 39.516 |
| 1 | Signed scalar | 39.323 |
| 4 | Node | 50.202 |
| 4 | Node label 제거 | 49.922 |
| 4 | Geometry label shuffle | 50.258 |
| 4 | Signed scalar | 49.874 |

클래스당 context가 하나이면 assignment shuffle이 class 이름의 순열일 수 있다. 현재 geometry는 class 이름에 불변인 통계를 쓰므로 k=1의 작은 변화는 수치 변동과 함께 해석한다. k=4에서는 class별 context 묶음 자체를 바꾸므로 다른 개입이다.

## Label와 feature 법칙별 결과

| 개발 cell | PCFI % | Node % | Label 제거 % | Node 공간 균등 % | Signed % |
|---|---|---|---|---|---|
| homo_smooth | 55.673 | 56.154 | 55.720 | 54.817 | 55.642 |
| homo_rough | 60.961 | 60.739 | 60.610 | 59.931 | 60.531 |
| hetero_smooth | 33.356 | 33.404 | 33.266 | 33.495 | 33.211 |
| hetero_rough | 29.098 | 29.110 | 28.736 | 29.012 | 29.011 |

Label heterophily와 feature roughness는 별도의 생성 축이다. 같은 그래프에서 라벨이 다른 이웃이 많다는 이유로 모든 numeric feature의 부호를 뒤집어야 한다고 해석하지 않는다.

## 실제 graph의 개발 평가

| 방법 | Cora % | CiteSeer % | Texas % | Wisconsin % |
|---|---|---|---|---|
| PCFI | 50.188 | 30.852 | 28.005 | 21.557 |
| Uniform I | 48.952 | 32.418 | 29.058 | 22.888 |
| Node | 49.800 | 31.275 | 28.178 | 22.114 |
| Node label 제거 | 50.009 | 31.046 | 28.315 | 22.760 |
| Node 공간 균등 | 48.709 | 32.099 | 26.690 | 21.101 |
| Signed scalar | 49.396 | 30.596 | 28.295 | 21.951 |
| Geometry label shuffle | 49.862 | 31.221 | 28.249 | 22.331 |
| native_pcfi | 52.030 | 33.867 | 33.260 | 24.445 |
| lp | 56.281 | 34.434 | 13.711 | 26.526 |
| negative_lp | 32.026 | 24.757 | 63.537 | 42.725 |
| mask_reference | 40.046 | 25.012 | 21.282 | 23.864 |

기존 네 real graph의 동일 입력과 세 context/mask 반복을 재사용했다. 새 일곱 arm의 확률은 모두 이번 실행에서 계산했다. Native-PCFI, LP와 Mask floor는 이전 동일 입력의 hash 검증된 예측을 유지했으며 신규 학습군이 아니다. Native-PCFI는 원본 고차원에서 먼저 전파하므로 동일표현의 공간 operator 대조와 구분한다. 실제 graph 결과는 개발용 점수이며 독립 holdout이나 다수 graph에 대한 유의성 주장이 아니다.

## Source가 없는 조건과 확률 품질

| Source | 방법 | Accuracy % | NLL | Brier |
|---|---|---|---|---|
| zero | PCFI | 42.769 | 1.34151 | 0.66429 |
| zero | Uniform I | 42.769 | 1.34151 | 0.66429 |
| zero | Node | 42.769 | 1.34151 | 0.66429 |
| zero | Node label 제거 | 42.769 | 1.34151 | 0.66429 |
| zero | Node 공간 균등 | 42.769 | 1.34151 | 0.66429 |
| zero | Signed scalar | 42.769 | 1.34151 | 0.66429 |
| zero | Geometry label shuffle | 42.769 | 1.34151 | 0.66429 |
| nonzero | PCFI | 45.058 | 1.30100 | 0.63720 |
| nonzero | Uniform I | 44.866 | 1.30250 | 0.63786 |
| nonzero | Node | 45.149 | 1.30215 | 0.63795 |
| nonzero | Node label 제거 | 44.842 | 1.30280 | 0.63863 |
| nonzero | Node 공간 균등 | 44.534 | 1.31117 | 0.64130 |
| nonzero | Signed scalar | 44.860 | 1.31844 | 0.64065 |
| nonzero | Geometry label shuffle | 45.190 | 1.30254 | 0.63810 |

Source0에서는 추정기가 새 숫자를 생성하지 않고 같은 PFN의 graph/context 예측으로 돌아간다. Source0와 nonzero는 과제 구성이 다르므로 두 평균의 차이를 피처 관측의 인과적 효과로 해석하지 않는다. Source2/8과 full의 secondary 결과는 statistics.json에 별도로 보존했다.

## 전파 계수와 검증

| Signed scalar 입력 | 음수 계수 비율 | 평균 gain | +1 clip 비율 |
|---|---|---|---|
| development | 0.000000 | 0.979035 | 0.669091 |
| real | 0.000000 | 0.977463 | 0.669207 |

직접 scalar는 이번 실행에서 음수 gain을 실제로 사용하지 않았고, +1 clipping이 약 67%였다. 이는 signed 함수족의 표현력이 부족하다는 증거가 아니다. 최적화가 부호를 활용하지 못한 이유도 분리되지 않았다. 따라서 이 대조만으로 2D transport가 signed 전파보다 필요하다는 결론은 내리지 않는다.

Audit PASS_WITH_MEASURED_GPU_REPLAY. 기존 파일 8353개와 입력 2580개, 새 예측 10668개를 hash로 확인했다. 새 frontend 9개의 초기 상태와 과제 순서가 기존 Node와 일치하고 총 9216 update였다. PFN과 실제 graph update는 0회다.

선택된 12개 예측의 replay에서 저장 확률 대비 최대 차이는 0.0104881, 같은 session 반복 최대 차이는 0.0068214, 저장 예측 대비 최대 argmax 변경은 0.300%다. 0.1 검사는 큰 손상 검사이며 엄격한 재현 보장은 아니다. CI는 이 전체 수치 변동을 포함하지 않는다.

Full과 source0의 PCFI 환원 1998건의 최대 확률 차이는 0.0000000다. Gain의 부호와 clipping은 학습 연산의 실제 사용을 나타내며 성능의 인과적 설명을 확정하지 않는다.

## 다음 판단의 범위

이 결과는 구성 요소의 필요성을 검토하는 개발 실험이다. 이미 검토한 실제 그래프와 같은 known family에서 좋은 arm을 고른 것을 독립 확인으로 부르지 않는다. 특정 부품이 유효하지 않다면 메소드와 주장을 단순화하고, 최종 구조를 고정한 뒤 새로운 그래프에서 확인한다. 이번 실행에서는 결과를 보고 새로운 loss, 초기화, 학습량을 추가하지 않았다.

[계산과 비교 정의](METHOD_KO.md), [원점수와 CI](results/statistics.json), [검증](results/audit.json), [고정한 프로토콜](protocol.json).
