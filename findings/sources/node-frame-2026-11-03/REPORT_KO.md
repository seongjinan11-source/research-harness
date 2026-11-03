# 노드 직교 frame GraphPFN 실험 결과

2026-11-03. 판정: **현재 구성의 확대 기준 미통과**.

Edge마다 restriction을 만드는 대신 노드마다 SO(2) frame 하나를 만들고 모든 incidence에 공유했다. 입력·전파 위치·학습 예산은 유지했다. 새 모델은3개, query CE update는3,072회이며 실제 graph 학습은0회다.

## 계산과 비교 조건

T_ij=Q_i^T Q_j로 구성해 단계별 순환 경로의 변환 곱을I로 제한한다. 노드 자신의 descriptor와 closed-neighborhood descriptor 평균을 같은72→32→1Phi에 넣는다. 기존 edge 모델은 두 endpoint의 descriptor를 넣었다. 동일 파라미터 수와 초기화는 확인했지만 두 함수족과 conditioning은 다르다.

기존 GraphPFN의 모든 feature/label token에 같은 geometry를 쓰며0/3/6/9층에서 갱신한다. Confidence·degree+1 정규화·좌우 mixer·ELU·step·관측 토큰 anchor도 유지했다. PCFI를 추가 입력하거나 라벨 전파만 분리하는 변경은 하지 않았다.

기존5개 비교군은 같은1,024update를 끝낸 checkpoint를 재사용했다. 기존64train graph·512episodes·seed별taskorder와 초기화가 정확히 같다. 새 합성64graph에서는 비교군도 모두 다시 추론했다. 실제4graph의 비교군 확률은 같은 입력의 기존 artifact다.

## 새 합성 graph의 primary 결과

동일 known family의 새64base graph다. Uniform/clustered90/99.5% whole-node missing의 결측 query accuracy를 평가했다. 4조건·3고정학습seed를 graph 안에서 평균하고 graph-paired bootstrap5,000회를 계산했다.

| 모델 | Accuracy % | NLL | Brier |
|---|---|---|---|
| mask | 53.873 | 1.09719 | 0.54448 |
| pcfi100 | 53.005 | 1.10975 | 0.55125 |
| identity | 54.484 | 1.07792 | 0.53313 |
| static | 54.777 | 1.07662 | 0.53242 |
| dynamic | 54.620 | 1.07498 | 0.53330 |
| node_frame | 54.599 | 1.07732 | 0.53336 |

| Node-frame − 비교군 | Accuracy 차이 %p | 95% CI |
|---|---|---|
| identity | +0.116 | [-0.433, +0.676] |
| dynamic | -0.020 | [-0.463, +0.406] |
| static | -0.178 | [-0.742, +0.356] |
| mask | +0.726 | [+0.072, +1.380] |
| pcfi100 | +1.594 | [+0.754, +2.480] |

CI가0을 포함한 비교를 동등성이나 효과 없음으로 해석하지 않는다. 이전 test bank의 절대 accuracy와 이번 accuracy를 빼서 모델 개선으로 쓰지 않는다.

## 실제 graph의 개발용 평가

| 모델 | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|
| mask | 40.05 | 25.01 | 21.28 | 23.86 |
| pcfi100 | 50.15 | 30.84 | 27.77 | 21.61 |
| identity | 37.16 | 24.24 | 21.43 | 23.28 |
| static | 36.35 | 25.43 | 19.64 | 24.47 |
| dynamic | 38.07 | 22.35 | 19.06 | 23.81 |
| node_frame | 36.51 | 24.42 | 19.32 | 22.23 |
| native_pcfi100 | 52.03 | 33.87 | 33.26 | 24.44 |
| public_native_pcfi100 | 50.12 | 32.15 | 32.19 | 20.49 |
| lp | 56.28 | 34.43 | 13.71 | 26.53 |
| negative_lp | 32.03 | 24.76 | 63.54 | 42.73 |

단위%. 기존4graph·3context/mask반복을 사용했다. 가중치는 고정이고 query/validation labels로 선택하지 않았다. Node-frame만 새로 실제 추론했으며 비교군은 hash 검증된 기존 확률이다. 실제 graph 수가4개이므로 노드 bootstrap으로 임의 graph 모집단의 유의성을 주장하지 않는다.

Heterophily 평균 Node-frame−PCFI는-3.915%p, −Dynamic은-0.663%p다. Homophily 평균 Node-frame−PCFI는-10.030%p다.

Native-PCFI는 원본 고차원 공간에서 전파한 뒤 같은RP16으로 압축한다. 주모델은 먼저RP16으로 압축된 관측값을 받으므로 native 비교는 동일표현의 geometry-only 효과가 아니라 전체 pipeline floor다. Negative LP는 고정 signed LP이며 CPGNN/CLP 재현은 아니다.

## Source 수와 결측 조건별 보조 결과

| 조건 | mask | pcfi100 | identity | static | dynamic | node_frame |
|---|---|---|---|---|---|---|
| uniform90 | 53.37 | 55.10 | 54.53 | 54.63 | 54.93 | 54.66 |
| uniform995 | 53.66 | 51.27 | 53.50 | 54.55 | 53.84 | 53.54 |
| cluster90 | 52.99 | 53.70 | 54.39 | 54.31 | 54.93 | 54.79 |
| cluster995 | 55.48 | 51.94 | 55.50 | 55.62 | 54.77 | 55.41 |
| sources2 | 54.51 | 54.43 | 54.96 | 54.77 | 55.37 | 55.26 |
| sources8 | 54.61 | 54.37 | 55.50 | 55.39 | 56.10 | 55.34 |

| Primary source 분할 | mask | pcfi100 | identity | static | dynamic | node_frame |
|---|---|---|---|---|---|---|
| zero | 45.15 | 43.55 | 44.59 | 46.08 | 45.90 | 45.69 |
| nonzero | 55.25 | 54.50 | 56.05 | 56.15 | 56.00 | 56.01 |

Source2/8은 별도 mask/context/query 조건이며 primary를 대체하지 않는다. 관측 피처가0인 episode도 유지했고 graph/context로 예측하는 것을 새로운 피처 정보를 복원한 것으로 해석하지 않는다.

## Frame이 실제로 작동했는가

| 모델/평가 | stage별 cos-negative 비율 | 첫 map과의 차이 | anchor 최대변경 |
|---|---|---|---|
| identity/test | 0.0000, 0.0000, 0.0000, 0.0000 | 0.0000, 0.0000, 0.0000, 0.0000 | 0.0 |
| identity/real | 0.0000, 0.0000, 0.0000, 0.0000 | 0.0000, 0.0000, 0.0000, 0.0000 | 0.0 |
| dynamic/test | 0.0456, 0.0271, 0.0496, 0.0353 | 0.0000, 0.1510, 0.2301, 0.1924 | 0.0 |
| dynamic/real | 0.0562, 0.0128, 0.0386, 0.0267 | 0.0000, 0.1705, 0.2355, 0.1913 | 0.0 |
| node_frame/test | 0.0577, 0.0173, 0.0297, 0.0290 | 0.0000, 0.8443, 0.5584, 0.6607 | 0.0 |
| node_frame/real | 0.0713, 0.0182, 0.0291, 0.0143 | 0.0000, 0.7877, 0.5481, 0.5671 | 0.0 |

Negative는 SO(2) transport의 cos항이 음수인 비율이며 행렬 전체의 부호나 올바른 geometry를 의미하지 않는다. 위 통계는 full을 제외한 secondary source조건까지 포함한다.

구현 검사에서 선형 공통좌표 항등식 오차는4.44e-16, 학습 map의 triangle cycle 오차는1.11e-16, 공유 incidence 오차는0였다. 이는 고정 frame 연산의 검사이며 전체 비선형 PFN의 gauge-equivariance나 성능 보장은 아니다.

## 동일 checkpoint의 개입 결과

| Graph | 원본 % | Feature 제거 % | Feature 셔플 % | Context 셔플 % | Frame 고정 % | Frame I % |
|---|---|---|---|---|---|---|
| Cora | 38.01 | 27.00 | 37.92 | 13.55 | 40.17 | 37.89 |
| CiteSeer | 23.37 | 19.96 | 23.36 | 16.99 | 27.15 | 22.47 |
| texas | 19.92 | 23.68 | 21.70 | 27.98 | 18.04 | 19.21 |
| wisconsin | 24.85 | 19.60 | 22.41 | 17.45 | 23.33 | 21.60 |

학습 seed11 하나의 진단이다. 모든 개입에서 원래 결측 query집합을 고정했다. Frame 고정은 같은 checkpoint에서 첫 단계 map을 재사용하는 조작이고, 별도로 학습한 static 모델과 다르다. 좋은 개입을 골라 새 최종 방법으로 사용하지 않았다.

## 확률 품질과 full feature

| Graph | 모델 | NLL | Brier | Macro F1 % |
|---|---|---|---|---|
| Cora | pcfi100 | 1.53131 | 0.66804 | 49.98 |
| Cora | identity | 1.67889 | 0.74824 | 37.23 |
| Cora | dynamic | 1.65589 | 0.74566 | 38.81 |
| Cora | node_frame | 1.70066 | 0.75678 | 36.89 |
| CiteSeer | pcfi100 | 1.87618 | 0.83783 | 27.91 |
| CiteSeer | identity | 1.85981 | 0.86475 | 21.09 |
| CiteSeer | dynamic | 1.96974 | 0.90354 | 19.11 |
| CiteSeer | node_frame | 1.89108 | 0.87202 | 21.06 |
| texas | pcfi100 | 2.53853 | 0.94260 | 16.96 |
| texas | identity | 2.40182 | 0.94595 | 15.63 |
| texas | dynamic | 2.56458 | 1.02748 | 14.08 |
| texas | node_frame | 2.46448 | 0.98753 | 14.22 |
| wisconsin | pcfi100 | 2.08644 | 0.94634 | 17.69 |
| wisconsin | identity | 1.94318 | 0.87835 | 19.32 |
| wisconsin | dynamic | 2.04402 | 0.90914 | 19.27 |
| wisconsin | node_frame | 2.01442 | 0.90041 | 17.38 |

| Full accuracy % | Synthetic | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|---|
| mask | 96.77 | 36.63 | 26.49 | 30.55 | 21.45 |
| pcfi100 | 97.41 | 38.22 | 26.83 | 31.11 | 26.73 |
| identity | 96.74 | 36.64 | 25.83 | 28.40 | 24.55 |
| static | 96.89 | 37.10 | 26.37 | 29.34 | 22.99 |
| dynamic | 96.67 | 37.63 | 26.92 | 31.12 | 25.40 |
| node_frame | 96.52 | 35.39 | 25.29 | 28.39 | 24.98 |

Full은 전체 query이며 primary와 query집합이 다르므로 절대 차이를 missing의 인과효과로 해석하지 않는다.

## 비용과 재현 검사

| 모델 | Graph | median ms | p10/p90 ms | 추가 GPU MiB |
|---|---|---|---|---|
| identity | Cora | 126.52 | 123.19/130.22 | 235.33 |
| identity | texas | 65.48 | 58.46/74.05 | 21.03 |
| dynamic | Cora | 124.45 | 120.05/127.85 | 235.81 |
| dynamic | texas | 71.44 | 68.49/77.08 | 21.05 |
| node_frame | Cora | 123.81 | 121.08/126.46 | 235.81 |
| node_frame | texas | 66.64 | 62.66/71.83 | 21.05 |

자체 GPU 작업 종료 후 직렬로3회 warmup·10회 측정했다. CPU의 visible RP16 입력부터 CPU 확률까지이며 confidence/frame/전파를 포함한다. 모델 load·최초RP는 제외한다. 다른 사용자 부하는 통제하지 못했다. 이번 구현은 기존 edge sparse 연산을 유지해 공통좌표 전파로 계산을 줄이는 최적화는 하지 않았다.

Audit: PASS_WITH_REPLAY_VARIATION. 입력1034개, 새 예측8484개와3checkpoint를 검증했다. 선택된 재실행의 저장 확률 최대차이는0.0211677, 동일 session 반복 최대차이는0.0177468, 저장 예측 대비 최대argmax변경은0.80%다.

0.1 검사는 큰 손상만 거르는 기준이며 엄격한 수치 재현 통과 기준이 아니다. CI는 저장 예측에 조건부이며 전체 수치 변동을 포함하지 않는다. 초기화·동일taskorder·frozen weights·기존 입력 보존·새test graph 중복 여부도 확인했다.

## 판정 범위

{"synthetic": false, "real_input": false, "negative_lp_floor": false, "expand": false}

이번 비교가 판단하는 것은 현재SO(2)/RP16/공유token geometry/1,024updates에서 node-frame 제약이 기존 edge map과 Identity를 넘어서는가다. 실패 원인을 고유하게 분리한 결과도, 모든 orthogonal frame의 실패/성공도 아니다. 미통과하면 이번 bank를 보고 loss·frame 차원·학습량을 바꾸어 구제하지 않는다.

[계산 정의](METHOD_KO.md), [원점수](results/statistics.json), [검증](results/audit.json), [개입](results/control_summary.json).
