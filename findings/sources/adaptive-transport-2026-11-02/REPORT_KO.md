# 정답 법칙 없는 적응 전달 — frozen PFN과 직접 추론 비교

참 법칙을 주지 않는 적응 PCFI→frozen PFN은 사전 primary를 통과하지 못했다. 피처만 쓰는 적응과50:50 평균을 정확도에서 확실히 넘지 못했고, NLL/Brier는 두 대조보다 악화했다. 같은 입력의 직접 Gaussian/probit 추론은 유의한 피처 활용 이득을 보였다. 이번 입력 가설 선택기를 새 PFN 메소드로 확대하지 않는다.

## 설정과 고정한 것

새64 base graphs×32조건=2,048 episodes다. 기존 sign-transport와 같은192-node connected bipartite graph와8개의 상관된 Gaussian-signal column family이며, graph seeds는 새로 생성했다. Label law t와 feature law g를 독립적으로 교차하고, 같은 t에서 g를 바꿀 때 graph·labels·context·query·mask를 고정했다. 개발 graph2개에서 계산을 확인한 뒤 source/config/checkpoint를 lock하고 최종 bank를 생성했다.

기존 matched192-update Mask Plain/PCFI Plain의 checkpoint를 각각 seeds11/12/13으로 재사용했다. 이번 새 모델/optimizer updates는0이다. 여기서 Plain은 공개 GraphPFN 기반 모델이며 과거 소형 typed-relation Plain과 다르다. 이 graph family는 기존 학습 prior 밖이다. 따라서 frozen 실패를 새 prior matched 학습이나 전체 PFN 구조의 실패로 해석하지 않는다.

Context는 클래스당1/4개다. Feature source는 context 밖의 정확히0/1/8개 또는 full이다. 1/192와8/192는 각각99.48%,95.83% whole-node missing이며 Bernoulli99.5% 조건이 아니다. Primary는 sources8·k4/class에서 자기 feature가 없는 query이며 네(t,g) 조합과 세 고정 training seeds를 각 graph 안에서 평균한다. 64개 graph 단위 paired bootstrap5,000회를 사용했다. Training-seed population CI가 아니며 subgroup CI는 다중비교 미보정이다.

## 어떤 적응을 계산했나

Adjacency만으로 bipartition b와 저주파 Gaussian covariance K를 구성한다. 관측행만으로 계산한 leading uncentered singular score를 feature evidence로 사용했다. (g,t,상대 방향 σ,amplitude)의24개 가설에서 Gaussian feature density와 context-label orthant probability를 결합하고, 같은 evidence로 query sign probability를 계산한다. Context 적분은 고정 Sobol/antithetic2,048 particles다. 법칙·query 정답·hidden feature는 적응 입력에 주지 않았다.

Feature-only q(g)는 labels/context를 읽지 않는 별도 함수다. Joint q(g)는 context labels까지 이용하고 t·σ·amplitude를 주변화한다. PFN 경로는 일반 PCFI와 parity-PCFI 두 completion에 같은 checkpoint를 각각 실행해 q(g)로 확률을 평균한다. 50:50 평균도 동일한 두 PFN branches를 사용한다. 참 g를 받은 reference는 별도 평가용이며 실행 가능한 적응 성과가 아니다.

직접 Gaussian/probit 대조는 같은24개 가설에서 query 확률을 직접 주변화하며 PFN을 사용하지 않는다. 이는 알려진 family의 Gaussian covariance/score-noise 가정을 사용하는 강한 통계 대조다. 모델은 원8columns의 joint, RMS 압축, median threshold, class-balanced context 선택의 정확한 likelihood가 아니다. 원 데이터 exact Bayes posterior·calibration·새 neural 추론의 성과로 부르지 않는다. CPGNN/CLP 재현도 아니다.

## Primary: 네 label/feature 법칙 평균

| 방법 | Accuracy % | NLL | Brier |
|---|---:|---:|---:|
| 기존 Mask Plain | 52.357424 | 0.705444 | 0.509802 |
| 기존 PCFI Plain·일반 복원 | 51.565242 | 0.715365 | 0.518345 |
| 같은 PCFI Plain·고정 반전 복원 | 52.205386 | 0.719126 | 0.520209 |
| 같은 PCFI Plain·50:50 두 복원 평균 | 52.354767 | 0.702633 | 0.508327 |
| 같은 PCFI Plain·피처만 사용해 적응 | 52.461961 | 0.704598 | 0.509727 |
| 같은 PCFI Plain·피처와 context로 적응 | 52.555985 | 0.705381 | 0.510321 |
| 참 g 복원 reference (privileged) | 52.292214 | 0.715630 | 0.517637 |
| 양의 LP | 52.796435 | 0.733438 | 0.523781 |
| Signed LP | 53.113760 | 0.765261 | 0.545701 |
| Context 적응 LP (graph-only) | 53.046855 | 0.694709 | 0.500752 |
| 직접 Gaussian/probit 추론 | 56.154827 | 0.672120 | 0.479871 |
| 같은 Gaussian/probit·피처 제거 | 53.687863 | 0.681201 | 0.488462 |
| 참 t LP reference (privileged) | 57.135862 | 0.688915 | 0.493230 |
| Uniform | 49.302684 | 0.693147 | 0.500000 |

적응 PCFI→PFN에서 대조를 뺀 paired 차이다. Accuracy 단위는%p이며 NLL/Brier는 낮을수록 좋다.

| 대조 | Accuracy 차이 [95% CI] | NLL 차이 [95% CI] | Brier 차이 [95% CI] |
|---|---:|---:|---:|
| 같은 PCFI Plain·50:50 두 복원 평균 | +0.201218 [95% CI -0.127618, +0.537932] | +0.002748 [95% CI +0.000562, +0.005176] | +0.001993 [95% CI +0.000218, +0.003933] |
| 같은 PCFI Plain·피처만 사용해 적응 | +0.094024 [95% CI -0.066909, +0.271023] | +0.000783 [95% CI +0.000185, +0.001435] | +0.000594 [95% CI +0.000087, +0.001160] |
| 기존 Mask Plain | +0.198560 [95% CI -1.302255, +1.684394] | -0.000063 [95% CI -0.009120, +0.008721] | +0.000519 [95% CI -0.007726, +0.008507] |
| 기존 PCFI Plain·일반 복원 | +0.990743 [95% CI +0.151828, +1.837583] | -0.009984 [95% CI -0.016127, -0.003995] | -0.008024 [95% CI -0.013078, -0.003209] |
| 같은 PCFI Plain·고정 반전 복원 | +0.350599 [95% CI -0.489938, +1.167805] | -0.013746 [95% CI -0.020602, -0.007319] | -0.009888 [95% CI -0.015299, -0.004838] |
| Context 적응 LP (graph-only) | -0.490871 [95% CI -2.550722, +1.614366] | +0.010672 [95% CI +0.002382, +0.018880] | +0.009569 [95% CI +0.001685, +0.017316] |
| 직접 Gaussian/probit 추론 | -3.598842 [95% CI -5.635105, -1.536745] | +0.033261 [95% CI +0.025097, +0.041242] | +0.030450 [95% CI +0.022878, +0.037792] |

사전 primary=False, strong floor=False다. 일반 PCFI 하나와의 개선은 보존하되, 같은 두 completion의 고정 평균과 feature-only 적응을 넘지 못한 결과를 새 context 적응 메소드의 성공으로 바꾸지 않는다. 참 g를 받는 reference도 강한 classification upper bound가 아니다. True feature law에 맞게 복원하는 것과 frozen classifier의 최종 예측에 좋은 입력은 다를 수 있다.

## 피처의 예측 기여: query 집합을 고정한 제거 대조

양쪽 예측 모두 원8-source primary의 missing-query 집합으로 제한했다. 관측 source 수가 바뀌면서 query 집합도 바뀌는 효과를 섞지 않았다.

| 경로 | 피처 있음 accuracy % | 피처 제거 accuracy % | 차이%p [95% CI] | NLL 차이 [95% CI] |
|---|---:|---:|---:|---:|
| 같은 PCFI Plain·피처와 context로 적응 | 52.555985 | 52.959920 | -0.403936 [95% CI -1.874849, +0.994962] | -0.051815 [95% CI -0.064284, -0.039279] |
| 직접 Gaussian/probit 추론 | 56.154827 | 53.687863 | +2.466964 [95% CI +0.884420, +4.166715] | -0.009081 [95% CI -0.013619, -0.004556] |

직접 통계 추론은 피처가 context와 분리된 상태에서도 accuracy·NLL·Brier의 추가 이득을 보였다. 반면 frozen PFN의 적응 completion은 피처 제거 대비 NLL/Brier를 개선했지만 accuracy 우위를 확인하지 못했다. 이 차이는 통계 모형/branch decoder/기존 prior mismatch 등 여러 요소가 섞여 있어 고유 원인을 분리한 결과는 아니다.

## Label/feature 법칙별 보조 결과

Sources8,k4/class. 이 네 조건의 평균이 위 primary이며 가장 좋은 한 cell로 primary를 교체하지 않는다.

| t | g | 일반 PCFI % | 50:50 % | 피처만 적응 % | Context 적응 % | 직접 Gaussian % | 적응 LP % |
|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | 54.179220 | 53.778113 | 53.951219 | 54.020622 | 55.815288 | 52.906972 |
| 0 | 1 | 53.473672 | 52.828206 | 52.696965 | 52.839194 | 55.815288 | 52.906972 |
| 1 | 0 | 49.351217 | 50.670317 | 50.627936 | 50.703210 | 56.494366 | 53.186739 |
| 1 | 1 | 49.256859 | 52.142430 | 52.571722 | 52.660912 | 56.494366 | 53.186739 |

## 법칙 식별과 source/context 수

현재 family는 feature column의 숨겨진 방향이 대칭적이다. Source0/1에서는 g를 바꾼 관측값의 부호를 숨겨진 orientation으로 흡수할 수 있어 g에 대한 추가 정보가 없다. Context를 넣어도 q(g)는0.5로 남았다. 이 식별 불가능성을 복잡한 router로 해결할 수 있다고 가정하지 않는다. 이는 현재 family의 g에 관한 결과이며 임의 데이터의 모든 one-source 식별 불가능성을 주장하지 않는다.

| k/class | Sources | Feature-only g 정확도 % | Joint g 정확도 % | Label-only t 정확도 % | Joint t 정확도 % |
|---|---|---:|---:|---:|---:|
| 1 | 0 | 50.000000 | 50.000000 | 48.828125 | 48.828125 |
| 1 | 1 | 50.000000 | 50.000000 | 48.828125 | 48.828125 |
| 1 | 8 | 68.359375 | 69.140625 | 48.828125 | 47.265625 |
| 1 | full | 100.000000 | 100.000000 | 48.828125 | 69.140625 |
| 4 | 0 | 50.000000 | 50.000000 | 60.937500 | 60.937500 |
| 4 | 1 | 50.000000 | 50.000000 | 60.937500 | 60.937500 |
| 4 | 8 | 60.156250 | 59.375000 | 60.937500 | 64.062500 |
| 4 | full | 100.000000 | 100.000000 | 60.937500 | 100.000000 |

법칙 정답은 이 사후 진단에만 사용했다. 수치상0.5 근처(1e−8)의 tie는 기대 정확도0.5로 집계했다. 이 보조 규칙은 weight 결과를 보기 전에 runtime/latent_accuracy_addendum.json에 기록했고 원 statistics/weights/예측을 보존했다. 실제 query argmax accuracy와 primary는 수정하지 않았다. 원 statistics의 strict q>0.5 latent accuracy 대신 latent_accuracy_audit.json을 참조한다. Source0/1의 g 확률은 최대 약4.4e−16만큼0.5와 달랐다.

8-source,k4에서 context가 g의 hard identification을 개선했다는 결과는 없다. Joint t 점추정이 label-only보다 높더라도 latent 식별 정확도를 최종 PFN 분류 개선으로 바꾸지 않는다. 같은 Gaussian 추론기가 직접 예측한 결과와 PFN 입력 가중치로 쓴 결과가 다르다.

## 비용

| 경로 | Median latency ms | 추가 peak GPU work memory MiB |
|---|---:|---:|
| 기존 Mask Plain | 57.775150 | 13.604 |
| 기존 PCFI Plain·일반 복원 | 63.089950 | 13.609 |
| 같은 PCFI Plain·50:50 두 복원 평균 | 128.198050 | 13.609 |
| 같은 PCFI Plain·피처만 사용해 적응 | 129.475550 | 13.609 |
| 같은 PCFI Plain·피처와 context로 적응 | 224.979750 | 13.609 |
| Context 적응 LP (graph-only) | 88.662100 | 0.000 |
| 직접 Gaussian/probit 추론 | 85.282000 | 0.000 |

RTX3090, seed11/첫 primary input,5warmups/30repeats다. CPU visible input→CPU 확률이며 kernel/PCA/orthant와 실제 PCFI/PFN branches를 포함하고 model load·bank I/O는 제외한다. Across-graph kernel cache는 없다. 같은 PCFI checkpoint를 쓰지만 적응/50:50는2회 PFN 실행이 필요하다. External GPU load는 통제하지 않았고 메모리는 추가 작업량이지 총 GPU 메모리가 아니다. 단일 입력의 비용을 일반 scaling 우위로 확대하지 않는다.

## 검증과 보존

저장된 neural variant probabilities 43,008개를 재채점했다. Classical controls 14,336개는 finite/정규화·metric 계산을 확인했다. 그중 label-only 네 대조는 graph/label/context가 같은256그룹에서 피처-law/source 변경에 불변인지 확인했다. 저장된 PFN mixture 재계산 오차는 0였다. Neural replay18건, working-model replay3건, 이전 sign-probe checkpoint replay18건을 확인했다. Source/config/입력/예측/checkpoint SHA와 모델 파라미터 불변, 새/기존 graph overlap0, source0/1 대칭성을 확인했다.

개발 검사에는 해석 가능한 Gaussian orthant 해, class permutation/global gauge, node permutation 수치오차,2048→4096 particle 근사 비교와 strict feature-only label blindness가 포함된다. 일부 classical 확률이 합산 roundoff로[0,1]을 최대1.47e−14 벗어나 audit의1e−12 수치 허용치로 기록했다. Raw probabilities를 바꾸지 않았으며 합 정규화 오차는8.89e−16 이하다.

## 판정과 닫지 않은 범위

참 법칙을 주지 않는 적응 PCFI→frozen PFN은 사전 primary를 통과하지 못했다. 피처만 쓰는 적응과50:50 평균을 정확도에서 확실히 넘지 못했고, NLL/Brier는 두 대조보다 악화했다. 같은 입력의 직접 Gaussian/probit 추론은 유의한 피처 활용 이득을 보였다. 이번 입력 가설 선택기를 새 PFN 메소드로 확대하지 않는다.

닫는 것은 이번 Gaussian/probit q(g)를 기존 frozen PCFI classifier의 두 completion 가중치로만 사용하는 구현이다. 모든 adaptive propagation, 새 prior에서 직접 학습한 PFN, context-conditioned feature inference를 기각하지 않는다. 반대로 아직 하지 않은 재학습을 이번 실패의 해결책이라고 단정하지도 않는다. 새 구조를 논의하려면 알려진 법칙/고정 Gaussian 추론이 이미 하는 기능과 구분되는 연산, 강한 직접 추론 대조, 동일-query 피처 활용·accuracy·비용 기준을 먼저 제시해야 한다. 이 bank에서 temperature/noise/amplitude/particle/checkpoint를 조정하거나 새 head/loss로 구제하지 않았다.
