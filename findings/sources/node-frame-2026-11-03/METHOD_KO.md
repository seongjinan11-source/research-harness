# 노드 직교 frame을 사용하는 GraphPFN 비교

2026-11-03. 사용자가 승인한 실험은 synthetic task에서만 학습하고 실제 그래프에서는 가중치를 고정한다. 기존 hidden-token NSD의 입력과 전파 위치를 유지하면서, edge별 restriction을 노드별 frame으로 제한한다. 앞서 제안한 PCFI 입력 복원 및 라벨 토큰 전용 전파는 이번에 함께 변경하지 않는다.

## 노드 frame과 전달 연산

노드마다 `Q_i=R(theta_i)`라는 SO(2) frame 하나를 계산한다. 해당 노드에 접한 모든 edge에서 restriction은 Q_i다. 따라서 `T_ij=Q_i^T Q_j`이고, 같은 단계의 순환 경로에서는 전달행렬의 곱이 I다. 이는 SO(2)라는 제한된 orthogonal family이며 일반 O(d)를 실험한 것은 아니다.

Descriptor는 기존과 같이 feature-token 평균과 label token을 연결해 LayerNorm 및 384→32 projection을 적용하고 관측 비율, context 여부, source 거리 confidence와 logdegree를 더한 36차원이다. 노드 자신의 descriptor와 자신을 포함한 이웃 descriptor의 평균을 연결하고, 기존과 같은 72→32→1 Phi로 각도를 계산한다. 기존 edge 모델은 두 endpoint의 descriptor를 연결했다. 따라서 파라미터는 같지만 conditioning과 함수족은 달라진다.

0/3/6/9층에서 현재 token의 frame을 다시 계산한다. Confidence conductance, degree+1 정규화, 좌우 mixer, ELU residual, step 크기, 관측 feature-pair 및 context-label anchor는 기존 모델과 동일하다. 모든 feature/label token이 같은 frame을 공유한다는 기존 제약도 유지한다.

고정 frame의 선형 연산에서 `u_i=Q_i h_i`로 바꾸면 공통 좌표에서 일반 확산을 한 뒤 되돌리는 연산과 같다. 이 항등식은 비선형 PFN 전체의 동치나 정보 보존을 보장하지 않는다. 직교 변환은 개별 벡터의 norm을 보존하지만 평균에서의 상쇄나 미관측 frame의 식별 문제를 없애지 않는다. 관련 이론은 [Neural Sheaf Diffusion](https://arxiv.org/html/2202.04579)에 있으며, 경로 독립성 자체의 novelty를 주장하지 않는다.

## 동일 학습 조건

새 node-frame 모델 3개만 학습한다. Seed11/12/13마다 기존 dynamic 모델과 named parameter 초기화, trainable 수 4,064,599개, frozen parameter, 512개 train episode와 1,024회 update 순서를 해시로 확인한다. Optimizer, learning rate, AMP, query CE도 유지한다. 기존 Identity/Static/Dynamic/Mask/PCFI checkpoint는 수정하지 않는다. 개발용 smoke 1회는 폐기한다.

학습 원본은 기존 corrected synthetic 64개 graph, 128nodes, 16features, 2~7classes, label-SBM과 feature filter를 교차한 family다. Uniform/clustered90/99.5%, full, uniform50, sources2/8의 8조건을 두 번 처리한다. 실제 graph update, 새 auxiliary loss, teacher는 없다.

## 평가와 사전 판정

새 graph seeds2038111300+i의 64개 합성 graph에서 기존 5개 비교군도 모두 다시 평가한다. 새로운 generator family 검증은 아니다. Primary는 uniform/clustered90/99.5%에서 자기 피처가 없는 query accuracy다. 조건과 3개 고정 학습 seed를 base graph 안에서 평균하고 5,000회 graph-paired bootstrap으로 차이를 구한다.

Node-frame이 Identity와 Dynamic 각각을 CI 하한>0으로 넘고, PCFI보다 평균 accuracy가 높으며, homophily에서 PCFI 대비 하락의 CI 하한이 −2%p를 넘으면 synthetic 기준을 통과한다. Static 차이는 별도로 보고한다. Sources2/8, full, zero-source 분할과 NLL/Brier는 secondary다.

실제 Cora/CiteSeer/Texas/Wisconsin은 이전 RP16 입력, context와 mask, split을 그대로 사용한다. 기존 train split에서 class당 최대4개 label만 context로 제공한다. Node-frame은 새로 추론하고 비교군의 기존 확률은 hash를 검증해 재사용한다. 이는 기존 개발용 bank 평가이며 독립적인 실제 graph 확인이 아니다. Heterophily 평균에서 Identity/Dynamic/PCFI/native-PCFI를 넘고 homophily의 PCFI 대비 하락이2%p 이내인지 본다. Negative LP도 강한 기준으로 유지한다.

Node permutation, 공유 incidence, orthogonality, triangle cycle, 고정 frame 선형 좌표변환 항등식, finite-difference gradient와 실제 optimizer step을 먼저 검사한다. 평가 후 선택된 예측을 재실행하고 수치 변동을 기록한다. 판독한 bank에서 loss, frame dimension, 예산을 조정해 구제하지 않는다.
