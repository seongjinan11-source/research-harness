# Confidence 가중 PCFI의 공간 전파를 Sheaf로 교체하는 실험

2026-11-03. 관측 숫자 피처에서 출발해 confidence 가중 graph sheaf diffusion으로 결측값을 추정하고, 그 결과를 PFN에 입력한다. PFN hidden token을 전파했던 앞선 NSD 실험과 구분한다. 실제 graph에서 가중치를 갱신하지 않는다.

## 유지하는 PCFI 계산

원 PCFI의 거리 기반 pseudo-confidence `c_if=.9^distance(i,observed_f)`와 self-loop를 유지한다. Source가 없는 column과 도달 불가능한 거리는 기존 CPU 구현의 convention을 그대로 쓴다. Confidence는 보정된 확률이 아니다. 양의 가중치는 `P_ij,f=A_ij*c_jf/sum_k A_ik*c_kf`이며 모든 방법에서 동일하다.

관측 cell은 `(x_if,0)`, 결측 cell은 `(0,0)`에서 시작한다. 총100회만 전파한다. 매 반복 관측 cell의 두 좌표는 `(원래 관측값,0)`으로 고정한다. 첫 좌표를 읽어 원 PCFI의 채널 상관 보정을 마지막에1회 적용하고 관측값을 다시 고정한다. 실제 CPU imputer에는 마지막 출력 clip이 없으므로 이를 추가하지 않는다. 관측 입력을 만드는 기존 RMS/clip 전처리는 유지한다.

## 교체하는 공간 메시지

결측 위치에서 `z_if^(t+1)=sum_j P_ij,f*T_ij,f*z_jf^t`로 갱신한다. Edge 방식은 `T_ij,f=R(theta_j,e,f-theta_i,e,f)`다. 역방향은 transpose이고 self-loop는I다. Node 방식은 노드/column당 frame 하나를 공유해 `T_ij,f=Q_if^T Q_jf`로 제한한다. 두 방식 모두 SO(2)의 직교 transport이며100회 전파 동안 map은 고정한다.

같은 confidence의 대칭 conductance를 `a_ij,f=A_ij*c_if*c_jf`로 잡으면 P는 그 conductance의 row normalization이다. 따라서 원 PCFI의 가중치를 유지한 connection random walk로 볼 수 있다. 각 transport의 norm은1이지만 평균에서 상쇄될 수 있고, 뒤의 채널 상관 보정까지 직교 연산이라는 뜻은 아니다.

Scalar 대조는 Edge와 같은 angle 함수를 쓰되 cross-stalk sin 항을 제거하고 cos 계수만 사용한다. 학습한 두 번째 좌표의 전달이 단순 scalar signed 전파를 넘는지 확인한다. PCFI 대조는 T=I이며 동일 Torch kernel을 사용한다. 개발 검사에서 I 환원은 정확히 같고 원 NumPy PCFI와의 최대 차이는2.98e-7이었다.

## Map의 입력과 학습

각 cell의 관측값·절댓값·관측 이웃 평균·mask·confidence·degree·context 여부 및 context-only LP의 entropy와 최대 support, context끼리의 class agreement를 요약한다. Edge 방식은 두 endpoint의 descriptor와 context affinity를, Node 방식은 자기/이웃 요약과 평균 affinity를 입력받는다. 모든 학습 arm은 같은737-parameter 공유 MLP(21→32→1)를 사용한다. 피처를 미리100회 채운 warm-up, 숨은 피처, query 정답, 참 feature law/homophily, PFN feedback은 입력하지 않는다.

Map은 I 근처의 작은 비영 초기값이다. 정확한I에서 scalar를 `(x,0)`으로 올려 첫 좌표만 읽으면 angle의 첫 미분이0이 될 수 있어, I로 강제한 환원 검사와 학습 초기화는 구분한다.

Seed11/12/13의 기존 synthetic-only PCFI-PFN checkpoint를 뒤의 분류기로 사용한다. 모든 PFN 파라미터를 학습 때도 고정하고 eval mode를 유지한다. 새로운 학습 대상은 대치 연산의737개 파라미터뿐이다. 같은 seed의 모든 arm은 정확히 같은 PFN checkpoint를 사용하며, query CE의 gradient는 PFN을 통과해 frontend에만 흐른다. 이 실험은 operator의 독립 효과를 보는 것이고 joint PFN 재학습의 효과는 평가하지 않는다.

기존512개 synthetic train episodes와 seed별 순서를 유지해 Scalar/Edge/Node 각각1,024회 학습한다. 총9개 frontend,9,216updates다. PCFI는 새 학습할 파라미터가 없으며 기존 PFN의 사전1,024회 학습 이력은 모든 arm에 공통이다. AdamW·lr.001·16step warm-up·cosine·clip1·AMP와 query CE를 사용한다. 새 auxiliary loss나 teacher는 없다.

## 평가와 판정

동일 family의 새64base graph(seeds2039111300+i)를 사용한다. Uniform/clustered whole-node90/99.5%의 자기 피처 없는 query accuracy가 primary다. 4조건과3고정학습seed를 base graph 안에서 평균하고5,000회 graph-paired bootstrap을 계산한다. Edge와 Node는 사전 지정한 별도 후보로 모두 보고하고 가장 좋은 후보만 골라 성공으로 바꾸지 않는다. 두 후보와 여러 대조의 CI는 다중비교 미보정이다.

각 후보는 PCFI와 학습된 Scalar를 각각 CI 하한>0으로 넘고 homophily의 PCFI 대비 CI 하한이−2%p를 넘어야 한다. 실제4graph의 heterophily 평균은 PCFI/Scalar/native-PCFI보다 높고 homophily 하락은2%p 이내인지 본다. Negative LP도 강한 기준으로 유지한다. Source2/8, zero-source, full feature, NLL/Brier, 원래 query를 고정한 feature/context 개입 및 비용은 별도로 보고한다.

실제 graph는 Cora/CiteSeer/Texas/Wisconsin의 기존 RP16·context·mask·split 개발 bank다. 기존 train mask에서 class당 최대4개 context만 제공하고 query/validation label은 적응 입력에 쓰지 않는다. 모든 실제 가중치는 고정한다. Native-PCFI는 원본 고차원 피처에서 먼저 전파하므로 동일 표현의 geometry-only 대조가 아닌 전체 pipeline floor다.

단계별 관측값 고정, source0의 정확한0 출력, I 환원, node permutation, frame 공유/cycle 및 역방향 일관성, frozen PFN gradient와 실제 optimizer step을 검사한다. 평가 후 입력·예측·checkpoint 해시와 선택된 예측의 재실행 변동을 공개한다. 이 bank를 보고 loss·차원·전파 횟수·학습 예산을 조절해 구제하지 않는다.
