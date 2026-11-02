# Confidence-weighted Sheaf-PCFI GraphPFN: 첫 zero-shot pilot

2026-11-03. 사용자가 선택한 범위는 **synthetic task에서만 학습하고 실제 graph에서는 가중치를 고정하는 PFN**이다. Heterogeneous graph는 제외한다. 이 문서는 실험 전 계산 정의이며 성공·novelty를 미리 주장하지 않는다.

## 기존 PCFI와 유지하는 부분

기존 PCFI의 channel shortest-path pseudo-confidence `c_if=.9^distance(i, observed_f)`, 매 반복의 관측값 clamp, 마지막1회 inter-channel correlation을 유지한다. 거리가 무한하거나 source가 없는 column의0 convention도 원 구현과 맞춘다. Confidence는 calibrated probability가 아니다. 원 [PCFI](https://arxiv.org/abs/2305.16618)의 전체 알고리즘과 표준 [Neural Sheaf Diffusion](https://arxiv.org/html/2202.04579)은 서로 다른 방법이며, 이번 구현은 둘의 특정 연산을 결합한 작은 pilot이다.

일반 PCFI의 공간 전파는 다음이다.

`P_ij,f = A_ij c_jf / sum_k A_ik c_kf`.

Self-loop를 포함한다. 먼저 일반 공간 전파100회로 warm 값을 만든다. Learned router의 입력은 이 값과 관측 mask·confidence·degree·context-only LP support의 대칭 요약이다. Query 정답, hidden feature, 참 feature law, 참 homophily는 주지 않는다. Context LP는 입력 통계이며 출력 offset이나 예측 혼합으로 사용하지 않는다.

## 실제 변경: scalar를2차원 stalk로 lift

각 physical cell을 `z_if=(x_if,0)`으로 lift한다. 관측 source는 매 반복 `(실제 관측값,0)`으로 clamp한다. 처음100회 warm 이후100회 추가 전파에서 다음 symmetric contraction을 사용한다.

`T_ij,f = (1-gamma) I + gamma Q(theta)`

`Q(theta)=[[cos(theta),sin(theta)],[sin(theta),-cos(theta)]]`.

`theta=pi*tanh(router0)`, `gamma=sigmoid(router1)`다. 모든 edge/channel에 같은459-parameter MLP(15→24→3)를 공유한다. Router 입력은 endpoint 교환에 대칭이므로 `T_ij=T_ji=T_ij^T`다. Self-loop는I다. 세 번째 출력은 `exp(.5*tanh(router2))`의 양수 edge weight이며 confidence weight와 곱한 뒤 다시 row-normalize한다.

이는 단순히 행렬이라는 이름을 붙인 signed 평균이 아니다. 두 번째 stalk가 반복 사이에 정보를 유지하며, 첫 좌표로 되돌아올 수 있다. 반면 scalar 대조는 동일한 첫 대각 성분만 사용하고 cross-stalk 전달을 제거한다. 첫 번째 좌표를 numeric completion으로 읽은 뒤 PCFI의 원래 inter-channel 보정을1회 적용한다.

`Q`는 orthogonal reflection이고 `0<=gamma<=1`이므로 `||T||_2<=1`이다. 따라서 각 edge의 quadratic form `||z_i||²+||z_j||²-2z_i^T T z_j`는 PSD다. 이는 `R_i=[I;0]`, `R_j=[T;sqrt(I-T^T T)]`라는4차원 edge-space restriction maps로 실현할 수 있다. Confidence conductance `A_ij c_if c_jf exp(score)`를 사용한 weighted sheaf Jacobi 전파로 해석할 수 있다. 원 NSD 전체 아키텍처의 재현은 아니다.

## 비교군과 계산량

| Arm | 공간 전파·입력 |
|---|---|
| mask | Numeric missing은0, 명시적mask; 같은 PFN |
| pcfi100 | 일반PCFI100회+correlation1회 |
| pcfi200 | 일반PCFI200회+correlation1회 |
| weighted | warm100+양수 가중치 조절100, T=I |
| signed | warm100+동일 router의 signed scalar100 |
| sheaf | warm100+2차원 cross-stalk100 |

실제공간계산에서 learned 방법은200회이며 원PCFI100보다 비싸다. pcfi200과 weighted·signed를 함께 비교해 추가 반복·양수 가중치·부호 허용·cross-stalk 효과를 구분한다. Static arm에도 동일 front parameter container가 있지만 사용하지 않는 weight는 gradient를 받지 않는다. 파라미터 container 개수가 같다는 것을 유효 함수 용량이 같다는 뜻으로 쓰지 않는다.

최종 예측은 모두 같은 공개GraphPFN1.3/LimiX wrapper를 사용한다. Transformer는 기존 policy대로 고정하고 graph adapter·encoder/decoder·공통 mask/TokenControl 및 learned router를 synthetic query CE만으로 학습한다. 초기화·task bank·노출 순서·384successful updates를 맞춘다. 세 fixed seed를 사용한다. 새 posterior head·distillation teacher·auxiliary loss는 없다.

## Synthetic prior

128nodes·16features·2~7classes의64train graphs를 사용한다. Graph는label-SBM에서 동종 선호6 또는 이종 선호.15와 expecteddegree8을 사용한다. 독립적으로 feature law의 graph filter 계수lambda를+.65/-.65로 바꾼다. Feature는 `(I-lambda P)^-1(classmean+.35noise)`이며 실제 feature energy와 label agreement를 기록한다. 명목 law 이름과 달성된 통계는 구분한다.

Train graph마다 uniform90/99.5%,cluster90/99.5%,full,uniform50의6episodes,총384개를각seed에서한번씩사용한다. Context는class당1/4개다. Dev2graphs의smoke optimizer step은폐기하고새32test graphs를사용한다. 실제graph에대한학습·validation선택은없다.

## 실제 graph와 정보 경계

로컬 Cora·CiteSeer 및 Texas·Wisconsin의 원본 graph를 사용하며 LCC만 골라내지 않는다. Planetoid는public split,WebKB는기존첫3split을사용한다. 각 split에서 기존train mask에속한노드만class당최대4개context로선택한다. 나머지train label과validation label은입력하지않는다. 평가는기존test mask다.

모든 row를 읽어 data bank를 만들지만 predictor에는 관측값·mask·context label만 전달한다. RP16은 데이터 값과 무관하게 사전 고정된Rademacher matrix이며 관측된 full row에만 적용한다. Hidden X로PCA나scaler를fit하지않는다. Compressed observed values의RMS만사용하고column centering은하지않는다. 이 첫pilot은전체고차원원본공간의Sheaf-PCFI검증이아닌**고정RP16인터페이스**다.

압축이원PCFI를불리하게만드는지보기위해같은pcfi100checkpoint에 **원본feature에서PCFI100→같은RP16** 입력을주는강한추가대조를유지한다. Full-feature는별도reference다. 실제전체bank는결과로선택한독립최종benchmark가아니며이4graph를개발용pilot으로기록한다.

## Missing·평가·제한

Uniform은Bernoulli whole-node90/99.5%,cluster는BFS-nearest의고정개수관측source다. 작은Texas에서99.5%는관측0개인경우도있다. 이를재추출하거나최소source를강제하지않는다. Mask와query집합을고정한feature제거·shuffle대조를별도로본다.

Primary는자기feature가없는test query accuracy다. Synthetic CI는base graph단위paired bootstrap5000회이며3trainingseed는fixed반복이다. Real4graph는dataset별점수와반복paired차이를보고하고노드를독립표본처럼bootstrap하지않는다. Full-feature성능·NLL·Brier·macroF1·시간을별도로본다. Reconstruction은숨겨진값을평가때만사용한다.

Synthetic gate는Sheaf가pcfi100/200/weighted/signed를모두CI하한>0으로넘고homo harm2pp이내인것이다. Real은hetero그룹pcfi100/signed대비이득과homo harm2ppguard를함께확인한다. 네개developmentgraph의이득을새schema·임의heterophily·모든missing으로확대하지않는다. PCFI가labelheterophily에서항상나쁘다는전제도두지않는다.

이pilot이통과해도FSD-CAP·CPGNN/CLP·원NSD의동일protocol재현까지이겼다는뜻은아니다. 이들은후속강한대조로남고특히target-graph학습을사용하는방법은zero-shotPFN과학습체계/비용을구분해야한다. 기존fixedGaussian/Grid6의범용실패나Sheaf의새원리를입증했다는주장도하지않는다.
