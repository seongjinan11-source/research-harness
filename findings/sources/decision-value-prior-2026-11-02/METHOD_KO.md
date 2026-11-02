# J / V 관측 prior — 실행한 최소 비교

2026-11-02. 사용자가 승인한 두 후보는 entropy 감소 `J`와 Bayes 분류 오류 감소 `V`를 이용한 **학습 mask prior**다. 배포 때 source를 선택하거나 새 관측을 획득하지 않는다. Architecture와 query CE는 기존 공개 GraphPFN wrapper 그대로다. 정확한 설정은 [protocol.json](protocol.json), 변경 불가능한 학습·생성 source hash는 `results/lock.json`에 있다.

## Teacher와 생성기의 일치

이전 Gaussian/probit working model은 원래 RMS·median·balanced-context 생성기의 exact posterior가 아니다. 이를 재사용하는 대신 이번에는 **64개 latent state의 유한 mixture**를 정의했다. 이는 이전 192-node family와 다른 작은 known-family 원리 검증이다.

96-node 연결 bipartite graph의 normalized adjacency에서 nonconstant 저주파 고유벡터 세 개를 구한다. 각 열의 RMS를1로 한 행렬을 `U`라 하고, `z∈{−1,+1}³`를 균일하게 뽑아 `v=Uz/√3`로 둔다. Graph로 결정되는 bipartition을 `b`라 한다. Feature parity `g`, label parity `t`, label orientation `σ`는 서로 독립이다. 세 부호계수와 세 이진 변수를 모두 열거하면64가설이다.

\[
X_{i1}=v_i b_i^g+\epsilon_{i1},\quad X_{i2}=0.6v_i b_i^g+\epsilon_{i2},\quad \epsilon\sim N(0,0.4^2),
\]

\[
Y_i\mid h\sim\mathrm{Bernoulli}\{\mathrm{sigmoid}(2\sigma v_i b_i^t)\}.
\]

주어진 가설에서 feature noise와 node labels는 독립이다. Context 위치는 label을 보기 전에 뽑는다. 총 context2/8개이며 클래스당 개수가 아니다. Class balancing·median threshold·observed-only RMS는 적용하지 않는다. Reference는64가설의 Gaussian density와 context Bernoulli likelihood를 직접 정규화한다. **Posterior 열거는 exact지만 J·V의 기대값은 Monte Carlo 근사**다. 고정 finite family를 모르는 상황이나 임의 Gaussian process의 exact teacher라고 부르지 않는다.

## 두 관측 설계

Context `C`, query24개 `Q`, source 후보12세트를 graph RNG만으로 정한다. Source 수1/4/8을 맞추며 `C,Q,O`는 겹치지 않는다. 후보마다 실제 학습 realization과 독립인256개 prior draw에서 계산한다.

\[
\widehat J(O)=\frac1{256|Q|}\sum_{m,q}\mathrm{KL}\{p(Y_q|Y_C,X_O^{(m)},G)\Vert p(Y_q|Y_C,G)\},
\]

\[
\widehat V(O)=\frac1{256|Q|}\sum_{m,q}[\max_c p(Y_q=c|Y_C^{(m)},X_O^{(m)},G)-\max_c p(Y_q=c|Y_C^{(m)},G)].
\]

유한 Monte Carlo에서 V가 음수로 나오더라도 원값을 남긴다. 두 prior 모두 확률0.5로 uniform 후보를 쓰고, 나머지0.5에서는 각 점수 상위3개 중 하나를 균일하게 쓴다. 단순 geometry도 같은 mixture를 사용하되 context/query에서 source까지 `exp(−shortest distance)`의 평균으로 순위를 정한다. **별도 RNG의 실제 X/Y realization**을 모든 prior에 공유한다. 구현은 graph마다 이 realization을 먼저 생성하지만, mask 선택 함수는 그 값·정답·latent state를 인자로 받지 않는다. 따라서 mask는 실제 realization과 독립이다. Protocol의 시간 순서 설명과 실제 실행 순서의 차이는 `runtime/scope_errata.json`에 남겼다. 선택에 사용한 정보나 분포는 바뀌지 않았다.

## 실제 학습과 대조

`random / geometry / entropy / decision / random_cost` × `Mask / PCFI` ×3 seeds =30개의 새 checkpoint다. 기존 wrapper의4,012,259개 trainable parameter와 초기화를 맞춘다. 일반 arm192updates, random_cost384updates. 각 context×source6cell을 같은 횟수 노출하고, 일반 arm은 같은 graph/setting 순서·forward RNG·학습률을 쓴다. 마지막 checkpoint만 평가한다. 추가 loss/head, reconstruction, teacher distillation, target 재학습은 없다.

비용 대조는 실측 cold scoring 및 bank generation 시간을 포함한다. Random_cost의 추가 학습 시간이 이 비용을 실제로 덮는지 별도로 확인한다. 추론 때는 J/V scorer가 없고 입력 interface에 따른 기존 PCFI 또는 Mask wrapper만 실행한다.

## 평가와 판정

새64graph의 모든 모델에 동일한 random 관측 배치를 준다. Primary는 source4/96개, 총 context8개의 query accuracy다. 즉95.83% whole-node missing이며99.5% Bernoulli 조건과 다르다. Query는 모두 자신의 피처가 없다. 총3개의 고정 학습 seed를 graph 안에서 평균한 뒤64graph paired bootstrap5000회를 사용한다.

J와 V 각각 Mask에서 random·geometry·random_cost 세 대조를 모두 정확도 CI 하한>0으로 넘어야 primary 통과다. 그다음 같은 family에서 학습한 PCFI arm들을 강한 pipeline 대조로 본다. Exact-family reference는 이미 family를 아는 정보 기준이며 학습기가 기대상 넘어야 하는 대상으로 설정하지 않는다. 같은 query의 피처 제거·셔플, 다른 source/context budget, clustered mask, full feature, NLL/Brier는 분리해 보고한다. Pointwise CI이며 두 후보와 여러 대조에 대한 multiplicity 보정은 하지 않은 exploratory pilot다.

J/V가 실제로 다른 mask를 얼마나 고르는지, 자체 scoring sample에서만 점수가 좋아진 것은 아닌지도 보고한다. 독립4096-draw 값 감사는 첫16개 train graph의 primary 설계에 한정하고, mask·하이퍼파라미터 수정에 사용하지 않는다. 과거 bank의 정보 부족을 이번 실험으로 소급해 확정하지 않는다.
