# Disjoint evidence를 연결하는 factor PFN의 계산

이 pilot은 공개 GraphPFN에 새 head를 붙인 모델이 아니라, synthetic graph task들에서 학습하는 약8천 파라미터의 별도 structured predictor다. Target graph에서는 parameter update 없이 관측값과 context label을 넣어 추론한다. 고정된 binary family의 원리 시험이며 범용 graph foundation model이 완성됐다는 뜻은 아니다.

## 입력과 잠재변수

입력은 `D=(G, X_obs, M, Y_C, known_family)`다. Query 정답, helper label, true density/signal, true relation role와 feature orientation은 입력하지 않는다.

각 원본 node에 class spin `s_i∈{-1,+1}`, 각 feature column에 `z_f∈{-1,+1}`, 전체 task에 relation-role hypothesis `h∈{00,10,01,11}`를 둔다. 알려진 family에 따라 h의 prior mass가 달라진다. 모든 학습 대조도 같은 family descriptor를 받는다.

## Conditional factor model

고정된 관측 입력에 대해 다음 structured distribution을 정의한다.

\[
\widetilde p_\theta(s,z,h\mid D)\propto
\pi(h)\,\mathbf1[s_C=2Y_C-1]
\exp\!\left[
\sum_{r,(i,j)\in\mathcal P_r}
\big(a_{ijr}^{(h)}+J_{ijr}^{(h)}s_i s_j\big)
+\sum_{i,f:M_{if}=1}\kappa_{if}s_i z_f
\right].
\]

`P_r`는 해당 schema에서 유효하며 edge/nonedge 상태를 모두 아는 native target–helper pair다. Schema상 존재하지 않는 pair를 observed nonedge로 세지 않는다.

Graph-only shared encoder와 pair network가 relation별 same/different class의 edge 확률을 만든다. 관측된 edge 또는 nonedge의 full log factor를 `a+J*s_i*s_j`로 바꾼다. **Role을 비교할 때 상쇄되지 않는 상수 a도 유지한다.** 이 확률은 현재 graph를 보는 network의 출력이므로 참 생성 likelihood를 복원했다는 주장은 하지 않는다.

관측 feature factor는 다음 형태다.

\[
\kappa_{if}=\operatorname{clip}\!\left(
x_{if}\,\operatorname{softplus}
g_\theta(|x_{if}|,\mathrm{RMS}_f,\mathrm{count}_f,h_i^G),-8,8\right).
\]

실제 구현에는 작은 양의 최소 계수를 더한다. 같은 함수를 모든 column에 공유한다. Column 값의 부호가 바뀌면 latent orientation의 해석이 바뀌고 label 예측은 보존된다. 관측하지 않은 cell에는 factor를 만들지 않는다.

이 모델은 결측 X의 수치 completion을 생성하지 않는다. Missing continuous feature를 정확히 적분하는 generative circuit도 아니다. 원래 cyclic graph의 참 posterior를 정확히 계산한다고 주장하지 않는다.

## 추론과 학습

유한24회 damped cavity BP를 실행한다. 일반 변수에서 Ising message update는 다음과 같다.

\[
u_{i\to j}^{t+1}=(1-\gamma)u_{i\to j}^{t}
+\gamma\operatorname{atanh}\!\left[
\tanh J_{ij}\,
\tanh\!\left(\sum_{k\ne j}u_{k\to i}^{t}\right)
\right],\qquad\gamma=0.5.
\]

Context node에서는 cavity spin을 관측 label로 clamp한다. 즉시 되돌아오는 edge message를 빼지만, loopy graph 전체에서 모든 중복 evidence가 제거된다는 보장은 없다.

같은 factor에서 node beliefs와 Bethe log partition score를 계산한다.

\[
q_\theta(h\mid D)=\operatorname{softmax}_h
\{\log\pi(h)+A_h+\widehat{\log Z}_h\},
\qquad
p_\theta(y_q\mid D)=\sum_h q_\theta(h\mid D)b_{q,h}(y_q).
\]

별도 neural posterior head는 없다. Finite BP에서 node/role beliefs는 근사값이고 Bethe score의 정확성도 보장하지 않는다. Predictive query CE만 쓰며 role-weight 경로에도 gradient를 흘린다. Posterior/likelihood/consistency auxiliary loss는 추가하지 않는다.

이 계산에서는 **context label → graph factor → 관측 feature node → shared column state → 다른 node**의 경로가 생긴다. O와 C가 겹치지 않아도 이 경로를 사용할 수 있는지가 검증 대상이다.

## 무엇과 비교하는가

| 대조 | 아는 정보와 목적 |
|---|---|
| Mask GraphPFN / PCFI GraphPFN | 같은 task, 초기 public model, exposure/update budget. PCFI는 실제 해당 입력으로 학습한다. |
| Fixed-mode factor BP | 같은 추론 연산에 historical training mode d=.25/eta=.72와 Gaussian 계수 .8/.6²를 넣는다. 학습한 potential의 추가 가치를 구분한다. |
| Grid6 exact/BP | 알려진 six-support parameter prior와 정확한 Gaussian feature law를 사용한다. 실제 test parameter를 받는 oracle과는 다르다. |
| Feature-only / LP | 한 evidence source만으로 설명되는 성과인지 확인한다. |

CSBM에서 node classes와 latent feature centroid를 함께 추론하는 것은 [기존 AMP–BP 연구](https://arxiv.org/html/2306.07948)에 있다. 그 연구의 Gaussian/high-dimensional 설정과 이 binary finite pilot은 다르지만, latent state나 BP를 썼다는 사실을 novelty로 주장할 수 없다. 원 논문 구현을 재현하거나 이겼다는 주장도 하지 않는다.
