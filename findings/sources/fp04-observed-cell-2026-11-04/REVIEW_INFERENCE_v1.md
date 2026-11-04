# FP04 추론·계산·누출 검토

검토일: 2026-11-04. 대상: `PLAN_v1.md`와 root가 전달한 수정 사항. 다른 현 주기 리뷰는 읽지 않았다. 구현·학습·실제 데이터 예측은 수행하지 않았다.

**판정: 수정 사항을 식과 검사로 고정하면 구현 가능한 실험이다.** 현재 문장만으로는 sparse/dense 계산 동치와 정규화가 유일하게 정의되지 않는다. 아래 항목을 V2에 명시한 뒤 구현해야 한다. Fully learned observed-cell 접근의 성능·새로움은 아직 입증되지 않았다.

## 구현 전 필수 수정

1. **Hard routing을 arm 차이로 보존한다.** 관측 mask를 M, 메시지 support를 A라 하면 observed는 A=M, mask와 PCFI는 A=1이다. Dense arm에 공통 M gate를 걸면 관측되지 않은 토큰을 학습하는 비교가 사라진다. M=1인 실제 값 0도 observed 셀이다. Native sparse feature 행렬의 nonzero 위치를 관측 집합으로 사용해서는 안 된다. Root는 이 구분과 완전관측 forward/gradient parity를 채택한다고 전달했다.

2. **셀 연산을 separable contraction으로 명시한다.** Rank4 값 가중치와 node/column 선형 projection 뒤의 비선형 함수를 구분해야 sparse scatter, dense matmul, feature chunk가 같은 연산이 된다. 셀마다 `MLP(concat(node_state,column_state,value))`를 적용하면 일반적으로 단순 matmul로 환원할 수 없다. Signed value basis의 합을 분모로 사용하지 말고, 양수 support count 또는 명시한 양수 가중합으로 정규화한다. Round 내 old/new state 순서와 네 round의 파라미터 공유도 고정한다.

3. **정규화와 intervention을 입력 계약으로 고정한다.** 열별 평균·분산·count는 M=1인 값만으로 계산한다. Source0, singleton, 관측값이 모두 0인 열은 서로 구별하고 scale floor/empty flag를 정의한다. Dense mask의 결측 0이 raw-space 0인지 normalized-space 0인지 정해야 한다. None/shuffle 뒤 통계와 PCFI를 재계산하며, PCFI cache key는 intervention까지 포함한 visible-input hash여야 한다. Hidden NaN은 곱하기 0으로 지우지 말고 통계·MLP 전에 선택에서 제외한다. Root가 이를 명시할 예정이라고 전달했다.

4. **공유 class 연산과 실제 learnable task를 고정한다.** Class chunk는 전체 K개의 logits를 모은 뒤 한 번의 softmax/CE를 적용한다. Class별 state 계산에는 class ID embedding이나 chunk별 softmax가 없어야 한다. Family B의 random nonlinear rule은 한 task에서 모든 노드에 공유하는 함수여야 한다. 노드마다 새 독립 함수를 뽑으면 context가 query 규칙을 전달할 수 없다. Generator 정의는 final 생성뿐 아니라 scored training 전에 고정한다.

5. **Native dimension에서 메모리와 산술량을 별도로 검사한다.** CiteSeer N=3327,F=3703,K=6,width32에서 `[K,N,F,width]`는 FP32 약 8.81 GiB이며 rank4까지 확장하면 약 35.25 GiB다. 이 activation을 만들면 안 된다. Feature/class chunk는 peak memory를 줄이지만 총 계산량은 줄이지 않는다. 두 방향 rank4 contraction을 네 round 수행하면 대략 75.7 billion MAC/model이다. Native-shaped 가상 입력 preflight로 실행시간·peak memory를 확인한 뒤 9 fits를 시작해야 한다.

## 동치를 보장할 수 있는 구체 식

다음은 계획 문장을 구현 가능한 형태로 만드는 한 예다. 다른 식을 선택해도 되지만 실제 검사 대상은 정확히 하나로 고정해야 한다. Node state H는 class별이고, value basis v는 class와 무관한 R=4 scalar다.

```text
v_ifr = phi_r(normalized_x_if, observed_if, observed_only_stats_f)
C_fc  = psi_C(sum_r P_r(sum_i A_if v_ifr H_ic / d_f), stats_f)
D_ic  = sum_r Q_r(sum_f A_if v_ifr C_fc / e_i)
H'_ic = psi_H(H_ic, neighbor_aggregate(H_c)_i, D_ic, seed_ic, static_context_counts_c)
logit_ic = shared_score(H_ic)
```

`d_f,e_i`는 0-support 처리까지 정의한 비음수 support 기반 분모다. 모든 feature chunk의 기여를 합친 뒤 `psi_C/psi_H`를 적용하며, 전체 열을 필요로 하는 분모·통계를 chunk별로 재정의하지 않는다. Per-class 계산이 독립이면 class chunking은 logits concat과 동치다. 동적 context pool을 삭제하고 정적 context label count만 허용한다는 root 수정은, 열 경로를 우회하는 상태 전달을 제거한다는 점에서 적절하다.

Dense mask의 missing token basis가 열마다 일정한 b_fr라면 아래 최적화도 정확하다. 이는 M hard gate를 공통 적용하는 것이 아니다.

```text
W_ifr = b_fr + M_if * (v_ifr - b_fr)
sum_i W_ifr H_ic = b_fr sum_i H_ic + sum_observed_i (v_ifr-b_fr) H_ic
```

역방향에도 같은 분해를 적용할 수 있다. PCFI의 missing 값은 노드마다 다르므로 일반적으로 이 상수 분해가 성립하지 않는다. 속도를 비교할 경우 dense mask도 이와 같은 정당한 최적화 대상이다.

## 필수 독립 검사와 해석 경계

- **동치:** 같은 파라미터·완전관측 입력에서 observed/mask/PCFI forward, 파라미터 및 입력 gradient 일치. 별도로 같은 support에서 scatter/dense 및 여러 feature/class chunk 크기의 forward/backward 일치. 평가의 feature-free query가 남은 입력은 완전관측 parity 사례가 아니다.
- **경계조건:** empty source/context, singleton, all-zero observed column, isolated/empty graph, K 중 미관측 class를 검사한다. 빈 context와 공유 class 함수이면 class logits가 같아야 한다. 같은 seed pattern의 미관측 class들도 같은 확률이어야 한다. Source0는 context가 있으면 graph 기반 비균등 예측이 가능하다.
- **누출·대칭:** hidden 값/NaN 및 query label poison 불변성, node/class/column permutation, 관측 0 보존, PCFI 관측값 clamp, none/shuffle 통계 재계산을 검사한다. K는 명시된 vocabulary에서 받고 query 정답의 unique/max로 추론하지 않는다.
- **경로·gradient:** 동적 context pooling 없이 두 disconnected graph components를 만들고, 한 성분의 context에서 다른 성분의 query로 이어지는 graph→source→column→source→graph 경로를 구성한다. 필요한 value/column/graph/head gradient가 유한하고 적절한 비퇴화 사례에서 0이 아닌지 확인한다. Graph/column 연결을 끊었을 때 경로가 사라지는지도 검사한다. 모든 사례·모든 파라미터의 gradient가 항상 nonzero여야 한다는 요구는 잘못이다.
- **Source1:** 기존 Gaussian centroid 모델의 singleton 무효 성질을 이 learned mixed prior에 강제하지 않는다. 한 관측값도 task prior나 graph-smoothed family에서 정보를 줄 수 있다. Singleton에서는 유한성·명시된 정규화를 검사하고 utility는 결과로 판단한다.
- **실행 가능성:** class-independent value basis는 class/round마다 불필요하게 재계산하지 않도록 cache 또는 chunk 재계산 방식을 고정한다. Native-shaped 가상 입력의 no-grad preflight와 작은 float64/FP32 gradient 검사를 분리한다. Chunk 크기 조정은 동일 식의 운영 변경으로 기록한다.

Train N을 128/256/512로 바꾸는 것은 계산상 가능하지만 task/edge/cell budget과 실제 schedule을 다시 고정해야 한다. 이 경우 N160 패널은 더 큰 그래프 외삽이 아니므로 size-generalization 표현을 수정해야 한다. Source>=2 disjoint utility subset은 node 수뿐 아니라 열별 공동 관측 support도 보고해야 한다. 총 두 source가 있어도 같은 열을 공유하지 않으면 의도한 column 경로가 없을 수 있다.

계획의 네 conjunctive gate는 이 모델의 관측 정보 사용 여부를 시험할 수 있다. 이를 통과해도 정확한 Bayesian inference, missingness mechanism 식별, 특정 내부 경로의 일반적 인과 설명, 보편적 missing robustness는 성립하지 않는다. 현재 수정 방향에 대한 수학적 veto는 없으며, 위 계약과 검사를 V2/구현에 반영한 뒤 실행하는 것이 적절하다.
