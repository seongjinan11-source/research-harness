# FP04 데이터·실험설계 검토

2026-11-04. Experiments/statistics reviewer. PLAN_v1과 기존 bank/PCFI/native 복원 코드를 읽고 source 수의 확률만 CPU로 계산했다. 새 데이터·모델 구현, GPU/학습, historical artifact/branch 변경을 수행하지 않았다. 현재 cycle의 다른 리뷰는 읽지 않았다.

**판정: 아래 고정 수정안을 반영하면 실행할 가치가 있다(4/5).** Fully trainable3-arm 대조와 native-feature real 평가, synthetic FAIL이어도 real을 실행한다는 범위에 동의한다. 원안의 small-N 99.5%와 미정 generator/primary 분모는 코드 전에 해결할 중요한 명세 공백이다. 이것은 새 observed-cell 방법의 성공이나 novelty를 미리 인정한다는 뜻은 아니다.

**V2 재검토:** 결과를 계산하기 전에 작성된 PLAN_v2/protocol도 읽었다. Larger-N/factorial96/Bernoulli nested masks/primary95·995/source0·1 포함/conditional utility, 정확한 A/B law, full observed init parity, RNG-isolated 실제 PCFI, native chunk/resource 검사를 채택해 주요 실험설계 공백은 해소됐다. 고정안의 구현과 수치·resource 검사로 진행하는 데 추가 blocker는 없다. 학습 schedule은 V2의 uniform sampling이며 아래의8 permutations는 채택된 명세가 아니라 설명용 예시다. 따라서 실제 base별 exposure 횟수와 unique(view,mask,context) 수를 그대로 기록하면 된다.

남은 통계 구현의 명확화는 final12 strata(family×shape×K) 안에서8 graphs를 bootstrap하고 utility에도 eligible denominator를 보존하는 것이다. 선언한 utility strata가 비면 그대로 재추출하거나 새 stratum으로 대체하지 않는다. Source0/1 경계를 보존한 채 네 gate를 모두 보고한다. Real은405 predictions를 synthetic FAIL 여부와 무관하게 수행한다. 본 검토 후 실제 data/model 구현은 아직 하지 않았다.

## 1. 권고할 수정안은 하나: N 확대 + Bernoulli mask + 고정 두 primary rates

Root와 사전 논의한 다음 한 안을 권고하며 결과에 따른 sweep 대안은 제시하지 않는다.

- Train N={128,256,512}, F={4,8,16}, K={2,3,4},384 base graphs.
- Final panel은 matched(N256,F8)와 held-shape(N384,F12), K2/3/4, familyA/B, 각8 graphs로 총96개다. N/F 모두 train 범위 안의 미관측 shape 조합이므로 semantic schema OOD나 범위 밖 size 일반화라고 하지 않는다.
- Whole-node source mask는 query를 뺀 eligible nodes에 독립 Uniform[0,1]을 한 번 뽑고 p=.05/.005 threshold로 nested 생성한다. Disjoint stratum에서는 context도 eligible에서 제외한다. Primary는95/99.5%의 **모든** case이며 source0/1도 포함한다. Whole90%, cell masks와 full/nonquery-visible은 사전 secondary 또는 training support로 정의한다.
- Final 각 family×shape×K cell의8 graphs 중4개를 disjoint utility stratum으로 미리 지정한다. 나머지 overlap 규칙과 context cap의 deterministic schedule은 protocol에 고정한다. Seed/실제 source 수/expected rate/realized rate를 모두 저장하고 관측 결과를 보고 재추출하지 않는다.

원안에서 `round(N*.005)`를 쓰면 N64/96은0, N128/160은1 source뿐이다. Source>=2 utility가 99.5%에서 구조적으로 비게 된다. Bernoulli라도 극단적 결측은 정보량이 적다. Query16개/context16개를 제외한 disjoint 예시에서 source 수 S의 정확한 확률은 다음과 같다.

| N | Eligible nodes | P(S=0) | P(S=1) | P(S>=2) |
|---|---:|---:|---:|---:|
| 64 | 32 | .8518 | .1370 | .0112 |
| 96 | 64 | .7256 | .2333 | .0411 |
| 128 | 96 | .6180 | .2981 | .0838 |
| 256 | 224 | .3254 | .3662 | .3084 |
| 384 | 352 | .1713 | .3030 | .5257 |
| 512 | 480 | .0902 | .2175 | .6923 |

N 확대가99.5%의 정보 부족을 없애지는 않는다. Final matched에서도 약69%가 source0/1인 예시이므로 primary는 정보가 없는 조건까지 포함한 전체 위험이고 utility subset은 다른 질문이다. Cell99.5%는 source node 수뿐 아니라 **column별 support0/1/2+ 및 같은 column을 함께 본 node 수**를 보고해야 한다. Source node2개라도 관측 column이 겹치지 않으면 intended column-memory 경로의 정보가 부족할 수 있다.

## 2. Generator는 “복잡함”보다 관측 정보의 회수 가능성을 고정해야 한다

Family A/B를 독립적으로 정확히 정의한 뒤 source/config를 잠가야 한다. A의 centroid draw·class histogram·homophily/degree·SNR·noise df와 nonlinear function은 task/node 중 어느 단위에 공유되는지 명시한다. 특히 per-column transformation은 node 사이에 공유되어야 column evidence를 배울 수 있다. Graph signal이 너무 강해 graph-only가 이미 포화하거나 centroid가 거의 같아 feature 정보가 없으면 arm 차이를 검증하기 어렵다. Dev 결과를 보고 좋은 regime만 final로 골라서는 안 된다.

B는 graph가 label-independent라는 의미와 graph-smoothed feature 생성법, random nonlinear rule의 입력이 own/neighbor feature 중 무엇인지, class threshold/bias 및 context 수로 추론해야 할 task parameter를 수식으로 고정한다. 임의의 고차원 함수를1~4labels/class만으로 식별할 수 있다고 가정하지 않는다. Hidden complete features가 label을 잘 설명한다는 사실은 **실제 sparse observations가 query에 정보를 준다**는 증거와 다르다. 최소한 별도 tiny example에서 visible-input 기반 interpolation/분류 reference 또는 명시적인 dependency path가 feature-free보다 유용할 수 있음을 확인하고, 전체384/96 draws는 결과와 관계없이 보존한다. 이 확인을 또 하나의 대형 tuning 단계로 늘릴 필요는 없다.

Declared K는 generator가 정한 vocabulary이며 realized absent class를 제거하지 않는다. Context는 가능한 class에만 cap1/2/4로 뽑고 실제 count와 absent class를 남긴다. Query16은 context 밖에서 label/feature 결과를 보고 선별하지 않고 먼저 고정한다. F·family·mask·source count·class coverage의 generator-side audit 정보는 forward whitelist 밖에 둔다. Class-balanced context는 모집단 class 빈도가 아니다.

Source0은 physical feature value 정보가 없다는 경계다. Source1은 A의 독립 exchangeable centroid 조건에서는 식별에 약할 수 있지만, B 또는 task-family uncertainty에서는 정보가 있을 수 있다. **FP01의 singleton-invariance를 새 arbitrary observed-cell predictor 전체에 강제하지 않는다.** Source0/1을 각각 기술하고 class별 source coverage/zero-support columns를 audit하되 query truth로 source를 재선택하지 않는다. 현재 MCAR mask 결과를 MNAR·class-dependent source shift의 성공으로 확대하지 않는다.

## 3. Primary/utility estimand와 bootstrap 분모를 코드 전에 고정

Family는 base graph의 속성이다. A/B를 한 graph 안에서 평균했다고 쓰지 않는다. Final12 strata(family2×shape2×K3), 각8 graphs를 보존하고 strata 내 base-graph bootstrap5000회가 적절하다. 한 graph 안에서95/99.5의 accuracy를 동일 가중 평균하고 fixed3 training-seed 지표를 평균한다. Seed probability ensemble, node 또는 mask를 독립 graph로 세지 않는다. Family/shape/K를 동일 가중하는 최종 평균과 strata bootstrap을 사전에 고정한다.

Observed−mask 및 observed−PCFI primary는 모든96 graphs/source0·1을 포함한다. **모든 arm은 동일 graph/context/query/source mask**를 받아야 한다. Utility는 미리 지정한 disjoint graphs 안에서 source>=2인 rates만 graph 내부 평균한 conditional estimand로 계산한다. 그래프마다 eligible rate 수가 다르므로 모든 case를 단순 pooling하지 않는다. 다음을 summary에 함께 저장한다.

- Primary graph96개/각 rate case 수 및 source0/1/2+ 분해.
- Utility disjoint 예정 graph48개 중 실제 eligible graph 수, stratum별 분모, 포함95/995 case 수.
- 각 graph의 utility 분모와 제외 이유(source count만), per-class source/context support.

Utility가 한 stratum에서 비면 graph를 다시 생성하거나 범위를 사후 확대하지 않고 gate undefined/inconclusive로 처리한다. Bootstrap은 primary의96 graphs와 utility의 사전 조건부 eligible graphs를 respective strata 안에서 resample한다. Rates마다 다른 실제 support를 감춘 단일 “n=96” CI를 쓰지 않는다.

네 positive-lower-bound 조건을 모두 만족해야 성공이라는 conjunction에 동의한다. 두 arm primary CI는 각각97.5%, utility observed−none/shuffle도 각각97.5%로 고정하되, 네 개의 **동시95% CI**를 제공한다고 쓰지는 않는다. 두 primary Bonferroni 해석과 utility의 별도 조건부 estimand를 구분하면 된다. 모든 조건을 요구하는 joint gate 자체와 개별 효과를 여러 개 주장하는 familywise coverage도 다른 개념이다. Accuracy 실패를 NLL로 바꾸거나 family/shape별 winner로 primary를 대체하지 않는다.

Whole-node shuffle는 같은 observed rows의 값을 row permutation하고 mask/context/query를 유지한다. Cell shuffle은 column 안의 observed values만 permute한다. Singleton/empty observed sets와 identity permutation도 그대로 남긴다. None/shuffle 뒤 PCFI는 기존 output 재사용 없이 해당 visible input에서 실제 재계산한다. Observed−none이 양수여도 shuffled와 차이가 없으면 위치에 맞는 관측 evidence 활용은 미확인이다. Source0의 observed/none 입력이 같으면 같은 arm의 deterministic forward가 같아야 한다.

## 4. Fully trainable matched controls와 예산

9 fits×768 updates×batch4=6,912 updates/27,648 presentations 계산은 맞다. 각 fit은3,072 presentations이고 train base graph는384개다. 이는독립27,648 tasks가 아니다. 간단한 고정 schedule은384 episodes의8 permutations이며, mask/context를 노출마다 바꾼다면 view seeds와 모든 unique episode/cache 수를 별도로 잠가야 한다. 동일 seed의3 arms는 초기 parameter뿐 아니라 task/view order·batch 구성·optimizer schedule도 같아야 한다.

전체 observed complete toy에서 observed/mask/PCFI가 동일한 init output을 내는 검사는 중요하다. 관측값 clamp, flag/moments, reductions의 차이로 input 외 architecture/init 차이가 섞이지 않아야 한다. 실제 PCFI가 dense input을 제공하는 것이 arm의 개입이며 FLOPs/벽시계가 같다는 뜻은 아니다. Trainable parameter 수·성공 updates·노출 수와 실제 per-arm/total wall·peak·preprocessing 비용을 함께 보고한다. Checkpoint는 final 하나이며 dev-best 선택 없이 smoke training은 폐기·비용 기록한다.

Vendor `pcfi()`는 Python/NumPy/Torch global RNG를 reset한다. Precompute cache 후 training seed를 정하거나 RNG state를 명시적으로 격리해 matched init/schedule을 보존해야 한다. Actual PCFI 구현/설정/cache hashes/finite checks를 잠그고 source0 fallback도 library-defined 동작 또는 명시된 처리로 테스트한다. Library failure를 조용한 zero-fill로 바꿔 PCFI arm이라고 쓰지 않는다. 완료된 unique PCFI cache는3 arms/seeds에 공유하며 물리적 재계산 비용을9배로 세지 않는다.

## 5. Native real 전이는 강한 실행 검증이지만 독립 holdout이 아니다

3 graph×3rep×5modes=45 episodes,9 trained models면 **405 new prediction records**다. Synthetic gate FAIL이어도 이45 episodes를 모두 수행한다는 약속을 유지한다. Cora/CiteSeer full과 PubMed 기존 induced2000를 구분하고3rep를 독립 graph9개로 세지 않는다. Context/source/query를 가능한 한 기존 episode 그대로 쓰고 native coordinates와 observed-only normalization을 보존한다. K6/7과 native F수백~수천은 train K2~4/F4~16보다 넓은 전이 조건이라는 한계도 명시한다.

실제 복원 경로는 존재한다. Cora/CiteSeer는 `factor_external_benchmark.run.inputs`가 raw x와 source/context-order RNG를 재구성하고 projection replay까지 검증한다. PubMed bank는 native/original_ids/value_original_ids를 보유한다. 이름이 `native_pcfi`인 과거 Factor 배열 중에는 실제로 RP16으로 내려온 값도 있으므로 이를 native full-feature PCFI output으로 재사용하면 안 된다. Public external pipeline의 native input 여부는 해당 method metadata를 개별 확인한다.

외부 score는 context/query/truth만 같아서는 부족하다. Graph edges, observed source IDs, **실제 observed native values(특히shuffle)**, raw data hash와 preprocessing/method 정보를 확인한다. Same episode이지만 기존 Factor RP16과 새 native 입력처럼 representation이 다르면 정보 표현이 다른 비교라고 적는다. Exact-identity를 검증할 수 없으면 unmatched로 별도 표시하거나 사전 범위 내 재계산하며 matched mean/CI에 조용히 섞지 않는다. 강한 restart LP와 public/PCFI/FISF-PFN을 보존하고 새 observed의 none 및 observed−none/shuffle도 같이 보고한다.

모델의 sparse/matmul 구현은 O(NFD)로 만들 수 있지만 **PCFI는 별도**다. 현 vendor는 F×F correlation 및 native feature matmul로 O(NF²) 비용도 있다. CiteSeer N3327/F3703에서는 cell×class×width tensor 하나도 매우 커지므로 전체 N×F×K×32를 저장하는 구현을 피하고 feature/class chunk를 정확하게 검증해야 한다. Source lock 전에 native95/99.5/none의 resource smoke를 한 번 고정하고 native PCFI cache/전처리 시간·cold inference·peak를 따로 기록한다. 숨은 projection이나 feature truncation으로 자원 문제를 덮지 않는다. 예상 resource 한계를 넘으면 incomplete/resource failure를 기록해야 하며 PCFI를 다른 fill로 대체하지 않는다.

## 최소 검증 세 가지

1. V2의 A/B 생성식·96 strata·nested Bernoulli mask·overlap·utility eligibility/분모와 actual0/1/2+ counts를 final 생성 전에 잠근다. Source0/1, support0 column, absent class는 draw를 버리지 않고 테스트한다.
2. Complete-input arm parity·finite parameter gradients·hidden-value/query-label poison·permutation·chunk/dense equivalence·PCFI RNG isolation 및 clamp/none/shuffle 재계산을 확인한다.
3. Final saved p의 네 gate CI를 independent base-graph rescore로 재현하고405 real predictions를 synthetic 결과와 무관하게 감사한다. 초기384 graphs·반복노출·실제 PCFI/native 비용과 reused-real 범위를 정확히 보고한다.

이 수정안은 **실제 end-to-end 학습과 native 관측 사용**을 시험할 수 있는 합리적인 pilot이다. 작은 생성기에서 미통과했다고 전체 missing-robust PFN을 닫거나, 두 synthetic family에서 통과했다고 보편 robustness/새 feature 정보의 실제 전이를 확립한 것으로 확대하지 않는다.
