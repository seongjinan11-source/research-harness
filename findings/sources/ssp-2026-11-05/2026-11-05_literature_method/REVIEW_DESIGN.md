# SSPd3 probing의 발전 가능성: 실험 설계·bandit 선행 검토

2026-11-05. `RESEARCH_GOAL.md`와 `count_support/DECISION_KO.md`를 읽고 아래6편의 공식 proceedings/저자 원문을 검색·부분 정독했다. 새 fit·objective 호출·bank 변경은 하지 않았다. 전체 appendix/proof를 완독했다는 뜻은 아니며 읽은 범위를 각 항목에 명시했다.

## 판정

**고정 간격의 farthest-count probing 자체는 강한 새 방법 기여로 보기 어렵다.** Model misspecification에서 탐색이 필요하다는 원리, 결정에 중요한 차이를 식별하는 실험 설계, 정보와 reward 비용을 조절하는 정책, nuisance를 제거하는 orthogonalization/dueling은 이미 있다.

그래도 목표를 포기할 이유는 없다. SSP에 남은 구체적인 질문은 **count shape 오지정이 어떤 composition/최적-count 판단을 왜곡하는지 진단하고, 유용한 count 일반화를 유지하면서 필요한 physical query만 사는가**다. 기존 수학으로도 이 연결을 새롭게 규명할 수 있다. 현재 정책을 새 이름으로 포장하기보다 그 연결의 예측력과 query 효율을 검증해야 한다.

## 현재 결과가 제한하는 방향

- 초기 narrow sigmoid의 conditional false certainty45/48은 실제 관측된 문제다. 그러나 SSPd3는 online feedback으로 회복한다. 잘못된 초기 posterior만 보여주며 전체 BO 실패를 주장할 수 없다.
- 같은 probe 원칙을 쓴 SSPd3_probe가 table_probe보다 좋다. 따라서 다음 후보의 필수 기준선은 SSPd3_mean뿐 아니라 **SSPd3_probe**다.
- Table은 step에서 강하고 composition을 보호하지만, smooth count에서 between-count 정보를 버리는 대가가 있다. 평균 composition 정확도나 interval 폭만으로 선택 성능을 판단하지 않는다.
- 기존24-target count-support bank는 이제 development evidence다. 아래 가설을 그 bank에서 고르고 같은 결과로 확인했다고 보고하지 않는다.

## 공식 원문6편과 직접 관련성

### 1. Bogunovic & Krause, NeurIPS2021 — Misspecified Gaussian Process Bandit Optimization

[공식 논문 페이지](https://papers.nips.cc/paper_files/paper/2021/hash/177db6acfe388526a4c7bff88e1feb15-Abstract.html) · [PDF](https://papers.nips.cc/paper_files/paper/2021/file/177db6acfe388526a4c7bff88e1feb15-Paper.pdf)

읽은 범위: abstract, §2의 bounded-RKHS approximation 설정, §3.2–3.3의 known/unknown misspecification 및 Algorithm2, Theorem2의 진술. 증명 전체는 미검토.

RKHS 함수로 uniform error ε 이내 근사되는 objective를 다룬다. ε를 아는 enlarged-confidence 접근과 ε를 모르는 phased uncertainty sampling/elimination을 구분한다. 후자는 episode마다 prior uncertainty를 다시 시작하며, 오지정 때문에 실제 최적 후보가 제거될 수도 있음을 인정하고 near-optimal 후보의 보존을 분석한다.

**경계:** “잘못된 모델의 UCB를 신뢰하면 위험하고 강제 탐색이 필요하다”는 주장 자체는 선행이다. SSP에서는 임의 오지정이 count 방향에만 있다는 구조, composition ranking으로의 전이, limited physical budget에서의 count별 탐색 비용이 추가 접점이다. 이 논문의 cumulative regret bound를 현재 mean incumbent regret에 그대로 옮길 수 없다.

### 2. Fiez, Jain, Jamieson & Ratliff, NeurIPS2019 — Sequential Experimental Design for Transductive Linear Bandits

[공식 논문 페이지](https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/8ba6c657b03fc7c8dd4dff8e45defcd2-Paper.pdf)

읽은 범위: §1의 measurement/recommendation sets 구분, §2/Theorem1의 allocation lower bound, §3 Algorithm1 RAGE 및 Theorem2 진술. Appendix proofs는 미검토.

관측할 수 있는 vectors와 최종 추천할 items가 달라도 되는 설정이다. 모든 parameter를 균일하게 잘 아는 것보다 살아 있는 후보의 **차이 방향**을 구분하도록 관측을 배분한다.

**경계:** candidate difference covariance를 줄이는 probe, active best-count elimination, “결정에 필요한 정보”라는 이름만으로 새 기여가 되지 않는다. 정확한 linear model과 fixed-confidence best-arm identification이라는 전제도 현재 misspecified SSP/16-query incumbent 목적과 다르다. Count별 shape bias를 covariance와 별도로 다뤄야 한다.

### 3. Wagenmaker, Katz-Samuels & Jamieson, AISTATS2021 — Experimental Design for Regret Minimization in Linear Bandits

[공식 논문 페이지](https://proceedings.mlr.press/v130/wagenmaker21a.html) · [PDF](https://proceedings.mlr.press/v130/wagenmaker21a/wagenmaker21a.pdf)

읽은 범위: abstract, §4.1–4.4의 Gaussian-width 설계, RegretMED Algorithm1/Eq(2), bandit 대 semi-bandit 계산 복잡도 설명과 정리 진술. 증명은 미검토.

관측의 정보 가치와 잘못된 arm을 사는 regret 비용을 함께 고려하는 설계다. Finite enumerable arm 경우와 거대 combinatorial action에서 효율적인 경우를 구분한다. Linear oracle를 이용한 계산 효율의 강한 결론은 특히 semi-bandit에서 나온다.

**경계:** “멀리 있는 count라도 reward 손실이 크면 probe하지 않는다”는 비용 인식 원리 자체는 기존이다. SSP는 aggregate subset feedback이므로 item별 semi-bandit 결과를 직접 재사용할 수 없다. 작은 common catalog에서의 정책 이득과 큰 feasible set에서의 계산 가능성도 별도로 검증해야 한다.

### 4. Krishnamurthy, Wu & Syrgkanis, ICML2018 — Semiparametric Contextual Bandits

[공식 논문 페이지](https://proceedings.mlr.press/v80/krishnamurthy18a.html) · [PDF](https://proceedings.mlr.press/v80/krishnamurthy18a/krishnamurthy18a.pdf)

읽은 범위: abstract, model 설명, §3의 deterministic-policy lower bound, BOSE의 orthogonalized estimator와 randomized action-selection 원리. Martingale proof 전체는 미검토.

Action-independent nuisance가 있는 linear reward에서 알려진 action propensity의 평균 feature를 빼 추정한다. Deterministic 선택이면 centered feature가0이 되어 정보가 없을 수 있어 충분한 randomization이 중요하다.

**직접 적용의 한계:** g(K)는 전체 subset action에 대해 action-dependent다. K를 먼저 정한 뒤 같은 K의 compositions만 비교하면 nuisance가 공통이 되어 접점이 생긴다. 이것으로 count 선택 자체는 해결되지 않는다. Same-count orthogonalization을 새 원리로 주장할 수 없고, 제약된 feasible slices에서 conditional sampling mean을 어떻게 알고 계산하는지와 query 기회비용이 남는다.

### 5. Russo & Van Roy, NIPS2014 — Learning to Optimize via Information-Directed Sampling

[공식 논문 페이지](https://papers.nips.cc/paper_files/paper/2014/hash/90720a2fcc41f9332e6a1558da327089-Abstract.html) · [PDF](https://papers.neurips.cc/paper/5463-learning-to-optimize-via-information-directed-sampling.pdf)

읽은 범위: introduction, §2–3의 information gain·expected regret와 Eq(5), optimization 설명. 전체 regret proofs/experiments는 미검토.

최적 action에 관한 정보와 즉시 regret를 결합하는 information ratio가 핵심이다. 일반 parameter entropy나 posterior variance만 키우는 것과 구분된다.

**경계:** Count-optimum entropy/expected improvement에 query 비용을 붙이는 아이디어는 IDS/KG/experimental design과 가까운 기존 원리다. 더구나 잘못된 SSP posterior로 정보량을 계산하면45/48의 과신을 그대로 공유할 수 있다. 임의 count 오지정에 대한 robust information 또는 bias certificate 없이 “decision-aware”라는 이름만 붙여서는 해결이 아니다. Bayesian cumulative regret와 현재 noisy-incumbent 목적도 같지 않다.

### 6. Kirschner & Krause, ICML2021 — Bias-Robust Bayesian Optimization via Dueling Bandits

[공식 논문 페이지](https://proceedings.mlr.press/v139/kirschner21a.html) · [PDF](https://proceedings.mlr.press/v139/kirschner21a/kirschner21a.pdf)

읽은 범위: introduction, §2의 feedback assumptions, §3의 two-point/one-point reductions, §4 초입 IDS 연결. 전체 regret proof는 미검토.

Action-independent additive bias를 randomized dueling feedback으로 바꾸는 선행이다. 두 점 평가 또는 한 점 무작위 평가에 각각 다른 bias/noise 가정이 필요하다.

**경계:** Same-K pair로 count nuisance를 없애 composition을 추정하는 사실은 이와 매우 가깝다. 전체 action에서 g(K)는 action-dependent이므로 across-K duel에서는 size 차이가 사라지지 않는다. Pair는 두 physical evaluations를 사용하며 차분 noise·loss를 함께 계산해야 한다. 공짜 pairwise oracle를 제공하거나 두 evaluations를 한 query로 세면 부당한 이득이다. Count 수준을 선택하는 바깥 문제가 별도로 남는다.

## 발전시킬 후보: posterior 분산과 count-shape bias를 구별하는 지원 정책

우선 SSPd3를 버리지 않고, **현재 추천을 바꿀 수 있는 count 오지정 위험에만 제한된 support query를 배분하는 정책**을 검토할 만하다. 이는 아직 검증되지 않은 연구 가설이다. Generic IDS/RAGE와 다른 부분을 수학적으로 명시하지 못하면 단순 조합으로 남는다.

가능한 표현은 f(x)=φ(x)'θ+r(K)다. φ에는 기존 SSP의 item/count basis를 넣고 r은 모델이 표현하지 못한 count 잔차다. 고정 linear ridge estimator의 V, 관측 feature matrix Φ, count incidence Z를 생각하면 두 후보의 feature 차이 d에 대한 residual bias에는 다음 구조가 나타난다.

    [e_Kx − e_Kz − Z' Φ V^{-1} d]' r.

같은 K이면 직접적인 r(Kx)−r(Kz)는 소거돼도, 학습 자료를 통한 composition 오염 항은 남는다. Shrinkage와 noise 항은 별도로 더해야 한다. 이 식을 완성된 SSP 정리나 새 정리로 제시하지 않는다. 기존 leakage/linear estimator algebra를 구조화된 bias 관점으로 쓴 출발점이다.

만약 외부에 정당화된 |r(K)|≤b_K가 있다면 이 bias 방향의 절댓값 합으로 gap 불확실성을 넓힐 수 있다. Probe가 그 위험을 실제로 줄이는지와 mean incumbent를 얻을 기회비용을 함께 평가한다. 중요한 제한은 **b_K를 관측되지 않은 count에서 자료만으로 정직하게 알 수는 없다**는 것이다. Smoothness·bounded variation·미리 정한 model class 또는 새 실제 관측이 필요하다. 사후 RMSE로 b_K를 정해 oracle 정보를 넣어서는 안 된다.

Generic augmented linear model `[Px,e_K]`는 arbitrary count effects를 표현하는 강한 기준선이다. 이 모델의 UCB/experimental design/IDS를 그대로 구현한 것을 새 SSP 원리라고 부를 수 없다. 가능한 차별화는 낮은 차원 count 구조가 맞을 때 정보 효율을 유지하면서, 틀린 부분의 결정 위험만 검증하는 더 싼 query/certificate와 그 조건을 밝히는 데 있다.

## 가장 작은 결정적 반증 실험

큰 새 online sweep 전에 작은 finite catalog의 새 targets에서 다음을 고정한다. 기존24/32/40-target banks는 사용하지 않는다.

1. **문제 구분:** 같은 h와 초기 관측 count의 labels를 공유하지만 미관측 count에서만 다른 두 g를 만든다. 추가로 먼 count가 중요하지 않은 조건과 실제 최적 count가 그곳에 있는 조건을 교차한다. Smooth/step뿐 아니라 nonmonotone shape를 넣고 amplitude가 작은 negative control을 둔다. 알고리즘은 truth 유형을 보지 못한다.
2. **필수 한계 확인:** 초기 자료가 완전히 같은 두 문제는 첫 query 전에 어떤 정책도 구분할 수 없다. 새 정책의 장점은 전지적 위험 판별이 아니라, 정한 가정 아래 더 나은 query 배분이어야 한다. 이 동일정보 제약을 위반하는 결과면 누수나 부당한 oracle를 의심한다.
3. **정책만 먼저 비교:** SSPd3 fitting·composition search·총 query 수를 고정하고 mean, 기존4-slot farthest probe, 같은4-slot random-count probe, posterior-variance probe, 새 정책을 비교한다. Adaptive trigger라면 총 probe 수도 기록하고, slot 고정 대조로 시점/대상 선택의 효과를 분리한다.
4. **강한 구조 대조:** 작은 `[Px,e_K]` 모델의 generic uncertainty/design 정책을 포함한다. Same-K dueling 후보를 넣으면2 physical evaluations를2개로 과금하고 기존 absolute labels를 버리는 비용까지 맞춘다. 계산량·objectives·dedup·potential noise를 공통 적용한다.
5. **endpoint:** 기존 noisy-observed incumbent의0..T 평균을 실제 primary로 유지한다. Count support 수, posterior calibration, composition MSE는 기전 보조 지표다. “같은 정보에서 위험 추정→실제 유용한 query→incumbent 개선”의 연결을 보고한다. 새 development 이후 고정한 independent confirmation이 필요하다.

**중단 조건:** 같은 평가/계산 예산에서 SSPd3_probe나 random-count보다 좋아지지 않거나, 개선이 단지 probes를 더 쓴 데서 나오거나, full-table generic baseline과 행동이 사실상 같다면 새 policy의 추가 필요성은 약하다. 관측되지 않은 count의 true error를 이용해야만 동작하거나, smooth low-signal 조건에서 불필요한 탐색 손해가 크면 일반 권고로 채택하지 않는다. 성공 후에만 제약된 큰 subset domain으로 확장한다.

## 최종 연구 권고와 읽기 한계

SSPd3_probe를 강한 단순 기준선으로 보존하고, count shape 오지정의 구조를 이용한 query의 추가 가치를 가장 작은 실험에서 먼저 반증하는 방향이 타당하다. 새 kernel은 필수가 아니다. 다만 fixed farthest spacing, nuisance cancellation, “정보와 보상을 함께 본다”는 일반 원리는 이미 선행이 있으므로 그 자체를 flagship novelty로 주장하지 않는다.

이번 검색은 위6편의 직접 primary 확인이며 subset BO 전체 문헌의 완전한 priority 조사나 proof 검증은 아니다. 기존 harness XLSX/JSON은 사용할 수 있고, 이전 검토의 same-information 대 fitting-shortfall 및 strong-baseline 설계 교훈을 재사용했다. 그 workbook에 직접 BO 선행이 없다는 사실은 novelty 근거로 쓰지 않았다. 이 문서는 실험 실행이나 새 보장의 성립을 보고하는 산출물이 아니다.
