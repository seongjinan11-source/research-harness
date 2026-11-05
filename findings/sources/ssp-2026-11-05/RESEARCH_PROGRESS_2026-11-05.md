# SSP 연구 진행 정리

기준일: 2026-11-05. 범위: 이번 대화에서 수행한2026-11-04~05연구 cycle, 현재 이론·문헌 판단, 다음 미검증 후보. 기존 원고의 모든 과거 수치를 다시 검산한 문서는 아니다. 활성 목표는 [RESEARCH_GOAL.md](C:/Users/seongjin/Desktop/code/ssp-share/RESEARCH_GOAL.md)다.

## 1. 현재 판단

현재 가장 강한 근거는 **적절한 size/count 모델의 joint fitting이 composition 추정과 실제 선택을 회복할 수 있다**는 것이다. 기존 SSPd3 확장의 tomography 확인과 유연한 count GP의 합성 online 확인이 이를 뒷받침한다.

추가로 시도한 nonlinear composition projection, cut inference의 추가 이득, 단일 contrast probe는 각자의 사전 기준을 넘지 못했다. 따라서 더 복잡한 모듈이 필수라는 주장은 현재 근거와 맞지 않는다. 적절한 joint count fitting 자체가 SSP의 유효한 해결 방식일 수 있다.

이론에서는 기존 fixed-prior retention/leakage와 새 fixed-archive 식별 조건을 구분했다. 후자는 count support와 within-count composition 정보가 다른 자원임을 설명하지만, 고전 추정·검정·설계 이론의 subset 특수화다. 새 일반 이론이나 현재 adaptive 정책의 regret 보장으로 주장하지 않는다.

연구 기여는 **size/composition 문제의 정량적 진단 → 충분한 교정 원리 → 실제 추천 회복**의 전체 연결로 평가한다. 모티브의 최초성, 다양한 실제 문제로의 전이, 새로운 관측 정책의 추가 효용은 아직 확정하지 않았다.

## 2. 수치 읽는 법

- Tomography의 AUC 개선율은 보고서에 정의한 **평균 AUC의 비율 변화**다. Regret 차이와 같은 척도가 아니다.
- Joint-count online과 contrast-probe의 regret는 true objective range로 정규화한 손실이며 작을수록 좋다. 0.007의 차이는 정규화 손실0.7percentage point이지 목적함수 성능+0.7%가 아니다.
- 같은 정책을 썼다는 것과 실제로 같은 count를 관측했다는 것을 구분한다. Tomography 주 비교는3200proposal의 실제 count가 일치했다. Joint-count online의 두 probe arm은 같은 규칙을 썼지만 경로는 달라질 수 있다. Contrast-probe는 실제 count schedule을 공유했다.
- 개발 관찰, 새 확인 bank, 고정 archive 실험, online 실행, 수학적 sanity check를 구분한다. 이후 해석에 사용한 bank를 새 방법의 독립 확인으로 재사용하지 않는다.

## 3. 실험·검토별 결과

| Cycle | 실제 검증 범위 | 결과와 판단 | 남은 범위 |
|---|---|---|---|
|SSP-TS|4family×2개 개발 target, 동일1초 MILP runner|Equal-family AUC 변화−3.2746%, gate 실패. TS=Thompson sampling이며 time series가 아니다|전체 TS의 부정이나 실패 원인의 확정이 아님. SetCover/MWIS solver gap도 유지|
|Tomography utility|교정된 physical operator에서 복잡한 관측 가치 정책 비교|추가 효용 gate 실패|정보 가치 기반 정책 전체가 불가능하다는 뜻은 아님|
|Tomography shape|12개 개발 scenario에서 size degree와 count 정책 비교|d2_scalar 주 비교 실패, d3_spread 보조 후보 확보|보조 승자를 같은 개발 bank에서 확인 성공으로 세지 않음|
|Tomography confirm|고정 d3_spread를 새40 IID scenario에서 확인|d1_spread 대비 AUC+1.30919%,95%CI[+1.04467%,+1.57140%],38/40승리. 모든 사전 gate 통과|다른 physical task, equal-wall-time 우위, 원고의 모든 ROI 수치 확인은 아님|
|Nonlinear composition GP|추가 projection의 효용 비교|gp_orth−gp_plus equal-family AUC+0.04237%, gate 실패|비선형 size 확장 전체의 실패가 아님|
|Size injection|새32target,같은 archive에서 size shape 교란|적절한 nonlinear joint fitting으로 실제 같은-count composition 추천 회복. Cubic에서 degree3의 실용 가치 확인|Step의 polynomial misspecification, 좁은 support 외삽 실패 유지|
|Count support|새24target의 count table/probe 비교|지정한 주 비교는 통과했지만 같은 probe의 SSPd3가 table보다 좋아 table 기본 채택 유보|Table은 step에서 보호하지만 count 간 일반화를 잃음|
|Flexible count|새24target,2880개 고정-archive 기록|Cut 추가 이득 주 기준 실패. 유연한 full joint GP는 wide-step 오류를 크게 회복|Narrow cubic/wave 외삽 한계와 optimizer finite non-success 보존|
|Joint count online|새24target,2304runs,36864newqueries|같은 probe 규칙의 SSPd3 대비 비다항 혼합 incumbent regret0.014104→0.007077. 사전 gate 통과|Narrow-wave 등 손해, 강제 probe의 기회비용, 합성 task 범위 유지|
|Decision radius|독립 증명·직접 선행 검토|Affine robustness의 기존 radius와 직접 대응. 새 이론 후보로 중단|수치 확인 실험을 수행했다고 세지 않음|
|Identifiability|고정 archive 명제 증명,새16설계64개 수학 검증|식별 조건·정확한 Gaussian two-point risk·within-count 정보 update 확인|새 정책 효능, 실제 BO 난이도 예측, adaptive regret 보장 아님|
|Contrast probe|새24target,3456runs,실제 count-matched6arm 비교|단일 contrast 규칙의 다섯 superiority gate 모두 실패. Stage2 미실행|현재 규칙의 추가 효용 실패이며 기존 joint GP 성공을 반박하지 않음|

각 cycle의 상세 결과·등록·리뷰는 [연구 ledger](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/README.md)에서 연결된다. 원본 source와 locked banks는 보존했다.

## 4. 가장 중요한 긍정 근거

### 4.1 Size 모델이 같은-count 선택에도 영향을 준다

Tomography 확인의 d3_spread와 d1_spread는3200개 paired proposal에서 실제 count가 모두 같았다. 따라서 +1.30919%AUC를 더 많은 count 탐색의 효과로 설명할 수 없다. Joint fitting에 따른 composition 계수와 이후 관측 경로의 차이가 관련된다. 어떤 추정 오류가 얼마만큼 줄었는지의 완전한 인과 분해는 남아 있다.

별도의 size-injection 실험은 count shape만 바꾸어 joint composition 회복을 관찰했다. 이는 위 해석에 기전 근거를 추가하지만, 모든 실제 subset objective가 선형 composition+size라는 보장은 아니다.

근거: [tomography 확인](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-04_tomo_confirm/DECISION_KO.md), [size injection](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_size_injection/DECISION_KO.md). 관련 선행은 [RegConfounding], [OrthogonalGP], [SetBO]이며, 일반 confounding/identifiability 원리와 subset BO의 구체적인 실패를 구분한다.

### 4.2 Flexible joint count GP의 online 확인

비다항3shape(sigmoid,step,wave)×4archive regime을 target 안에서 동일 가중했다.24target이 통계 단위다.

| 지표 | 결과 |
|---|---:|
|SSPd3+probe mean incumbent regret|0.014104149|
|Joint count GP+probe mean incumbent regret|0.007076945|
|Baseline−candidate 손실 감소|0.007027205|
|Paired bootstrap95%CI|[0.006036082,0.007998785]|
|Target 평균 승리|24/24|
|정규화 regret의 상대 감소|약49.82%|

이49.82%는 objective/AUC가 약50% 좋아졌다는 뜻이 아니다. Polynomial/zero control의 비열등 기준도 통과했다. 하지만 비다항12개 shape×regime strata 중5개에서는 후보 평균이 더 나빴다. 특히 narrow-wave 손해가 남았고, joint_mean이 joint_probe보다 좋은 혼합 평균도 있어 강제 probe가 항상 필요한 것은 아니다.

근거: [online 확인](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_joint_count_online/DECISION_KO.md). 이 결과를 cut 또는 projection의 우월성으로 해석하지 않는다. [SemiModular], [CutSemiparametric]는 가능한 feedback 제어의 선행이며, 여기서는 full joint가 충분한 경우를 실제로 확인했다.

## 5. 실패가 알려 준 것

### 5.1 Cut과 projection을 필수 해법으로 삼을 근거가 약하다

Flexible-count 고정 archive에서 joint−cut same-count regret 차이는−0.000784,95%CI[−0.001974,+0.00000883]이었다. Joint가 더 낮은 평균 손실을 보였고 cut의 추가 우위 기준을 통과하지 못했다. Wide-step에서 SSPd3의 composition 손실0.24408은 joint0.008188,cut0.008972로 둘 다 크게 회복했다. Count shape를 적절히 처리하는 단순 joint 모델을 먼저 기준으로 삼아야 한다.

GP projection의 +0.04237%AUC도 해당 projection의 추가 효용 gate 실패다. 두 실패를 전체 nonlinear size 모델링의 실패로 확대하지 않는다.

근거: [flexible count](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_flexible_count/DECISION_KO.md), [nonlinear GP](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_nonlinear_gp/DECISION_KO.md).

### 5.2 단일 contrast probe는 actual regret 개선을 입증하지 못했다

Initial44observations,16newqueries,동일16count schedule에서 probe4회의 composition 선택 규칙을 바꿨다. 아래는 비다항 혼합 평균이다.

| 정책 | 실제 composition recommendation regret | Noisy-observed incumbent의 latent utility regret |
|---|---:|---:|
|Greedy|0.003275414|0.012856750|
|Random|0.003356129|0.015909759|
|A-opt|0.002918175|0.012898546|
|Joint query variance|0.002976601|0.015869658|
|Joint contrast reduction|0.003394102|0.015471281|
|새 robust contrast|0.003203461|0.012919557|

Greedy−contrast gain은+0.000071953,99%CI[−0.000127103,+0.000271976]로0을 포함하고 사전 최소효과0.0001에 못 미쳤다. 다섯 primary 우위 gate가 모두 실패해 conditional online Stage2를 실행하지 않았다.

Safety의 비열등 기준을 넘었다고 손해가 없는 것은 아니다. Greedy 대비 incumbent regret 증가량은+0.000062807,99%CI[+0.000006267,+0.000141468]로 작지만 양수였다.

좁은 count support의288contexts에서는 네 probe가 각각 첫 신규 count여서 robust score가0이었다. 후보의2304probe 중1152회가 greedy fallback이었고, 이288contexts에서 greedy/A-opt/contrast의 전체 query 경로가 같았다. Wide에서는 실제 행동을 바꿨지만 우위를 입증하지 못했다. 한 개의 애매한 item swap에 집중하는 것이 전체 composition 손실과 맞지 않을 수 있다는 것은 다음 **가설**이며 확정 원인이 아니다.

근거: [contrast probe](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_contrast_probe/DECISION_KO.md). A-opt의 평균이 더 낮았다는 관찰을 새 독립 확인 성공으로 승격하지 않는다. [LinearBAI], [TransductiveLB], [NuisanceDesign]의 표준 설계 원리를 직접 대조했다.

## 6. 이론에서 확보한 것과 범위

### 6.1 기존 SSP 이론

원고에는 size 방향의 variance share와 fixed-prior retention/leakage 분석이 이미 있다. Higher-order size basis와 matrix retention도 기존 appendix에 포함돼 있으므로 새 발명으로 다시 세지 않는다.

### 6.2 Fixed archive의 composition 식별 조건

선형 composition과 arbitrary observed-count effect를 가정한다. B는 item 공통 방향을 제거한 orthonormal basis, U=XB, Z는 관측 count indicator라 두면

\[
Q=I-Z(Z^\top Z)^{-1}Z^\top,\qquad M=U^\top Q U.
\]

전체 nontrivial fixed-K slice의 composition 차이를 모두 식별할 필요충분조건은 rank(M)=n−1이며, 따라서 N−m≥n−1이 필요하다. N은 관측 수,m은 관측 count 종류,n은 item 수다. 이 예산 조건만으로 충분하지 않으며 실제 within-count excitation이 필요하다. 제약된 feasible slice는 필요한 contrast만 식별되면 된다.

Fixed nonzero signal norm s,Gaussian noise sigma,실제 추천을 바꾸는 방향 v에서 서로 반대인 두 환경을 구성하면, 그 두 환경을 아는 최적 rule의 정확한 normalized regret는

\[
\Phi\!\left(-s\sqrt{v^\top Mv}/\sigma\right).
\]

Null 방향이 feasible contrast에 영향을 주면 동일 관측 분포를 갖는 두 문제 중 적어도 하나에서 expected normalized regret≥1/2다. 전체 unknown-parameter class에 대한 알고리즘 upper bound는 아니다.

처음 보는 count의 첫 관측은 이 arbitrary-nuisance composition 정보 M에0을 더한다. 기존 count에 t개 관측이 있으면 새 u=B'x의 정보 증분은 t/(t+1)(u−ubar)(u−ubar)'이다. 신규 count의 level 학습이나 proper GP prior 아래 정보가 무가치하다는 뜻은 아니다.

근거: [증명과 수학 검증](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_identifiability/THEOREMS.md). [NuisanceDesign], [TransductiveLB], [BAIComplexity]의 기존 정보·설계·검정 원리와 연결된다. Adaptive archive에 fixed-design Gaussian 분포를 그대로 조건부 적용하지 않는다.

### 6.3 중단한 이론 후보

최소 size perturbation으로 fitted ranking이 바뀌는 radius는 affine multiclass margin/normal-norm 공식에 대응했다. [DeepFool], [RobustnessBounds]를 직접 선행으로 인정해 새 이론 후보에서 내렸다. 수치 실험을 실행한 실패로 세지 않는다.

## 7. 문헌과 현재 주장 사이의 연결

문헌 정리의 유일 원본은 research-harness의 papers/*.yaml이고, Excel/Citations/BibTeX/그래프는 그 원본에서 생성한다. 아래 ID는 그래프와 Excel의 paper ID에 대응한다.

| Paper ID | SSP에서의 역할 | 현재 확대하지 않는 주장 |
|---|---|---|
|SetBO|Variable-cardinality set kernels와 subset BO의 직접 선행|기존 set BO가 variable size를 전혀 다루지 않았다는 주장|
|RegConfounding|Nuisance shrinkage가 관심 계수 추정에 오염을 만들 수 있다는 선행|SSP의 subset-specific 분석과 같은 정리라는 주장|
|OrthogonalGP|Mean/GP identifiability와 orthogonality의 직접 선행|우리 projection이 새 일반 원리이거나 항상 좋아야 한다는 주장|
|SemiModular|Misspecified module feedback 제어의 선행|Cut이 현재 joint보다 우월하다는 주장|
|CutSemiparametric|Parametric/nonparametric 모듈의 feedback 제어와 GP 사례|Semiparametric cut 자체가 새로운 SSP 기여라는 주장|
|SemiparametricBandits|Action-independent nuisance 아래 sequential orthogonalization 선행|Count-dependent g(K)에 보장을 그대로 적용|
|LinearBAI|Arm 차이와 fixed-confidence experimental design 선행|Item contrast 목표의 발명|
|TransductiveLB|측정 feature와 추천 feature가 다른 선형 식별/설계|[B'x,e_K] 관측 모델이 기존 선형 틀 밖이라는 주장|
|NuisanceDesign|Nuisance 제거 후 treatment contrast 정보/설계 선행|Schur complement와 within-group 정보의 발명|
|BAIComplexity|Best-arm identification의 구별하기 어려운 두 환경·Gaussian testing 기반|이번 pair equality가 범용 adaptive BO 보장이라는 주장|
|DeepFool|Affine score의 정확한 margin/normal radius 직접 선행|Decision-radius 공식의 독립적인 큰 신규성|
|RobustnessBounds|분류기 robustness radius와 선형 exact case의 선행|Fitted 안정성이 true decision 정확성을 보장한다는 주장|

각 논문의 정확한 metadata·전원 저자·검증 URL과 생성 citation key는 아래 인용 목록 및 research-harness의 Citations 탭에 있다. 문헌상 알려진 원리, 우리의 subset 적용, 실제 새 실험 결과를 구분한다.

## 8. 다음 후보와 미해결 과제

현재 미실행 후보는 **같은 count의 paired observations와 여러 composition 방향을 다루는 단순 설계의 결합**이다. 네 probe를 두 count의 쌍으로 배분하면 response 차이에서 g(K)가 소거되지만, 신규 count coverage를 줄이는 비용이 생긴다. Full joint fit을 유지하고 greedy/A-opt 및 기존 joint mean/probe와 비교해야 한다.

이는 등록된 새 확인 실험이나 검증된 해결책이 아니다. [다음 가설](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-05_contrast_probe/NEXT_HYPOTHESIS.md)에 미구현 상태로 분리했다.

남은 우선 과제는 다른 실제/physical task에서의 확인, size-dependent composition interactions에 대한 범위 확인, count coverage와 composition 정보의 실제 budget tradeoff, 강한 단순 baseline 대비 추가 메소드의 필요성이다. 실패한 후보를 삭제하거나 새 이름으로 같은 bank에서 재튜닝하지 않는다.

## 9. 인용 목록

- **[SetBO]** Poompol Buathong, David Ginsbourger, Tipaluck Krityakierne (2020). [Kernels over Sets of Finite Sets using RKHS Embeddings, with Application to Bayesian (Combinatorial) Optimization](https://proceedings.mlr.press/v108/buathong20a.html). Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics. Citation key: `buathong2020kernels`.

- **[RegConfounding]** P. Richard Hahn, Carlos M. Carvalho, Jingyu He, David Puelz (2016). [Regularization and confounding in linear regression for treatment effect estimation](https://arxiv.org/abs/1602.02176). arXiv:1602.02176. Citation key: `hahn2016regularization`.

- **[OrthogonalGP]** Matthew Plumlee, V. Roshan Joseph (2018). [ORTHOGONAL GAUSSIAN PROCESS MODELS](https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n23/j28n23.html). Statistica Sinica. Citation key: `plumlee2018orthogonal`.

- **[SemiModular]** Christian Carmona, Geoff Nicholls (2020). [Semi-Modular Inference: enhanced learning in multi-modular models by tempering the influence of components](https://proceedings.mlr.press/v108/carmona20a.html). Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics. Citation key: `carmona2020semi`.

- **[CutSemiparametric]** Linda S. L. Tan, David J. Nott, David T. Frazier (2025). [Optimization-centric cutting feedback for semiparametric models](https://arxiv.org/abs/2509.18708). arXiv:2509.18708. Citation key: `tan2025optimization`.

- **[SemiparametricBandits]** Akshay Krishnamurthy, Zhiwei Steven Wu, Vasilis Syrgkanis (2018). [Semiparametric Contextual Bandits](https://proceedings.mlr.press/v80/krishnamurthy18a.html). Proceedings of the 35th International Conference on Machine Learning. Citation key: `krishnamurthy2018semiparametric`.

- **[LinearBAI]** Marta Soare, Alessandro Lazaric, Rémi Munos (2014). [Best-Arm Identification in Linear Bandits](https://proceedings.neurips.cc/paper_files/paper/2014/hash/f8d84caae48546d0934d637ab54f7086-Abstract.html). Advances in Neural Information Processing Systems 27. Citation key: `soare2014best`.

- **[TransductiveLB]** Tanner Fiez, Lalit Jain, Kevin G. Jamieson, Lillian Ratliff (2019). [Sequential Experimental Design for Transductive Linear Bandits](https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html). Advances in Neural Information Processing Systems 32. Citation key: `fiez2019sequential`.

- **[NuisanceDesign]** Samuel Rosa, Radoslav Harman (2015). [Optimal approximate designs for estimating treatment contrasts resistant to nuisance effects](https://arxiv.org/abs/1504.06079). arXiv:1504.06079. Citation key: `rosa2015optimal`.

- **[BAIComplexity]** Emilie Kaufmann, Olivier Cappé, Aurélien Garivier (2016). [On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models](https://jmlr.org/papers/v17/kaufman16a.html). JMLR. Citation key: `kaufmann2016complexity`.

- **[DeepFool]** Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard (2015). [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599). arXiv:1511.04599. Citation key: `moosavidezfooli2015deepfool`.

- **[RobustnessBounds]** Matthias Hein, Maksym Andriushchenko (2017). [Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation](https://proceedings.neurips.cc/paper/2017/hash/e077e1a544eec4f0307cf5c3c721d944-Abstract.html). Advances in Neural Information Processing Systems 30. Citation key: `hein2017formal`.

## 10. 재현·동기화

이 문서의 수치는 각 cycle의 DECISION/SUMMARY에 근거한다. 연구 원본과 locked bank는 수정하지 않았다. 문헌 그래프의 명시적 citation edge는 연구 기록에서 해당 문헌을 사용한 관계이고, 자동 revisit edge는 재검토 검색 단서다. 두 연결 모두 새로운 방법의 성공이나 novelty 증명으로 해석하지 않는다.

이번 정리의 원문 snapshot·source hashes·문헌 추가 내역·검증 결과는 research_cycles/2026-11-05_research_summary에 남긴다.

[SetBO]: https://proceedings.mlr.press/v108/buathong20a.html
[RegConfounding]: https://arxiv.org/abs/1602.02176
[OrthogonalGP]: https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n23/j28n23.html
[SemiModular]: https://proceedings.mlr.press/v108/carmona20a.html
[CutSemiparametric]: https://arxiv.org/abs/2509.18708
[SemiparametricBandits]: https://proceedings.mlr.press/v80/krishnamurthy18a.html
[LinearBAI]: https://proceedings.neurips.cc/paper_files/paper/2014/hash/f8d84caae48546d0934d637ab54f7086-Abstract.html
[TransductiveLB]: https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html
[NuisanceDesign]: https://arxiv.org/abs/1504.06079
[BAIComplexity]: https://jmlr.org/papers/v17/kaufman16a.html
[DeepFool]: https://arxiv.org/abs/1511.04599
[RobustnessBounds]: https://proceedings.neurips.cc/paper/2017/hash/e077e1a544eec4f0307cf5c3c721d944-Abstract.html
