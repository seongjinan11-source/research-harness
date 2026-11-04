# 계산 원리와 주장 범위

이 cycle은 새로운 Bayesian posterior나 새 PFN pretraining을 주장하지 않는다. 기존 학습된 관측 factor를 유지하면서 context의 held-out 예측으로 동일 조건부 score family의 관계 강도를 보정하는 작은 실험이다.

1. Yao, Vehtari, Simpson, Gelman의 [Using stacking to average Bayesian predictive distributions](https://arxiv.org/abs/1704.02030)과 [저자 원문](https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf)은 LOO predictive distributions의 결합을 다루며 stacking과 pseudo-BMA를 구분한다. 본 실험은 stacking weight 최적화나 PSIS 재현이 아니다. 고정 후보별 macro mean log score를 softmax하는 보수적인 predictive weighting이다. 이 식이 학습되거나 새롭다고 말하지 않는다.
2. Ribatet, Cooley, Davison의 [Bayesian inference from composite likelihoods, with an application to spatial extremes](https://www3.stat.sinica.edu.tw/statistica/j22n2/j22n217/j22n217.html)은 composite likelihood를 Bayes 식에 넣은 분포가 full posterior와 다를 수 있고 coverage 조정이 필요함을 다룬다. 우리 평균 점수의 scale1은 graph dependence에 대한 정합적 coverage 보정이 아니다. 큰 entropy/ESS도 그 자체로 calibrated role uncertainty의 증거가 아니다.
3. FP02의 Gamma/Dirichlet 갱신과 구별되는 변화는 자기 예측을 population/affinity 통계로 다시 사용하는 경로를 없애고, 실제 context label을 가린 예측으로 contrast family를 비교하는 것이다. Degree-opportunity는 주어진 graph에서 고정하며 query labels·전체-label homophily·참 population을 쓰지 않는다.
4. 새 family와 old full/edge의 차이 전체를 calibration 효과로 돌리지 않는다. 같은15개 조건부 예측의 prior mixture·fixed-edge·MAP와 adaptive mixture를 비교해야 가중치 변경의 실증효과가 분리된다. Graph-only calibration이므로 feature 조건부 posterior adaptation도 아니다.

성능·비용 결과가 나오기 전에는 방법적 우월성·독립적인 학습의 필요성·미지 schema 전이·정확 posterior나 calibration을 주장하지 않는다. 원문에서 가져온 것은 계산상의 경계와 비교 기준이며 본 구현은 위 두 논문의 재현이 아니다.
