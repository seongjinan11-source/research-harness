# LP-conditioned structural residual: 소형 합성 실험

공식 GraphPFN이 아니라 기존 일방향 quadratic prototype을 공통 warm start로 쓴 실험이다. 3개 arm × 2개 초기화, 각 384 updates × 4개 새 synthetic episodes. 실제 그래프에서 weight update는 없다. LP는 native relation별 LP 확률의 균등 평균이며, context에서는 자기 seed의 전역 기여를 제거한 LOO를 사용했다.

- plain: context one-hot을 내부 추론에 쓰는 기존 일방향 모델.
- lp_offset: 내부는 plain과 같고 출력에 log LP만 더한다.
- lp_conditioned: 내부 ridge target과 전파 clamp를 onehot−LP_context_LOO로 바꾸고 log LP에 학습된 보정을 더한다. 피처가 없어도 구조적 오차 보정을 허용한다.

세 arm은 같은 10,469개 파라미터·초기 tensor·학습 과제·optimizer·update 수를 사용했다. 출력 offset만의 효과와 내부 LP conditioning의 증분을 분리한다. CE로 학습한 logit 보정이며 엄밀한 posterior나 중복 증거 제거를 보장하지 않는다.

| Synthetic schema | LP | plain | LP offset only | LP-conditioned |
|---|---:|---:|---:|---:|
| matched | 66.08 | 74.61 | 74.29 | 74.26 |
| heldout_shape | 63.03 | 70.42 | 70.97 | 71.32 |

| Comparison | Mean Δ (%p) | Graph bootstrap interval |
|---|---:|---|
| matched: lp_conditioned_minus_plain | -0.35 | [-0.97, 0.23] |
| matched: lp_conditioned_minus_lp_offset | -0.03 | [-0.28, 0.21] |
| matched: lp_conditioned_minus_LP | 8.18 | [6.71, 9.62] |
| heldout_shape: lp_conditioned_minus_plain | 0.90 | [0.01, 1.84] |
| heldout_shape: lp_conditioned_minus_lp_offset | 0.35 | [0.00, 0.70] |
| heldout_shape: lp_conditioned_minus_LP | 8.29 | [6.42, 10.31] |

## 결측 조건별 accuracy (%)

| Schema / condition | LP | plain | LP offset | LP-conditioned |
|---|---:|---:|---:|---:|
| matched / full | 66.08 | 85.51 | 85.80 | 85.65 |
| matched / node_0.9 | 66.08 | 71.01 | 70.44 | 70.54 |
| matched / node_0.995 | 66.08 | 71.20 | 70.37 | 70.42 |
| matched / cell_0.995 | 66.08 | 70.72 | 70.55 | 70.44 |
| heldout_shape / full | 63.03 | 84.04 | 83.71 | 83.97 |
| heldout_shape / node_0.9 | 63.03 | 65.82 | 66.88 | 67.05 |
| heldout_shape / node_0.995 | 63.03 | 66.46 | 66.87 | 67.29 |
| heldout_shape / cell_0.995 | 63.03 | 65.37 | 66.42 | 66.95 |

각 schema에서 새 그래프 36개, 결측 조건 4개를 평가했다. Matched/heldout_shape는 같은 generator template이고 후자는 relation 수와 feature 폭의 외삽이다. 일반적인 unseen schema나 실데이터 검증은 아니다.

위 interval은 두 학습 seed와 네 결측 조건을 graph별로 평균한 뒤, family별로 층화한 2,000회 graph bootstrap이다. 한 prior와 두 학습 seed의 소형 feasibility 결과이며, 학습 seed 전반의 불확실성을 충분히 대표하지 않는다. 새 semantic type 조합·실제 그래프 전이는 검증하지 않았다. λ tuning, residual norm regularization, learned fallback은 실행하지 않았다.

## 추가 해석

**내부 LP conditioning의 일관된 우위를 확인한 것은 아니다.** Matched에서는 plain보다 −0.35%p, offset-only보다 −0.03%p다. Heldout shape에서는 plain 대비 +0.90%p, offset-only 대비 +0.35%p의 작은 긍정 신호가 있다. Offset-only 대비 bootstrap 하한은 실제 약 +0.003%p로 0에 매우 가깝다. 두 학습 seed, 한 생성기의 pilot이며 다중 비교 보정도 하지 않았으므로 확정적 일반화 근거로 과대해석하지 않는다.

99.5% whole-node 조건만 보면 matched는 plain 71.20 / offset 70.37 / conditioned 70.42%, heldout shape는 66.46 / 66.87 / 67.29%다. 전체 평균의 순위가 결측 조건에서도 항상 유지되는 것은 아니며, matched에서는 conditioned가 기존 일방향 모델보다 낮다.

NLL과 Brier는 두 schema 모두 offset-only가 conditioned보다 좋았다. Accuracy 증분과 확률 품질을 구분해야 한다. 잔차 norm 정규화나 learned fallback을 평가한 것은 아니고, 더 큰 모델이나 새로운 실제 그래프에 대한 성능을 보여준 것도 아니다.

LP만보다 8%p 높다는 비교를 새 기여로 삼으면 안 된다. 기존 plain도 이미 LP보다 높다. 핵심 대조는 같은 파라미터·초기화·학습 과제의 plain과 offset-only이며, 이후 연구에서 입증해야 할 것은 이들 대비 반복 가능한 증분이다.

Frozen 실데이터 비율식과 이 합성 학습 결과는 [분리해서 확인한다](C:/Users/seongjin/Desktop/code/gts_pfn/counterfactual_residual/REPORT.md). 비율 식의 실패로 학습된 보정을 부정하지도 않고, 이 작은 shape 결과로 실제 heterograph 전이를 주장하지도 않는다.
