# Observed-feature intervention: ACM / IMDB

99.5% whole-node target feature missing, 클래스당 context 라벨 20개, seed 11/12/13, 전체 public test. 기존 Full 12개와 새 frozen GraphPFN 추론 48개를 비교했다. 모델 학습·selector 조정은 하지 않았다. 모든 예측을 저장한 뒤 test 정답을 읽었다. 두 그래프는 개발용이다.

**Full은 완전 관측 그래프가 아니라, 원래 99.5% 결측 마스크에서 남아 있는 실제 관측값을 사용하는 기준이다.** 같은 context/query의 기존 metapath LP 평균 정확도는 ACM 85.55%, IMDB 35.41%다.

## 현재 판정

**IMDB의 LP 대비 평균 우위는 실제 관측 피처를 제거해도 거의 유지됐다.** 원래 FISF+PFN 44.55% → No-observed 44.51%, 감소 0.04%p다. 독립 injection을 고정하고 clamp 위치를 유지한 controlled Full−Zero도 0.03%p였다. 현재 설정에서 약 9%p 우위를 실제 관측값의 순이득으로 설명할 근거는 약하다.

같은 context/query에서 구조 입력만 남은 PCFI No-observed는 41.55%로, 기존 구조 입력 GraphPFN 예측과 정확히 일치했다. LP 35.41% 대비 Full FISF의 차이 9.13%p는 산술적으로 구조 입력 PFN의 차이 6.13%p + 피처 없는 FISF 파이프라인과 구조 입력 PFN의 차이 2.96%p + 실제 관측 경로의 순차이 0.04%p로 적을 수 있다. 이는 서로 다른 파이프라인 간 정확도 차이의 회계이며, 합성 주입 하나의 인과적 기여 분해가 아니다.

**평균 변화가 작다는 것은 피처가 예측에 영향을 주지 않는다는 뜻이 아니다.** IMDB의 End-to-end 제거 전후 예측은 평균 30.47% 바뀌었다. Controlled Full−Zero에서도 26.89%가 바뀌며, Full이 맞고 Zero가 틀린 노드 9.95%와 그 반대 9.92%가 거의 상쇄된다. Query별 양·음의 영향은 있지만, 이를 관측 가능한 신호로 구분할 수 있는지는 이번 실험에서 검증하지 않았다.

ACM의 controlled alignment 차이는 평균 +3.01%p였지만 seed별 +4.29/+5.53/−0.78%p로 일관되지는 않았다. Controlled 실제 값 효과도 평균 +1.03%p가 한 seed의 +6.62%p에 의존했다. 또한 해당 PFN은 feature 없는 LP 85.55%에 미치지 못한다.

**Gate C는 계속 보류한다.** Missing-feature recovery를 메인 가설로 밀기보다, 구조 입력·합성 입력 경로가 현재 PFN의 우위를 어떻게 만드는지 먼저 분리하는 편이 현재 결과에 더 부합한다. 관측 피처가 본질적으로 무용하거나 query별 선택이 불가능하다는 결론은 아니다.

## End-to-end: 원래 파이프라인 전체를 다시 계산

Full은 기존 원본 좌표 PCFI/FISF+typed paths+GraphPFN이다. No-observed는 모든 실제 피처를 NaN으로 바꾼 뒤 원래 알고리즘을 재실행한다. Shuffled는 관측 노드들 사이에서 행 벡터 전체를 단일 cycle 순열로 옮긴다. Mask·관측 개수·피처값 multiset은 유지되며 고정점은 없다. 같은 순열을 모든 arm에 사용했다. 피처 보간, SVD32, path 집계, quantile/PCA64는 각 조건에서 다시 계산했다.

수치는 세 seed 평균 정확도(%), Δ는 Full−intervention(%p)다. Δ가 양수면 해당 intervention으로 정확도가 하락한 것이다.

| Graph | Imputer | Full | No-observed | Shuffled | Δ feature | Δ alignment |
|---|---|---:|---:|---:|---:|---:|
| ACM | pcfi | 75.45 | 74.53 | 74.31 | 0.92 | 1.14 |
| ACM | fisf | 74.36 | 74.14 | 73.38 | 0.22 | 0.98 |
| IMDB | pcfi | 42.61 | 41.55 | 42.67 | 1.06 | -0.06 |
| IMDB | fisf | 44.55 | 44.51 | 44.30 | 0.04 | 0.25 |

## Controlled FISF: 관측값과 독립인 injection을 고정

S₀의 채널은 raw 폭만을 사용하는 고정 RNG 순열로 고르고, anchor·값은 원래 FISF의 RandomState(0) / Torch Generator(0)로 만든다. Full 값·라벨·Full의 low-variance 선택은 S₀에 들어가지 않는다. 모든 조건이 같은 S₀와 원래 관측 위치 M으로 계산한 q(M)=0.9^거리 가중치를 공유한다. No-observed에서는 실제 관측 clamp를 모두 제거하지만 q(M)는 유지한다. Synthetic anchor가 실제 관측 cell을 덮어쓰는 원래 동작도 유지했다.

측정값은 Acc[T(Xobs,M,S₀,q(M))]−Acc[T(∅,∅,S₀,q(M))]와 shuffled 대비 차이다. SVD·quantile·PCA는 arm별로 새로 적합한다. 따라서 고정 injection·고정 source-confidence 아래 실제 관측 경로와 그 후속 변환을 통한 효과이며, 원래 adaptive FISF의 natural direct effect나 mask geometry 제거 효과는 아니다. Controlled Full도 원래 Full과 다른 기준 모델이므로 두 regime의 Full을 섞어 contrast를 계산하지 않는다.

**Test 성능 평가 전에 추가한 zero-observed 대조:** 원래 실제 관측 위치의 clamp를 유지하되 값을 전부 0으로 바꾼다. S₀와 q(M)는 동일하다. Full−Zero는 zero boundary 대비 실제 숫자값 경로의 효과, Zero−No는 zero clamp 유지 경로의 효과다. 후속 변환 재적합은 포함한다. 이 보조 arm 6개는 식에서 clamp availability 효과가 남는다는 점을 발견하여 zero_addendum.json에 고정했고, 새 test 성능을 보며 추가한 것은 아니다.

| Graph | Controlled Full | No-observed | Zero-observed | Shuffled | Δ Full−No | Δ Full−Zero | Δ alignment | Δ Zero−No |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACM | 75.31 | 74.08 | 74.29 | 72.30 | 1.23 | 1.03 | 3.01 | 0.21 |
| IMDB | 43.78 | 43.47 | 43.75 | 43.07 | 0.31 | 0.03 | 0.71 | 0.28 |

## Seed별 contrast: 같은 그래프의 반복임에 주의

| Graph | Imputer/regime | Intervention | Δ s11 | Δ s12 | Δ s13 |
|---|---|---|---:|---:|---:|
| ACM | pcfi/e2e | no_observed | 0.71 | 2.11 | -0.06 |
| ACM | pcfi/e2e | shuffled | 2.42 | 1.43 | -0.43 |
| ACM | fisf/e2e | no_observed | 0.03 | 2.58 | -1.96 |
| ACM | fisf/e2e | shuffled | -0.47 | 2.11 | 1.30 |
| ACM | fisf/controlled | no_observed | -1.24 | 5.22 | -0.28 |
| ACM | fisf/controlled | shuffled | 4.29 | 5.53 | -0.78 |
| ACM | fisf/controlled | zero_observed | -1.74 | 6.62 | -1.80 |
| IMDB | pcfi/e2e | no_observed | 1.15 | 3.45 | -1.41 |
| IMDB | pcfi/e2e | shuffled | -1.50 | 3.28 | -1.96 |
| IMDB | fisf/e2e | no_observed | 1.32 | -0.32 | -0.89 |
| IMDB | fisf/e2e | shuffled | 0.78 | -0.17 | 0.14 |
| IMDB | fisf/controlled | no_observed | -0.35 | 0.89 | 0.37 |
| IMDB | fisf/controlled | shuffled | -0.81 | 2.56 | 0.37 |
| IMDB | fisf/controlled | zero_observed | -1.55 | 1.67 | -0.03 |

## 실제 PFN 입력 차원

원래의 constant-column dropping을 유지했다. No-observed PCFI는 zero imputation이 되어 PFN에 전달하는 비상수 열이 줄어든다. End-to-end 차이에는 이런 입력 표현 변화도 포함된다. 조건부 상호정보량이나 순수한 정보 효과를 추정한 것은 아니다.

| Graph | Imputer/regime | Full | No-observed | Zero-observed | Shuffled |
|---|---|---|---|---|---|
| ACM | pcfi/e2e | 64,64,64 | 5,5,5 | — | 64,64,64 |
| ACM | fisf/e2e | 64,64,64 | 64,64,64 | — | 64,64,64 |
| ACM | fisf/controlled | 64,64,64 | 64,64,64 | 64,64,64 | 64,64,64 |
| IMDB | pcfi/e2e | 64,64,64 | 2,2,2 | — | 64,64,64 |
| IMDB | fisf/e2e | 64,64,64 | 64,64,64 | — | 64,64,64 |
| IMDB | fisf/controlled | 64,64,64 | 64,64,64 | 64,64,64 | 64,64,64 |

## 결과 그림

![Intervention accuracy](C:/Users/seongjin/Desktop/code/gts_pfn/feature_intervention/results/intervention_accuracy.png)

막대는 세 seed 평균, 점은 각 seed다. Controlled FISF는 별도의 Full을 갖는다. Zero-observed는 controlled FISF에만 해당한다.

## 해석 범위

- 정확도 외에 balanced accuracy, macro-F1, NLL, Brier, mean confidence/margin을 metrics.csv에 저장했다. Brier가 낮다는 사실만으로 calibration이 더 좋다고 단독 판정할 수는 없다.
- Seed별 mask와 context가 함께 달라지고 test는 중첩된다. 동일 그래프의 세 seed를 독립 그래프 표본처럼 취급하지 않는다. 순열도 seed당 하나이며, 작은 Δ를 확정적 효용으로 판정하지 않는다.
- Full이 제거 arm보다 높으면 해당 pipeline에서 관측 경로의 기여를 뒷받침한다. 차이가 작거나 음수여도 원래 피처에 예측 정보가 없다는 증명은 아니다. 모델이 그 정보를 활용하지 못할 수 있다.
- No-observed FISF에도 synthetic features가 남는다. PCFI와 FISF 반응 차이는 synthetic 경로의 가능성을 시사하지만, adaptive channel 선택·거리 가중치·후속 변환도 달라지므로 합성 주입 하나의 인과 효과로 확정할 수 없다.
- Zero-observed는 배포 성능을 겨루는 baseline이 아니라 clamp 위치를 보존하는 기전 대조다. 0 벡터는 실제 관측값의 분포와 다를 수 있다.
- Gate C와 양방향 모델 학습은 보류 상태다. 새 실제 그래프와 다른 missing 메커니즘의 일반화는 검증하지 않았다.

프로토콜: protocol.json, zero_addendum.json. 구현 검증: implementation_checks.json. 지표·paired node 결과·검증: results/. 재현: run_all.ps1.
