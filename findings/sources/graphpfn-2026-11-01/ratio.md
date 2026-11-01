# LP prior + counterfactual PFN residual: frozen MVP

ACM/IMDB, 99.5% whole-node missing, k=20/class, seed 11/12/13, 전체 public test. λ=1, ε=1e-12를 새 결과 계산 전에 고정했다. λ/temperature/규칙 tuning은 하지 않았다. 기존 audited 예측을 사용하며 새 PFN forward와 모델 학습은 모두 0회다. 그래프들은 개발용이다.

## 현재 판정

**Frozen counterfactual 비율 보정은 성공하지 못했다.** FISF end-to-end의 ACM 82.35%는 LP 85.55%보다 낮고, IMDB 34.45%는 LP 35.41%와 Full PFN 44.55%보다 낮았다. Controlled Full−Zero residual도 ACM 82.22% / IMDB 34.50%로 이 실패를 해결하지 못했다. PCFI의 IMDB 38.40%는 LP를 넘었지만 Full 42.61%와 50:50 40.80%에는 못 미쳤다.

FISF end-to-end에서 ACM은 LP 오답→정답 3.50%, 정답→오답 6.70%였고 IMDB는 11.13% / 12.10%였다. 보정의 방향과 크기가 기존 정답을 보호하지 못했다. Shuffled residual이 실제 Full residual보다 높은 경우도 있으므로, 비율을 feature-dependent evidence로 읽는 해석은 더욱 조심해야 한다.

**사후 민감도 진단:** LP의 top-1/top-2 log margin 중앙값은 ACM seed별 0.740–0.908, IMDB 0.134–0.151이었다. FISF Δ norm 평균은 IMDB가 더 작았지만, argmax 변경률은 ACM 11.13% / IMDB 34.68%였다. 약한 prior margin이 작은 교란에도 민감하다는 관측과 일치하나, 그래프 간 성능 차이의 인과적 설명을 입증한 것은 아니다. LP/neutral의 probability floor 적용 cell은 모든 case에서 0개였다. 이 진단으로 λ나 규칙을 바꾸지 않았다.

이 결과는 별도 합성 학습을 하는 LP-conditioned 모델의 반증이 아니다. 내부 LP conditioning을 검증한 소형 학습 실험은 [별도 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/REPORT.md)에 구분했다.

## 정의와 neutral의 의미

p_final = softmax(log(max(p_LP,ε)) + log(max(p_evidence,ε)) − log(max(p_neutral,ε))). 각 log는 float64로 계산한다.

- PCFI/FISF end-to-end: Full 대비 실제 관측값을 제거하고 전체 파이프라인을 다시 계산한 No-observed. FISF 합성 입력은 제거 arm에도 남고, injection/전처리 변화도 residual에 포함된다.
- Controlled FISF No-observed: 실제 값과 독립인 injection과 원래 mask 거리 가중치를 고정하고 실제 clamp만 제거한다.
- Controlled FISF Zero-observed: 같은 injection·거리 가중치·실제 clamp 위치를 유지하고 실제 숫자값만 0으로 바꾼다. 후속 변환은 각 arm에서 새로 적합했다.
- Shuffled는 미리 고정한 alignment 음성 대조다. Neutral은 그대로 두고 numerator만 기존 Shuffled PFN 예측으로 바꾼다. Test 결과로 채택할 방법을 고르는 용도가 아니다.

## λ=1 결과: 평균 accuracy (%)

| Graph | PFN pair | LP | Full PFN | Neutral PFN | 50:50 LP/Full | LP + Full residual | LP + Shuffled residual |
|---|---|---:|---:|---:|---:|---:|---:|
| ACM | pcfi_e2e_no_observed | 85.55 | 75.45 | 74.53 | 83.65 | 79.54 | 78.73 |
| ACM | fisf_e2e_no_observed | 85.55 | 74.36 | 74.14 | 82.20 | 82.35 | 84.08 |
| ACM | fisf_controlled_no_observed | 85.55 | 75.31 | 74.08 | 83.12 | 82.08 | 82.66 |
| ACM | fisf_controlled_zero_observed | 85.55 | 75.31 | 74.29 | 83.12 | 82.22 | 82.48 |
| IMDB | pcfi_e2e_no_observed | 35.41 | 42.61 | 41.55 | 40.80 | 38.40 | 37.72 |
| IMDB | fisf_e2e_no_observed | 35.41 | 44.55 | 44.51 | 42.54 | 34.45 | 33.33 |
| IMDB | fisf_controlled_no_observed | 35.41 | 43.78 | 43.47 | 41.26 | 35.39 | 37.14 |
| IMDB | fisf_controlled_zero_observed | 35.41 | 43.78 | 43.75 | 41.26 | 34.50 | 35.88 |

## 보정 방향과 크기: Full numerator

Rescued는 LP 오답→보정 정답, Harmed는 LP 정답→보정 오답이다. 차이가 accuracy 증분과 일치한다. Δ는 클래스 공통 상수를 제거한 뒤 L2 norm을 계산했다. Rival advantage는 정답 클래스와 LP의 가장 강한 오답 클래스 사이의 Δ 차이다. 이 분석에만 test 정답을 사용했다.

| Graph | Pair | Rescued (%) | Harmed (%) | Changed (%) | Mean ‖Δ centered‖₂ | Rival advantage > 0 (%) |
|---|---|---:|---:|---:|---:|---:|
| ACM | pcfi_e2e_no_observed | 3.19 | 9.21 | 14.17 | 0.605 | 46.38 |
| ACM | fisf_e2e_no_observed | 3.50 | 6.70 | 11.13 | 0.432 | 56.35 |
| ACM | fisf_controlled_no_observed | 3.54 | 7.02 | 11.59 | 0.482 | 57.54 |
| ACM | fisf_controlled_zero_observed | 3.69 | 7.02 | 12.02 | 0.502 | 58.14 |
| IMDB | pcfi_e2e_no_observed | 11.27 | 8.28 | 28.80 | 0.359 | 51.84 |
| IMDB | fisf_e2e_no_observed | 11.13 | 12.10 | 34.68 | 0.312 | 47.60 |
| IMDB | fisf_controlled_no_observed | 10.38 | 10.40 | 31.36 | 0.290 | 48.84 |
| IMDB | fisf_controlled_zero_observed | 9.94 | 10.85 | 31.51 | 0.307 | 47.56 |

## Seed별 정확도 증분: corrected minus LP (%p)

| Graph | Pair | s11 | s12 | s13 |
|---|---|---:|---:|---:|
| ACM | pcfi_e2e_no_observed | -5.84 | -0.84 | -11.37 |
| ACM | fisf_e2e_no_observed | -4.07 | -1.99 | -3.54 |
| ACM | fisf_controlled_no_observed | -5.25 | -3.08 | -2.11 |
| ACM | fisf_controlled_zero_observed | -6.03 | -1.52 | -2.45 |
| IMDB | pcfi_e2e_no_observed | 1.44 | 6.70 | 0.83 |
| IMDB | fisf_e2e_no_observed | -2.82 | -1.52 | 1.44 |
| IMDB | fisf_controlled_no_observed | -0.46 | 2.70 | -2.30 |
| IMDB | fisf_controlled_zero_observed | -2.42 | 1.35 | -1.67 |

## 보정이 LP에 준 이득과 손해

![Correction changes](C:/Users/seongjin/Desktop/code/gts_pfn/counterfactual_residual/results/correction_changes.png)

## 해석의 범위

- 같은 checkpoint의 두 실행이지만 neutral intervention은 p(y|G,Y_C)를 marginalize한 분포라고 보장되지 않는다. 따라서 이 비율을 엄밀한 likelihood ratio, Bayes posterior 또는 중복 증거 제거의 증명으로 해석하지 않는다.
- 평균 Full/Neutral accuracy가 같아도 확률 벡터와 Δ가 작다는 뜻은 아니다. Centered Δ와 paired gain/loss를 함께 봐야 한다.
- LP-conditioned residual head는 별도 가설이다. Frozen 출력 비율의 성공·실패가 그 학습 모델의 성공·불가능성을 증명하지 않는다. 이번 실험에서는 학습하지 않았다.
- 같은 graph/test를 공유하는 세 seed의 평균이다. 독립 그래프 일반화 CI를 계산하지 않는다. ACM/IMDB 결과에 맞춘 λ, clipping, 부호 반전은 하지 않았다.
- 실제 배포에서는 Full/Neutral 두 PFN 실행이 필요하다. 50:50 baseline보다 정보·추론 비용이 많다. 이번 실행은 저장 예측 재활용 진단으로, 동일 budget 최종 비교가 아니다.

지표: results/metrics.csv. Residual magnitude quintile: results/magnitude_bins.csv. Query별 오류: results/queries/. 고정 규칙: protocol.json. 이론·선행연구: METHOD.md.
