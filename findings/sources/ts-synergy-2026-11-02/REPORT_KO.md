# 공동 정보·nuisance TSFM: frozen 비교 결과

2026-11-02. **이 product bank에서는 제안2×2 학습으로 진행하지 않는다. 학습법은 미실행이며 실패로 세지 않는다.** 먼저 확인하려던 “유용한 쌍을 잘 활용하지만 nuisance 추가에 취약”이라는 두 조건이 함께 나타나지 않았다. 전체 후보 곱 특징+Lasso는 이 toy를 거의 noise floor까지 해결했다. 이 결과는 일반적인 공동정보 학습 또는 다른 TSFM의 실패 판정이 아니다.

## Primary: product64개 독립 task

History512, task당16개 rolling1-step origin, lag0..3, 유용2채널+동일 iid Rademacher nuisance최대8개다. 숫자는 checkpoint21quantile 평균 **twice pinball loss**, 낮을수록 좋다. 입력 개입 u/v/uv에는 유용 변수 정체가 사용되지만 full uvN8에는 쌍 표시가 없다. [계산·프로토콜](METHOD_KO.md).

| 방법 | H | H+u | H+v | H+uv | +N3 | +N8 |
|---|---:|---:|---:|---:|---:|---:|
| raw_ridge | 0.52949 | 0.53433 | 0.53654 | 0.54127 | 0.55213 | 0.56693 |
| interaction_ridge | 0.52949 | 0.53433 | 0.53654 | 0.08431 | 0.11067 | 0.38917 |
| interaction_lasso | 0.52464 | 0.52605 | 0.52644 | 0.08350 | 0.08499 | 0.08552 |
| chronos2 | 0.52529 | 0.52462 | 0.52391 | 0.52376 | 0.52387 | 0.52422 |
| old_null_lora | 0.52529 | 0.52463 | 0.52391 | 0.52378 | 0.52391 | 0.52426 |
| correlation_chronos2 | 0.52529 | 0.52462 | 0.52391 | 0.52376 | 0.52372 | 0.52365 |

참 쌍·지연·부호·noise를 아는 privileged oracle의 실현 loss는 0.081099다. 실용 baseline이 아니며 정상적인 회귀 fit과 정보량이 다르다.

| 방법 | G_joint, 95%task-paired CI | P_null8, 95%task-paired CI |
|---|---:|---:|
| raw_ridge | -0.013196 [-0.016315, -0.010151] | +0.025662 [+0.018315, +0.033486] |
| interaction_ridge | +0.443764 [+0.436215, +0.451623] | +0.304855 [+0.289376, +0.320575] |
| interaction_lasso | +0.439141 [+0.431403, +0.447187] | +0.002025 [+0.000650, +0.003395] |
| chronos2 | -0.000555 [-0.001054, -0.000031] | +0.000451 [-0.000443, +0.001342] |
| old_null_lora | -0.000566 [-0.001061, -0.000048] | +0.000478 [-0.000380, +0.001340] |
| correlation_chronos2 | -0.000555 [-0.001054, -0.000031] | -0.000110 [-0.000716, +0.000489] |

Frozen Chronos는 clean pair에서도 history 대비 차이 +0.001527 [-0.000420, +0.003614]다. 이는 평균 예측이 완전히 동일하다는 뜻은 아니지만, clean pair 활용의 안정적인 이득은 확인되지 않았다. N8의 손실 증가는 +0.000451 [-0.000443, +0.001342]로 CI가0을 포함한다. 작은 nuisance penalty 자체를 성공으로 세면 context를 거의 사용하지 않는 모델을 보상하게 된다.

Lasso는 참 쌍만 받은 oracle이 아니다. 10채널×4lag의 raw40항, target history4항과 모든 채널쌍×lag쌍720항을 함께 받았다. full-context loss차이(Lasso−Chronos)는 -0.438694 [-0.444621, -0.432558]다. Raw-Ridge history에서 privileged oracle까지 가능한 평균 gap의 99.01%를 회수했다. 이 gap-recovery는 평균으로 계산한 기술통계이며 별도 CI를 붙이지 않았다.

같은 전체 interaction 사전을 받은 Ridge는 N8에서 0.38917로 악화했다. 따라서 **전체 곱 특징을 만들기만 하면 항상 충분하다**고 결론 내리지 않는다. 이 sparse known-family에서는 전체 사전 위 희소 회귀까지 포함한 강한 대조가 작동했다.

이전 null-invariance LoRA도 실제 checkpoint를 사용했으며 이번 과제의 clean synergy 활용을 회복하지 못했다. 이는 Alibaba에서 학습한 adapter의 이 bank 전이에 관한 결과다. 공동정보 과제를 포함한 새 학습의 실패가 아니다.

## Additive32task 양성 대조

| 방법 | H | H+uv | +N8 |
|---|---:|---:|---:|
| raw_ridge | 0.55050 | 0.08849 | 0.09276 |
| interaction_ridge | 0.55050 | 0.08949 | 0.38828 |
| interaction_lasso | 0.54882 | 0.08797 | 0.09267 |
| chronos2 | 0.55621 | 0.53498 | 0.54225 |
| old_null_lora | 0.55621 | 0.53692 | 0.54552 |
| correlation_chronos2 | 0.55621 | 0.53498 | 0.53498 |

Additive에서 Chronos clean G는 +0.016244 [+0.010754, +0.022476], N8 penalty는 +0.007261 [+0.001533, +0.013717]였다. 상관 screening은 유용한 쌍을 회수해 clean/full을 사실상 같게 만들었다. 입력 경로가 전혀 정보를 전달하지 않는다는 설명에는 반대되는 대조지만, Chronos가 이 선형 과제를 잘 해결한다는 뜻도 아니다. Task-specific raw Ridge는 훨씬 낮은 loss다. 이 결과를 product primary 대신 승격하지 않는다.

## 비용과 검증

각 task16origin 전체 예측의 반복 측정이다. Lasso는 feature 생성·과거 alpha 선택·refit·residual quantile calibration까지 포함한다. Chronos는 공통 checkpoint 로딩을 제외한다. CPU입력→CPU출력,8task×5반복·3warmup,4CPUthreads,공유RTX3090 환경이다.

| 방법 | median ms/task16origins | p10–p90 ms |
|---|---:|---:|
| interaction_lasso | 42.90 | 37.49–47.22 |
| chronos2 | 119.90 | 118.32–121.11 |

학습 체계는 다르다: Lasso는 task별 과거를 fit하고 Chronos는 zero-shot이다. 그러나 이 toy의 해결에 대형 모델을 추가 학습할 실용적 이유가 생겼는지를 확인하는 기준으로는 유지한다. 실데이터의 일반적인 품질·비용 우위 주장이 아니다.

27,648 TSFM group forecast,1,728개 classical final fit(내부 alpha후보 fit 별도)을 실행했다. TSFM parameter update는0이다. 저장된 모든 loss를 raw forecast에서 재계산했고, 두 family의36forecast/model을 다시 실행했다. Replay 최대 오차는 1.07e-06. Future 값 교란 시 입력·screening·feature가 그대로인지96task에서 확인했다. Direct forward와 public pipeline 차이는0, covariate 순열·다른 task의 batch 혼합 차이는 약2.4e-7이었다. [Audit](results/audit.json),[통계](results/statistics.json),[비용](results/benchmark.json).

## 판정과 남은 범위

- 사전 joint gate: False; nuisance gate: False; interaction toy-solution floor: True; 2×2 진행: False.
- 닫는 범위는 이 product toy를 근거로 곧바로 conditional-null consistency 학습을 확대하는 경로다. D/A/B/C의 training 성능은 측정하지 않았다.
- 기존 모델이 비선형 연산을 원천적으로 못한다거나 사전학습 prior가 선형뿐이라는 결론은 아니다. 확장 Chronos 원고에는 MLP/RNN 변환이 명시돼 있다.
- 현재 bank는 iid·짧은 고정lag·희소한 한 쌍이다. 조밀하거나 regime별로 바뀌는 상호작용, 실데이터, unseen mechanism, t0·다른 TSFM은 평가하지 않았다. 결과를 살리기 위해 이 bank에서 사후 재조정하지 않는다.
- 향후 재개에는 강한 모든 후보 interaction/sparse regression을 유지하면서, clean 공동정보 활용과 nuisance 손해가 함께 존재하고 단순 과거-fit 특징화만으로 해결되지 않는 독립 과제가 필요하다. 그때만 원래2×2의 독립적 학습 기여를 검증한다.
