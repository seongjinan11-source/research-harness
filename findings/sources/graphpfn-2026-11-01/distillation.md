# Role-Conditional Distillation: 고정 아키텍처 학습 ablation

> 2026-10-31 · Stage 1 exact teacher · 새 모델63개 + 기존 기준선18개 재사용 · 3개 prior × 3개 학습 seed

**판정: 역할별 conditional 감독은 의도한 오차를 줄였지만, 최종 예측의 유의미한 증분은 확인하지 못했다. 현재 staged decoder에 이 loss를 추가하는 방식으로 Stage 2를 확장할 근거는 약하다.** Posterior 추론의 예측 가치가 존재한다는 이전 결과를 부정하는 것은 아니다.

- C−B의 weighted conditional KL은 세 prior에서 모두 감소했다. 그러나 최종 Bayes predictive KL 차이의 CI는 세 prior 모두 0을 포함한다.
- D−C의 posterior KL은 세 prior에서 모두 감소했지만, 이번 고정 예산의 predictive KL은 모두 증가했다.
- Exclusive에서는 JR 계열이 Large Plain distill보다 여전히 좋다. 하지만 B·기존 A에도 이미 존재하는 차이여서 추가 conditional/posterior loss의 성과로 돌릴 수 없다.
- 동일 family에서 좋은 component fit이 좋은 최종 predictive approximation을 자동 보장하지 않는다. 아래 비교는 재학습된 fixed-prior/factorized 대조군도 포함한다.

## 1. 먼저 바로잡은 기존 baseline

**기존 JR A는 sampled-role auxiliary CE가 아니라 exact posterior KL을 이미 사용했다.** 따라서 이번에 새로 추가한 핵심 감독은 exact marginal predictive target과 role-conditional predictive target이다. A−B는 target 변경과 posterior auxiliary 제거가 함께 일어나므로, predictive target의 순수 효과를 위해 A0도 유지했다.

| Arm | 최종 감독 | Conditional weight | Posterior weight | 파라미터 |
|---|---|---:|---:|---:|
| A 기존 JR | sampled-label CE | 0 | 0.1 | 14,503 |
| A0 CE only | sampled-label CE | 0 | 0 | 14,503 |
| B Predictive | exact predictive KL | 0 | 0 | 14,503 |
| C Conditional | exact predictive KL | 1 | 0 | 14,503 |
| D All | exact predictive KL | 1 | 0.1 | 14,503 |
| Factorized D | D와 같은 세 loss, product-marginals aggregation | 1 | 0.1 | 14,503 |
| Known-prior D | D와 같은 세 loss, fixed prior aggregation | 1 | 0.1 | 14,503 |
| Plain distill | exact predictive KL | 0 | 0 | 10,469 |
| Large Plain distill | exact predictive KL | 0 | 0 | 14,655 |

Known-prior D도 같은 posterior head를 학습하지만 최종 집계에는 쓰지 않는다. Large Plain은 실제 relation MLP 폭을78로 넓히고 feature/label diffusion을80/48회로 늘린 기존 대조군이다. 다른 모델은20/12회다. 정확한 FLOP equality 주장은 하지 않는다.

## 2. 학습 연산과 동일 예산

\[\mathcal L_{pred}=\frac1{|Q|}\sum_{i\in Q}KL(p_i^*\Vert p_{\theta,i}),\]
\[\mathcal L_{cond}=\frac1{|Q|}\sum_{i\in Q}\sum_h q_h^*KL(p_{ih}^*\Vert p_{\psi,ih}),\quad \mathcal L_{post}=KL(q^*\Vert q_\phi).\]

Query별 marginal 분포를 감독한다. **Joint query-label 분포의 KL을 학습했다고 주장하지 않는다.** Conditional role weight는 detached exact posterior이며 네 가설을 모두 열거한다. 작은 양의 posterior를 잘라내지 않는다. 다만 Exclusive/Coupled의 prior-zero 상태는 conditional weight가0이다. Uniform role conditional KL도 평가에 별도로 저장했다.

Alpha=1, beta=0.1을 사전에 고정했다. AdamW lr=0.001, weight decay=1e−4, gradient clip5, 192 updates × batch16, seed17/29/43을 사용했다. A~D는 전체 초기 state·파라미터 수·과제 순서·학습 step이 같다. 역할별 branch는 독립 모델이 아니라 기존 adapter 가중치를 공유한다. 새 아키텍처·head는 추가하지 않았다.

A와 A0의18개 체크포인트는 기존 것을 재사용했다. 학습 bank/source hash, 초기 adapter hash, update 수와 실제 task schedule hash를 확인했다. B/C/D와 나머지 대조군63개를 새로 학습하여 실제 새 optimizer update는12,096회다. A를 다시 학습한 것처럼 새 GPU 시간에 포함하지 않는다.

Teacher는 기존 소형 SBM의 정확한 bank를 재사용한다. 모든 latent query/helper label과 네 role state를 주변화한 값이다. Student의 inference input에는 graph와 관측 context label만 들어가며 sampled hidden role/query truth는 들어가지 않는다. Conditional supervision은 Plain의 marginal target보다 더 풍부한 training annotation이라는 점도 구분한다.

## 3. 새 graph에서 주 평가

각 prior의 새 graph192개(seed8,100,000부터, family별100,000 stride)를 사용했다. Feature는 완전히 없고 target8/helper4/role2/class2다. 주 평가는 훈련과 같은 context signal0.72·총 context1/2/4개다. **클래스당 k 조건이나 실제 heterograph 평가가 아니다.** 각 graph에서 context와 학습 seed를 평균했다.

| Prior | Arm | Acc % | NLL | Brier | Bayes KL | Conditional KL | Posterior KL |
|---|---|---:|---:|---:|---:|---:|---:|
| independent | A 기존 JR | 56.39 | 0.66370 | 0.47181 | 0.06616 | 0.09392 | 0.19801 |
| independent | A0 CE only | 56.35 | 0.66079 | 0.46896 | 0.06322 | 0.09995 | 0.72602 |
| independent | B Predictive KL | 56.35 | 0.65891 | 0.46749 | 0.06192 | 0.10181 | 0.69306 |
| independent | C + Conditional | 56.26 | 0.66040 | 0.46872 | 0.06211 | 0.09384 | 0.47543 |
| independent | D + Posterior | 56.46 | 0.66263 | 0.47101 | 0.06615 | 0.09308 | 0.19121 |
| independent | Factorized D | 56.48 | 0.66340 | 0.47174 | 0.06680 | 0.09307 | 0.18911 |
| independent | Known-prior D | 56.11 | 0.66525 | 0.47323 | 0.06831 | 0.09315 | 0.18616 |
| independent | Plain distill | 55.88 | 0.67217 | 0.47960 | 0.07304 | — | — |
| independent | Large Plain distill | 56.04 | 0.67003 | 0.47752 | 0.07140 | — | — |
| exclusive | A 기존 JR | 57.92 | 0.66338 | 0.47109 | 0.05034 | 0.06628 | 0.05930 |
| exclusive | A0 CE only | 57.92 | 0.66303 | 0.47076 | 0.05035 | 0.06630 | 0.06449 |
| exclusive | B Predictive KL | 57.94 | 0.66274 | 0.47059 | 0.04964 | 0.06698 | 0.07960 |
| exclusive | C + Conditional | 57.97 | 0.66263 | 0.47042 | 0.04946 | 0.06449 | 0.06959 |
| exclusive | D + Posterior | 57.95 | 0.66307 | 0.47079 | 0.04994 | 0.06438 | 0.05932 |
| exclusive | Factorized D | 54.99 | 0.67190 | 0.47947 | 0.05941 | 0.06463 | 0.06286 |
| exclusive | Known-prior D | 57.58 | 0.66539 | 0.47289 | 0.05320 | 0.06446 | 0.05847 |
| exclusive | Plain distill | 53.86 | 0.68464 | 0.49156 | 0.07185 | — | — |
| exclusive | Large Plain distill | 53.91 | 0.67706 | 0.48460 | 0.06555 | — | — |
| coupled | A 기존 JR | 55.10 | 0.67438 | 0.48204 | 0.08837 | 0.10829 | 0.20235 |
| coupled | A0 CE only | 55.26 | 0.67427 | 0.48191 | 0.08771 | 0.11047 | 0.41247 |
| coupled | B Predictive KL | 55.25 | 0.67292 | 0.48069 | 0.08664 | 0.11158 | 0.48053 |
| coupled | C + Conditional | 55.10 | 0.67275 | 0.48057 | 0.08606 | 0.10789 | 0.26078 |
| coupled | D + Posterior | 55.13 | 0.67328 | 0.48107 | 0.08762 | 0.10774 | 0.19556 |
| coupled | Factorized D | 55.13 | 0.67347 | 0.48139 | 0.08629 | 0.10793 | 0.19645 |
| coupled | Known-prior D | 55.11 | 0.67559 | 0.48290 | 0.08974 | 0.10785 | 0.19346 |
| coupled | Plain distill | 55.22 | 0.67335 | 0.48106 | 0.08736 | — | — |
| coupled | Large Plain distill | 55.19 | 0.67339 | 0.48110 | 0.08732 | — | — |

Conditional KL은 exact role posterior로 가중한 query marginal KL이다. Posterior KL은 exact joint-role posterior와의 차이다. 둘 다 Plain의 독립적인 예측과 직접 같은 지표로 비교하지 않는다.

## 4. 추가 loss의 독립적인 증분

아래는 new−control이다. Accuracy를 제외한 지표는 음수가 좋다. 95% CI는4,000회 paired graph bootstrap이며 세 학습 seed는 graph 내부에서 평균한다. 따라서 이 세 fitted seed에 조건부인 graph sampling CI다. 여러 개발용 비교의 CI이며 전체 학습 불확실성이나 multiple-testing 보정을 포함하지 않는다.

| Prior | 비교 | Predictive KL 차이 [CI] | Conditional KL 차이 [CI] | Posterior KL 차이 [CI] |
|---|---|---:|---:|---:|
| independent | B_predictive - A0_ce_only | -0.001297 [-0.002174, -0.000481] | +0.001866 [+0.001389, +0.002374] | -0.032964 [-0.044188, -0.022529] |
| independent | C_conditional - B_predictive | +0.000188 [-0.000440, +0.000795] | -0.007978 [-0.010204, -0.005829] | -0.217628 [-0.260438, -0.177333] |
| independent | D_all - C_conditional | +0.004039 [+0.002707, +0.005418] | -0.000754 [-0.001322, -0.000204] | -0.284227 [-0.348550, -0.224022] |
| exclusive | B_predictive - A0_ce_only | -0.000712 [-0.001217, -0.000182] | +0.000676 [+0.000478, +0.000873] | +0.015110 [+0.008983, +0.021674] |
| exclusive | C_conditional - B_predictive | -0.000181 [-0.000683, +0.000329] | -0.002484 [-0.003197, -0.001750] | -0.010015 [-0.013152, -0.007092] |
| exclusive | D_all - C_conditional | +0.000479 [+0.000013, +0.000952] | -0.000110 [-0.000168, -0.000052] | -0.010266 [-0.015125, -0.005666] |
| coupled | B_predictive - A0_ce_only | -0.001069 [-0.001388, -0.000766] | +0.001109 [+0.000591, +0.001615] | +0.068057 [+0.056686, +0.079916] |
| coupled | C_conditional - B_predictive | -0.000582 [-0.001401, +0.000182] | -0.003686 [-0.006164, -0.001217] | -0.219753 [-0.261740, -0.175102] |
| coupled | D_all - C_conditional | +0.001561 [+0.000368, +0.002823] | -0.000158 [-0.000305, -0.000008] | -0.065214 [-0.085285, -0.045271] |

Exclusive에서 C의 conditional KL은0.06698→0.06449로 약3.7% 줄지만 predictive KL은0.04964→0.04946이다. Predictive 차이 −0.000181의 CI [−0.000683,+0.000329]는0을 포함한다. D에서는 posterior KL을 더 낮춰도 predictive KL은0.04994로 다시 커진다.

즉, supervision이 component fitting에는 작동했다. 하지만 그 개선이 최종 예측의 실질적 개선으로 이어졌다는 주장은 이 실험에서 지지되지 않는다. Loss weight를 test에 맞춰 다시 조정하지 않았다. 이 단일 예산/weight의 결과로 모든 conditional distillation이 불가능하다고 결론 내리지는 않는다.

### Strong teacher baseline 및 집계 대조군

| Prior | D − Large Plain: KL [CI] | D − Factorized: KL [CI] | D − Known prior: KL [CI] |
|---|---:|---:|---:|
| independent | -0.005255 [-0.007244, -0.003328] | -0.000655 [-0.000793, -0.000522] | -0.002163 [-0.003198, -0.001150] |
| exclusive | -0.015609 [-0.018464, -0.012696] | -0.009470 [-0.010775, -0.008213] | -0.003263 [-0.004600, -0.001991] |
| coupled | +0.000292 [-0.000851, +0.001565] | +0.001322 [+0.000523, +0.002102] | -0.002120 [-0.003550, -0.000654] |

Exclusive에서 D는 Large Plain distill보다 정확도 약4.04%p 높다. 그러나 기존 A도57.92%, B도57.94%, D는57.95%로 거의 같다. 이는 기존 JR 구조의 장점이 남는다는 결과이며, 새 conditional/posterior distillation의 고유 이득이라는 증거가 아니다.

## 5. 훈련에 없던 context 수와 signal

- Unseen context: signal0.72, 총 context3/6개.
- Unseen signal: signal0.36/0.96, 총 context1/2/4개.
- Unseen both: signal0.36/0.96, 총 context3/6개.

모든 모델을 같은 새 paired graph에서 평가했고 추가 학습은 없다. Oracle teacher는 평가 signal을 알고 계산하지만 student에는 signal metadata를 제공하지 않는다. 따라서 OOD에서의 Bayes KL은 새 생성분포에 대한 이상적 기준과의 차이이며, 학습 prior의 정확한 posterior를 보장하는 지표는 아니다. 새 semantic schema 일반화와 구분한다.

| OOD group | Prior | B KL | C KL | D KL | Large Plain KL |
|---|---|---:|---:|---:|---:|
| unseen_context | independent | 0.07974 | 0.07985 | 0.08497 | 0.09412 |
| unseen_context | exclusive | 0.06392 | 0.06347 | 0.06467 | 0.08445 |
| unseen_context | coupled | 0.09950 | 0.09888 | 0.09775 | 0.10041 |
| unseen_signal | independent | 0.05921 | 0.05916 | 0.06292 | 0.06666 |
| unseen_signal | exclusive | 0.05147 | 0.05190 | 0.05231 | 0.06310 |
| unseen_signal | coupled | 0.07957 | 0.07892 | 0.07876 | 0.08047 |
| unseen_both | independent | 0.07554 | 0.07555 | 0.07987 | 0.08640 |
| unseen_both | exclusive | 0.06936 | 0.07004 | 0.07127 | 0.08322 |
| unseen_both | coupled | 0.09346 | 0.09283 | 0.08845 | 0.09427 |

일부 OOD cell에서는 conditional/posterior 감독이 유리하지만, 모든 prior/shift에서 일관된 개선은 아니다. 세부 cell·각 seed·paired CI는 [전체 분석](results/analysis.json)에 보존했다.

## 6. 감독만으로 사라지지 않는 decoder 표현 한계

현재 feature-free staged adapter에서는 feature latent와 ridge readout이0이다. 남은 연산은 양의 relation weight와 양의 normalization을 쓰는 label diffusion이다. 관측 context label이 모두 class a이면 모든 query에서 $p_{model}(a)\geq1/2$다. 여러 role 가설의 비음수 확률 혼합도 이 제약을 유지한다.

반면 exact SBM은 nonedge를 이용해 반대 class를 지지할 수 있다. Bayes 분포를 위 반공간으로 투영한 KL은 이 decoder family의 유효한 하한이다. 이는 full family의 정확한 최적값은 아니며, 모든 남은 오차를 설명하는 값도 아니다.

| Prior | 주 평가의 부분 KL 하한 | D predictive KL |
|---|---:|---:|
| independent | 0.01924 | 0.06615 |
| exclusive | 0.01569 | 0.04994 |
| coupled | 0.02362 | 0.08762 |

이 구조에서는 teacher를 더 정확하게 주는 것과 그 target을 표현할 수 있는 것은 별개의 문제다. 특히 총 라벨1개와 관측 class가 하나뿐인 context를 기존 클래스당 k=1 문제와 혼동하지 않는다. [모든 저장 모델의 확률 제약 검사](results/family_bound_check.json)를 별도로 통과했다.

## 7. 최종 연구 판단

**현재 loss-only 제안을 논문 메인 메소드로 확정하거나 teacher-free scaling으로 확장하는 것은 보류한다.** 정확한 teacher가 있어도 현재 아키텍처에서 conditional 감독의 최종 predictive 증분이 작다. Posterior 분해의 정확도를 높였다는 사실만으로 분류 predictor의 우월성을 주장할 수 없다.

이전 Exact Bayes 분석이 보여준 role inference의 가치는 여전히 유효하다. 다음에 decoder를 바꾼다면 conditional target을 표현할 수 있는지부터 명시해야 하며, 그 새 구조에도 Plain predictive distillation 기준선을 유지해야 한다. 이번 요청에서는 아키텍처를 바꾸지 않았다.

## 검증과 재현

- 초기 전체 JR state, trainable parameter 수, train task schedule, fixed final checkpoint를 검사했다.
- 원래 bank/source/checkpoint는 불변이다. A/A0 재사용의 provenance도 검사했다.
- 새 평가 graph와 train graph는 disjoint이고 query-target incidence는 signal 간 paired다.
- Query/helper sampled truth는 student input과 distillation target 계산 경로에 들어가지 않는다.
- Context/helper teacher row를 query loss에서 먼저 제외했다. 완전한 teacher를 student로 넣으면 세 KL loss가0이다.
- 모든 모델의 relation/class 순열, 저장 확률 재집계, bootstrap 직접 계산 일치를 검사했다.
- 학습 전 검사 모듈 이름 충돌과 KL 마스킹 수정은 [수정 기록](results/pretraining_check_correction.json)에 있다. 수정 전 optimizer update는0이다.
- [감사](results/audit.json), [고정 프로토콜](protocol.json), [학습 기록](results/training.json), [재사용 기록](results/reused.json), [전체 분석](results/analysis.json).

```powershell
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/build_distill.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/train_distill.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/evaluate_distill.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/analyze_distill.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/audit_distill.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/family_bound.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/distillation/report_distill.py
```

기존 결과가 있으면 덮어쓰기를 거부한다. 새 경로에서 재현한다. Teacher-free scaling·큰 graph·R3/4·unseen role semantics·실제 heterograph는 이번 실험 범위가 아니다.

![Distillation ablation](results/distillation_ablation.png)
