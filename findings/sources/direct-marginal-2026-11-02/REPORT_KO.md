# 직접 주변화하는 conditional graph circuit — 완료 결과

**판정: 정규화·주변화 유효성 및 두 학습된 직접 예측기 대비 개선은 확인했다. Grid6 대비 메인 메소드 경쟁력은 충족하지 못했다.**

- 새 12-node binary typed-SBM family: target8/helper4, relation2, 연속 feature2. 세 prior-family descriptor는 모든 모델에 제공했다.
- Matched primary는 새48 base graphs의 whole-node uniform/clustered50/90%, 총 context1/2/4 평균. OOD까지 총192 base graphs다.
- Circuit/direct graph/public Mask PFN을 3 fixed seeds로 학습했다. 모델마다 2048 task exposures와256 성공 updates, query CE만 사용했다. Parameter 수는 동일하지 않다.
- Circuit은 56,385 parameters, public Mask PFN은 4,012,259 trainable parameters다. Public PFN에는 typed context support까지 제공했다.

## 1. 같은 모델 내부의 주변화는 유효하다

Graph와 알려진 family에만 조건화한 normalized tree mixture를 만들고, 관측 feature와 context label은 leaf evidence로 평가한다. 각 원본 변수 scope는 한 번씩만 사용한다.
소형 열거, 전체 질량1, label 합 규칙, 연속 feature 적분, conditional ratio의 최대 오차는 약10^-15 수준이었다. 단일 class0 context에서 음의 transition을 주면 반대 class 확률0.95257을 표현할 수 있었다.
이는 현재 conditional tree model 안의 정확한 계산이다. 원래 cyclic SBM의 참 posterior가 정확하다는 뜻은 아니며 GSPN/MaM 원 논문의 재현도 아니다.

## 2. 예측 성능

| 방법 | Accuracy (%) | NLL | Brier | Grid6-exact predictive KL |
|---|---:|---:|---:|---:|
| mask_pfn | 53.4846 | 0.687080 | 0.493828 | 0.147684 |
| direct_graph | 51.1512 | 0.693134 | 0.499986 | 0.154796 |
| circuit | 61.3839 | 0.605128 | 0.430740 | 0.064716 |
| grid6_bp | 69.5602 | 0.540755 | 0.375136 | 0.000099 |
| grid6_exact | 69.3680 | 0.541211 | 0.375627 | 0.000000 |

Circuit−Mask PFN accuracy: +7.899306 [+3.889009, +11.790761] %p. NLL: -0.081952 [-0.101972, -0.062402].
Circuit−Direct graph accuracy: +10.232722 [+5.688365, +14.589172] %p. NLL: -0.088005 [-0.106528, -0.069288].
따라서 사전 neural-baseline gate는 통과했다. 서로 다른 과제인 feature-memory 실험의 accuracy와 절대값을 비교하지 않는다.

## 3. 실제 관측값을 사용했는가

아래 intervention은 context2/clustered50, 전체192 base graphs의 보조 비교이며 primary와 조건이 다르다.

| Intervention−원래 입력 | Accuracy %p [95% CI] | NLL [95% CI] |
|---|---|---|
| no_observed | -15.451389 [-19.386574, -11.805556] | +0.172725 [+0.147664, +0.197846] |
| shuffled | -13.686343 [-17.534722, -9.953704] | +0.352257 [+0.280898, +0.432729] |

이 family에서는 실제 feature와 node 대응 관계를 예측에 활용한다. 이를 IMDB 등 기존 실제 graph에서의 feature 기여로 옮겨 주장하지 않는다.

## 4. 강한 기준선이 제한하는 주장

Grid6-BP 대비 circuit NLL은 +0.064373 [+0.024247, +0.105302] 더 높고, accuracy는 -8.176257 [-13.111376, -3.230252] %p 낮았다.
Grid6는 이전 six-support prior를 유지하되 이 family의 Gaussian feature orientation을 적분하도록 확장한 known-family baseline이다. 실제 test density/signal을 주는 별도 oracle과 구분한다.
Primary 설계 고정 뒤 해석용으로 추가한 feature-only/constant-tree 대조는 아래와 같다. Gaussian magnitude0.8/sd0.6라는 정확한 training-family 상수를 아는 model-based 대조이며 학생 모델의 성과가 아니다.

| 추가 대조 | Accuracy (%) | NLL |
|---|---:|---:|
| known_feature_only | 58.1329 | 0.605147 |
| known_constant_tree | 60.0116 | 0.685252 |

Circuit−feature-only NLL: -0.000019 [-0.013783, +0.013696]; accuracy: +3.251075 [-0.142816, +6.531240] %p. 추가 NLL 이득은 확인되지 않았고 accuracy CI도0을 포함한다. 이것이 두 방법의 동등성을 증명하는 것은 아니다.
관측 feature가 label context 바깥에만 있는 primary 사후 분할에서는 circuit52.69%/NLL0.69955, Grid6-BP64.87%/0.62460이었다. Visible mask/context로 나눈 기술적 분석이며 새 primary gate나 독립 확인으로 쓰지 않는다. 올바른 주변화 연산이 있어도 구조를 통해 떨어진 feature와 label을 연결하는 예측이 충분하지 않을 수 있음을 보여주는 미해결 조건이다.
원인으로 tree 표현력·graph-conditioned potential fit·학습량 중 하나를 확정하지 않는다.

## 5. 비용과 범위

같은 GPU의 단일 graph visible-input→확률 median: circuit 12.369ms, Grid6-BP 13.443ms. 약간의 측정값 차이가 있지만 사전2배 속도 기준을 충족하지 않는다. Grid6-exact predictive KL도 0.064716로 .01 목표보다 크다.
BP-only 비용에서는 reference 생성에만 필요한 exact query marginal 출력을 제외했고 예측 차이0을 확인했다. 모델 구성·bank I/O·사전학습 비용은 이 latency 밖이며 외부 GPU 부하는 통제하지 않았다.
학습9개, 총 2,304 성공 updates와 18,432 task exposures, 저장 예측 44,928개를 감사했다. Test4608개 중 실제 physical feature 관측이0개인 episode는 1,353개다.
대형·실제 heterograph, 새 semantic schema, 가변 feature 폭, 독립적으로 학습한 MaM, full sparse cell-token 모델은 아직 실행하지 않았다. 현재 결과로 이를 해결했다고 쓰지 않는다.
근거: protocol.json, results/preflight.json, paired.json, feature_controls.json, descriptive_breakdowns.json, benchmark.json, final_integrity.json.
