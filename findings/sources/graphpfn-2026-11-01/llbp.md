# Learned-Likelihood BP / EAJ 연구: 실패·중단·부분 성공 기록

> **정리 기준일:** 2026-11-01  
> **범위:** Role-Conditional Distillation → Learned-Likelihood BP → Joint Role–Parameter Grid → EAJ → EAJ predictive 학습 → 자기-update R1 → 재학습 T1 → 강한 Grid 기준선과의 최종 비교.  
> **목적:** 어떤 가설을 시도했고, 무엇이 개선됐으며, 어떤 주장 또는 메소드 후보가 끝내 성립하지 않았는지 기록한다. 이미 실패한 갈래를 이름만 바꿔 반복하지 않기 위한 연구 회고다. **새 실험을 자동으로 시작하라는 지시서가 아니다.**

## 0. 최종 결론

**현재 설정에서 경쟁력 있는 독립 메인 메소드는 아직 확보하지 못했다.**

그렇다고 모든 구현이나 모든 실험이 실패한 것은 아니다. 다음 세 사실은 동시에 참이다.

1. Learned potential을 BP에 연결하는 것은 작동했고, 기존 positive/signed decoder보다 좋은 확률 예측을 얻었다.
2. 같은 learned potential에서 role weights를 다시 계산하는 R1은 추가 학습 없이 기존 방식보다 좋아졌고, 새 graph bank에서 재현됐으며, Constant+update보다도 유의하게 좋았다.
3. 그러나 작은 Joint Grid6이 R1보다 KL·NLL·Brier에서 좋았고, 병렬 구현에서는 R1에 뚜렷한 속도 우위도 없었다. 따라서 R1을 우수한 범용 추론법 또는 독립적인 메인 메소드로 승격하는 것은 중단한다.[^s02][^u03][^u04]

**정확한 종료 대상은 “현재 문제·구조·예산에서 loss/head/재학습을 추가해 메인 메소드로 만드는 갈래”다. BP 전체, learned potential 전체, 모든 후속 연구가 불가능하다는 결론은 아니다.**

---

## 1. 이 문서를 읽는 방법

### 1.1 ‘실패’의 종류를 구분한다

| 판정 | 의미 | 이번 연구의 예 |
|---|---|---|
| 추가 이득 미확인 | 개선을 주장할 증거가 부족하다. 동등성이나 불가능성이 증명된 것은 아니다. | T1−R1 CI가 0을 포함 |
| 유의한 악화 | 지정된 비교·조건에서 오차가 증가했다. | 초기 posterior loss 추가, 일부 auxiliary loss |
| 부분 성공·목표 미달 | 중간 목표는 달성했지만 최종 목적에는 부족하다. | EAJ B의 predictive KL 개선과 실용성 부족 |
| 경쟁력 미달 | 강한 대안 대비 품질 또는 품질–비용 우위를 확보하지 못했다. | R1 대 Grid6 |
| 해석 기각 | 실험은 유효하지만 그 결과에 붙이려던 설명이 성립하지 않는다. | predictive 개선을 posterior 복원 개선으로 해석 |
| 미실행·미검증 | 제안만 있었거나 실험 증거가 없다. 실패로 세지 않는다. | 관측과정 자체를 학습하는 BP 확장 |

아래의 ‘실패 목록’은 독립적인 학습 실행 수가 아니라 **검증한 가설·기각한 주장·중단한 갈래의 목록**이다. 같은 실험이 여러 가설을 동시에 검사한 경우가 있다.

### 1.2 근거 수준

- **원문 확인:** 초기 distillation, learned-potential MVP, Joint-grid BP, 최초 EAJ 보고서 네 개는 이번 정리에서 본문을 직접 읽었다.
- **사용자 결과 보고:** EAJ A→B와 convex oracle, frozen 자기-update, T1, 최종 Grid6 비교는 이 대화에 사용자가 제공한 수치와 판정을 근거로 한다. 해당 최신 원시 배열·checkpoint·전체 보고서·CI 계산 코드는 이번 정리에서 재실행하거나 감사하지 않았다.
- **이전 assistant 문서:** `learned_likelihood_bp_R1_next_steps.md`, `learned_likelihood_bp_R1_review.md`는 당시의 계획·판정 기록이다. 원시 실험 증거를 대체하지 않는다. 특히 “R1을 메인 방법으로 유지”라는 당시 문구는 최종 Grid 비교 후 **메인 후보 승격 중단**으로 갱신됐다.
- 상대 감소율 등 단순 산술은 보고된 반올림 평균으로 계산했다. 별도로 제공된 paired 차이와 CI는 그대로 보존했다. 원시 데이터에서 재생성한 결과인 것처럼 표시하지 않는다.

### 1.3 범위에서 제외한 연구

대화 앞부분의 **결측 인지 GraphPFN 피처 결측 MVP**와 이 문서의 **feature-free 관계 그래프 BP/EAJ 실험**은 별개다. 이 문서에서 전자의 실패를 판정하지 않는다. ASNO, 시계열, 다른 연구 주제도 포함하지 않는다.

실험 날짜는 원문에 적힌 날짜와 대화상 진행 순서를 따른다. 파일명의 시각을 실제 학습 시작·종료 시각으로 간주하지 않는다. 회수하지 못한 더 이른 실험을 추정해 채우지도 않는다.

---

## 2. 방법·teacher·이름을 먼저 고정한다

### 2.1 서로 다른 A/B를 섞지 않는다

| 표기 | 의미 | 주의 |
|---|---|---|
| 초기 JR의 `A`, `A0` | 기존 joint-role 모델, sampled-label CE와 posterior auxiliary 유무 | 뒤의 EAJ A와 다르다. |
| 초기 `B_predictive` | JR 구조에서 exact predictive KL로 학습 | EAJ B와 다르다. |
| 초기 `C_conditional` | B에 role-conditional target 감독 추가 | 새로운 BP 구조가 아니라 기존 staged decoder의 학습 ablation |
| 초기 `D_all` | C에 posterior KL 감독 추가 | 이후 learned BP의 frozen role posterior 출처 |
| Learned BP / MRP | 관측 통계로 local potential을 예측하고 BP로 query 예측 | 초기 보고서에는 auxiliary 포함/미포함 arm이 함께 있다. |
| EAJ A | 고정 grid/BP branches 위에서 exact joint posterior KL로 학습한 evidence head | validation checkpoint 선택 기준은 grid predictive KL |
| EAJ B | 같은 구조에서 predictive 목표로 학습한 비교 arm | predictive weights의 품질과 posterior 복원 품질을 구분 |
| R0 | 기존 learned potential + 기존 frozen role weights | 후속 R1/T1 비교의 기준 |
| R1 | R0와 같은 potential·conditional BP, 평가 때만 자기 role-update | 추가 학습 없음; 학습 단계까지 head가 없었던 것은 아님 |
| T1 | 자기 role-update를 학습·평가에 모두 포함 | 같은 초기화·teacher·schedule·192 updates로 처음부터 학습 |
| Constant+update | train-fitted constant potential + 해당 potential의 role update | frozen-q constant와 다른, 더 강한 기준선 |
| Joint Grid81 | broad parameter prior의 9×9 grid + joint role–parameter update | Grid6과 prior 자체도 다름 |
| Joint Grid6 | training-support 여섯 parameter + 사전에 정한 prior mass | 단순히 Grid81의 저해상도 버전이라고 부르지 않음 |

초기 learned-BP 보고서의 `Learned BP + frozen q`는 auxiliary 포함 arm이다. 후속 frozen 자기-update 표의 기존 값은 predictive-only arm과 대응한다. **이름이 비슷하다고 서로 다른 checkpoint의 수치를 한 곡선으로 연결하지 않는다.**[^s01][^s02][^s04][^u02]

### 2.2 서로 다른 KL을 섞지 않는다

**True-parameter Bayes predictive KL**은 참 생성 parameter를 아는 teacher의 query 예측을 기준으로 한다.

$$
L_{\mathrm{true}}
=\frac{1}{|Q|}\sum_{q\in Q}
\mathrm{KL}\!\left(
 p^*(Y_q\mid D,\xi_{\mathrm{true}})
 \Vert p_{\mathrm{student}}(Y_q\mid D)
\right).
$$

Student가 참 parameter를 받지 않으므로, 이 teacher와의 차이에는 parameter 식별 불확실성이 포함된다. 잔여 KL 전부를 encoder나 optimizer의 실패로 돌리지 않는다.

**Unknown-parameter exact-grid predictive KL**은 선언된 parameter prior를 적분한, 유한 grid에 조건부인 exact teacher를 기준으로 한다. EAJ A→B 비교는 이 기준이다.

$$
p^*_{\mathrm{grid}}(Y_q\mid D)
=\sum_{h,g}w^*_{hg}(D)\,p^*_{hg}(Y_q\mid D).
$$

**Posterior KL**은 role 또는 role–parameter weights의 차이다. 최종 query predictive KL과 동일하지 않다.

**자기 exact update**는 learned potential이 유도하는 role weights를 정확히 계산한다는 뜻이다. 참 생성모형의 oracle role posterior라는 뜻이 아니며, conditional prediction까지 exact로 교체했다는 뜻도 아니다.[^s02][^s03][^s04]

### 2.3 공통 범위와 제한

초기 원문들의 중심 설정은 known-family binary SBM, feature-free, target 8개 + helper 4개, 두 관계와 네 role 가설이다. Context 1/2/4는 **클래스당 수가 아니라 총 context 수**다. 작은 latent state는 exact enumeration으로 검증하고 conditional BP는 고정 반복으로 계산했다.[^s01][^s02][^s03]

관측되지 않은 pair와 관측된 nonedge는 다른 상태다. 이 실험군의 relation observation mask와 앞서 논의한 노드 feature 결측 mask도 같은 것이 아니다. **Low edge density와 낮은 observation rate를 혼동하지 않는다.**

---

## 3. 실패·중단 가설 한눈에 보기

| ID | 당시 기대 또는 가설 | 실제 관찰 | 최종 판정 |
|---|---|---|---|
| F01 | Role-conditional 감독을 더하면 최종 예측이 개선된다. | Conditional KL은 감소했지만 predictive 증분 CI는 세 prior 모두 0 포함 | 추가 이득 미확인 |
| F02 | Posterior까지 정확하게 학습하면 최종 예측도 더 좋아진다. | Posterior KL은 감소했지만 predictive KL은 세 prior 모두 증가 | 현재 설정에서 유의 악화 |
| F03 | Masked-edge likelihood auxiliary가 learned BP의 핵심 학습 기여다. | Predictive-only보다 Independent/Exclusive에서 악화; Coupled 추가 이득 미확인 | 새 loss 기여 주장 중단 |
| F04 | Learned potential + 기존 frozen q만으로 adaptive OOD 추론이 충분하다. | Matched에서는 좋아졌지만 density/signal 외삽과 constant 비교에서 일관성 부족 | 범용성 미확보 |
| F05 | Parameter 후보를 혼합하되 기존 role q를 유지해도 충분하다. | 같은 BP branches에서 coherent joint role update가 크게 우수 | Frozen-hybrid가 충분하다는 기대 기각 |
| F06 | EAJ가 exact-grid를 비슷한 품질·훨씬 낮은 전체 비용으로 대체한다. | 약한 head보다 좋지만 exact 및 강한 constant에 미달; 전체 비용 절감 제한 | 대체·확장 중단 |
| F07 | EAJ를 직접 predictive 학습하면 실용성과 posterior 해석을 함께 회복한다. | Grid predictive KL은 개선; posterior KL은 악화; signal-high 실용 기준선 열세 | 부분 성공, 메소드 구제 실패 |
| F08 | 같은 potential의 자기-update가 OOD 문제를 전반적으로 해결한다. | High density 등의 개선은 확인; low density는 exact update에서도 악화 | 보편적 OOD 해결 주장 기각 |
| F09 | 자기-update 경로까지 학습하는 T1이 R1보다 우수하다. | 추가 KL 개선 CI가 0 포함; 여러 조건에서 점추정 악화 | T1 우수성 미확인 |
| F10 | Low-density 실패는 partition 근사 또는 role weights만의 문제다. | Exact update도 실패; learned conditional도 손해; 상호작용 존재 | 단일 병목 설명 기각 |
| F11 | R1은 강한 model-based 기준선보다 좋은 메인 메소드다. | Grid6이 KL·NLL·Brier에서 유의하게 우수 | 품질 경쟁력 미달 |
| F12 | R1은 품질이 낮아도 큰 속도 이점으로 정당화된다. | 병렬 Grid6과 latency가 비슷하고 단일 graph에서는 더 느림 | 큰 속도 우위 미확보 |

이 표는 F01/F02의 초기 loss 실험과 F07의 EAJ predictive 실험을 별도 단계로 센다. 또한 **R1−R0의 재현된 개선과 R1−Constant의 유의한 개선 자체를 실패로 표시하지 않는다.**

---

## 4. 단계별 상세 기록

### 4.1 초기 Role-Conditional Distillation: 구성요소를 잘 맞추면 최종 예측도 좋아질까?

**목표:** 기존 staged joint-role 구조를 유지한 채 teacher 감독을 정교하게 만들어 최종 예측을 개선한다.

새 아키텍처를 추가한 것이 아니라 학습 목표를 순서대로 바꿨다. 기존 A는 이미 posterior KL auxiliary를 사용했으므로, 이를 새 감독인 것처럼 계산하지 않았다. 순수 predictive target 효과를 확인하기 위해 CE-only A0를 별도로 유지했다.[^s01]

| Arm | 최종 예측 감독 | Conditional loss | Posterior loss |
|---|---|---:|---:|
| A0 | Sampled-label CE | 0 | 0 |
| B_predictive | Exact predictive KL | 0 | 0 |
| C_conditional | Exact predictive KL | 1 | 0 |
| D_all | Exact predictive KL | 1 | 0.1 |

초기화·task schedule·192-update 예산을 맞췄다. 기존 A/A0 18개를 재사용하고 새 모델 63개를 학습했다. 이 단계의 새 optimizer update는 12,096회다.[^s01]

#### F01. Conditional loss의 최종 predictive 증분 미확인

차이는 `C_conditional − B_predictive`다. KL에서 음수는 개선이다.

| Prior | Conditional KL 차이 | Predictive KL 차이 | Predictive 차이의 paired 95% CI |
|---|---:|---:|---|
| Independent | −0.007978 | +0.000188 | [−0.000440, +0.000795] |
| Exclusive | −0.002484 | −0.000181 | [−0.000683, +0.000329] |
| Coupled | −0.003686 | −0.000582 | [−0.001401, +0.000182] |

**실패한 주장:** conditional target을 더 정확하게 맞추는 것이 현재 구조에서 의미 있는 최종 predictive 개선으로 이어진다.

**남은 사실:** 추가 감독은 의도한 conditional 오차를 실제로 줄였다. 모든 conditional distillation이 불가능하다는 결론은 아니다.[^s01]

#### F02. Posterior loss 추가는 component 개선과 predictive 악화를 동시에 만들었다

차이는 `D_all − C_conditional`이다.

| Prior | Posterior KL 차이 | Predictive KL 차이 | Predictive 차이의 paired 95% CI |
|---|---:|---:|---|
| Independent | −0.284227 | +0.004039 | [+0.002707, +0.005418] |
| Exclusive | −0.010266 | +0.000479 | [+0.000013, +0.000952] |
| Coupled | −0.065214 | +0.001561 | [+0.000368, +0.002823] |

**실패한 주장:** posterior를 더 정확하게 복원하도록 추가 감독하면 최종 predictor도 더 좋아진다.

**정확한 판정:** 현재 architecture, loss weights, 예산에서는 posterior fit 개선이 최종 predictive 악화와 함께 나타났다. Posterior와 conditional의 component 오차를 줄이는 것만으로 최종 품질을 보장하지 못했다.[^s01]

#### 감독만으로 해결되지 않은 decoder 제약

당시 feature-free staged decoder의 남은 연산은 양의 relation weight와 normalization을 쓰는 label diffusion이었다. Context label이 전부 클래스 a이면 모든 query에 대해 다음 제약이 유지됐다.

$$
p_{\mathrm{model}}(a)\geq \tfrac12.
$$

하지만 해당 exact SBM teacher는 nonedge를 통해 반대 클래스를 지지할 수 있었다. 따라서 teacher를 정확하게 주는 것과 decoder가 그 target을 표현할 수 있는 것은 다른 문제였다. 보고서의 반공간 projection 하한은 이 구조의 **부분적 KL 하한**이지, 전체 family의 exact optimum이나 모든 오차의 설명은 아니다.[^s01]

**당시 종료:** 현재 staged decoder에 conditional/posterior loss만 추가해 메인 메소드나 teacher-free scaling으로 확장하는 갈래를 중단했다. 이후에는 decoder 자체를 바꾸는 방향으로 이동했다.

---

### 4.2 Learned-Likelihood BP: decoder 전환은 성공했지만 새 auxiliary loss는 실패했다

**변경:** 관측 통계로 local potential을 학습하고, observed edge와 observed nonedge가 주는 evidence를 고정 반복 BP로 결합했다. 기존 D_all role posterior는 고정했다.

이 단계는 4 arms × 3 priors × 3 seeds, 총 36개 모델을 같은 192-update/task budget으로 학습했다. 다만 Plain과 potential 모델은 mask 입력·감독 정보와 FLOPs까지 동일한 대조가 아니었다. 따라서 모든 차이를 단일 architectural 요인으로 단정하지 않았다.[^s02]

#### 먼저, 실제로 된 것

초기 MRP matched 평가의 Bayes predictive KL:

| 방법 | Independent | Exclusive | Coupled |
|---|---:|---:|---:|
| Frozen positive D | 0.063955 | 0.052586 | 0.085548 |
| New Large Plain distill | 0.071414 | 0.073581 | 0.085079 |
| Learned signed decoder | 0.057612 | 0.049333 | 0.079295 |
| **Learned BP, predictive-only** | **0.015946** | **0.011013** | **0.028391** |
| Learned BP + auxiliary | 0.016319 | 0.011275 | 0.028654 |
| Constant potential + frozen q | 0.018514 | 0.012652 | 0.028160 |

이는 기존 positive/signed decoder보다 learned BP 구성이 유효했다는 결과다. Signed decoder도 기존 positive decoder보다 개선될 수 있었지만, **단순 signed diffusion만으로 BP의 개선을 설명할 수는 없었다.** Signed arm은 실패한 독립 신메소드로 세기보다 설명을 검증한 대조군으로 기록한다.[^s02]

#### F03. Masked-edge likelihood auxiliary의 독립적 이득은 없었다

차이는 `BP + auxiliary − BP predictive-only`다.

| Prior | Predictive KL 차이 | Paired 95% CI | 판정 |
|---|---:|---|---|
| Independent | +0.000374 | [+0.000187, +0.000587] | 유의 악화 |
| Exclusive | +0.000261 | [+0.000144, +0.000364] | 유의 악화 |
| Coupled | +0.000263 | [−0.000028, +0.000656] | 추가 개선 미확인 |

**실패한 주장:** masked-edge likelihood loss를 추가한 것이 새로운 학습 원리이며 핵심 성능 증분을 제공한다.

**남은 사실:** auxiliary 없이도 potential을 학습하고 BP로 좋은 예측을 만들 수 있었다. 실패한 것은 학습 가능성이 아니라 **추가 loss의 우수성 주장**이다. 이 결과를 보고 test에 맞춰 auxiliary weight를 다시 조정하지 않았다.[^s02]

#### F04. Learned potential + frozen q의 범용 OOD 적응은 미완성이었다

Matched에서 learned potential의 constant 대비 개선은 일부 prior에서 있었지만, Coupled에서는 auxiliary 포함 learned BP가 constant보다 나빴다. Density 0.12 외삽에서는 일부 Plain/signed 대조보다도 나빴고, density/signal 이동에서 oracle potential과의 차이가 커졌다.[^s02]

**실패한 주장:** 관측 통계 encoder를 붙여 potential을 학습하면 고정된 role head와 결합해도 넓은 조건에서 강한 추론기가 된다.

**아직 확정하지 못한 원인:** potential point estimate, frozen posterior의 분포 이동, conditional BP 근사, 이들의 상호작용. 어느 하나에 전체 오차를 귀속하지 않는다.

#### 실패가 아니라 유효한 설계 근거로 남은 대조

같은 learned potential과 frozen q를 유지하고 decoder coupling만 바꾼 초기 MRP 결과:

| Prior | Full edge/nonedge BP KL | Edge-only BP KL | 관측률 0.2: 올바른 mask KL | 같은 조건: unknown을 nonedge로 처리한 KL |
|---|---:|---:|---:|---:|
| Independent | 0.016319 | 0.082736 | 0.000226 | 0.044486 |
| Exclusive | 0.011275 | 0.061489 | 0.000208 | 0.029325 |
| Coupled | 0.028654 | 0.124520 | 0.000386 | 0.058414 |

Observed nonedge를 버리거나 unknown pair를 nonedge로 취급하면 크게 악화됐다. 이는 **올바른 observation-state 처리의 필요성**에 관한 증거다. 의도적으로 잘못된 처리를 적용한 ablation을 개발 중 발생한 버그나 별도 메소드 실패로 적지 않는다. Frozen q에는 기존 구조 정보가 남아 있어 end-to-end nonedge 정보를 완전히 제거한 실험도 아니다.[^s02]

---

### 4.3 Joint-grid 진단: parameter 혼합만으로는 부족했고, 강한 비학습 기준선이 등장했다

**변경:** 알려진 SBM family에서 parameter 후보별·role별 BP prediction을 계산하고, 같은 branches에 서로 다른 weights를 적용했다. 새 학습은 없었다.

#### F05. Frozen role q를 유지한 parameter 혼합이 충분하다는 기대는 성립하지 않았다

초기 개발 bank의 Coupled / high density 예:

| 구성 | True-parameter Bayes predictive KL |
|---|---:|
| 기존 MRP | 0.112820 |
| Full-grid parameter mixture + frozen role q | 0.078075 |
| **같은 BP branches + coherent joint weights** | **0.017615** |
| Exact joint의 같은 marginals를 곱한 대조 | 0.019049 |

Parameter 후보를 늘리는 것만으로도 일부 개선됐지만, 같은 conditional 계산에서 role marginal까지 coherent하게 갱신한 차이가 더 컸다.[^s03]

**해석상 기각한 주장:** `0.078075 → 0.017615`의 전체 개선이 H–parameter dependence 보존만의 효과라는 주장.

그 dependence만 분리한 same-marginal 비교의 차이는 −0.001434, CI [−0.001946, −0.000872]였다. **Role marginal update 효과와 dependence 추가 효과는 다르다.** Frozen hybrid도 정규화된 분포이며 일부 조건부 dependence를 보존하므로, 이를 단순 독립 head의 곱이라고 부르지 않는다.[^s03]

#### Grid 자체의 한계도 확인했다

Broad Grid81은 matched Exclusive와 sparse observation 등에서 기존 방법보다 나쁠 수 있었다. Training-support Grid6은 이 조건에서 더 좋았지만 low density에서는 broad보다 나쁠 수 있었다. 모든 조건을 지배하는 하나의 prior를 확보한 것은 아니다.[^s03]

**이 단계는 새 learned 메소드의 성공이 아니었다.** 알려진 family와 선언된 prior를 사용하는 model-based 기준선이 강해졌고, 이후 learned 방법이 넘어야 할 기준이 명확해진 단계였다.

---

### 4.4 EAJ: joint posterior를 학습해 exact-grid를 대체하려 했지만 품질·비용 목표에 못 미쳤다

EAJ는 **Evidence-Amortized Joint PFN**이다. 고정된 81×4 grid/role BP branches 위에서 공유 evidence head를 학습했다.

$$
q_\theta(h,\xi_g\mid D)
=\operatorname{softmax}_{h,g}
\left[\log\pi_H(h)+\log\pi_G(g)+\widehat\ell_\theta(D,h,\xi_g)\right].
$$

원래 EAJ A의 학습 목표는 exact joint posterior KL이고, validation checkpoint 선택 기준은 exact-grid predictive KL이었다. Shared joint와 independent-head 대조를 합해 18개 모델, 3,456 updates를 수행했다. 같은 encoder 초기값과 schedule을 사용했다.[^s04]

#### F06-1. 약한 head보다 좋아졌지만 exact-grid 대체에 실패했다

Density-high / signal-high의 여섯 prior×condition에서 EAJ는 다음 대조보다 좋았다.

- Independent heads.
- 학습 없는 analytic initializer.
- 자신의 두 marginals를 곱한 own-product 대조.

하지만 **teacher의 정확한 marginals를 곱한 대조보다 여섯 조건 모두 나빴고, exact joint와도 차이가 컸다.** Joint 표현의 장점은 있었으나 posterior 근사의 전체 정확도가 충분하지 않았다.[^s04]

실용적으로도 signal-high에서 더 단순한 constant+updated-role이 세 prior 모두 EAJ보다 좋았다. 아래 표는 **true-parameter Bayes KL**이며, 다음 EAJ A→B의 grid predictive KL과 다른 지표다.

| Signal-high prior | Constant+update | EAJ A | EAJ−Constant 95% CI |
|---|---:|---:|---|
| Independent | 0.046959 | 0.093380 | [+0.038260, +0.055812] |
| Exclusive | 0.091788 | 0.131321 | [+0.033992, +0.045969] |
| Coupled | 0.032537 | 0.057417 | [+0.015283, +0.035029] |

**실패한 주장:** learned joint evidence가 강한 classical control까지 대체하는 좋은 범용 predictor가 됐다.[^s04]

#### F06-2. Posterior 계산만 빠른 것으로는 전체 가속이 충분하지 않았다

초기 EAJ 비용 보고에서 posterior-only는 exact evidence보다 약 1.9–2.9배 빨랐지만, 공통 81×4 BP 계산이 남았다. 같은 BP 시간에 posterior 시간을 더한 전체 추정 시간 감소는 약 6–12%였다.

| Prior | EAJ posterior | Exact evidence | 공통 BP | EAJ+BP / Exact+BP |
|---|---:|---:|---:|---:|
| Independent | 33.08 ms | 96.50 ms | 437.12 ms | 470.20 / 533.63 ms |
| Exclusive | 30.95 ms | 66.49 ms | 307.44 ms | 338.40 / 373.93 ms |
| Coupled | 34.15 ms | 66.15 ms | 438.15 ms | 472.30 / 504.30 ms |

이것은 당시 64-graph 측정 및 합산 추정치다. 뒤의 RTX 3090 단일 graph / Batch16 latency와 직접 섞지 않는다.

**실패한 주장:** posterior amortization만으로 비슷한 품질과 큰 end-to-end 가속을 동시에 얻는다. Quality gap이 남았고, candidate/BP 계산의 대부분도 그대로였다.[^s04]

---

### 4.5 EAJ 직접 predictive 학습 B: 목적을 바꾸는 효과는 있었지만 메소드를 구하지 못했다

**다음 가설:** posterior KL이 아니라 최종 predictive KL을 직접 학습하면 EAJ의 실용적 gap이 줄어든다.

A는 재사용하고 B 9개만 동일 예산으로 학습했다. 다음은 동일한 **unknown-parameter exact-grid teacher**에 대한 predictive KL이다.[^u01]

| Prior | Density-high A | Density-high B | Signal-high A | Signal-high B |
|---|---:|---:|---:|---:|
| Independent | 0.02028 | 0.01577 | 0.01835 | 0.01249 |
| Exclusive | 0.02653 | 0.01964 | 0.01170 | 0.00916 |
| Coupled | 0.01672 | 0.01409 | 0.02258 | 0.01709 |

여섯 조건 모두 B−A의 paired 95% CI가 0보다 낮았다. 반올림 평균으로 계산한 상대 감소는 약 15.7–31.9%다. **직접 predictive 학습이 작동하지 않았다는 결론은 틀리다.**

#### F07. 실용성과 posterior 해석을 함께 회복하지 못했다

사용자가 보고한 세 판정은 다음과 같다.

| 질문 | 결과 | 판정 |
|---|---|---|
| A보다 teacher predictive를 잘 근사하는가? | 여섯 조건 모두 개선 CI가 0 제외 | 성공 |
| 강한 실용 기준선보다 좋은가? | Signal-high 세 prior 모두 Constant+role-update보다 NLL·Brier가 유의하게 나쁨 | 실패 |
| Accuracy도 좋아졌는가? | A 대비 여섯 조건 모두 개선 미확인 | 우수성 주장 불가 |
| Posterior를 더 정확히 복원했는가? | Posterior KL은 여섯 조건 모두 악화 | 실패 |

따라서 B의 출력은 **decoder에 맞춘 aggregation weights**로 남긴다. 개선된 latent posterior라고 해석하지 않는다.[^u01]

Predictive가 좋아지고 posterior가 나빠진 결과는 decoder 오차 보정과 일관될 수 있지만, 그 원인이 확정됐다는 뜻은 아니다. 서로 다른 weights가 비슷한 prediction을 만드는 경우와 유한 학습 예산도 구분하지 못했다.

#### Convex oracle: 실패한 실험이 아니라, 남은 gap에 대한 진단

사용자는 graph-shared convex oracle 12,096개의 최적성 certificate를 확인했다고 보고했다. 다만 대화에서 **최적성 오차 bound의 숫자가 깨져 있으므로 정확한 허용치는 이 문서에서 미상으로 둔다.** Certificate 원본을 이번에 재검증한 것은 아니다.[^u01]

여기서 12,096은 oracle 문제 수다. 초기 distillation의 optimizer update 12,096회와는 우연히 숫자가 같을 뿐 같은 실험이 아니다.

동일한 teacher·branches·집계에서 A, B, oracle 손실을 각각 \(L_A,L_B,L_*\)라 하면,

$$
\rho=\frac{L_A-L_B}{L_A-L_*}
$$

는 **A에서 oracle까지 가능한 개선 여지 중 B가 회수한 비율**이다. 보고값은 약 16–32%다. 나머지 68–84%는 A→oracle gap 중 미회수 몫이지, 전체 예측 오류의 비율이 아니다.

이 결과로 기각한 설명은 “이미 포화돼 더 좋아질 여지가 없다”다. 그러나 teacher를 이용해 그래프마다 최적 weights를 구할 수 있다는 사실이, 관측 정보만으로 EAJ가 그 weights를 쉽게 학습할 수 있음을 보장하지도 않는다.

**당시 종료:** EAJ에 새 surrogate loss를 더하거나 모델/학습량을 늘려 경쟁력을 추격하는 갈래를 닫았다. A/B checkpoint와 oracle 진단은 보존했다. **개선 가능성 자체의 부재가 아니라, 현재 갈래의 추가 투자 근거 부족**으로 판정했다.

---

### 4.6 R1: 별도 posterior head를 제거하는 추론 변경은 성공했으나 low density는 남았다

**새 접근:** EAJ head를 더 잘 학습하는 대신, 이미 학습한 local potential에서 role weights도 계산한다. Potential과 conditional BP는 고정하고 role weights만 교체했다.

$$
q_\theta^{\mathrm{self}}(h\mid D)
\propto \pi_H(h)\,\widehat Z_\theta(h;D),
\qquad
p_\theta(Y_q\mid D)
=\sum_h q_\theta^{\mathrm{self}}(h\mid D)
 p_{\mathrm{BP},\theta}(Y_q\mid D,h).
$$

\(\widehat Z\)를 exact로 계산한 진단과 Bethe 근사로 계산한 경로를 비교했다. 같은 potential을 사용한다는 뜻이지, 데이터 의존 potential이 자동으로 정규화된 생성 likelihood가 된다는 뜻은 아니다.

사용자가 보고한 Exclusive의 frozen 검증:

| 조건 | 기존 frozen q | Constant+update | 자기 exact update | 자기 Bethe update |
|---|---:|---:|---:|---:|
| Matched | 0.01101 | 0.00541 | 0.00354 | 0.00315 |
| High density | 0.08276 | 0.05073 | 0.04029 | 0.04035 |
| High signal | 0.11042 | 0.09179 | 0.07917 | 0.08090 |
| Low density | 0.00915 | 0.00917 | 0.01159 | 0.01132 |

High density에서는 세 prior 모두 자기 exact update가 Constant+update보다 낮은 KL을 얻었고 paired CI도 0을 제외했다. Accuracy는 high-density Independent에서 Constant 대비 +0.83%p가 확인됐지만, 전반적인 주 효과는 확률 예측 품질이었다.[^u02]

**이 단계에서 된 것:** learned potential의 가치를 기존 frozen q가 충분히 활용하지 못했으며, 추가 학습 없이 그 활용 방식을 개선할 수 있었다.

#### F08. 자기일관성만으로 OOD가 해결된다는 기대는 실패

Exclusive low density에서는 자기 exact update 0.01159가 기존 frozen q 0.00915와 Constant+update 0.00917보다 나빴다. 따라서 Bethe를 exact로 바꾸는 것만으로 이 실패를 없앨 수 없었다.[^u02]

**실패한 주장:** potential과 weights를 맞추면 분포 이동도 전반적으로 해결된다.

**남은 사실:** 자기-update는 특정 조건에서 유효한 추론 개선이다. 같은 potential을 공유하는 것과 그 potential이 새 조건에 잘 맞는 것은 다르다.

---

### 4.7 T1: 추론 구조에 맞춘 재학습의 추가 이득은 확인되지 않았다

**가설:** 자기-update를 평가 때만 적용하는 R1보다, 학습 때도 적용하는 T1이 더 좋아질 것이다.

T1 9개를 같은 초기화·teacher·task schedule·192 updates로 처음부터 학습했다. Role-update 경로까지 gradient가 흐르는 것도 검증했다. 기존 checkpoint에 학습을 더한 방식이 아니므로 추가 update 양의 이득과 혼동하지 않는다.[^u03]

| 방법 | Potential 학습 시 role weights | 평가 시 role weights |
|---|---|---|
| R0 | 기존 frozen q | 기존 frozen q |
| R1 | 기존 frozen q | 자기-update |
| T1 | 자기-update | 자기-update |
| Constant+update | Train-fitted constant | 해당 potential의 update |

새 graph bank의 전체 조건 평균:

| 방법 | Bayes predictive KL ↓ | Accuracy |
|---|---:|---:|
| R0 | 0.033483 | 61.849% |
| R1 | 0.020488 | 62.225% |
| T1 | 0.020295 | 62.131% |
| Constant+update | 0.022231 | 62.133% |

#### F09. T1의 R1 대비 우수성 미확인

$$
\Delta_{\mathrm{T1-R1}}=-0.000192,
\qquad 95\%\ \mathrm{CI}=[-0.000576,+0.000164].
$$

사용자가 보고한 accuracy 차이는 약 −0.093%p다. 표시 평균끼리 빼면 KL −0.000193, accuracy −0.094%p가 되지만, 반올림 차이를 임의로 보정하지 않고 보고된 paired 값을 유지한다.

T1은 low-density 세 prior에서는 R1보다 좋아졌지만, 별도의 16조건에서는 KL 점추정이 악화됐다. Low density에서도 R0에는 못 미쳤다. 이 16조건 모두가 유의하게 악화됐다는 뜻은 아니며, 제공되지 않은 나머지 condition의 상태를 추정하지 않는다.[^u03]

**실패한 주장:** 자기-update에 맞춰 학습하면 현재 R1보다 전반적으로 우수한 새 학습법이 된다.

**성공한 부분:** 학습 단계부터 별도 posterior head 없이 구성할 수 있고 실제 gradient도 흐른다.

**운영 판정:** 현재 예산에서 T1의 추가 이득은 확인되지 않았다. 더 큰 예산에서도 절대 개선되지 않는다는 증명은 아니지만, T1을 살리기 위해 loss·예산을 계속 늘릴 근거는 부족하다.

#### R1의 성과는 별도로 재현됐다

보고된 평균 기준 R1−R0의 KL 차이는 −0.012995, 약 38.8% 감소다. Accuracy 차이는 +0.376%p다. T1이 R1을 더 개선하지 못했다는 사실이 이 R1 추론 변경의 효과를 지우지 않는다.[^u03]

---

### 4.8 Low-density 교차 분석: 실패 원인을 weights 하나로 줄일 수 없었다

Constant/learned weights를 \(q_C,q_L\), conditional predictions를 \(r_C,r_L\)라 하고 네 조합을 비교했다.

| Conditional | Constant weights \(q_C\) | Learned weights \(q_L\) |
|---|---|---|
| Constant \(r_C\) | Constant+update | Weights 교체 진단 |
| Learned \(r_L\) | Conditional 교체 진단 | Learned 자기-update |

사용자 보고의 방향성:

| Prior | Learned weights 교체 효과 | Learned conditional 교체 효과 |
|---|---|---|
| Independent | 해당 교차 비교에서 손해 | 해당 교차 비교에서 손해 |
| Exclusive | 해당 교차 비교에서 손해 | 해당 교차 비교에서 손해 |
| Coupled | 해당 교차 비교에서 도움 | 해당 교차 비교에서 손해 |

비가법적인 상호작용도 확인됐다.[^u03]

#### F10. 단일 병목 설명 기각

**기각한 설명 1:** low-density 실패는 role posterior만 고치면 해결된다.  
Conditional을 바꾸는 것 자체도 손해를 줄 수 있었다.

**기각한 설명 2:** Bethe partition 근사가 주된 원인이므로 exact partition이면 해결된다.  
Exact role update에서도 실패했다.

**기각하지 못한 가능성:** potential 추정 편향, parameter uncertainty 부족, conditional BP 오차, 과확신, 각 요소의 상호작용. 이번 자료만으로 어느 하나를 최종 원인으로 확정하지 않는다.

각 교차 효과는 상대 구성요소를 고정한 상태에서의 효과다. 손실을 구성요소별 고정 기여도로 더할 수 없다. 예를 들어 네 조합 손실의 interaction

$$
I=L(q_L,r_L)-L(q_L,r_C)-L(q_C,r_L)+L(q_C,r_C)
$$

가 0이 아니면 단순 가법 분해가 성립하지 않는다. 이 문서에는 interaction의 정확한 수치가 제공되지 않았으므로 방향·유의성 크기를 만들어 넣지 않는다.

또한 exact role update가 conditional BP를 exact inference로 바꿔주는 것은 아니다. **Partition 문제만으로 설명되지 않는다는 사실을 BP 전체의 근사오차가 무관하다는 뜻으로 확대하지 않는다.**

---

### 4.9 최종 강한 기준선 비교: R1을 메인 메소드로 승격하지 않기로 했다

**마지막 질문:** R1은 Constant보다 유의하게 좋은가? 그리고 더 강한 model-based 방법보다 품질 또는 비용에서 경쟁력이 있는가?

추가 학습 없이, 이미 검토한 동일 bank에서 R1·Constant·Grid81·Grid6을 비교했다. 따라서 이번 최종 Grid 비교 자체를 새 독립 hold-out 검증으로 부르지 않는다.[^u04]

| 방법 | Bayes predictive KL ↓ | NLL ↓ |
|---|---:|---:|
| R1 | 0.020488 | 0.615840 |
| Constant+update | 0.022231 | 0.617605 |
| Joint Grid81 | 0.014249 | 0.610493 |
| **Joint Grid6** | **0.011939** | **0.608083** |

#### R1−Constant의 유의한 개선은 남는다

$$
\Delta_{\mathrm{R1-Constant}}=-0.001744,
\qquad 95\%\ \mathrm{CI}=[-0.002072,-0.001425].
$$

약 7.8%의 평균 KL 감소다. 반올림 평균끼리의 차이 −0.001743보다, 사용자가 보고한 paired 차이 −0.001744를 우선한다.[^u04]

#### F11. 강한 기준선 대비 품질 우위 확보 실패

Grid6−R1의 표시 평균 차이는 KL −0.008549, NLL −0.007757이다. KL 평균 기준으로 Grid6이 약 41.7% 낮다. 사용자 보고상 Grid6은 KL·NLL·Brier 모두 우수했고 해당 paired CI가 모두 0을 제외했다.

Brier 평균과 이 세 비교의 CI 양 끝값은 메시지에 없으므로 이 문서에서 만들어 넣지 않는다. Grid6의 accuracy 우위도 보고되지 않았으므로 주장하지 않는다.[^u04]

**실패한 주장:** R1을 강한 대안보다 우수한 메인 메소드로 확정할 수 있다.

**정확한 판정:** R1의 유의한 개선과 최종 경쟁력 부족은 동시에 성립한다. Constant만 넘는 것으로는 충분하지 않았다.

#### F12. 품질 열세를 상쇄할 큰 속도 우위도 없었다

RTX 3090, 병렬화한 Grid6과의 보고값:

| 항목 | R1 | Grid6 | 표시 평균 기준 해석 |
|---|---:|---:|---|
| 단일 graph latency | 5.17 ms | 4.66 ms | R1이 약 10.9% 느림 |
| Batch16 latency | 6.41 ms | 6.81 ms | R1이 약 5.9% 빠름 |
| Batch16 추가 작업 메모리 | 1.44 MiB | 64.04 MiB | R1이 약 44.5배 적은 메모리 사용 |

작업 메모리 차이는 62.60 MiB, 약 97.8% 절감이다. **이는 총 GPU 메모리 차이가 아니라 측정된 추가 작업 메모리 차이**다.

Latency 반복 측정의 분산·CI는 제공되지 않았다. Batch16의 5.9% 점추정 차이를 확실한 큰 속도 우위로 확대하지 않는다. 단일 graph에서는 오히려 R1이 느렸다.[^u04]

**실패한 주장:** R1이 품질에서 밀려도 큰 추론 가속으로 정당화된다.

**남은 장점:** 추가 작업 메모리 절감. 다만 더 큰 graph 처리, 실제 메모리 한계 극복, 배포상 이익으로 이어지는지는 검증하지 않았다. 이 숫자만으로 새 메모리 효율 메인 주장을 만들지는 않는다.

#### Grid6이 Grid81보다 좋은 이유를 과장하지 않는다

초기 명세에서 Grid81은 broad continuous prior의 9×9 quadrature이고, Grid6은 training-support 여섯 parameter와 질량 `(0.5, 0.1, 0.1, 0.1, 0.1, 0.1)`을 사용한다. **격자 수뿐 아니라 prior 자체가 다르다.**[^s03]

최종 비교가 이 명세를 유지했다는 것은 원시 config 대조로 다시 확인할 수 있지만, 이 문서 작성에서는 최신 config를 직접 열지 않았다. 따라서 단순히 “6점이 81점보다 정밀하다”, “parameter uncertainty는 필요 없다”, “작은 grid가 일반적으로 더 낫다”고 해석하지 않는다.

알려진 family의 prior·likelihood를 활용하는 Grid도 BP를 사용한다. 이번 결과는 BP 전체의 실패가 아니라, **현재 learned-potential 구성의 강한 대안 대비 경쟁력이 부족했다는 결과**다.

---

## 5. 반복해서 성립하지 않았던 기대

### 5.1 Component fitting이 좋아지면 전체 predictor도 좋아질 것이라는 기대

초기 conditional/posterior loss, EAJ predictive 학습의 posterior 악화, low-density 교차 분석에서 반복해서 분리됐다. 어떤 module의 오차 감소만으로 최종 predictor의 우위를 주장할 수 없었다.

### 5.2 더 직접적인 loss가 메소드 경쟁력을 해결할 것이라는 기대

직접 predictive 학습은 실제로 일부 지표를 개선했다. 그러나 EAJ에서는 강한 Constant+update의 실용 품질을 넘지 못했고, T1에서는 R1 대비 추가 이득도 확인하지 못했다. **Loss 변경의 효과와 메소드의 경쟁력을 다른 질문으로 유지해야 했다.**

### 5.3 일관된 추론이면 OOD도 해결될 것이라는 기대

Self-update는 효과가 있었지만 low density를 해결하지 못했다. 같은 potential에 기반한 추론은 잘못 맞는 potential을 자동으로 교정하지 않으며, 새 조건에서의 적합성을 보장하지 않는다.

### 5.4 Neural amortization 또는 branch 수 감소가 곧 큰 속도 우위라는 기대

EAJ에서는 공통 BP 계산이 비용을 지배했다. 최종 R1 비교에서는 병렬 Grid6이 비슷한 latency를 보였다. 연산량·branch 수·posterior-only 시간만으로 실제 wall-clock 우위를 대신할 수 없었다.

### 5.5 기준선 한 개를 이기면 메인 방법으로 정할 수 있다는 기대

R1−R0와 R1−Constant는 중요한 성공이었다. 그러나 Grid6과의 품질·비용 비교에서 독립적인 메인 메소드로 승격할 근거가 부족해졌다. 약한 비교에서의 유의성과 강한 비교에서의 경쟁력은 다른 기준이었다.

---

## 6. 우리가 연구 운영에서 잘못 판단한 부분

이 절은 새로운 실험 결과가 아니라, 이번 진행 순서와 이전 assistant 조언에 대한 회고다.

### 6.1 강한 Grid 기준선 비교가 늦었다

Joint-grid가 이미 존재하고 강한 결과를 보였는데도, R1 frozen 개선 후 T1 재학습을 먼저 추진했다. **R1 대 병렬 Grid6/81의 품질·비용 비교를 T1보다 먼저 했어야 했다.**

이는 T1 실험이 무의미했다는 뜻은 아니다. Head 없는 학습 가능성과 추가 이득 미확인을 알게 됐다. 다만 메인 메소드 성립 여부를 빠르게 판별하는 순서로는 더 중요한 비교가 앞에 있어야 했다.

### 6.2 Assistant가 ‘유망한 후보’를 ‘메인 방법’으로 너무 빨리 표현했다

R1이 frozen q와 Constant보다 좋아지는 결과를 보았을 때 “메인 방법으로 유지”라고 한 것은 성급했다. 그 시점의 정확한 표현은 **“추론 변경 효과가 확인된 후보; 강한 Grid와 비교 대기”**였다.

반대로 강한 Grid에 밀렸다고 “남은 게 없다”처럼 연구 전체를 부정하는 것도 과도하다. 후보 승격 실패와 국소 개선의 실재를 분리해야 했다.

### 6.3 서로 다른 목표의 성공을 이어 붙여 하나의 강한 주장으로 만들 위험이 있었다

Posterior KL 개선, predictive KL 개선, accuracy 개선, 비용 절감, 새로운 family 일반화는 각각 다른 검증이 필요했다. 어느 한 단계의 성공을 다음 단계의 성공처럼 사용해서는 안 됐다.

### 6.4 Oracle gap을 다음 투자의 자동 근거로 삼을 수 없었다

Certified oracle까지 여지가 크다는 것은 현재 mixture가 최적이 아니라는 뜻이다. 그 gap을 관측 정보와 현재 모델로 회수할 수 있는지, 회수해도 강한 baseline을 넘는지는 별개다. “아직 gap이 크다”만으로 loss 탐색을 계속할 이유는 없었다.

### 6.5 남은 장점 하나에 맞춰 목표를 사후 변경할 위험이 있었다

최종 메모리 이점은 사실이지만, 원래 목표가 무너진 뒤 이를 검증 없이 메인 주장으로 바꾸지 않았다. 마찬가지로 low-density 실패를 본 뒤 조건별 gate나 temperature를 추가해 평가 조건에 맞추는 방향도 현재 갈래의 기본 연장으로 채택하지 않는다.

---

## 7. 실패로 오해하면 안 되는 결과와 아직 하지 않은 것

### 7.1 보존할 실증 결과

| 보존 대상 | 확인된 내용 | 확대해서는 안 되는 주장 |
|---|---|---|
| Learned BP checkpoint | 원래 positive/signed decoder보다 좋은 matched 확률 예측 | 모든 생성 family에서 learned likelihood가 필요하다. |
| R1 자기-update | 같은 potential의 활용 방식을 바꾸는 것만으로 개선; 새 graph에서 재현 | 학습 단계도 head가 필요 없었다. |
| R1−Constant paired 결과 | 최신 비교에서 KL 개선 CI가 0 제외 | 강한 모든 baseline보다 우수하다. |
| T1 | Head 없는 학습 경로와 gradient 흐름 확인 | R1보다 우수한 학습법이다. |
| EAJ A→B | 직접 predictive 학습의 효과 확인 | Posterior 복원도 개선됐다. |
| Convex oracle | 보고된 certificate와 큰 미회수 gap | 현재 모델이 쉽게 그 gap을 회수한다. |
| Nonedge/mask controls | 관측 상태에 맞는 factor 사용의 중요성 | 모든 현실 데이터에서 해당 관측 구분이 주어진다. |
| 작업 메모리 | 지정 Batch16 측정에서 큰 절감 | 실제 처리 가능한 graph 규모가 더 크다. |

### 7.2 미실행·미검증이며 실패로 세지 않을 항목

**관계의 관측과정 자체를 학습하는 BP**는 제안됐지만, 이 대화와 확인한 보고서에서는 완료된 실험이 없다. Relation missingness가 label/role에 의존하는 설정, unseen semantic schema, 실제 heterograph, 큰 graph scaling도 여기서 실패한 실험으로 기록하지 않는다.

새 temperature/shrinkage/density gate, 더 큰 모델, 추가 surrogate loss, 더 긴 T1 학습 역시 **추가 실행을 제안하지 않거나 중단한 것**이지, 모든 변형을 실제로 학습해서 실패를 증명한 것은 아니다.

Low density의 최종 원인도 아직 단일 원인으로 확정하지 않았다. 이 문서는 실패 기록이지 모든 실패 원인의 완전한 인과 분해가 아니다.

---

## 8. 확인 가능한 실행·재사용 내역

| 단계 | 신규 학습 | 재사용 | 근거 |
|---|---|---|---|
| Role-Conditional Distillation | 63개, 12,096 optimizer updates | A/A0 18개 checkpoint | 원문 보고서 |
| Learned-potential MVP | 36개, 6,912 optimizer updates | 기존 D_all posterior 고정 | 원문 보고서 |
| Joint-grid BP | 0 | 기존 bank와 비교 결과; 새 analytic 계산 | 원문 보고서 |
| 최초 EAJ | 18개, 3,456 optimizer updates | 기존 train/development bank | 원문 보고서 |
| EAJ predictive B | B 9개, A와 같은 예산 | A checkpoint | 사용자 보고; update 원기록 미열람 |
| Frozen self-update | 0 | Learned potential·conditional BP | 사용자 보고 |
| T1 | 9개 × 192 updates | R0/R1 checkpoint와 대조 | 사용자 보고 |
| 최종 Grid 비교 | 0 | 이미 검토한 동일 bank | 사용자 보고 |

전체 GPU 시간이나 중복 제거한 누적 학습 비용은 합산하지 않았다. 기존 모델을 재사용한 비용을 새 학습 비용으로 세지 않고, 단계별 원본 manifest를 대조하지 않은 총합을 감사된 값처럼 쓰지 않는다.

---

## 9. 이 기록을 재사용할 때 지켜야 할 사항

**지표:** 평균 KL의 상대 감소와 조건별 상대 감소율의 평균은 다르다. KL·NLL·Brier 개선을 accuracy 또는 ECE 개선으로 바꾸어 쓰지 않는다. Teacher가 다른 KL끼리 크기를 직접 비교하지 않는다.

**통계:** 초기 원문들은 주로 graph 안에서 context와 fitted-seed 지표를 평균한 뒤 4,000회 paired graph bootstrap을 사용했다. 이 CI는 해당 fitted-seed 집합에 조건부이며, 전체 training randomness와 multiple testing의 불확실성을 모두 포함하지 않는다. 최신 사용자 보고의 resampling 세부를 초기 방식과 같다고 자동 가정하지 않는다.

**Bank:** 원래 개발 bank, T1 단계의 새 graph bank, 그 bank를 재사용한 최종 Grid 비교를 구분한다. 매번 새로운 독립 평가가 이루어진 것처럼 적지 않는다.

**추론:** Exact role weights, exact conditional inference, true-parameter teacher, finite-grid exact reference, Bethe approximation을 각각 구분한다. 한 부분이 exact라는 이유만으로 전체 predictor가 exact인 것은 아니다.

**비용:** Serial/parallel grid, batch size, posterior-only/end-to-end, 추가 작업 메모리/총 메모리를 구분한다. 수학적인 branch 수와 실제 wall-clock ratio를 동일시하지 않는다.

**최신 판정:** 이전 MD에 있는 “R1 메인 유지”와 “R1−Constant CI 미확인”은 역사적 상태다. 현재는 CI가 확인됐고, Grid6 비교 후 메인 메소드 승격을 중단했다.

---

## 10. 보존할 자료와 상태

### 10.1 현재 상태

| 항목 | 상태 |
|---|---|
| 기존 staged decoder에 conditional/posterior loss 추가 | 현재 갈래 종료 |
| Masked-edge auxiliary를 핵심 새 학습 기여로 주장 | 중단 |
| EAJ에 새 surrogate loss 추가·규모 확대 | 중단 |
| T1의 우수성을 확보하기 위한 추가 학습 | 현재 중단 |
| R1을 강한 기준선보다 우수한 메인 메소드로 승격 | 중단 |
| R1 구조 개선과 paired 결과 | 보존 |
| Learned BP·A/B·T1 checkpoints와 재현 코드 | 보존 |
| Joint Grid6/81 및 Constant 기준선 | 이후 비교 자산으로 보존 |
| 분석 결과만으로 논문 성립 | 아직 판정하지 않음 |
| 새로운 문제에서의 learned potential/BP 전체 가능성 | 이 기록의 판정 범위 밖 |

### 10.2 보존해야 할 연결 관계

원본 결과를 지우거나 실패 실험을 표에서 빼지 않는다. 각 checkpoint가 어떤 initialization, teacher, prior, schedule, 학습 예산, evaluation bank와 연결되는지 유지한다.

권장 보존 대상은 source/config/manifest, per-graph predictions와 metrics, paired differences/CI, oracle certificate, cost measurement 원기록, 실패·audit 기록이다. 초기 보고서에 등장하는 `results/aggregate.json`, `results/contrasts.json`, `results/audit.json` 등은 **해당 원래 실행 디렉터리 안의 상대경로**다. 이 MD에 원시 파일이 동봉됐다는 뜻이 아니며, 최신 실험의 정확한 경로를 추정해 만들지 않는다.

원자료 복원 시 특히 확인할 미상 정보는 EAJ B의 전체 CI와 posterior KL 수치, oracle certificate의 정확한 bound, 최신 조건별 R1/T1 표, 최종 Grid6의 Brier 값과 CI endpoints, latency 반복 측정 변동, 마지막 비교의 exact/Bethe 및 Grid config다. **이 정보가 빠져 있다는 이유로 새 학습을 시작하지 않는다.**

---

## 11. 종료 문구

> 이 연구에서는 역할별 conditional 감독, posterior 감독, masked-edge likelihood auxiliary, learned joint evidence, 직접 predictive 학습, 자기-update 추론 및 그 재학습을 순차적으로 검증했다. 일부 구성요소의 정확도와 확률 예측은 개선됐으나, 그 개선이 더 강한 기준선 대비 독립적인 메소드 경쟁력으로 연결되지는 않았다. R1은 같은 learned potential을 더 효과적으로 사용하는 추론 개선이며 Constant+update보다 유의하게 좋은 결과를 얻었다. 그러나 최종 비교에서 Joint Grid6이 KL·NLL·Brier에서 우수하고 latency도 비슷했으므로, 현재 R1/EAJ 갈래를 우수한 범용 추론법으로 확대하는 것은 중단한다. Checkpoint, 통제 실험, oracle·교차 분석, 작업 메모리 이점은 보존하되, 이를 완성된 새 메소드의 증거로 과장하지 않는다.

**한 줄 요약: 학습과 추론의 몇몇 개선은 실제로 얻었지만, 현재 문제에서 작은 명시적 grid를 넘는 새 메인 메소드는 만들지 못했다.**

---

## 출처와 문서 이력

아래 출처 표기는 외부 문헌 리뷰가 아니라, 이번 회고를 뒷받침하는 실험 보고서와 사용자 보고의 위치를 나타낸다. 출처의 상대 이미지·결과 링크와 원시 파일은 이 MD에 복사하지 않았다.

[^s01]: **Role-Conditional Distillation: 고정 아키텍처 학습 ablation**. 보고서 날짜 2026-10-31. Library 파일 `붙여넣은 텍스트(1)(20261031-220527).txt`, file ID `file_00000000549c8206b63f8d044842f1a2`. 이번 정리에서 전체 본문 확인. 주요 근거: §1 기존 baseline 정정, §2 동일 예산, §3–4 결과·paired CI, §6 decoder 제약, §7 최종 판정, 검증과 재현.

[^s02]: **Mask-aware Relational Potential PFN — learned MVP**. 보고서 날짜 2026-11-01. Library 파일 `붙여넣은 텍스트(1)(20261031-224249).txt`, file ID `file_000000001270820681f66daedaca18ad`. 이번 정리에서 전체 본문 확인. 주요 근거: §1–3 모델·teacher·예산, §4 matched 결과와 auxiliary CI, §5 module 교체, §6 nonedge/mask, §7 OOD, §9 판정.

[^s03]: **Joint Role–Parameter Grid Posterior — same-decoder BP test**. 보고서 날짜 2026-11-01. Library 파일 `붙여넣은 텍스트(1)(20261031-234727).txt`, file ID `file_00000000b4d0820684cdac8062a8901d`. 이번 정리에서 전체 본문 확인. 주요 근거: §1 grid/prior와 평가 bank, §2 joint/frozen/product 구분, §3–6 결과와 한계, §7 teacher/accuracy, §8 비용·exact enumeration 제한.

[^s04]: **Evidence-Amortized Joint PFN — fixed-branch learned MVP**. 보고서 날짜 2026-11-01. Library 파일 `붙여넣은 텍스트(1)(20261101-005341).txt`, file ID `file_00000000c4848206963dae0b69b1c247`. 이번 정리에서 전체 본문 확인. 주요 근거: 고정 설계, 두 reference 구분, 주요 predictive 결과, posterior 근사, 비용, checkpoint·검증. 동일 내용으로 검색된 다른 파일은 독립 실험으로 중복 계산하지 않았다.

[^u01]: 이 대화에서 사용자가 **“비교를 끝냈어. 직접 predictive 학습의 효과는 확인됐지만, 현재 EAJ에 새 surrogate loss를 추가하는 갈래는 종료하는 게 맞아”**로 시작해 제공한 결과. A→B 여섯 조건의 grid predictive KL, B 9개 학습, posterior KL 악화, signal-high 실용 기준선 열세, graph-shared convex oracle 12,096개와 16–32% 회수율의 출처. Certificate bound는 메시지에서 글자가 손상돼 미상이다. 최신 원문·원시 배열은 이번 정리에서 미열람.

[^u02]: 이 대화에서 사용자가 **“첫 번째 안을 구현해 확인했어. 이 방향은 계속할 가치가 있어”**로 시작해 제공한 frozen 자기-update 결과. Exclusive 네 조건의 frozen/constant/exact/Bethe 표, high-density 세 prior의 CI 판정, Independent accuracy +0.83%p의 출처. 최신 원문·원시 배열은 이번 정리에서 미열람.

[^u03]: 이 대화에서 사용자가 **“완료했어. 추론 변경의 이득은 새 seed에서 재현됐지만, T1 재학습의 추가 우위는 확인되지 않았어”**로 시작해 제공한 결과. T1 9개·192 updates, gradient 검증, 전체 평균, T1−R1 CI, low-density 및 교차 분석의 출처. 이전 두 R1 MD에도 해당 요약이 있으나, 여기서는 사용자 보고를 일차 근거로 사용했다.

[^u04]: 이 대화에서 사용자가 **“강한 기준선까지 비교했어. R1의 개선은 유효하지만, 독립적인 메인 메소드로 확정하기에는 부족해”**로 시작해 제공한 최종 비교. R1/Constant/Grid81/Grid6 KL·NLL, R1−Constant CI, Grid6의 KL·NLL·Brier 우위, RTX 3090 단일/Batch16 latency와 추가 작업 메모리의 출처. 최종 비교는 이미 검토한 같은 bank를 재사용했다.
