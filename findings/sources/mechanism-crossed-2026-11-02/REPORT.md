# Mechanism-crossed GraphPFN — controlled prior pilot

2026-11-02. 실제 학습 완료: 12 checkpoints × 256 successful updates. Gate A=False, Gate B=False.

모델은 기존 공개 GraphPFN Mask Plain 그대로다. Official Mask, Coupled Mask, Crossed Mask, Coupled PCFI를 같은 초기화·update 수로 학습했다. 추가 posterior head·refiner·auxiliary loss는 없다.

## 주요 결과

| Test family / 조건 | 모델 | NLL ↓ | Accuracy % | Brier ↓ |
|---|---|---:|---:|---:|
| controlled:offdiagonal:uniform | official_mask | 0.70976 | 52.20 | 0.51364 |
| controlled:offdiagonal:uniform | coupled_mask | 0.70308 | 50.72 | 0.50880 |
| controlled:offdiagonal:uniform | crossed_mask | 0.70189 | 52.69 | 0.50700 |
| controlled:offdiagonal:uniform | coupled_pcfi | 0.69793 | 51.07 | 0.50360 |
| controlled:offdiagonal:uniform | uniform_prior | 0.69315 | 47.56 | 0.50000 |
| controlled:offdiagonal:uniform | context_prior | 0.69565 | 47.56 | 0.50244 |
| controlled:offdiagonal:uniform | LP | 0.69761 | 56.05 | 0.50086 |
| controlled:offdiagonal:clustered | official_mask | 0.70267 | 53.71 | 0.50763 |
| controlled:offdiagonal:clustered | coupled_mask | 0.69634 | 49.79 | 0.50330 |
| controlled:offdiagonal:clustered | crossed_mask | 0.69345 | 53.22 | 0.50019 |
| controlled:offdiagonal:clustered | coupled_pcfi | 0.71085 | 52.99 | 0.51298 |
| controlled:offdiagonal:clustered | uniform_prior | 0.69315 | 47.56 | 0.50000 |
| controlled:offdiagonal:clustered | context_prior | 0.69565 | 47.56 | 0.50244 |
| controlled:offdiagonal:clustered | LP | 0.69761 | 56.05 | 0.50086 |
| official:any:uniform | official_mask | 0.69252 | 53.30 | 0.49886 |
| official:any:uniform | coupled_mask | 0.69303 | 52.70 | 0.49969 |
| official:any:uniform | crossed_mask | 0.69042 | 51.95 | 0.49726 |
| official:any:uniform | coupled_pcfi | 0.70163 | 51.12 | 0.50606 |
| official:any:uniform | uniform_prior | 0.69315 | 52.15 | 0.50000 |
| official:any:uniform | context_prior | 0.69315 | 52.15 | 0.50000 |
| official:any:uniform | LP | 0.70639 | 53.52 | 0.50865 |
| official:any:clustered | official_mask | 0.69373 | 53.47 | 0.50016 |
| official:any:clustered | coupled_mask | 0.69113 | 52.49 | 0.49796 |
| official:any:clustered | crossed_mask | 0.69004 | 53.04 | 0.49691 |
| official:any:clustered | coupled_pcfi | 0.69333 | 52.34 | 0.49998 |
| official:any:clustered | uniform_prior | 0.69315 | 52.15 | 0.50000 |
| official:any:clustered | context_prior | 0.69315 | 52.15 | 0.50000 |
| official:any:clustered | LP | 0.70639 | 53.52 | 0.50865 |

## Crossed의 paired NLL 차이

| 조건 | 대조군 | Δ NLL [95% CI] |
|---|---|---:|
| official:any:uniform | crossed_mask:standard - official_mask:standard | -0.00210 [-0.00970, +0.00530] |
| official:any:uniform | crossed_mask:standard - coupled_mask:standard | -0.00261 [-0.00625, +0.00056] |
| official:any:uniform | crossed_mask:standard - coupled_pcfi:standard | -0.01121 [-0.02889, +0.00020] |
| official:any:uniform | crossed_mask:standard - uniform_prior:standard | -0.00272 [-0.00898, +0.00316] |
| official:any:uniform | crossed_mask:standard - context_prior:standard | -0.00272 [-0.00898, +0.00316] |
| official:any:uniform | crossed_mask:standard - LP:standard | -0.01597 [-0.03218, -0.00234] |
| official:any:clustered | crossed_mask:standard - official_mask:standard | -0.00369 [-0.00928, +0.00194] |
| official:any:clustered | crossed_mask:standard - coupled_mask:standard | -0.00108 [-0.00333, +0.00117] |
| official:any:clustered | crossed_mask:standard - coupled_pcfi:standard | -0.00328 [-0.00891, +0.00198] |
| official:any:clustered | crossed_mask:standard - uniform_prior:standard | -0.00310 [-0.00820, +0.00245] |
| official:any:clustered | crossed_mask:standard - context_prior:standard | -0.00310 [-0.00820, +0.00245] |
| official:any:clustered | crossed_mask:standard - LP:standard | -0.01635 [-0.03103, -0.00352] |
| controlled:offdiagonal:uniform | crossed_mask:standard - official_mask:standard | -0.00787 [-0.01482, -0.00143] |
| controlled:offdiagonal:uniform | crossed_mask:standard - coupled_mask:standard | -0.00119 [-0.00603, +0.00402] |
| controlled:offdiagonal:uniform | crossed_mask:standard - coupled_pcfi:standard | +0.00396 [-0.00917, +0.01896] |
| controlled:offdiagonal:uniform | crossed_mask:standard - uniform_prior:standard | +0.00874 [-0.00591, +0.03062] |
| controlled:offdiagonal:uniform | crossed_mask:standard - context_prior:standard | +0.00624 [-0.00596, +0.02307] |
| controlled:offdiagonal:uniform | crossed_mask:standard - LP:standard | +0.00428 [-0.01163, +0.02196] |
| controlled:offdiagonal:clustered | crossed_mask:standard - official_mask:standard | -0.00921 [-0.01959, -0.00046] |
| controlled:offdiagonal:clustered | crossed_mask:standard - coupled_mask:standard | -0.00288 [-0.00817, +0.00207] |
| controlled:offdiagonal:clustered | crossed_mask:standard - coupled_pcfi:standard | -0.01740 [-0.03587, -0.00337] |
| controlled:offdiagonal:clustered | crossed_mask:standard - uniform_prior:standard | +0.00031 [-0.01090, +0.01184] |
| controlled:offdiagonal:clustered | crossed_mask:standard - context_prior:standard | -0.00220 [-0.01222, +0.00729] |
| controlled:offdiagonal:clustered | crossed_mask:standard - LP:standard | -0.00415 [-0.01764, +0.00824] |

## 관측 피처 개입 (90% whole-node, 고정 checkpoint)

| 조건 | Crossed 개입 | Δ NLL intervention−standard [95% CI] |
|---|---|---:|
| official:any:uniform | no_observed - standard, paired episode subset | +0.00378 [-0.00604, +0.01313] |
| official:any:uniform | shuffled - standard, paired episode subset | -0.00126 [-0.00579, +0.00297] |
| official:any:clustered | no_observed - standard, paired episode subset | +0.00621 [-0.00026, +0.01290] |
| official:any:clustered | shuffled - standard, paired episode subset | +0.00277 [-0.00089, +0.00666] |
| controlled:diagonal:uniform | no_observed - standard, paired episode subset | +0.00582 [-0.00420, +0.01562] |
| controlled:diagonal:uniform | shuffled - standard, paired episode subset | +0.00302 [-0.00224, +0.00831] |
| controlled:diagonal:clustered | no_observed - standard, paired episode subset | -0.00296 [-0.01189, +0.00615] |
| controlled:diagonal:clustered | shuffled - standard, paired episode subset | +0.00307 [-0.00161, +0.00818] |
| controlled:offdiagonal:uniform | no_observed - standard, paired episode subset | -0.01189 [-0.02658, +0.00132] |
| controlled:offdiagonal:uniform | shuffled - standard, paired episode subset | -0.00083 [-0.00463, +0.00312] |
| controlled:offdiagonal:clustered | no_observed - standard, paired episode subset | -0.00234 [-0.01112, +0.00672] |
| controlled:offdiagonal:clustered | shuffled - standard, paired episode subset | -0.00129 [-0.00446, +0.00208] |

## 정보량·실행 범위

Controlled train은 64 base graphs × 4 styles다. Coupled와 Crossed의 concrete feature field·rule instance 주변 exposure를 정확히 맞췄다. Crossed는 style×rule×mask 64조합이 균등하다. One graph/update이므로 pair loss·paired minibatch 효과는 없다.

Feature law는 graph Gaussian field의 4개 변환이며 label rule은 self linear/neighbor linear/product/two-hop contrast다. Label rule 함수는 style ID를 받지 않고, sample quantile threshold나 class-balance rejection을 쓰지 않는다. Context는 별도로 클래스당 최대4개를 고른다. 전체 realized graph가 single-class여도 binary output을 유지했다.

Gaussian/linear subset의 관측 sign-only 진단은 알려진 rule을 받는 privileged reference다. 실제 RMS 입력이 보존하는 첫 관측 sign만 사용하고 unknown magnitudes를 적분한다. 전체 context posterior나 실제 배포 predictor가 아니다. 정보량이 적은 과제와 zero-observation 과제는 제거하지 않았다.

Test32 fresh base graphs의 controlled diagonal/off-diagonal과 official native tasks를 동일 입력으로 평가했다. Style/rule 종류는 학습에서 본 것이며 새로운 graph/function seed다. 실제 graph·미지 semantic schema 성과가 아니다.

Official vs Crossed는 feature/label 함수족까지 다르므로 coupling만의 효과를 분리하지 못한다. Coupled vs Crossed가 통제된 비교다. Constructed biased prior를 고쳤다는 결과를 기존 공식 prior 실패 원인의 확인으로 확대하지 않는다.

Bootstrap은 base graph32개를 paired resampling5000회 했고 rates·고정 training seeds3개를 먼저 평균했다. 동일 base graph의 tasks를 독립 표본으로 세지 않았다. Training-seed 모집단 CI나 다중 비교 보정은 없다. 마지막 checkpoint만 사용했다.

PCFI 기준선도 같은 update 수로 해당 입력을 학습한 모델이다. FISF-trained 대조와 real graph 비교는 이 pilot에 없다. 원래 공개 모델의 공식 preprocessing 재현이 아닌 공통 RMS+mask interface다.

Sampling 대조는 사전 지정 official99.5% 조건의 4-view zero와 iid-unit noise다. Primary predictor는 1-view이므로 호출 수를 구분하며, 병렬 실행 시간을 방법의 latency 우위로 주장하지 않는다.

모든 12288 saved probability outputs를 독립 재채점했다. Frozen original transformer 보존과 같은 seed의 initialization·update schedule·trainable parameter 수를 검증했다. 성공 optimizer updates 합계=3072.
