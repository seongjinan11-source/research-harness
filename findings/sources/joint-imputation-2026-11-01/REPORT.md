# Joint Imputation + Public GraphPFN — matched pilot

No-Go for scaling this implementation: prespecified primary gate not passed.

공개 GraphPFN 1.3의 adapter/feature encoder/label decoder를 합성 과제로 추가 학습했다. 원래 LimiX transformer weights는 고정했다. 실제 그래프 전이·새 schema·논문 novelty는 미검증이다.

전체 graph를 사용했다. 부분그래프 선택은 변경하지 않았다. 공통 입력은 observed-only RMS, feature-pair numeric encoding과 원래 mask embedding이다. 공개 기본 전처리와 다르므로 이전 보고서 숫자를 전후 성능으로 비교하지 않는다.

Mask Plain: zero+mask와 일반 token residual. PCFI Plain: PCFI+mask와 같은 residual. Frontend: PFN 전에 공유 refiner 4회. Interleaved: 같은 refiner를 layer 0/3/6/9 앞에 넣는다. 두 refiner 모델은 파라미터·갱신 횟수가 같고 PFN은 모두 한 번(12 layers) 실행한다.

## 결과

| 조건 | 모델 | Accuracy % | NLL ↓ | Brier ↓ |
|---|---|---:|---:|---:|
| uniform | mask_plain | 49.04 | 0.88681 | 0.56854 |
| uniform | pcfi_plain | 47.35 | 0.89527 | 0.57272 |
| uniform | frontend | 47.35 | 0.89276 | 0.57608 |
| uniform | interleaved | 46.92 | 0.89745 | 0.57692 |
| clustered | mask_plain | 50.46 | 0.88318 | 0.56489 |
| clustered | pcfi_plain | 49.95 | 0.90248 | 0.57577 |
| clustered | frontend | 48.84 | 0.89339 | 0.57334 |
| clustered | interleaved | 49.27 | 0.89508 | 0.57245 |
| cell | mask_plain | 49.22 | 0.87600 | 0.56287 |
| cell | pcfi_plain | 50.10 | 0.88912 | 0.57107 |
| cell | frontend | 46.61 | 0.88628 | 0.57227 |
| cell | interleaved | 49.25 | 0.88375 | 0.56690 |
| k1 | mask_plain | 40.30 | 1.00109 | 0.63214 |
| k1 | pcfi_plain | 41.76 | 1.16219 | 0.68597 |
| k1 | frontend | 41.60 | 0.99158 | 0.62778 |
| k1 | interleaved | 43.03 | 0.93901 | 0.60018 |
| full | mask_plain | 51.66 | 0.84795 | 0.54149 |
| full | pcfi_plain | 50.94 | 0.84622 | 0.54053 |
| full | frontend | 51.14 | 0.84692 | 0.54118 |
| full | interleaved | 51.56 | 0.84715 | 0.54135 |

## Interleaved의 paired 차이

| 조건 | 대조군 | Δ NLL [95% CI] |
|---|---|---:|
| uniform | interleaved - mask_plain:standard | +0.01063 [-0.00717, +0.02525] |
| uniform | interleaved - pcfi_plain:standard | +0.00217 [-0.02237, +0.02302] |
| uniform | interleaved - frontend:standard | +0.00469 [-0.01612, +0.02143] |
| uniform | interleaved - interleaved:no_feedback | +0.00306 [+0.00063, +0.00581] |
| uniform | interleaved - interleaved:no_refinement | -0.23718 [-0.32406, -0.16659] |
| clustered | interleaved - mask_plain:standard | +0.01190 [-0.00048, +0.02546] |
| clustered | interleaved - pcfi_plain:standard | -0.00741 [-0.03982, +0.01992] |
| clustered | interleaved - frontend:standard | +0.00169 [-0.01132, +0.01571] |
| clustered | interleaved - interleaved:no_feedback | +0.00010 [-0.00359, +0.00314] |
| clustered | interleaved - interleaved:no_refinement | -0.24408 [-0.32087, -0.17547] |
| cell | interleaved - mask_plain:standard | +0.00775 [-0.00571, +0.02076] |
| cell | interleaved - pcfi_plain:standard | -0.00537 [-0.03377, +0.02076] |
| cell | interleaved - frontend:standard | -0.00253 [-0.01433, +0.00870] |
| cell | interleaved - interleaved:no_feedback | +0.00006 [-0.00249, +0.00277] |
| cell | interleaved - interleaved:no_refinement | -0.37644 [-0.48799, -0.27941] |
| k1 | interleaved - mask_plain:standard | -0.06208 [-0.13935, -0.00259] |
| k1 | interleaved - pcfi_plain:standard | -0.22317 [-0.39541, -0.09165] |
| k1 | interleaved - frontend:standard | -0.05257 [-0.12008, +0.00099] |
| k1 | interleaved - interleaved:no_feedback | +0.00259 [-0.00122, +0.00720] |
| k1 | interleaved - interleaved:no_refinement | -1.07398 [-1.39690, -0.79098] |
| full | interleaved - mask_plain:standard | -0.00080 [-0.00763, +0.00625] |
| full | interleaved - pcfi_plain:standard | +0.00093 [-0.00288, +0.00469] |
| full | interleaved - frontend:standard | +0.00023 [-0.00194, +0.00267] |
| full | interleaved - interleaved:no_feedback | +0.00000 [+0.00000, +0.00000] |
| full | interleaved - interleaved:no_refinement | +0.00000 [+0.00000, +0.00000] |

## 단순 확률 대조

| 조건 | Context prior NLL | Uniform prior NLL |
|---|---:|---:|
| uniform | 0.89707 | 0.91181 |
| clustered | 0.89707 | 0.91181 |
| cell | 0.89707 | 0.91181 |
| k1 | 0.91181 | 0.91181 |
| full | 0.89707 | 0.91181 |

## 학습·검증 범위

64 train / 32 fresh test graphs, seeds [11, 12, 13], 192 updates/arm/seed. 동일 seed에서 task schedule·common initialization·forward RNG를 맞췄다. 마지막 checkpoint만 사용했다. Query CE만 학습했고 feature reconstruction/auxiliary loss는 사용하지 않았다.

Graph별로 결측 조건과 3 seed 지표를 평균한 뒤 5,000회 paired bootstrap했다. CI는 graph 변동이며 training-seed 모집단 CI나 다중 검정 보정을 포함하지 않는다. Per-seed와 개별 mask 결과는 summary.json에 모두 남겼다. Zero-observation episode를 제외하거나 다시 뽑지 않았다.

Frontend/Interleaved의 정확한 동일 파라미터 비교와 Plain의 유사 용량 비교를 구분한다. General token residual과 explicit numeric refiner는 서로 다른 함수족이다. No-feedback와 no-refinement는 이미 학습한 checkpoint의 기전 진단이며 별도 재학습 대조가 아니다.

| 모델 | Trainable parameters | New parameters | Train sec, mean | Peak allocated MiB, max |
|---|---:|---:|---:|---:|
| mask_plain | 4012259 | 10200 | 65.9 | 566.5 |
| pcfi_plain | 4012259 | 10200 | 60.6 | 581.5 |
| frontend | 4013679 | 11620 | 73.4 | 700.5 |
| interleaved | 4013679 | 11620 | 87.6 | 689.0 |

Validation: hidden-feature poisoning invariance; no query labels in input dataclass; observed-value clamp; node relabeling; nonzero refiner/encoder gradients; identical common initialization; identical frontend/interleaved parameter counts; unchanged frozen weights; all saved probabilities independently re-scored. Details: preflight.json / training.json / run_manifest.json.

후속 방향은 이 구현의 결과 범위를 지킨다. Primary gate 미통과 시 test 결과를 보고 loss·반복 수·학습량을 바꿔 자동 구제하지 않는다. 성공하더라도 joint imputation 자체의 novelty를 주장하지 않는다.

## 피드백 제거 대조의 수치 보정

최초 no-feedback 대조의 `h + new_embedding - old_embedding`은 FP16에서 cancellation 오차가 있었다. `h + (new_embedding - old_embedding)`으로 묶고 입력 token의 같은 표현도 일치시켜 해당 대조만 3 seed 모두 다시 평가했다. 모델 학습과 standard/no-refinement 예측은 바꾸지 않았다. 위 표의 no-feedback 행은 보정된 값이다. 원래 predictions·paired.json은 보존했고, 현재 대조는 paired_corrected.json과 feedback_correction/evaluation.json을 사용한다.

보정 후 full-observed 96개 검사에서 no-feedback와 같은 process의 standard 차이는 정확히0이었다. 별도 process의 FP16 checkpoint replay와 원래 예측의 최대 확률 차이는 0.00246981였다. 따라서 해당 진단은 같은 process에서 다시 계산한 standard와 짝지었다. 주 비교의 원래 예측·성공 기준·선택한 checkpoint는 보존했다.

## 실행 보완 및 최종 provenance 검사

첫 Mask Plain seed11의 학습 완료 후 실행 프로세스가 종료되어 저장된 checkpoint에서 평가만 재개했다. 이미 완료한 모델을 추가 학습하지 않았다.

최초 Interleaved seed11은 FP16의 nonfinite gradient를 clipping 단계가 먼저 거부하면서 멈췄다. GradScaler가 scale을 낮춰 대응하기 전에 종료된 실행 문제였다. 원래 코드와 traceback은 runtime/에 보존했다. 이후 같은 과제를 낮은 scale로 재시도하고, finite gradient로 optimizer update가 성공한 경우만 192회에 포함했다. 모델·데이터·loss·LR schedule은 바꾸지 않았다.

수정 전 완료된 seed11의 Mask Plain·PCFI Plain·Frontend 결과는 보존했다. Interleaved는 완료 checkpoint가 없어 원래 초기화에서 다시 시작했다. 수치 재시도는 추가 학습 효과가 아니지만 추가 연산 비용이므로 시간 측정에 포함했다.

| 모델 | Seed | 성공 updates | Overflow 재시도 |
|---|---:|---:|---:|
| mask_plain | 11 | 192 | 0 |
| pcfi_plain | 11 | 192 | 0 |
| frontend | 11 | 192 | 0 |
| interleaved | 11 | 192 | 2 |
| mask_plain | 12 | 192 | 1 |
| pcfi_plain | 12 | 192 | 0 |
| frontend | 12 | 192 | 0 |
| interleaved | 12 | 192 | 3 |
| mask_plain | 13 | 192 | 0 |
| pcfi_plain | 13 | 192 | 0 |
| frontend | 13 | 192 | 4 |
| interleaved | 13 | 192 | 5 |

세 seed 모두 Frontend와 Interleaved의 초기 trainable weights·task schedule·parameter counts가 정확히 일치했다. 모든 12개 checkpoint의 hash와 480개 episode 파일 hash를 다시 확인했다. 총 4,032개 saved probability outputs의 metrics는 analyze 단계에서 독립 재계산했다.

추론 시간은 cached input 이후의 forward이며 PCFI 계산·bank 생성 비용을 포함하지 않는다. 이를 end-to-end 속도 우위로 해석하지 않는다. 일반 input recentering을 쓰지 않은 공통 RMS/mask interface이므로 official GraphPFN recipe의 성능 재현이라고 부르지 않는다.

원래 피처 수가 홀수면 feature pair를 위해 observed constant-zero channel 하나를 padding했다. Raw 관측량 통계는 padding을 제외한다. Uniform node99.5의 test32개 중 13개는 실제 관측 피처가 전혀 없었고 10개는 관측 노드가 하나였다. Clustered node99.5는 32개 모두 관측 노드가 하나였다. 해당 사례는 성능을 보고 제거·재추출하지 않았다.

원인 해석에 주의한다. 결측 state의 숫자가 생성됐다는 것은 참 피처를 복원했다는 증거가 아니다. 이 모델은 feature reconstruction target 없이 query CE만으로 학습한 task-conditioned numeric state다. Native NaN-input GraphPFN의 동일 예산 추가 학습, 큰 그래프, 실제 데이터와 새로운 semantic schema는 별도 미평가 범위다.
