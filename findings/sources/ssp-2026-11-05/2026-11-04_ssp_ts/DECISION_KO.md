# SSP-TS 개발 pilot 결과와 판정

판정: **이번 조건에서 SSP-TS의 확대를 중단하고 기존 SSP-mean을 보존한다.**
사전에 고정한 세 성능 조건을 모두 충족하지 못했다. 이 결과를 TS 일반의
무용성, SSP prior의 실패, 또는 확인되지 않은 특정 실패 원인으로 확대하지 않는다.

## 완료한 작업과 검증

- 계획을 과학/선행연구, 수학/구현, 실험/통계의 세 관점으로 검토하고 수정했다.
- 기존 논문·코드·결과는 변경하지 않고 이 cycle에 새 구현과 원시 결과를 저장했다.
- 원본 benchmark 네 클래스의 정확한 텍스트와 출처 SHA256을 보존했다.
- 저장된 과거 400개 평가와 objective 값이 오차 0으로 일치했다.
- posterior mean/covariance, 60,000개 sample, scalarization, size–contrast covariance,
  극단적 scale, single-count, 작은 문제의 exhaustive MILP 등을 포함한 **37/37 검사 통과**.
- 4 families × 2 new development targets × 5 arms = **40 runs, 4,000 charged evaluations**.
- 모든 run에서 100개 unique feasible evaluation; random fallback 0회.
- 사후 별도 4,000개 objective replay와 AUC 재계산 감사 통과.
- 소요 wall time **483.19초(약 8.05분)**, 최대 네 CPU worker, GPU/신경망 학습 없음.
- 종료 후 locked input/code 및 원본 raw ZIP의 SHA256이 유지됨을 확인했다.

## 핵심 성능

각 값은 family당 개발 target 두 개의 평균이다. 통계적 추론이나 최종 일반화
성능을 나타내지 않는다. 이 runner는 모든 arm에 같은 1초 full-set MILP와
no-good exclusion을 적용하며, 논문의 5초 v32 결과와 직접 비교할 수 없다.

| Family | SSP-mean AUC | SSP-TS AUC | TS 대 mean |
|---|---:|---:|---:|
| MOKP | 0.641556 | 0.597609 | −6.85% |
| MDK | 0.593699 | 0.540499 | −8.96% |
| SetCover | 0.886316 | 0.915558 | +3.30% |
| MWIS | 0.282360 | 0.280703 | −0.59% |

Family별 ratio-of-means 개선율을 동일 비중으로 평균한 **G = −3.2746%**.
사전 기준 G≥1%를 충족하지 못했고, 개선 family는 요구한 3/4보다 적은 1/4이며,
두 family가 허용 하한 −2%보다 낮았다. Eight paired targets 중 다섯 개에서 악화했다.

| Family | Seed 2026110400 | Seed 2026110401 |
|---|---:|---:|
| MOKP | −7.23% | −6.46% |
| MDK | −8.21% | −9.71% |
| SetCover | +3.75% | +2.84% |
| MWIS | −3.98% | +2.95% |

## 통제 비교와 solver 진단

| Family | L0에서 TS 대 mean | SSP-TS 대 L0-TS | SSP-TS 대 permuted |
|---|---:|---:|---:|
| MOKP | −5.62% | +23.36% | −0.08% |
| MDK | −3.95% | +14.16% | −0.19% |
| SetCover | −10.59% | +58.72% | +2.43% |
| MWIS | −13.44% | +40.52% | +0.68% |

Size prior의 이점은 TS 아래에서도 관찰되지만, SSP-mean을 개선했다는 증거는 아니다.
Permutation 대조는 size perturbation과 contrast norm을 보존하는 제한적 구조
대조다. 완전 무정보 대조가 아니며 covariance 일반의 가치를 판단할 수 없다.

| Family | Certified share: mean / TS | Mean relative gap: mean / TS | Distinct counts: mean / TS |
|---|---:|---:|---:|
| MOKP | 100% / 100% | 0.000061 / 0.000021 | 4.5 / 12.0 |
| MDK | 83.75% / 100% | 0.001493 / 0.001190 | 5.0 / 8.5 |
| SetCover | 0% / 0% | 0.369583 / 0.332194 | 13.5 / 16.5 |
| MWIS | 0% / 0% | 0.449551 / 0.222349 | 31.0 / 21.0 |

Certification은 동일 full feasible-set solve의 relative gap≤0.002 기준이다.
MOKP에서는 두 arm의 모든 제안이 이 기준을 만족했고 TS가 악화했다. MDK도
TS가 더 잘 certified되므로 knapsack 손실을 TS의 solver 품질 저하로 설명할
근거는 없다. SetCover와 MWIS에는 큰 solver gap이 남아 acquisition 자체와
후보 최적화의 영향을 분리할 수 없다. SetCover의 작은 이득은 이 1초 runner의
개발 관찰로만 남긴다.

Fit 400회 중 392회가 optimizer success를 보고했다. 나머지 8회도 finite
posterior/선형계 residual 검사를 통과했으며 그대로 기록했다. SSP-mean/TS만
보면 각각 79/80 및 80/80 success다. 이 flag만으로 실제 구현 오류를 확정하지 않는다.
SSP-mean/TS의 summed run wall time은 각각 586.64/577.72초로 거의 같았고,
sampling 자체 비용은 작았다. 전체 실행은 네 worker가 병렬 처리한 시간이다.

## 결과 리뷰와 해석

- **수학/추론 reviewer:** 실제 sampler/수치 bug의 증거를 발견하지 못했다.
  저장된 16개 SSP-mean/TS run의 각 round를 재구성한 추가 읽기 전용 진단에서
  최대 solve residual은 5.63e−14였다. 선택된 다음 batch의 predictive residual이
  심한 variance 오보정을 지지하지 않았다. 이것을 전체 posterior calibration의
  증명으로 확대하지 않는다. 현재 gate 실패를 유지한다.
- **실험/통계 reviewer:** 여덟 paired contrast와 equal-family 집계가 일치했다.
  중복/불공정 fallback으로 설명할 차이는 없었다. SetCover/MWIS의 solver gap을
  해석에 반드시 포함하고, 원인이 확인되지 않은 상태에서 온도나 예산을 바꾸어
  같은 개발 target을 반복 사용하지 않는 것이 적절하다.
- **과학/신규성 reviewer:** 기존 TS 선행을 인정하고 음성 결과를 보존한다.
  SSP-TS가 L0-TS보다 좋다는 사실과 SSP-mean의 upgrade 여부를 구분한다.
  SetCover만 사후 선택해 범용 성공으로 주장하지 않는다.

현재 데이터는 '과도한 탐색', 'EB 과신', 'size 정보가 불필요함' 중 어느 하나를
실패 원인으로 확정하지 못한다. Count 다양성 증가는 AUC 개선의 충분조건이 아니다.

## 다음 후보의 상태

Decision-value/KG 기반 관측 배분은 **미실행 가설**로 남긴다. 이 cycle에서는
구현하거나 실험하지 않았다. 해당 후보를 연구하려면 별도 계획·리뷰·새 평가
명세가 필요하며, 현재 pilot을 보고 설계했다면 현재 target들은 개발 자료다.
온도 sweep, 새 batch 정책, 더 긴 solver limit 또는 대규모 confirmation을
자동으로 이어서 실행하지 않는다.

## 재현 자료

- `PLAN.md`, `REVISION_LOG.md`: 사전 계획과 검토 반영.
- `MANIFEST.json`, `SOURCE_PROVENANCE.json`: 코드·입력·원본 출처 해시.
- `NUMERICAL_CHECKS.json`: 37개 검증 결과.
- `runs/`: 40개 원시 run의 X/Y, posterior fit, 제안별 solver 기록.
- `per_run.csv`, `SUMMARY.json`, `EXECUTION.json`: 감사된 수치와 실제 실행 내역.
