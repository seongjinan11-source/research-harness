# 실패 병목과 논문 강점: 재검토 단서

자동 매칭은 논문 metadata의 강점·태그·키워드 검색이다. 해결·novelty·재개 승인이 아니다. 새 논문 원문에서 필요한 장치와 가정을 확인하고, 아래 최소 재검증을 통과해야 한다. 기존 논문도 포함하므로 목록에 나타났다는 사실이 새 해결책의 출현을 뜻하지 않는다.

원인 수준: 직접 확인 / 진단이 지지 / 미검증 가설 / 원인 미상. 직접 확인은 적힌 병목·대조를 확인했다는 뜻이며 모든 인과 원인을 밝혔다는 뜻은 아니다.

## RATIO-F01 — Full−Neutral log probability가 LP에 더할 새로운 feature evidence다.

- 판정: Frozen 비율 보정 중단
- 원인·병목 [진단이 지지]: Neutral이 정확한 feature marginalization이 아니므로 비율을 likelihood ratio로 해석할 근거가 없다. IMDB LP log margin은0.134~0.151로 작고, residual argmax 변경률34.68%였다. 원인별 기여는 분리하지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 통제 입력의 의미와 feature marginalization을 정의하고 확률 비율이 유효한 증거가 되는 조건을 제시하는 방법이 나올 때.
- 먼저 이겨야 할 기준선: LP/PFN alone, 50:50, controlled neutral
- 최소 재검증: 먼저 feature intervention으로 순효용을 확인하고, 동일 구조·주입 아래 ratio가 정답 방향 보정인지 새 graph에서 평가한다.
- 닫지 않은 것: LP-conditioned 학습과 일관된 생성모형의 likelihood ratio 일반은 별개다.
- 원문: [findings/sources/graphpfn-2026-11-01/ratio.md](../findings/sources/graphpfn-2026-11-01/ratio.md); [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## FEAT-F01 — IMDB의 PFN 우위는 남은 실제 관측 feature의 평균 추가 효용이다.

- 판정: 해당 feature-reliability 동기 약화
- 원인·병목 [진단이 지지]: 평균상 실제 값 효과가 거의 상쇄됐다. 구조 입력 PFN과 synthetic-input 경로의 차이를 feature-only evidence로 오인할 수 없다. Node별 양·음의 효과가 왜 상쇄되는지는 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 실제 관측 feature의 양·음 utility를 누수 없이 분리하거나, 유효 feature가 남는 조건을 특정한 방법이 나올 때.
- 먼저 이겨야 할 기준선: Structure-only PFN, no-observed end-to-end, independent-injection controlled Full/Zero
- 최소 재검증: Mask·context·query·graph를 맞춘 제거/셔플 개입에서 실제 값 경로와 synthetic injection 변화를 분리한다.
- 닫지 않은 것: 다른 결측률·graph와 query별 feature utility는 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## FMARG-F01 — 단일 대치 대신 graph-correlated feature completion의 예측을 평균하면 두 주 결측 조건에서 point/diagonal/zero/PCFI 대조를 모두 넘는다.

- 판정: 현재 frozen surrogate completion operator의 사전 primary gate 미통과·확대 중단
- 원인·병목 [원인 미상]: 명시한 Gaussian-process surrogate와 4-sample frozen PFN 연산에서 필요한 동시 우위가 없었다. Feature-law misspecification, zero-fill 학습 PFN의 입력 활용 한계, 유한 Monte Carlo, 실제 feature 정보 부족의 원인은 분리하지 못했다.
- 반증·배제한 설명: Observed clamp·label 분리·covariance sampler·완전 관측 일치·weights 보존을 확인했다. 이것이 surrogate의 통계적 적합성을 보장하지는 않는다.; 모든 arm이 동일 checkpoint와 4개 feature-position view를 사용했다. 별도의 대조군별 재학습 예산 차이는 없다.
- 재검토 조건: 관측된 feature evidence와 graph의 joint missing-feature law를 검증 가능하게 연결하고, point fill 및 같은 예산의 covariance 제거·구조 random-feature 대조를 안정적으로 넘는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 PFN의 zero/PCFI/mean/native-NaN·independent sampling·context prior. 후속 positive 주장은 별도의 PCFI/FISF-trained 대조, 비조건부 구조 random features, GCNmf/GSPN/PEMix와의 차별화 필요.
- 최소 재검증: Source·data·예산을 먼저 고정한 새 graph에서 feature conditional의 품질과 predictive marginalization의 증분을 분리한다. Frozen replacement와 distributional-input 재학습을 같은 성과로 세지 않는다.
- 닫지 않은 것: Context labels에 조건화한 feature law, joint predictive training, cross-feature dependency, 정확한 SCM conditional, 실제 graph·unseen schema는 평가하지 않았다. GP likelihood는 공식 SCM posterior가 아니다.
- 원문: [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md); [findings/sources/feature-marginal-2026-11-02/paired.json](../findings/sources/feature-marginal-2026-11-02/paired.json)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## XFER-F01 — DBLP에서 통했던 context-LOO path 선택을 다른 graph로 옮기면 강하다.

- 판정: 고정 선택 규칙의 전이 실패
- 원인·병목 [진단이 지지]: 소수 context의 ranking이 query utility와 맞지 않았다. 후보 pool 확대와 accuracy 목적 변경만으로는 해결되지 않았다. 추정 분산·risk mismatch의 상대 기여는 미상이다.
- 반증·배제한 설명: Uniform/union 후보가 없어서만 실패했다는 설명은 확대 pool 대조로 충분하지 않았다.
- 재검토 조건: 극소 context의 추정 불확실성과 실제 query risk를 다루며 task 내 utility를 식별하는 방법이 나올 때.
- 먼저 이겨야 할 기준선: Uniform/union LP, individual paths, fixed GraphPFN path inputs
- 최소 재검증: 독립 graph에서 동일 context·후보 pool·선택 예산으로 사전 규칙을 고정해 검증한다.
- 닫지 않은 것: Context가 유용한 관계를 알려주는 다른 설정이나 relation 의미 추론 일반은 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/transfer.md](../findings/sources/graphpfn-2026-11-01/transfer.md); [findings/sources/graphpfn-2026-11-01/expanded.md](../findings/sources/graphpfn-2026-11-01/expanded.md); [findings/sources/graphpfn-2026-11-01/accuracy_selector.md](../findings/sources/graphpfn-2026-11-01/accuracy_selector.md)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## XFER-F02 — LP와 PFN 확률의 고정 혼합이 각자의 강점을 보존한다.

- 판정: 고정 혼합 메인 방향 중단
- 원인·병목 [원인 미상]: 서로 맞히는 query가 다르다는 사실이 고정 확률 평균의 utility를 보장하지 않았다. 중복 증거가 원인일 수 있으나 직접 분리한 결과는 아니다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 중복 evidence와 decision risk를 실제 관측량으로 구분하며 최강 단일 predictor를 보존하는 원리가 나올 때.
- 먼저 이겨야 할 기준선: LP alone, PCFI/FISF+PFN alone, compute-matched fixed mixtures
- 최소 재검증: 같은 query·context에서 within-task 오류 보완성과 독립 개발/확인 graph의 평균·worst-case utility를 함께 본다.
- 닫지 않은 것: 모든 ensemble이 불가능한 것도, 각 branch가 feature-only/structure-only인 것도 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/fusion.md](../findings/sources/graphpfn-2026-11-01/fusion.md)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## RFILL-F01 — 현재 frozen conditional Gaussian completion이 분산을 맞춘 random-fill/graph-prior null보다 안정적으로 좋은 예측을 제공한다.

- 판정: 사전 random-control gate 미통과; 현재 frozen conditional의 독립 가치 미입증
- 원인·병목 [원인 미상]: 동일 관측 입력·checkpoint·분산/forward-budget 대조에서 요구한 conditional law 우위가 없었다. 관측값에 조건화한 복원, 구조 random feature, 확률 smoothing의 원인별 기여를 완전히 식별한 결과는 아니다.
- 반증·배제한 설명: Null은 분산·동일 PFN 호출 수를 맞췄다. 이것이 정확한 true posterior 또는 모든 feature law에 대한 대조를 뜻하지는 않는다.
- 재검토 조건: Context와 관측 피처가 joint feature law에 제공하는 추가 정보를 proper feature score와 실제 query improvement로 입증하고, 같은 예산의 random/구조 입력보다 좋은 연산이 나올 때.
- 먼저 이겨야 할 기준선: Variance-matched iid/graph prior, iid unit, zero/PCFI/FISF 입력 및 같은 예산·별도 학습의 강한 PFN. 실제 관측 입력 직접 경로와 proposal 경로를 구분.
- 최소 재검증: 새 학습 가설을 source/data/budget 고정 후 별도 bank에서 비교한다. Label-blind law 대조와 true-label-context conditional-law 증분을 분리하며 query labels를 forward에 사용하지 않는다.
- 닫지 않은 것: 실제 관측 피처의 직접 효용·context-conditioned feature-law 학습·shared suffix·coherence 일반은 평가하지 않았다. 분산 맞춤 null에는 second-moment 정보가 남고 unit null에는 실제 값 clamp가 남는다.
- 원문: [findings/sources/random-fill-controls-2026-11-02/REPORT.md](../findings/sources/random-fill-controls-2026-11-02/REPORT.md); [findings/sources/random-fill-controls-2026-11-02/paired.json](../findings/sources/random-fill-controls-2026-11-02/paired.json)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## EVID-F01 — Relation 출처를 보존하는 head가 강한 Plain의 class decision을 개선한다.

- 판정: 현재 Evidence Head No-Go
- 원인·병목 [원인 미상]: Plain 정답을 잃는 mass5.36%가 구제2.48%보다 컸다. 정보 중복과 representation 간섭 중 원인은 식별하지 못했다. Confidence를 낮춰서 NLL이 좋아졌다는 설명은 후속 분석에서 기각됐다.
- 반증·배제한 설명: 과신을 줄여 확률 점수가 좋아졌다는 설명: Plain은 confidence63.99% vs accuracy88.96%로 과소확신이었다.
- 재검토 조건: Plain의 강한 결정 능력을 유지하면서 출처별 정보의 독립 가치를 회수하는 장치가 나올 때.
- 먼저 이겨야 할 기준선: Plain, Mean/Separate 동일 용량 control, fixed context-weighted evidence
- 최소 재검증: 동일 task에서 harmed/rescued partition과 NLL/Brier/accuracy를 함께 비교하고 conflict 강도별 증분을 확인한다.
- 닫지 않은 것: Relation source 정보가 불필요하다는 결론이나 다른 내부 추론 연산의 기각은 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/evidence.md](../findings/sources/graphpfn-2026-11-01/evidence.md); [findings/sources/graphpfn-2026-11-01/evidence_utility.md](../findings/sources/graphpfn-2026-11-01/evidence_utility.md)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 복수 태그 일치 (약한 단서); tags=graph-structure-use, uncertainty; keywords=-
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

## SRC-F01 — Source covariance를 보정하면 분류가 개선된다.

- 판정: 독립적인 task-level 이득 미확인
- 원인·병목 [원인 미상]: 오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.
- 먼저 이겨야 할 기준선: Vanilla continuation, matched/log/diagonal, covariance-shuffled controls
- 최소 재검증: 분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.
- 닫지 않은 것: 출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

- [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf): 구체 키워드+태그 일치 (미검증); tags=uncertainty; keywords=covariance
  - 기록된 강점: Covariance에 unlabeled geometry를 반영하면서 out-of-sample 확률 예측을 제공한다.

