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

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## FEAT-F01 — IMDB의 PFN 우위는 남은 실제 관측 feature의 평균 추가 효용이다.

- 판정: 해당 feature-reliability 동기 약화
- 원인·병목 [진단이 지지]: 평균상 실제 값 효과가 거의 상쇄됐다. 구조 입력 PFN과 synthetic-input 경로의 차이를 feature-only evidence로 오인할 수 없다. Node별 양·음의 효과가 왜 상쇄되는지는 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 실제 관측 feature의 양·음 utility를 누수 없이 분리하거나, 유효 feature가 남는 조건을 특정한 방법이 나올 때.
- 먼저 이겨야 할 기준선: Structure-only PFN, no-observed end-to-end, independent-injection controlled Full/Zero
- 최소 재검증: Mask·context·query·graph를 맞춘 제거/셔플 개입에서 실제 값 경로와 synthetic injection 변화를 분리한다.
- 닫지 않은 것: 다른 결측률·graph와 query별 feature utility는 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## COUP-F01 — Label→feature 반복 피드백이 일방향 추론보다 유용하다.

- 판정: 피드백 증분 미확인
- 원인·병목 [진단이 지지]: 피드백의 추가 정보를 확보하지 못했다. 관측 feature가 전혀 없으면 latent H가0에 머물고 피드백 제거와 출력이 같았다. 이 구조적 퇴화가 모든 조건의 미개선을 설명하지는 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 라벨 피드백이 실제로 새로운 feature 상태를 만들고 그 정보가 일방향 모델에 없음을 보이는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 예산 일방향 모델, 피드백 제거, 강한 context ridge baseline
- 최소 재검증: Zero-observation 퇴화와 nonzero 조건을 나눠 같은 초기화·task·예산으로 실제 예측 증분을 비교한다.
- 닫지 않은 것: 비선형 label-conditioned feature inference와 다른 유효 prior는 평가하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## FMARG-F01 — 단일 대치 대신 graph-correlated feature completion의 예측을 평균하면 두 주 결측 조건에서 point/diagonal/zero/PCFI 대조를 모두 넘는다.

- 판정: 현재 frozen surrogate completion operator의 사전 primary gate 미통과·확대 중단
- 원인·병목 [원인 미상]: 명시한 Gaussian-process surrogate와 4-sample frozen PFN 연산에서 필요한 동시 우위가 없었다. Feature-law misspecification, zero-fill 학습 PFN의 입력 활용 한계, 유한 Monte Carlo, 실제 feature 정보 부족의 원인은 분리하지 못했다.
- 반증·배제한 설명: Observed clamp·label 분리·covariance sampler·완전 관측 일치·weights 보존을 확인했다. 이것이 surrogate의 통계적 적합성을 보장하지는 않는다.; 모든 arm이 동일 checkpoint와 4개 feature-position view를 사용했다. 별도의 대조군별 재학습 예산 차이는 없다.
- 재검토 조건: 관측된 feature evidence와 graph의 joint missing-feature law를 검증 가능하게 연결하고, point fill 및 같은 예산의 covariance 제거·구조 random-feature 대조를 안정적으로 넘는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 PFN의 zero/PCFI/mean/native-NaN·independent sampling·context prior. 후속 positive 주장은 별도의 PCFI/FISF-trained 대조, 비조건부 구조 random features, GCNmf/GSPN/PEMix와의 차별화 필요.
- 최소 재검증: Source·data·예산을 먼저 고정한 새 graph에서 feature conditional의 품질과 predictive marginalization의 증분을 분리한다. Frozen replacement와 distributional-input 재학습을 같은 성과로 세지 않는다.
- 닫지 않은 것: Context labels에 조건화한 feature law, joint predictive training, cross-feature dependency, 정확한 SCM conditional, 실제 graph·unseen schema는 평가하지 않았다. GP likelihood는 공식 SCM posterior가 아니다.
- 원문: [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md); [findings/sources/feature-marginal-2026-11-02/paired.json](../findings/sources/feature-marginal-2026-11-02/paired.json)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## VIEW-F01 — Support metadata·다중 전파 view·PCFI/FISF 입력 결합이 범용 증분을 준다.

- 판정: 전체 조건 진입 기준 미통과
- 원인·병목 [원인 미상]: 출력에 보인 증분이 작고 조건별 부호가 달랐다. 표현 정보 중복, projection 손실, frozen backbone의 활용 한계 중 무엇이 주원인인지는 분리하지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 관측 provenance를 정보 손실 없이 표현·학습하고 equal-width 값 대조를 넘는 장치가 나올 때.
- 먼저 이겨야 할 기준선: PCFI/FISF parents, equal-width value controls, 같은 비용의 single predictor
- 최소 재검증: 같은 차원·추론 비용으로 새 실제 graph에서 parent 둘과 metadata 없는 값 대조를 함께 비교한다.
- 닫지 않은 것: Native cell-wise provenance encoder를 학습하는 방법과 입력 자체를 바꾸는 학습은 평가하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md); [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missing
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## LLBP-F07 — EAJ를 직접 predictive 학습하면 실용성과 posterior 해석을 함께 회복한다.

- 판정: 부분 성공·메소드 구제 실패 (predictive 개선을 posterior 복원 개선으로 보는 해석 기각)
- 원인·병목 [진단이 지지]: 실제 decoder에 맞춘 predictive weights와 정확한 Bayesian posterior 복원은 다른 목적이었다. Direct predictive 개선은 있었지만 classical 기준선을 넘지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Decoder 오차를 다루면서 posterior 해석과 predictive quality를 함께 보장하는 방법이 나올 때.
- 먼저 이겨야 할 기준선: Direct predictive EAJ, Constant+update, exact-grid
- 최소 재검증: 두 KL과 실제 NLL/Brier를 함께 보고 calibrated posterior라는 주장을 별도 검증한다.
- 닫지 않은 것: 직접 predictive 학습이 작동하지 않았다는 결론은 틀리다. B의 출력은 decoder에 맞춘 aggregation weights로 남긴다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, uncertainty; keywords=posterior, predictive
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## MISS-F01 — Missingness prior 추가가 공식 입력 경로와 미학습 MNAR에서 유리하다.

- 판정: 범용 우위 미확보·해당 라인 중단
- 원인·병목 [진단이 지지]: Raw-NaN 기준선이 공식 imputation보다 약했고, official fill만으로도 mask가 잘 드러났다(AUROC 0.97). Context-query mask 관계 불일치에서 mixed-prior 이득이 유지되지 않았다. 이것만으로 모든 원인을 확정하지는 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 관측 메커니즘의 context-query shift를 다루면서 official preprocessing 대비 추가 가치를 보이는 학습 원리가 나올 때.
- 먼저 이겨야 할 기준선: 공개 GraphPFN + official imputation, 동일 예산 B4
- 최소 재검증: 동일 입력 경로·초기화·예산의 새 bank에서 context-query mismatch와 미학습 mask를 paired 평가한다.
- 닫지 않은 것: Imputed-input에 맞춘 prior의 효과는 미평가다. 더 넓은 MNAR 식별 문제도 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md); [findings/sources/graphpfn-2026-11-01/missing_pilot.md](../findings/sources/graphpfn-2026-11-01/missing_pilot.md); [findings/sources/graphpfn-2026-11-01/missing_context.md](../findings/sources/graphpfn-2026-11-01/missing_context.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missingness
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## RFILL-F01 — 현재 frozen conditional Gaussian completion이 분산을 맞춘 random-fill/graph-prior null보다 안정적으로 좋은 예측을 제공한다.

- 판정: 사전 random-control gate 미통과; 현재 frozen conditional의 독립 가치 미입증
- 원인·병목 [원인 미상]: 동일 관측 입력·checkpoint·분산/forward-budget 대조에서 요구한 conditional law 우위가 없었다. 관측값에 조건화한 복원, 구조 random feature, 확률 smoothing의 원인별 기여를 완전히 식별한 결과는 아니다.
- 반증·배제한 설명: Null은 분산·동일 PFN 호출 수를 맞췄다. 이것이 정확한 true posterior 또는 모든 feature law에 대한 대조를 뜻하지는 않는다.
- 재검토 조건: Context와 관측 피처가 joint feature law에 제공하는 추가 정보를 proper feature score와 실제 query improvement로 입증하고, 같은 예산의 random/구조 입력보다 좋은 연산이 나올 때.
- 먼저 이겨야 할 기준선: Variance-matched iid/graph prior, iid unit, zero/PCFI/FISF 입력 및 같은 예산·별도 학습의 강한 PFN. 실제 관측 입력 직접 경로와 proposal 경로를 구분.
- 최소 재검증: 새 학습 가설을 source/data/budget 고정 후 별도 bank에서 비교한다. Label-blind law 대조와 true-label-context conditional-law 증분을 분리하며 query labels를 forward에 사용하지 않는다.
- 닫지 않은 것: 실제 관측 피처의 직접 효용·context-conditioned feature-law 학습·shared suffix·coherence 일반은 평가하지 않았다. 분산 맞춤 null에는 second-moment 정보가 남고 unit null에는 실제 값 clamp가 남는다.
- 원문: [findings/sources/random-fill-controls-2026-11-02/REPORT.md](../findings/sources/random-fill-controls-2026-11-02/REPORT.md); [findings/sources/random-fill-controls-2026-11-02/paired.json](../findings/sources/random-fill-controls-2026-11-02/paired.json)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## SEL-F01 — Context Brier·local reliability로 PFN/LP의 정답을 선택할 수 있다.

- 판정: Gate B 실패, Gate C 보류
- 원인·병목 [진단이 지지]: Brier와 accuracy 목적이 불일치하는 사례를 직접 관측했다. IMDB seed11에서 LP의 test Brier0.6624가 PFN0.6672보다 좋지만 accuracy는35.54% vs44.65%였다. Context noise만으로 설명할 수 없다.
- 반증·배제한 설명: Context Brier 추정 noise만의 문제라는 설명은 실제 test Brier/accuracy 역전으로 충분하지 않다.
- 재검토 조건: 극소 context에서 정확도 효용과 uncertainty를 함께 식별하는 추정법이 나올 때.
- 먼저 이겨야 할 기준선: 최강 단일 PFN/LP, global/local selector, fixed mixture
- 최소 재검증: Query 정답 없는 입력만으로 실제 선택 utility를 독립 graph에서 평가하고 proper score와 accuracy를 분리한다.
- 닫지 않은 것: Oracle headroom을 실제로 회수하는 다른 risk estimator는 기각하지 않는다. 학습 selector는 이 실험에서 미실행이다.
- 원문: [findings/sources/graphpfn-2026-11-01/reliability.md](../findings/sources/graphpfn-2026-11-01/reliability.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## SRC-F01 — Source covariance를 보정하면 분류가 개선된다.

- 판정: 독립적인 task-level 이득 미확인
- 원인·병목 [원인 미상]: 오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.
- 먼저 이겨야 할 기준선: Vanilla continuation, matched/log/diagonal, covariance-shuffled controls
- 최소 재검증: 분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.
- 닫지 않은 것: 출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

- [StructuralMissingPFN](https://arxiv.org/abs/2601.18500): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask, uncertainty; keywords=-
  - 기록된 강점: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.
