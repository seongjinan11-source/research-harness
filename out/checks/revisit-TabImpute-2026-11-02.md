# 실패 병목과 논문 강점: 재검토 단서

자동 매칭은 논문 metadata의 강점·태그·키워드 검색이다. 해결·novelty·재개 승인이 아니다. 새 논문 원문에서 필요한 장치와 가정을 확인하고, 아래 최소 재검증을 통과해야 한다. 기존 논문도 포함하므로 목록에 나타났다는 사실이 새 해결책의 출현을 뜻하지 않는다.

원인 수준: 직접 확인 / 진단이 지지 / 미검증 가설 / 원인 미상. 직접 확인은 적힌 병목·대조를 확인했다는 뜻이며 모든 인과 원인을 밝혔다는 뜻은 아니다.

## COUP-F01 — Label→feature 반복 피드백이 일방향 추론보다 유용하다.

- 판정: 피드백 증분 미확인
- 원인·병목 [진단이 지지]: 피드백의 추가 정보를 확보하지 못했다. 관측 feature가 전혀 없으면 latent H가0에 머물고 피드백 제거와 출력이 같았다. 이 구조적 퇴화가 모든 조건의 미개선을 설명하지는 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 라벨 피드백이 실제로 새로운 feature 상태를 만들고 그 정보가 일방향 모델에 없음을 보이는 추론 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 예산 일방향 모델, 피드백 제거, 강한 context ridge baseline
- 최소 재검증: Zero-observation 퇴화와 nonzero 조건을 나눠 같은 초기화·task·예산으로 실제 예측 증분을 비교한다.
- 닫지 않은 것: 비선형 label-conditioned feature inference와 다른 유효 prior는 평가하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## VIEW-F01 — Support metadata·다중 전파 view·PCFI/FISF 입력 결합이 범용 증분을 준다.

- 판정: 전체 조건 진입 기준 미통과
- 원인·병목 [원인 미상]: 출력에 보인 증분이 작고 조건별 부호가 달랐다. 표현 정보 중복, projection 손실, frozen backbone의 활용 한계 중 무엇이 주원인인지는 분리하지 못했다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 관측 provenance를 정보 손실 없이 표현·학습하고 equal-width 값 대조를 넘는 장치가 나올 때.
- 먼저 이겨야 할 기준선: PCFI/FISF parents, equal-width value controls, 같은 비용의 single predictor
- 최소 재검증: 같은 차원·추론 비용으로 새 실제 graph에서 parent 둘과 metadata 없는 값 대조를 함께 비교한다.
- 닫지 않은 것: Native cell-wise provenance encoder를 학습하는 방법과 입력 자체를 바꾸는 학습은 평가하지 않았다.
- 원문: [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md); [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=missing
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## JIMP-F01 — PFN 내부의 반복 대치가 동일한 대치기를 PFN 앞에서 실행하는 것과 강한 동일 예산 Plain을 넘는다.

- 판정: 현재 구현 확대 보류·내부 대치 증분 미입증
- 원인·병목 [원인 미상]: 실제 공개 PFN과 대치 상태를 연결해 gradient를 학습했지만, 위치를 바꾼 독립 예측 이득은 확인하지 못했다. 정보 중복·현재 numeric 함수족·prior·학습 예산 중 원인은 분리하지 못했다.
- 반증·배제한 설명: 대치 모듈이 label readout에 연결되지 않은 dead module이라는 설명은 nonzero gradient와 동작 대조로 지지되지 않는다. 이것이 유용한 정보의 학습을 뜻하지는 않는다.; Frontend와 Interleaved의 초기 가중치·학습 과제 순서·파라미터 수·성공 update 수 차이로 설명할 수 없다. 실제 학습 시간과 수치 재시도 비용은 같다고 주장하지 않는다.
- 재검토 조건: 대치 상태를 PFN 추론 중 갱신할 때 사전 대치에 없던 예측 정보를 보존·전달하고, 동일 예산 Plain/PCFI 대조를 넘는 연산이 나올 때.
- 먼저 이겨야 할 기준선: 동일 synthetic missing prior로 추가 학습한 PCFI Plain·Mask Plain·Frontend; native NaN GraphPFN의 같은 예산 학습도 향후 필요
- 최소 재검증: 공통 전처리·초기화·학습 과제·성공 update 수를 맞춘 새 bank에서 반복 위치 효과를 비교한다. 같은 checkpoint의 피드백 제거와 재학습 대조를 구분하고 NLL/Brier/accuracy를 따로 보고한다.
- 닫지 않은 것: 다른 대치 함수족, pretrained feature decoder, latent belief representation, native NaN-input 대조와 실제 그래프 전이는 별개다. 이 실험은 이전 10K quadratic prototype과 다른 공개 GraphPFN 구현이다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/paired.json](../findings/sources/joint-imputation-2026-11-01/paired.json)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=observation-mask, tokenization; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## JIMP-F02 — PFN의 query 분류 목적에 맞춘 사전 대치기로 PCFI를 교체하면 주 결측 조건에서 독립 개선을 얻는다.

- 판정: 현재 task-trained frontend의 주 조건 우위 미입증
- 원인·병목 [원인 미상]: Query CE로 학습된 네 단계 대치가 고정 PCFI나 일반 mask/token adapter보다 유용하다는 증거가 부족했다. 정확한 원인은 미상이며 복원 목적이 아닌 분류 목적을 썼다는 사실만으로 개선되지 않았다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: PFN에 필요한 정보와 단순 복원값을 구분하는 대치·입력 연산이 강한 동일 학습 대조보다 안정적인 예측 이득을 보일 때.
- 먼저 이겨야 할 기준선: PCFI Plain, Mask Plain, native NaN input, 동일 예산·초기화·query CE
- 최소 재검증: 대치 연산의 개선과 backbone 추가 학습을 분리하고, 먼저 새 그래프 bank에서 prior-only 확률 대조까지 함께 비교한다.
- 닫지 않은 것: 모든 task-aware imputation과 PFN용 전처리가 실패했다는 뜻은 아니다. 피처 복원 MSE를 최소화한 모델과 비교한 결과도 아니다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/frontend_comparisons.json](../findings/sources/joint-imputation-2026-11-01/frontend_comparisons.json)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=observation-mask, tokenization; keywords=imputation
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## LLBP-F09 — 자기-update 경로까지 학습하는 T1이 R1보다 우수하다.

- 판정: 추가 이득 미확인
- 원인·병목 [원인 미상]: Self-update를 미분해 학습하는 경로는 작동했지만 R1 대비 이득이 없었다. Optimization·model bias 중 고유 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 동일 계산 budget에서 inference-aware 학습의 독립 이득을 보이는 원리가 나올 때.
- 먼저 이겨야 할 기준선: Frozen R1, R0, Constant+update, Grid6
- 최소 재검증: 추론 시 head 제거 효과와 새 학습 효과를 분리하고 initial state·schedule·예산을 맞춘다.
- 닫지 않은 것: head 없는 학습 경로와 gradient 흐름은 확인됐다. 더 큰 예산에서 절대 개선되지 않는다는 증명은 아니다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=amortized-inference; keywords=training
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## MISS-F01 — Missingness prior 추가가 공식 입력 경로와 미학습 MNAR에서 유리하다.

- 판정: 범용 우위 미확보·해당 라인 중단
- 원인·병목 [진단이 지지]: Raw-NaN 기준선이 공식 imputation보다 약했고, official fill만으로도 mask가 잘 드러났다(AUROC 0.97). Context-query mask 관계 불일치에서 mixed-prior 이득이 유지되지 않았다. 이것만으로 모든 원인을 확정하지는 않는다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 관측 메커니즘의 context-query shift를 다루면서 official preprocessing 대비 추가 가치를 보이는 학습 원리가 나올 때.
- 먼저 이겨야 할 기준선: 공개 GraphPFN + official imputation, 동일 예산 B4
- 최소 재검증: 동일 입력 경로·초기화·예산의 새 bank에서 context-query mismatch와 미학습 mask를 paired 평가한다.
- 닫지 않은 것: Imputed-input에 맞춘 prior의 효과는 미평가다. 더 넓은 MNAR 식별 문제도 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md); [findings/sources/graphpfn-2026-11-01/missing_pilot.md](../findings/sources/graphpfn-2026-11-01/missing_pilot.md); [findings/sources/graphpfn-2026-11-01/missing_context.md](../findings/sources/graphpfn-2026-11-01/missing_context.md)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 구체 키워드+태그 일치 (미검증); tags=observation-mask; keywords=imputation, missingness
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.

## SRC-F01 — Source covariance를 보정하면 분류가 개선된다.

- 판정: 독립적인 task-level 이득 미확인
- 원인·병목 [원인 미상]: 오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: 공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.
- 먼저 이겨야 할 기준선: Vanilla continuation, matched/log/diagonal, covariance-shuffled controls
- 최소 재검증: 분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.
- 닫지 않은 것: 출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

- [TabImpute](https://arxiv.org/abs/2510.02625v6): 복수 태그 일치 (약한 단서); tags=amortized-inference, observation-mask; keywords=-
  - 기록된 강점: 새 table에서 fitting 없이 대치하며 row-column 정보를 cell 단위로 공유한다.
