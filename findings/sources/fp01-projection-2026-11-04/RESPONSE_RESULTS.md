# FP01: 세 독립 결과 리뷰에 대한 답변

검토 파일은 RESULT_METHOD.md, RESULT_INFERENCE.md, RESULT_EXPERIMENTS.md다. 세 리뷰는 실제 artifact manifest 완성 후 수행됐으며 root의 결론 문서를 승인하기 위한 형식 절차로 사용하지 않았다. 실험 리뷰어는 저장된 primary 확률 2,304개에서 accuracy/NLL과 paired bootstrap을 별도로 재현했고, 추론 리뷰어는 manifest가 가리킨 46개 파일의 해시 일치를 확인했다.

| 지적 | Root의 처리 |
|---|---|
| C−B +0.952%p의 CI가 0을 포함하고 +1%p 조건도 미달 | 사전 전체 gate 미통과를 그대로 확정한다. 평균의 방향을 확실한 우위로 쓰지 않는다. |
| C−C_global +1.187%p CI[+0.213,+2.446]는 양성 | 제한된 적응 효과와 NLL 감소를 보존한다. 전체 실패에 합쳐 삭제하지 않는다. |
| B−A는 약 −0.017%p | 투영 노출 학습만으로 회복했다는 주장을 철회한다. 수학적 RP non-closure 자체와 그 실제 기여도는 구분한다. |
| C−none +0.579%p의 보조 CI도 0 포함 | 안정적인 추가 피처 정보 활용으로 판정하지 않는다. C−B와 효용 차이의 대수적 중복도 별개 근거로 세지 않는다. |
| Real의 개선은 대체로 기존 feature harm 감소와 양립 | Cora/CiteSeer 개발 accuracy 개선은 남기고 observed−none 및 shuffle을 함께 보고한다. 이 설명도 유일한 기전으로 확정하지 않는다. |
| Coauthors edge의 큰 개선은 A/B에도 존재 | 기존 구조적 성과를 새 centroid encoder의 공으로 이전하지 않는다. Dataset별 full/edge winner를 선택하지 않는다. |
| Minority metric과 LP 반례, synthetic edge NLL 악화 | DECISION_KO.md에 PubMed/Physics 및 edge NLL 반례를 포함했다. Accuracy만으로 전반적 우위를 주장하지 않는다. |
| Finite grid와 유한 BP는 continuous exact Bayes가 아님 | Lock 전에 측정한 grid/loopy 오차를 제한으로 유지한다. 새 버그의 증거 없이 같은 final을 튜닝하지 않는다. |
| Audit PASS는 scientific success가 아님 | 구현/provenance와 가설 판정을 분리한다. 필수 code repair가 필요한 새 결함은 발견되지 않았다. |
| 실패 로그 []가 운영 중단 비용 0을 뜻하지 않음 | 별도 EXECUTION_MANIFEST.json과 EVALUATION_EXECUTION.md에 중단·strict resume·one-orphan·replay를 남겼다. Fit worker-time/예측 시간으로 전체 실험 속도 우위를 주장하지 않는다. |

최종 결정은 **FP01 현재 명세의 확대를 닫되 부분 성과와 코드·결과를 보존**하는 것이다. Factor 전체나 다른 Gaussian/graph inference를 닫는 판정이 아니다. 새 과학적 변경은 다음 cycle/branch의 사전 계획과 세 리뷰를 거친다. 실패 원인은 현재 unknown으로 기록하며, 기존 관찰을 설명하는 가설을 확인된 원인으로 격상하지 않는다.

버전 관리: root의 `research/fp01-projection-centroid`는 코드·계획·리뷰·작은 결과를 포함한다. 무거운 bank/checkpoint/prediction은 로컬 고유 경로와 해시로 연결한다. 축적 저장소는 별도 `research/fp01-projection-centroid-record` 브랜치에 원문 snapshot·구조화 finding·재생성 지식 그래프를 기록한다. 원래 연구 기록과 ID는 덮어쓰지 않는다.
