# FP04 results review response

2026-11-04. 아래는 결과 검토에 대한 root 판정이다. 고정 과학 소스·bank·checkpoint·예측·원보고서와 primary를 수정하지 않았다.

| 지적 | 조치와 판단 |
|---|---|
| 관측-cell 경로의 미분 가능성과 유용한 feature inference가 다르다. | 동의. correctness PASS와 네 performance gate FAIL을 분리했다. Bridge 검사를 성능 증거로 사용하지 않는다. |
| Observed의 arm 우위는 source0/none에서도 생긴다. | 동의. 자체 대조보다 높은 real accuracy를 feature-value 기여로 설명하지 않고, 같은 checkpoint의 none/shuffle를 병기한다. |
| Primary의 작은 양의 효과와 NLL/Brier 개선을 버리거나 과장하지 말아야 한다. | +.423%p와 CI를 보존했다. CI의 0 포함은 무효과/동등성의 증명이 아니지만 고정 조건은 미충족이다. Family별 악화와 source0/1도 유지한다. |
| 실험한 모델은 rank4/4-round separable predictor이며 full sparse Transformer가 아니다. | 실패 범위를 그 구조와 예산으로 한정했다. 모든 observed-set PFN의 실패라고 일반화하지 않는다. |
| 모든 query가 항상 feature-free이며 full도 nonquery-visible다. | 학습된 완전-query feature 조건의 성능을 먼저 검증하지 않았음을 명시했다. 가능한 후속 질문으로만 남기고 실패 원인으로 확정하지 않는다. |
| Real에서는 수치적으로 거의 동률인 class probabilities와 4-hop 대칭 노드가 많다. | 원예측과 primary를 유지하면서 리뷰의 secondary margin/distance 분석을 제한적으로 인용했다. 작은 TV에서 큰 argmax 변화가 가능하며 semantic evidence와 같지 않다. FP32 버그로 확정하거나 사후 tie 변경으로 구제하지 않는다. |
| 외부 public PFN/PCFI/FISF/LP는 preprocessing와 예산이 다르다. | physical input identity 확인과 causal 비교의 한계를 구분했다. Restart LP와 PCFI/FISF 공개 기준선의 우위를 숨기지 않았다. |
| 실데이터는 재사용한 개발 데이터다. | 새로운 독립 real population 검증이나 graph-level CI로 부르지 않는다. 합성 final은 fresh96 graphs지만 설계에 사용한 뒤 새 후보의 개발 근거가 된다. |
| 검증/출처와 통계 집계를 독립적으로 확인해야 한다. | 두 별도 CPU 전수 재검산에서 8,181 records와 4 CI를 재현했다. Method reviewer는 모델 구현에도 참여했음을 자신의 문서에 명시했다. 세 관점을 완전히 독립된 구현 재현 세 건이라고 부르지 않는다. |
| 같은 final에서 구조나 예산을 바꾸어 계속 구제하면 안 된다. | 현 후보는 종료. 다음 trainability→transfer 질문은 untested 기록만 추가했다. 새 학습이나 GPU 작업을 시작하지 않았다. |

원결과의 Markdown 표 한 곳에 separator 칸 수 오류가 있지만 수치는 JSON/NPZ와 일치한다. 원보고서와 scientific lock은 보존하고, 읽기 쉬운 표는 별도 `DECISION_KO.md`에 작성했다.

최종 판정: **감사 PASS / 네 사전 gate FAIL / 현재 후보 확대 중단 / 계산·부분 점 차이·원결과 보존**.
