# FP01: 투영 입력 학습과 centroid Factor 교체 결과

12개 모델의 학습·평가와 저장 확률 재채점을 완료했다. 데이터·입력·정답·체크포인트·고정 graph state·공통 role weights 감사는 PASS다. **C의 전체 사전 성공 기준은 통과하지 못했지만, C_global 대비 적응형 encoder의 이득은 확인됐다.** 이 둘을 함께 보존한다.

## 무엇을 비교했나

- A: 기존 preferred-class Factor를 원래 합성 피처로 학습.
- B: 같은 Factor를 투영된 합성 피처로 학습.
- C_global: class별 centroid 잠재변수와 3개의 전역 학습 계수.
- C: 같은 centroid 연산에 관측 통계 기반 공유 encoder를 사용.

3개 seed, arm마다 512개 task 노출/256 updates, 새 학습 총 3,072 updates/6,144 task 노출이다. Graph 모듈과 graph-only role weights는 공통으로 고정했고 실제 그래프에서 가중치 업데이트는 0회다. 연속 Gaussian exact Bayes나 기존 AMP–BP의 재현이 아니라, 41개 centroid state와 유한 반복 loopy BP를 쓰는 조건부 predictor다.

## 합성 주평가: 부분 성공과 전체 미통과

새 48개 base graph, projected/full, 50/75/90/95% 결측, 피처가 없는 query, source 2개 이상을 사전에 주평가로 고정했다. 아래 accuracy는 동일한 base graph 안에서 결측률·고정 3 seed를 평균한 값이다.

| 방법 | Accuracy | NLL |
|---|---:|---:|
| A | 52.817% | 1.00329 |
| B | 52.800% | 1.00335 |
| C_global | 52.565% | 0.99801 |
| C | 53.752% | 0.97400 |

| 비교 | Accuracy 차이 | 사전 97.5% paired CI | 판정 |
|---|---:|---:|---|
| C − B | +0.952%p | [−0.585, +2.696] | CI 하한 및 +1%p 요건 미통과 |
| C − C_global | +1.187%p | [+0.213, +2.446] | 이 비교의 요건 통과 |

단순 투영 노출 B−A는 −0.017%p였다. 따라서 이번 budget에서 입력 노출만 바꿔 기존 Factor의 문제를 해결했다는 근거는 없다. 이를 효과가 정확히 0이라는 증명으로 읽지는 않는다.

C의 관측 피처−피처 제거 차이는 +0.579%p이고, 보조 95% CI는 [−0.855, +1.929]다. 평균이 양수라는 사실과 안정적인 추가 피처 효용을 입증했다는 주장은 다르다. 공통 none 예측 때문에 C−B와 두 방법의 피처 효용 차이는 동일한 비교다.

주평가 NLL은 C가 두 비교군보다 낮았지만, secondary synthetic edge panel에서는 NLL이 B보다 약 +0.465 nats, C_global보다 약 +0.209 nats 높았다. Full에서만 학습한 모델의 별도 edge 평가라는 범위를 함께 남기며, 이 반례를 숨기거나 graph마다 유리한 panel을 선택하지 않는다.

## 실제 그래프: 피처 손상을 줄인 효과와 추가 정보의 구분

다음은 95/99/99.5% 결측의 동일 가중 평균이다. 전부 기존에 본 development graph이며 독립 실데이터 holdout으로 부르지 않는다. Cora/CiteSeer는 전체 그래프, 나머지는 기존 2,000-node induced graph다.

| 그래프 | Full B | Full C | Edge B | Edge C |
|---|---:|---:|---:|---:|
| Cora | 58.919 | 59.735 | 57.884 | 60.159 |
| CiteSeer | 44.608 | 45.827 | 35.824 | 36.348 |
| PubMed | 61.727 | 61.798 | 65.839 | 66.034 |
| CoauthorCS | 61.691 | 61.750 | 72.573 | 72.673 |
| CoauthorPhysics | 46.500 | 46.387 | 86.258 | 86.240 |

95% 결측에서 C의 **관측 피처−피처 제거** accuracy 차이는 다음과 같다.

| 그래프 | Full | Edge |
|---|---:|---:|
| Cora | −0.024%p | +0.245%p |
| CiteSeer | +0.003%p | −0.139%p |
| PubMed | −0.440%p | −1.120%p |
| CoauthorCS | 0.000%p | +0.015%p |
| CoauthorPhysics | +0.205%p | +0.095%p |

Cora/CiteSeer의 B 대비 개선은 보존하지만, 이 표는 추가 피처 정보를 안정적으로 활용했다는 해석보다 기존 피처의 손상을 완화했다는 해석에 가깝다. Cora/CiteSeer full에서 C는 PCFI+공개 GraphPFN의 52.898/35.262%보다 높다. 그러나 PubMed full C는 restart LP의 73.690%에 크게 못 미치고, Physics full도 harmonic LP 85.411%에 크게 못 미친다. Physics edge C의 accuracy는 86.240%지만 balanced accuracy 58.111%는 restart LP의 63.911%보다 낮고, NLL 0.9307은 harmonic LP의 0.5955보다 높다. Accuracy만으로 우월하다고 하지 않는다. 모든 외부 비교와 minority metric, shuffle 대조는 원본 결과 표에 남겼다.

## 이번에 닫는 범위와 남는 것

현재 학습 budget·고정 graph core·공통 graph-only weights·41-state centroid/BP24 구성에서 C를 성공한 메인 missing-feature 방법으로 확대하지 않는다. 실제 실패 원인의 기여도를 분리하지 않았으므로, 이를 “Gaussian centroid가 불가능하다”, “피처 정보가 없다”, “PCFI를 원천적으로 이길 수 없다”로 확대하지 않는다.

남기는 성과는 C_global 대비 C의 제한된 합성 개선, 여러 실데이터에서 피처 손상 감소, 일반 투영이 기존 one-preferred/equal-other mean family를 보존하지 않는다는 수학적 사실, 그리고 전체 해시·예측이 연결된 재현 자산이다. 수학적 mismatch의 존재가 그것만 고치면 실제 피처 효용을 회복한다는 보장은 아니었다.

같은 최종 bank에서 temperature·grid 크기·추가 loss·학습량을 조정해 성공 판정을 구제하지 않는다. 다음 방법 변경은 별도 cycle/branch와 세 리뷰를 거친다. 재검토할 때는 **graph-only보다 유용한 feature evidence를 회수하는 구체적인 계산 변화**와 B·C_global·강한 통계 추론기·PCFI/FISF-PFN·LP를 포함하는 비교를 먼저 고정해야 한다. 현재 결과만으로 다음 head나 더 큰 모델의 효과를 예측하지 않는다.

원본 수치: [자동 재채점 보고서](results/REPORT_KO.md), [전체 요약](results/summary.json), [paired 분석](results/paired.json), [감사](results/audit.json), [큰 파일 경로·해시](results/artifact_manifest.json). 실행 중 Git metadata만 캐시한 변경과 1개 orphan 보존·재계산은 [실행 기록](EVALUATION_EXECUTION.md)과 [manifest](EXECUTION_MANIFEST.json)에 별도로 기록했다. 세 독립 결과 리뷰에 대한 답변은 RESPONSE_RESULTS.md에서 마무리한다.
