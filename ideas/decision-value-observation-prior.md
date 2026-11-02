# 관측의 분류 가치로 구성하는 학습 prior — 재독 메모

> 후속: 사용자가 J/V 두 prior의 실행을 승인하여30개 모델을 학습했다. 아래는 실행 전 재독 메모를 보존한 것이며, 최신 결과는 [pilot 보고서](../findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md)와 [구조화 기록](../findings/decision-value-prior.yaml)을 따른다.

2026-11-02. **미실행 연구 가설**이다. 새 모델, teacher, graph bank, optimizer update나 성능 결과는 없다. 56편의 하네스 항목을 선별해 검토했고, 아래 연결의 핵심 원문을 다시 확인했다. 초록·발췌만 확인된 추가 heterogeneous 논문을 전체 본문 검토나 해결 근거로 취급하지 않는다.

## 원문을 다시 읽고 남긴 판단

| 논문 | 가져올 관점 | 넘겨짚지 않을 것 |
|---|---|---|
| [SSGP](https://www.ijcai.org/Proceedings/07/Papers/171.pdf), [GGP](https://proceedings.neurips.cc/paper/2018/file/1fc214004c9481e4c8073e85323bfd4b-Paper.pdf) | Graph covariance와 label likelihood를 결합하는 명시적 추론은 강한 기준선이다. | Gaussian 분류·kernel 적응·GP를 PFN에 연결했다는 사실만으로 novelty가 생기지 않는다. |
| [CSBM inference](https://arxiv.org/html/2306.07948) | Gaussian covariate centroid와 node class의 공동 추론은 BP–AMP로 이미 다뤘다. | 기존 Factor의 연결 기능을 다시 새 기여로 제안하지 않는다. Sparse CSBM의 점근적 최적성은 conjecture다. |
| [Spatial Context or Memory](https://arxiv.org/html/2610.12723v2) | 같은 관측의 최적 오차로 정보 부족과 fitting gap을 구분한다. | 회귀의 Gaussian conditional variance를 분류 정확도의 정보 가치로 바꾸지 않는다. |
| [PFN separability](https://arxiv.org/html/2610.14989) | Feature law와 label mechanism의 결합은 prior 설계의 선택이다. | Tabular iid 정리를 graph-dependent rows에 그대로 적용하거나 우리 실패 원인으로 단정하지 않는다. |
| [PEMix](https://arxiv.org/html/2610.18113) | 같은 missing budget에서도 관측 경계와 배치가 전파를 제한할 수 있다. | Boundary size·복원 rank가 classification utility와 같다고 가정하지 않는다. |

추가 검색에서 [Gaussian Neural Process](https://arxiv.org/abs/2101.03606), [learned EP messages](https://proceedings.neurips.cc/paper/2013/hash/1714726c817af50457d810aae9d27a2e-Abstract.html), [Neural Sheaf Diffusion](https://papers.nips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html)도 확인했다. 각각 predictive correlation, learned message operator, matrix-valued transport의 가까운 선행 원리다. 이 요소를 새 이름으로 붙이는 것을 1순위로 삼지 않는다. 세 논문의 전체 서지·한계 검토를 완료해 canonical paper로 추가한 것은 아니다.

## 현재 제안의 대상

Missing rate는 관측값의 양을 제어하지만, 그 관측값이 label 결정을 바꿀 수 있는지는 제어하지 않는다. 현재 Gaussian 비교는 한 family에서 feature 사용 이득을 보존했다. 그러나 **이전 학습 bank의 대부분이 무정보 과제였다는 사실은 확인하지 않았다.** 이를 실패 원인으로 가정하지 않는다.

후보는 같은 graph/feature/label 생성 family 안에서 학습 관측 배치를 구성하는 원리다. 배포 때 관측값을 추가 획득하거나 mask를 고르는 정책은 아니다. 기존 PFN 구조와 query CE를 유지하여 prior의 효과부터 분리한다.

구조적 설계를 \(S=(G,C,Q,M)\), 관측 cell 집합을 \(O(M)\), 합성 task prior를 \(\Pi\)라 하자. \(S\)를 고정하고 latent mechanism·feature·label의 가능한 realization을 해당 prior에서 평균한다. Query별 0–1 Bayes risk를 평균한 값을 사용한다.

\[
R_0^\star(S)=\frac1{|Q|}\sum_{q\in Q}
\mathbb E_\Pi\left[1-\max_c p_\Pi(Y_q=c\mid S,Y_C)\mid S\right],
\]

\[
R_X^\star(S)=\frac1{|Q|}\sum_{q\in Q}
\mathbb E_\Pi\left[1-\max_c p_\Pi(Y_q=c\mid S,Y_C,X_O)\mid S\right].
\]

관측의 기대 분류 가치는

\[
\boxed{V_\Pi(S)=R_0^\star(S)-R_X^\star(S)\ge0}
\]

이다. 같은 prior의 nested information sets와 Bayes 결정에 관한 기본 성질이지 새 정리가 아니다. 특정 dataset에서 feature를 넣었을 때 항상 accuracy가 오른다는 보장도 아니다.

Log loss에서는 대응되는 차이가

\[
J_\Pi(S)=\frac1{|Q|}\sum_q I_\Pi(Y_q;X_O\mid Y_C,S)
\]

이다. \(J_\Pi\)는 양수인데 \(V_\Pi\)는 작을 수 있다. 이전 NLL/Brier와 accuracy의 반대 움직임을 고려하면 **분류 연구의 task 설계 기준을 entropy 감소 하나로 대체하지 않는다.**

## 학습 연산과 검증 단위

1. Graph·feature·label family와 source/context 예산을 먼저 고정한다.
2. 실제 realization을 보기 전에, graph와 node/cell 위치로 정의한 후보 \(S\)들의 기대 \(V_\Pi(S)\)를 계산하거나 검증된 근사로 추정한다.
3. 무정보·낮은 가치·중간 가치·높은 가치 설계의 혼합을 사전에 정한다. 높은 가치의 쉬운 과제만 남기지 않는다. 그 선택은 학습 mask prior를 바꾸는 것이므로 평가 분포와의 차이도 명시한다.
4. 같은 PFN과 CE를 사용해 학습한다. 배포에서는 주어진 불완전 입력을 그대로 사용하며 \(V\) 계산이나 추가 관측을 요구하지 않는다.

Mask 설계가 실제 hidden feature, query 정답, instance-specific latent role을 보고 선택되면 mask가 정답을 전달할 수 있다. 따라서 **관측값을 보기 전 prior 평균으로 설계를 평가하고, 같은 구조적 설계가 모든 독립 realization에 적용되도록 해야 한다.** Context 추출·class balancing의 selection law도 teacher와 generator에서 일치시킨다.

현재 Gaussian/probit는 원8-column·RMS·median-threshold generator의 exact posterior가 아니다. 그 출력을 곧바로 \(R^\star\)라고 쓰면 안 된다. 첫 구현에는 teacher와 joint law가 일치하는 작은 family, 수치 적분 감사와 별도 Monte Carlo 검증이 필요하다. Teacher가 알려진 모델 밖에서는 \(\widehat V\)라는 근사로 남긴다. 관측이 거의 없어 식별 불가능한 과제를 학습으로 해결한다고 주장하지 않는다.

## 이전에 닫은 범위와의 차이

| 기존 기록 | 실제로 바꿨던 것 | 이번 후보의 차이 |
|---|---|---|
| REG-F01 | 학생-reference regret로 cached finite tasks의 sampling priority를 변경 | 학생 오차와 무관한 기대 결정 가치로 새로운 관측 배치를 생성 |
| MCP-F01 | Feature-style과 label-rule의 coupling을 변경 | 두 law를 arm 간 유지하고 관측 설계 prior만 변경 |
| OBS-F01 | 추론 때 선택한 subgraph 뒤에 PCFI와 frozen PFN을 적용 | 배포 subgraph 정책을 학습하지 않음 |
| SRC-F01 | Source-overlap/covariance proxy를 attention score에 보정 | Covariance head나 output correction을 추가하지 않음 |
| ADTR-F01 | 두 PCFI completion의 frozen prediction을 q(g)로 평균 | Predictor selector나 completion ensemble을 추가하지 않음 |

이는 기존 실패의 해결 증거가 아니다. 하네스의 자동 check도 이 기록들을 검색 단서로 반환했으며 재개 승인이나 novelty 판정을 하지 않았다.

## 먼저 넘어야 할 대조와 현재 한계

같은 새 task family·초기화·PFN·학습 exposure에서 random-mask prior, 결측률 조절만 하는 prior, 거리/coverage 규칙, geometry를 가능한 한 맞춘 random 설계와 비교한다. Teacher·mask 탐색의 CPU/GPU 생성 비용을 포함한 cost-matched random training도 필요하다. Cold scoring을 공짜로 취급했던 후보를 반복하지 않는다.

목표는 같은 heldout query에서 observed-value 제거·셔플에 반응하며 accuracy를 개선하는가다. NLL/Brier·feature recovery·추론 비용은 분리한다. Classical Gaussian/CSBM inference, 새 prior로 학습한 PCFI/FISF-PFN과 direct Mask PFN을 강한 비교군으로 유지한다. 최종 mask는 학습 설계기로 골라 주지 않는 독립 분포에서도 검증해야 한다.

Value-of-information·정보 기반 acquisition 자체는 선행 원리다. [PFN 기반 acquisition 평가](https://arxiv.org/abs/2610.07406)도 존재한다. 여기서 검토할 차이는 시험 시 acquisition이 아닌 **사전학습 observation prior**를 구성한다는 점이지만, 이것만으로 독립적인 메소드 novelty가 확정되지는 않는다. 학습된 uncertainty를 그대로 task 가치로 쓰면 추정 오차와 정보 부족을 혼동할 위험도 있다.

이 후보는 관측의 쓸모를 통제하는 **학습 가설**이며 새 architecture가 아니다. 이미 구성할 수 있는 좋은 Gaussian predictor를 PFN이 이길 이유까지 제공하지도 않는다. 작은 명시적 추론기가 동일 품질·비용을 제공하거나 단순 coverage 설계로 효과가 사라진다면, 이를 메인 메소드로 확대할 근거는 약하다. 현재는 이 후보를 미실행으로 보존하고 새 loss/head나 GPU 학습을 시작하지 않았다.
