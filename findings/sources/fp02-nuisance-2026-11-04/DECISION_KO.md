# FP02 판정과 homophily 적응의 정확한 목표

**현재 Gamma–Poisson nuisance 갱신은 종료한다.** 실제5개 그래프의 rep0, none/95% 결측10개 개발 episode에 새 추론30개를 실행했다. 새 신경망 학습은0회이며, 관측 graph에서 통계 파라미터를 갱신하는 추론은 포함한다. 독립 수치 검사는PASS이고 저장100예측의 감사도PASS이나, DC·SBM의 성능 gate는 각각7/7개FAIL이다.

| None | 기존 full Factor | 기존 edge-only Factor | 새 DC update |
|---|---:|---:|---:|
| Physics accuracy % |48.551|88.801|20.290|
| CiteSeer accuracy % |52.898|41.728|35.090|
| 다섯 graph 평균 accuracy % |55.691|63.750|43.805|

기존 nonedge 손해를 차수·클래스 비율·affinity 갱신으로 해결할 수 있다는 이번 가설은 지지되지 않았다. 같은 graph score를 유지한 채 새 loss/head/학습량을 추가하지 않는다. 첫 갱신 전부터 잘못된 class mass가 생겼고, 추론된 population은 참 population의 측정값이 아니었다. 원인을 community 가정 하나로 확정하지 않는다.

## 사용자 제안: homophily를 context에서 읽고 적응할 수 있는가?

그 목적은 타당하다. 그러나 homophily이면 nonedge를 유지하고 heterophily이면 삭제한다는 고정 규칙은 틀린다. 단순 Bernoulli SBM에서

\[
J_E=\log(p_{in}/p_{out}),\quad J_0=\log[(1-p_{in})/(1-p_{out})].
\]

Homophily면 두 부호는 양/음이고 heterophily면 음/양이다. 어느 쪽에서도 올바르게 관측한 nonedge는 정보가 될 수 있다. 삭제가 도움이 되는지는 클래스 opportunity·차수·관측 방식과 likelihood/근사 적합성에도 달렸다. Physics 개발 graph는 사후 h=.9581710258인 강한 homophily인데도 기존 Factor는 nonedge 제거로 크게 개선됐다.

실제로 허용된 context끼리의 edge 수는 Cora3개,CiteSeer0개,PubMed1개,CS2개,Physics2개다. PubMed의1개는 다른 class이므로 raw 추정은 h=0이 된다. 전체 정답으로 계산한 h를 모델에 주거나, 두 edge의 h=1을 확정값으로 쓰면 적응형 PFN의 근거가 되지 못한다. 경로 통계도 서로 독립인 label 표본이 아니므로 경로 수를 그대로 confidence로 쓰지 않는다.

따라서 다음에 검증할 적응의 대상은 **context가 지지하는 관계 compatibility·증거 강도와 그 불확실성**이다. Homophily 점추정 하나로 edge/nonedge를 동시에 제어하는 selector는 만들지 않는다. 새 conditional field와 held-context contrast calibration은 각각 미실행 후보로 보존한다. 다음 방법의 성공으로 세거나 두 후보를 한꺼번에 바꾸지 않는다.

세 검토문,응답,실행 source/lock,검사,확률 hash와 비용은 별도 로컬 branch에 남긴다. 현재 real bank는 재사용 development이며 독립 일반화나 새로운 PFN 학습의 성과가 아니다.
