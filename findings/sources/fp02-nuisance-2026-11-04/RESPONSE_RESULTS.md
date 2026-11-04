# 세 결과 검토에 대한 응답

세 검토자는 정상 수치 구현·저장 결과 감사와 scientific gate 실패를 구분했다. Root는 현재 FP02를 Stage1 학습으로 확대하지 않는 데 동의한다. 원래 예측·source lock·은행·비교군은 수정하지 않는다.

1. **확인한 것:** 동일 query/context의 새30개 및 기존70개 예측, 260개 artifact hash가 일치한다. DC와 SBM의 사전7개 개발 조건은 전부 실패했다. Physics none의 실제 query class2는1063/1518=70.02635%인데 DC hard prediction은52/1518=3.42556%다. 첫 Gamma 갱신 이전부터 잘못된 분할이 시작되므로 Gamma 갱신 하나만 고쳐 해결된다고 말하지 않는다.
2. **확인하지 못한 것:** Semantic class=structural block 가정, 초기화, positive/negative contrast, finite loopy BP/MF 및 Poisson working model 가운데 유일 원인은 분리하지 못했다. 구현 검사는 큰 graph의 정확 Bayes를 보장하지 않는다. Degree correction 일반이나 learned Factor의 불가능성으로 확대하지 않는다.
3. **사용자 homophily 제안:** Adaptive 관계 추론이라는 목적은 채택하되 h 하나를 full/edge gate로 쓰지 않는다. Homo에서는 edge/nonedge LLR 부호가 양/음, hetero에서는 음/양이다. 삭제의 효용은 이 부호만으로 결정되지 않는다. Physics의 사후 전체-label h=.95817과 nonedge 삭제 개선은 공존한다. 배포 시 이 전체-label 값은 사용할 수 없다. Context만의 edge 수3/0/1/2/2는 직접 h 추정의 근거가 매우 적다는 것을 보여 준다.
4. **다음 제안의 불일치:** Method/experiment reviewer는 graph-conditioned semantic field와 synthetic prior 변경을 권했다. Inference reviewer는 held-context conditional predictive calibration으로 두 contrast의 불확실성을 평균하는 더 작은 classical 수정을 권했다. 둘 다 미실행이다. 전자는 p(A|semantic class)의 가정을 바꾸고, 후자는 초기 contrast와 자기 예측의 nuisance feedback을 바꾼다. 합치면 원인이 다시 섞이므로 한 번에 구현하지 않는다. 사용자 요청의 context adaptation을 구체화할 때도 기존 Plain router·CLP·Gaussian/probit·Grid6와 겹치는 기능을 새 발견으로 부르지 않는다.

이 기록은 다음 방법을 이미 구현·학습했다는 뜻이 아니다. 다음 cycle은 변경할 조건부 연산, context 선택 과정·유효 support·불확실성, 강한 fixed/LP/PCFI-PFN 대조, synthetic-only 동일 예산과 독립 final 조건을 먼저 고정해야 한다. 현재 FP02의 실패한 10개 개발 사례에서 beta/eta/온도/반복 횟수를 골라 구제하지 않는다.
