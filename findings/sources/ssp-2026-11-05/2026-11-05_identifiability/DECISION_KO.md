# 판단: SSP의 식별 조건을 이론 보강으로 확보

2026-11-05. 사용자 요청은 이론·신규성을 강화하기 위한 구체적 도움이다. 기존 목표, 원고, 성공/실패 및 문헌 리뷰를 바탕으로 두 후보를 검토했다. 이전 decision-radius 후보는 직접 affine-robustness 선행과 겹쳐 수치 구현 전에 중단했다. 이번 count/composition 식별 후보는 독립 plan/code/result 검토와 별도 확인용 수치 검증으로 진행했다.

## 확보한 정확한 내용

고정된 비적응적 archive에서 선형 composition과 arbitrary count-level effects를 고려한다. n개 item, N회 관측, m개의 서로 다른 관측 count라면, full nontrivial fixed-K slice의 모든 composition contrast가 likelihood에서 식별될 필요충분조건은

    rank(M)=n−1,
    M=(XB)'Q(XB), Q=I−Z(Z'Z)^−1Z'.

따라서 **N−m>=n−1**이 필요하다. 이 조건만으로 충분하지 않다. 예를 들어 item1/2가 항상 함께 등장하는30개의 서로 다른 query로도 둘의 효과를 구분하지 못할 수 있다. 제약된 feasible slice는 필요한 contrast만 식별되면 되므로 full-rank를 일괄 요구하지 않는다.

Null 방향이 실제 추천을 바꾸면 동일 관측 분포를 갖는 두 문제를 만들 수 있다. 어떤 추천 rule도 두 문제 모두에서 expected normalized simple regret1/2 미만을 보장할 수 없다. 정보가 양수인 방향 v와 fixed nonzero signal norm s에 대해서는 해당 두 환경만을 아는 최적 rule의 정확한 minimax risk가

    Phi(−s sqrt(v'Mv)/sigma)

다. 이는 전체 unknown-parameter class의 learner에 대한 upper bound가 아니다.

M은 같은 count 내부의 composition scatter다. 처음 보는 count의 첫 관측은 이 unrestricted-nuisance composition 정보에0을 더한다. 기존 count에서 새로운 composition을 관측하면 t/(t+1)*(u−ubar)(u−ubar)'를 더한다. Count 자체의 가치를 배우거나 proper GP prior 아래 infer하는 정보가0이라는 뜻은 아니다.

## 실행·감사

- 세 독립 plan 리뷰 후 수정, 세 코드 리뷰 및 source freeze 후 확인 실행. 개발 seed2026110570의2fixture와 확인 seed2026110571의16fixture 분리. 기존 bank/source/원고 변경 없음.
- 확인16fixture×2archives. n8,counts2..6. N10 deficient 설계 rank4또는5; N30 excited 설계는 모두 rank7. 예측과 선택 방향·평균·실제 feasible catalog를 noisy labels 전에 저장했다. 임의 replacement나 unresolved case는 없었다.
- 총64 environment-averaged risk tests, 각 환경20000개의 Gaussian observation vectors. 합2560000관측벡터/64000000Gaussian scalar draws. 한 CPU, execution1.8507초. 새로운 BO objective 실행은 아니다.
- 행렬/평균/손실 항등식 모두 사전 tolerance1e-9 통과. 확인 artifact의 최대 identity residual4.15e-13. 모든 risk gate 통과, 최대 Monte Carlo risk 오차0.00605, 사전 familywise Hoeffding tolerance0.021603. 이 gate는64평균test의 오류확률 상한1e-6을 사용하며,128개 environment 개별 rate는 보조다.
- 독립 root audit가 전체 Gaussian labels와 null coin, raw projections, 추천 및 true catalog regret를 재생했다. 최대 재구성 residual7.99e-15, 저장 risk 차이0. 모든13개 frozen file 및 raw artifact hashes가 일치했다. 같은 세 reviewer의 결과 검토에서도 추가 blocker가 없었다. Empirical reviewer도 X까지 seed에서 전량 재생해 일치를 확인했다.

| 사전 SNR d | 이론 risk Phi(-d) |16fixture 평균 실제 risk |
|---|---:|---:|
|0(null pair)|0.500000|0.499595|
|0.5|0.308538|0.308908|
|1|0.158655|0.158089|
|2|0.022750|0.022583|

SNR에 맞춰 sigma를 정했으므로 이 표는 **공식의 수치 검증**이다. 실제 BO의 난이도를 독립적으로 예측했다는 증거가 아니다. Gaussian binary testing을 맞춘 사실이 성능·priority·ICML 채택을 입증하지 않는다.

## 신규성과 논문 위치

Estimability/FWL, Schur complement, Gaussian testing, online scatter, nuisance-resistant experimental design은 기존 이론이다. 측정 feature=(B'x,e_K), 목표 feature=(B'z,0)로 쓰면 transductive linear identification에 들어간다. 새 일반 minimax 이론이라고 주장하지 않는다. primary source 대조는 PLAN_REVIEW_NOVELTY.md에 있다.

기존 SSP의 fixed-prior retention/leakage와 달리, 이번 결과는 **관측 구조 자체의 식별 한계**를 설명한다. Count를 유연하게 처리하는 것, count support를 확보하는 것, composition을 구분하는 관측을 확보하는 것은 서로 다른 역할이다. 이 구분을 문제→교정 원리→결정 회복의 연결에 보조 proposition으로 포함할 가치는 있다. 이것만으로 전체 novelty 공백이 해결되지는 않는다.

## 메소드의 실제 다음 후보

현재 유연한 joint count fit을 유지하고, 같은 probe count에서 어떤 composition을 관측할지 decision-relevant contrast information으로 고르는 후보를 PAPER_POSITION_KO.md에 식으로 명세했다. 표준 experimental-design baseline과의 비교가 필요하다. **이 정책은 이번 cycle에서 구현하거나 검증하지 않았다.** 기존 고정-frequency probe의 optimality나 adaptive regret 보장도 얻지 않았다.

우선순위는 같은 count schedule/예산에서 probe composition만 바꾸는 새 반증 실험이다. 실제 같은-count 추천 손실과 online 추천 손실이 줄어야 하며, 정보 행렬만 좋아진 것으로 성공 판정하지 않는다. 기존 cut 실패와 narrow-wave/강제 probe의 손해는 그대로 보존한다.

근거: THEOREMS.md, PLAN.md, MANIFEST.json, CONFIRMATION/PREDICTIONS.json, CONFIRMATION/SUMMARY.json, AUDIT.json, 각 PLAN/CODE/RESULT_REVIEW 문서. 이 디렉터리의 파일은 완료 기록이며 background 실행이나 예약을 뜻하지 않는다.
