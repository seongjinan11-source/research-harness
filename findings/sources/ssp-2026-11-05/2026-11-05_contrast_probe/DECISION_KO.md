# 결과: 단일 contrast probe의 추가 이득은 입증되지 않음

2026-11-05. 사용자 승인에 따라 joint fitting을 유지한 decision-directed within-count probe를 구현했다. 별도 개발/확인 데이터, 세 독립 plan/code/result 관점의 리뷰와 source freeze를 거쳤다. **Stage1의 다섯 primary gate가 모두 실패했다. 현재 후보를 확대하지 않으며 conditional Stage2는 실행하지 않는다.**

## 무엇을 실제로 바꿨나

Probe count 안에서 현재 top-K의 가장 애매한 selected/unselected item 교환 한 개를 찾고, 그 방향의 regularized within-count 정보분산을 가장 많이 줄이는 unseen subset을 관측했다. A=M+I, M은 같은 count 내부 scatter다. 기존 joint count GP의 EB fitting/start/bounds/selected finite rule은 그대로다.

Greedy, uniform random, A-opt, joint posterior query variance, joint posterior target-contrast reduction의 다섯 대조군을 사용했다. Shared initial labels만으로16count schedule을 먼저 저장했고 모든 arm이 같은 schedule/예산을 사용했다. 이후 관측값에 따라 fitted composition과 실제 nonprobe query도 달라질 수 있지만, 다른 선택 규칙을 적용한 시점은 사전 probe4회다. Adaptive count BO 실험은 아니다.

## 주 결과

새24target,6size shapes×4archive regimes×6arms=3456runs. Primary는 비다항3shape×4regime에서 checkpoints1..16의 실제 full-K composition recommendation regret 평균이며, target24개가 통계 단위다. 작을수록 좋다.

| 정책 | Mean composition regret | Mean noisy-incumbent latent utility regret |
|---|---:|---:|
|Greedy|0.003275414|0.012856750|
|Random|0.003356129|0.015909759|
|A-opt|0.002918175|0.012898546|
|Joint query variance|0.002976601|0.015869658|
|Joint contrast reduction|0.003394102|0.015471281|
|새 robust contrast|0.003203461|0.012919557|

Greedy−새 후보의 primary gain은 **+0.000071953**, 사전99% paired target-bootstrap CI **[−0.000127103,+0.000271976]**다. 최소효과0.0001에 못 미치며 CI도0을 포함한다. Random 대비+0.000152668, A-opt 대비−0.000285286, jointvar 대비−0.000226860, jointcontrast 대비+0.000190641이고, 다섯 비교 모두99%CI가0을 포함한다. 다중 대조를 위해 요구한 all-five primary 조건을 만족하지 않았다. CI는 각 metric family 내 Bonferroni95% 범위이며 모든 family 전체95%coverage를 주장하지 않는다.

Incumbent safety/control safety gate는 통과했다. 그러나 incumbent가 전혀 손해 없다는 뜻은 아니다. Greedy 대비 후보의 mean incumbent regret 증가량은 **+0.000062807**,99%CI **[+0.000006267,+0.000141468]**로 작은 손해가 있지만 사전 허용폭0.002 안이다.

A-opt의 평균 composition 손실이 가장 작았다는 것은 서술적 결과다. 이 bank에서 승자를 골랐다는 사실을 새로운 독립 확인 성공으로 승격하지 않는다. 원래 후보를 A-opt로 바꿔 Stage2 gate를 사후 통과시키지 않는다.

## 왜 이 규칙으로 부족했나

확인 전 REVISION_LOG에서 이미 지적한 구조적 한계가 실제로 나타났다. Narrow regime의288contexts 모두에서 네 probe가 각각 첫 신규 count였다. 후보의2304probe 결정 중1152회(50%)에서 robust-M 정보 증가량이0이어서 명세대로 greedy로 돌아갔다. 이288contexts에서 greedy/A-opt/contrast의 **전체16query 경로가 정확히 같다.** 버그나 누락이 아니다.

Wide에서는 후보가 실제로 행동을 바꿨지만, 전체 추천 손실 우위를 입증하지 못했다. 특히 wide-aligned nonpoly의 평균 composition regret는 greedy0.004756→contrast0.004412인 반면 A-opt0.003488이었다. Wide-uniform에서는 contrast0.001966으로 greedy0.001910보다 나빴다. 단일 가장 애매한 swap을 겨냥하는 것이 전체 K의 실제 선택 손실 감소와 잘 맞지 않을 수 있다는 **가설**을 남긴다. 이 원인을 인과적으로 분리한 결과는 아니다.

따라서 first-count fallback만 없애면 해결된다고 단정하지 않는다. Narrow를 사후 제외해 primary를 구제하지 않는다. Exact rank-one 분산 감소식의 정확성도 true regret 개선을 보장하지 않는다.

## 검증·비용·보존

- 개발2targets의288runs,4896checkpointfits와48schedulefits 및6unit checks 통과. 새 확인 seed2026110581/target0..23과 분리했다. Performance-based tuning 없음.
- 확인58752checkpointfits+576schedulefits=59328model fits,55296newqueries,207360arm-charged observations. Main execution95.2178초,4CPU workers/BLAS1. Common schedule fit3.1281CPU-seconds는 arm runtime 비율에서 제외했다. Candidate/greedy median context runtime ratio1.01394, 비용 gate 통과.
- Selected finite optimizer non-success190/58752, schedule fit non-success1/576. 각 arm non-success36/25/37/26/27/39개를 모두 보존했다. Nonfinite posterior나 dropped case 없음.
- 모든 관측은 요청한 catalog ID와 고정 potential noise를 재계산해 검증했다. Label error<=8.89e-16.10368개 독립 parameter-space posterior 재계산 최대오차3.19e-9. Exact action replay는 frozen policy 함수를 재사용하며, 독립 공식·posterior 검증과 구별한다. Reviewer의 직접 rank-one/Gaussian conditional-update score 검증도 일치했다.
-25source files,50development artifacts와600raw/schedule files의 hash를 보존했다. 기존 source/원고/RESEARCH_GOAL/역사적 bank를 변경하지 않았다. 결과 리뷰는 RESULT_REVIEW_THEORY/NOVELTY/EMPIRICAL.md에 있다.

## 판단

이 구현은 현재 joint size correction에 유의한 새 결정가치를 추가하지 못했다. 기존 flexible joint fitting의 성공이나 count/composition 식별 정리의 타당성을 반박하는 결과는 아니다. 수학적 정보 지표를 개선하는 것과 좋은 추천을 하는 것의 차이를 보여주는 음성 근거로 보존한다.

다음 가설은 **신규 count에서 같은-count 비교를 만들 paired observation budget과, 여러 composition 방향을 함께 다루는 단순 설계의 결합**이다. 이번 고정 challenger를 가중치/temperature로 재튜닝하지 않는다. 새 count coverage 감소의 비용과 실제 incumbent 손실까지 포함한 별도 계획·fresh bank가 필요하다. 미검증 제안은 NEXT_HYPOTHESIS.md에 분리했다. 이 cycle은 완료됐고 background job이나 예약은 없다.
