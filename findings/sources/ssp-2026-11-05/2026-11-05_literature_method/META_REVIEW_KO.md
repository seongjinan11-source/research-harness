# SSPd3-probe를 발전시키기 위한 문헌 검토와 다음 판단

2026-11-05. 기존 문헌 엑셀을 실제로 다시 읽고, 세 독립 reviewer가 직접 선행·추정 원리·관측 설계를 추가 검색했다. 이번 산출물은 문헌 검토와 메소드 제안이다. 새 방법의 성능이나 최초성을 확인한 실험은 아니다.

## 결론

**더 발전시킬 근거는 있다. 가장 유망한 목표는 size 함수족이 틀려도 composition을 보호하면서, count 간 유용한 일반화를 유지하는 것이다.** 현재 SSPd3와 소수 count probe는 강한 기준선으로 남긴다. 알려진 cut inference나 active design을 붙이는 사실 자체보다, subset count 구조의 식별 문제와 실제 선택 회복을 연결하는 데 기여를 둔다.

우선 유연한 count curve를 완전히 joint fit하는 단순 대조를 세운다. 그것으로 해결되는 조건에서는 그 해법이 충분하다. 그 모델에서도 오지정 feedback이 남고, composition 보호가 실제로 필요하다는 근거가 있을 때 보호 추정과 query 설계를 강화한다. ‘더 복잡해야 novelty가 높다’는 기준은 사용하지 않는다.

## 기존 결과가 요구하는 해법

최신 [count-support 판단](../2026-11-05_count_support/DECISION_KO.md)에서 table+probe는 SSPd3+spread에 대한 사전 주 비교를 통과했지만, 같은 probe 규칙의 SSPd3에는 열세였다. Narrow sigmoid mean incumbent regret는 SSPd3 probe0.012200, table probe0.026517이었다. 반면 wide step에서는 각각0.027990,0.003866으로 table이 유리했다. Table의 composition 보호와 count 일반화 손실을 함께 설명해야 한다.

초기 narrow sigmoid48 context의 conditional interval false certainty45/48도 보존한다. 그러나 online 관측으로 SSPd3가 상당히 회복했으므로 초기 오류를 탐색 전체의 지속 실패로 확대하지 않는다. 독립 목표는 composition MSE 하나를 줄이는 것이 아니라, 실제 관측 비용 안에서 유용한 subset을 더 빨리 얻는 것이다.

## 문헌이 이미 해결한 원리와 남는 구체적 질문

| 문헌 계열 | 이미 알려진 것 | SSP에서 별도로 검증할 것 |
|---|---|---|
| [Buathong et al.2020](https://proceedings.mlr.press/v108/buathong20a.html) | Variable-cardinality sets를 입력으로 하는 kernel/BO | 입력 표현 가능성과 실제 size 학습·same-count leakage 교정은 다르다 |
| [Pelamatti et al.2020](https://arxiv.org/abs/2003.03300) | Variable-size subproblems와 예산 배분 | 고정 ground set의 count와 공유 composition 정보 사이의 오염·일반화 |
| [Hahn et al.](https://arxiv.org/abs/1602.02176), [Robinson1988](https://doi.org/10.2307/1912705) | Shrinkage confounding과 partially linear nuisance 제거 | K=1'X의 common-direction singularity, finite adaptive archive의 정보와 의사결정 |
| [Plumlee & Joseph2018](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n23.pdf) | Trend/GP residual 식별과 orthogonality | Prior projection만으로 해결되지 않는 size 오지정·observed support·선택 문제 |
| [Carmona & Nicholls2020](https://proceedings.mlr.press/v108/carmona20a.html), [Tan et al.2509.18708](https://arxiv.org/abs/2509.18708) | Cut/semi-modular inference, semiparametric cut 및 GP confounding | 보호된 composition posterior와 count prediction의 상관 전파, 실제 count decision에 대한 이득 |
| [Bogunovic & Krause2021](https://papers.nips.cc/paper_files/paper/2021/hash/177db6acfe388526a4c7bff88e1feb15-Abstract.html) | Misspecified GP bandit의 confidence·탐색 문제 | 오지정이 count 방향에 집중된 구조와 composition ranking으로 전이되는 경로 |
| [Fiez et al.2019](https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html), [Wagenmaker et al.2021](https://proceedings.mlr.press/v130/wagenmaker21a.html), [Russo & Van Roy2014](https://papers.nips.cc/paper_files/paper/2014/hash/90720a2fcc41f9332e6a1558da327089-Abstract.html) | Decision difference를 위한 설계와 정보·regret 절충 | Count level을 알기 위한 관측과 composition contrast를 알기 위한 관측의 배분 |
| [Krishnamurthy et al.2018](https://proceedings.mlr.press/v80/krishnamurthy18a.html), [Kirschner & Krause2021](https://proceedings.mlr.press/v139/kirschner21a.html) | Action-independent nuisance의 orthogonalization/dueling | g(K)는 full subset action에 의존한다. Same-K cancellation만으로 count 선택을 해결할 수 없다 |

이것은 각 논문의 확인한 부분에 대한 비교다. 모든 원문 증명이나 전체 관련 분야의 조사를 마쳤다는 뜻은 아니다. 정확한 확인 범위와 미확인 항목은 세 reviewer 보고서에 있다. 검색에서 같은 문장을 못 찾았다는 사실은 novelty 증명이 아니다.

## 엑셀을 실제로 어디에 썼는가

Papers A1:U59의58편, Claims A1:H415의414행을 read-only import했다. [출처/hash](WORKBOOK_SOURCE.json)와 [원본 행별 활용](WORKBOOK_TRANSFER_KO.md)을 남겼다. 이 엑셀은 PFN/graph/TS 중심이므로 직접 BO 선행은 추가 검색으로 보완했다.

- SpatialContextMemory(Papers46): 같은 정보 아래 최적 기준과 fitted 오류를 분리하는 평가 원리. SSP에서는 archive가 composition을 식별할 수 없는 경우와 잘못된 size model이 오염시킨 경우를 구분한다.
- PFNSeparability(Papers39): coupling은 유용한 정보도 줄 수 있고 오지정 손해도 줄 수 있다는 원리. 해당 원문 §3은 adaptive BO가 선택한 입력을 task-generated feature evidence로 취급하는 문제도 다룬다. 이는 SSP의 size leakage와 같은 정리가 아니다.
- GFM FairEval(Papers15): 강한 단순 baseline과 실제 계산 비용 통제. SSPd3+probe를 빼고 새로운 조합의 필요성을 주장하지 않는다.
- MaM(Papers27)/MCSSL(Papers29): marginal/정보 일관성은 참고할 선행이지만, 현재 Gaussian SSP에 추가 신경 모듈을 도입할 직접 필요성을 찾지 못했다.

## 1순위: flexible joint 대 protected composition 비교

현재 모델을 f(x)=x'p+g(K)로 보면, g에 cubic 이상의 유연성을 주는 것은 타당한 다음 대조다. 일반적인 count GP나 spline을 써도 된다. 새 kernel 발명은 필요하지 않다.

비교할 원리는 두 가지다.

1. **Full joint:** 모든 관측을 이용해 p와 유연한 g를 함께 추정한다. Correctly specified count 구조에서 between-count 정보를 가장 잘 활용할 수 있는 강한 단순 대조다.
2. **Protected composition:** 같은 K 안의 X와 y를 함께 residualize하여 p를 추정하고, p의 불확실성을 유지한 채 g(K)를 예측한다. 잘못된 count 모델의 역방향 feedback을 막는 cut 원리를 사용한다. 이는 기존 modular inference의 적용이며 그 자체가 새로운 원리는 아니다.

보호 분포의 최소 표현은 다음과 같다.

    q(p,g | data) = q(p | within-count contrasts)
                   × p(g | p, count-level summaries).

두 번째 항은 p별로 정규화해야 한다. 단순 residual을 count GP에 넣고 두 분산을 독립적으로 더하면 p–g 상관을 잃는다. [정확한 Gaussian 식과 식별 조건](GAUSSIAN_CUT_CANDIDATE.md)을 정리했고 독립 이론 reviewer가 확인했다. 이는 수식 검토이며 코드 검증이나 실제 성능 검증이 아니다.

**반증 조건:** 같은 count covariance와 query 조건에서 full joint가 composition·incumbent 모두 충분히 회복하면 cut의 추가 필요성은 약하다. 반대로 보호 추정이 composition만 좋아지고 count 선택을 계속 악화시키면 문제 해결이 완료된 것이 아니다. 보호가 버린 between-count 정보는 downstream GP만으로 p에 되살아나지 않는다.

## 2순위: count 관측의 두 목적을 구별하는 query

현재 farthest-count probe는 count 간 거리를 본다. 그러나 새 count의 첫 관측은 그 count의 level을 알려 주고, 같은 count의 추가 관측은 composition contrast를 식별할 수 있다. 두 정보는 같지 않다.

고정 archive에서 u=B'x를 zero-sum composition coordinates라 하고 M을 within-count scatter라 하자. m_K개 관측한 count에 새 x를 추가할 때

    Delta M = m_K/(m_K+1) (u-mean_u_K)(u-mean_u_K)'.

미관측 count에서는 m_K=0이므로 보호된 composition 정보의 즉시 증분이0이다. 이때 level 관측의 가치는 여전히 있다. 같은 K에서 어떤 조합을 더 평가할지와 어느 K를 처음 평가할지를 함께 설계할 이유가 된다. 이 rank-one 식 자체는 표준 통계 항등식이다.

후보 개선은 현재 선택을 바꿀 수 있는 count level 불확실성, composition gap 오염, query 기회비용을 함께 보는 것이다. Generic IDS/RAGE나 augmented [B'x,e_K] linear bandit과의 차이를 명시해야 한다. 잘못된 cubic posterior의 분산만 그대로 사용하면 기존 과신을 공유할 수 있다.

**반증 조건:** 같은 query 수에서 기존 farthest probe나 random-count probe보다 나아지지 않거나, 더 많은 forced probes를 쓰는 것이 이득의 전부면 추가 정책의 필요성은 약하다. Same-K pair 두 번 평가는 두 query로 과금한다.

## 기여로 발전시킬 수 있는 이론과 한계

- Arbitrary count shift에 대한 composition 추정의 불변성은 고정 archive·적절한 stage-1 scale 처리 아래 성립하는 검증 가능한 성질이다. 일반 partialling-out과 구분되는 새로운 원리라고 주장하지 않는다.
- Count별 관측과 within-count rank가 어떤 composition 방향을 식별하고 어떤 방향을 못 식별하는지 밝힌다. 공통 방향은 K에 흡수되므로 zero-sum gauge가 필요하다.
- Count-shape residual이 직접 across-K gap과 학습된 composition에 미치는 두 경로를 분리한다. 기존 matrix retention/leakage appendix를 새 발명으로 다시 쓰지 않는다.
- Arbitrary 미관측 g(K)의 honest confidence는 관측만으로 얻을 수 없다. 정당화한 smoothness/variation/shape 가정 또는 추가 관측이 필요하다.
- Adaptive archive에서 fixed-design noise 독립성이나 ordinary posterior update를 자동 적용하지 않는다. Algebraic invariance, sequential update 규칙, frequentist calibration을 각각 구분한다.
- f(x)=x'p+g(K)가 깨지는 size-dependent composition interactions를 negative control로 둔다. 현재 제안의 보호 성질이 모든 비선형 subset objective를 해결한다는 주장은 하지 않는다.

## 다음 실행으로 넘길 최소 설계

아래는 문헌 검토의 권고이며 확정된 experiment registration이 아니다. 구현에 들어가기 전에 별도 cycle에서 equations·규칙·budget·gates를 명세하고 독립 리뷰한다.

1. 고정 archive에서 SSPd3, joint flexible count, 동일 count covariance의 cut, count-table을 비교한다. 같은 X·y에서 arbitrary count shift·rank 결손·correctly specified control·곡선 밖 shape를 확인한다. 전체 Gaussian conditioning과 cross covariance를 독립 계산으로 검증한다.
2. 새 development targets에서 같은 query schedule을 먼저 사용해 모델 효과를 분리한다. Composition MSE/ranking과 count prediction, 실제 noisy-observed incumbent를 함께 측정한다. 최종 확인용 bank는 별도로 남긴다.
3. 보호 필요성이 확인된 뒤 query만 바꾼다. SSPd3 probe, 같은 slot 수 random-count, variance probe, generic count-augmented design을 대조한다. 관측 count에서 완전히 같은 labels를 갖고 미관측 count에서만 다른 두 size 함수를 포함해 공짜 외삽 판별이 없음을 점검한다.
4. Fresh confirmation에서 smooth-size 기존 장점을 유지하고 step-like 오지정의 실제 선택 손실을 줄이는지 확인한다. 초기 static failure, composition accuracy, true-best-observed를 실제 incumbent의 대체 primary로 쓰지 않는다.

## 논문 주장에 대한 현재 판단

현재 SSPd3+주기적 probe만으로 강한 method novelty가 확보됐다고 말하지 않는다. 새 kernel이 없어서 기여가 불가능하다고 보지도 않는다. 가장 설득력 있는 전체 연결은 **count 구조가 만드는 추정·식별 문제를 규명하고, 필요한 정보 공유와 보호를 구별해, 제한된 관측으로 실제 선택을 회복하는 것**이다.

그 연결에서 어떤 부분은 기존 SSP/appendix와 현재 통제 실험이 뒷받침하고, 어떤 부분은 아직 제안이다. 후자의 성공을 전제하지 않는다. Flexible joint size만으로 충분한 결과가 나오면 그것을 정직한 단순 해법으로 채택하고 문제 진단·조건·증거의 기여를 강화한다.

## 독립 검토

- [REVIEW_NOVELTY.md](REVIEW_NOVELTY.md): 직접 subset/variable-size BO와 confounding/partialling-out 선행.
- [REVIEW_THEORY.md](REVIEW_THEORY.md): cut/semi-modular·semiparametric inference, 수식·식별·adaptive 한계.
- [REVIEW_DESIGN.md](REVIEW_DESIGN.md): misspecified bandits·dueling·정보 설계와 최소 반증 실험.

원본 workbook·원고·메소드와 locked banks는 변경하지 않았다. 이번 검토를 기존 확인 결과의 재튜닝이나 새 확인 성공으로 사용하지 않는다.
