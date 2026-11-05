# Flexible count: 보호 추정의 추가 필요성은 확인되지 않음

2026-11-05. 사용자의 진행 지시에 따라 기존 문헌 검토에서 제안한 full-joint count GP와 composition 보호 추정을 구현하고, 새 독립24 target에서 비교했다. 세 관점의 계획·코드 리뷰, 개발 전용 수치 검사, source/analysis freeze, 원시 결과 전량 감사와 독립 결과 리뷰를 수행했다. 원래 SSP/원고/기존 확인 bank는 변경하지 않았다.

## 메타 판단

**Cut 보호 추정은 승격하지 않는다. 이번 조건에서는 보통의 유연한 count 모델을 joint fitting하는 것만으로 size-shape에 따른 composition 오염을 크게 줄였다.** 그보다 강한 보호가 필요하다는 사전 가설은 통과하지 못했다. 새 모듈의 복잡성을 기여로 대신하지 않고 이 음성 결과를 보존한다.

다만 joint GP를 새로운 기본 방법으로 즉시 확정하지도 않는다. 좁은 count support 밖의 cubic·비단조 곡선에서는 여전히 큰 추천 손실이 있었다. 이번 결과는 고정 archive의 posterior 추천에 대한 것이며, 추가 관측으로 회복하는 SSPd3+probe보다 온라인으로 낫다는 증거가 아니다.

## 실행과 대조

- 새 seed2026110551, 독립 target24개. Wide/narrow support × uniform/aligned archive의4조건, 각44 unique subset 관측.
- Zero control과 linear/cubic/sigmoid/step/nonmonotone-wave의6 datasets/조건. Size amplitude4, noiseSD.05, n16, K2..12. Shape와 regime은 target 안에서 의존하므로 독립 표본으로 세지 않았다.
-5arms: 기존 SSPd3, observed-count table, Gaussian cut+count GP, cut과 정확히 같은 hyperparameters의 full joint, 모든 scales를 full evidence로 fit하는 full joint EB.
- Cut은 within-count X와y를 함께 residualize하여 composition posterior를 추정한다. Count GP는 그 posterior uncertainty를 적분해 hyperparameters를 선택하고, 정규화된 conditional과 composition–count 상관을 유지한다. 이는 기존 cut inference와 Gaussian conditioning의 적용이다.
- 새 GP는 ordinary Matern3/2 count covariance와 SD10의 proper constant prior를 사용한다. 새 kernel 발명이나 unrestricted count-shape 안전 보장은 주장하지 않는다.
-576공유 datasets,2,880 arm records,2,304 selected optimizer problems/4,608 starts. Main4CPU16.707초. Online query·외부 objective 호출은0.

## 사전 primary: wide step의 same-count composition regret

각 target에서 wide_uniform/aligned 두 값을 먼저 평균했다. 값이 낮을수록 좋다.

| Arm | Mean same-count regret | Mean posterior recommendation regret |
|---|---:|---:|
| SSPd3 | 0.24408065 | 0.15136400 |
| Table | 0.00897169 | 0.00234383 (관측 count 안에서만 추천) |
| Cut+count GP | 0.00897169 | 0.00235376 |
| Joint matched | 0.00818761 | 0.00157777 |
| Joint EB | 0.00818761 | 0.00157777 |

Primary delta는 **joint EB − cut**이다. 양수가 cut의 개선이다.

- 평균 −0.0007840726.
- Target paired percentile bootstrap95% CI [−0.0019743353,+0.0000088348].
-24 target 중 cut승2,동률16,패6. Win/tie는 등록한1e−12 비교 기준을 사용한다.
- 사전 요구 mean≥0.01, CI lower>0,18/24승을 모두 통과하지 못했다.

같은 prior/scales를 쓰는 matched joint에서도 primary delta는 동일했다. 따라서 이 bank에서 보호를 추가해야 한다는 근거는 없다. SSPd3 대비 GP joint의 큰 회복은 보조 비교이며 이것으로 실패한 cut primary를 교체하지 않는다.

## 함께 요구한 narrow sigmoid count 의사결정 비열등성

Cut와 joint EB의 full-grid posterior recommendation regret는 모두0.00808336이었다. Cut−joint의24 target paired 차이와 CI는 모두0으로, 사전 평균 손해 허용치0.02는 통과했다. 독립 reviewer가 두 narrow regime의48 context에서 실제 추천 subset/count가 모두 동일함을 확인했다. Posterior 분포 자체가 동일하다는 뜻은 아니다.

그러나 selected optimizer failure gate는 통과하지 못했다. 두 gate context에서 cut stage1/2와 joint EB를 별도로 세는288 selected results 중1개가 실패 상태였다: target8/narrow_uniform/sigmoid joint EB, finite 결과의 L-BFGS-B line-search termination, ell 상한8. 결과를 삭제하거나 재시작·재튜닝하지 않았다. 전체2,304 selected problems에서는 joint EB3개가 같은 상태였고 모두 보존했다. **Wide-step 주 비교에는 selected failure가 없으므로 primary 성능 실패를 이1개 상태 탓으로 돌릴 수 없다.**

최종 사전 판정: primary mean/CI/wins 실패, NI 통과, optimizer 실패, audit 통과. Cut 승격 조건은 불충족이다.

## 무엇을 배웠고 무엇이 해결되지 않았는가

### 유연한 joint size 처리가 충분한 조건이 있었다

Wide step의 joint GP composition regret0.00819는 SSPd3의0.24408보다 훨씬 작았다. Wide sigmoid도0.07065→0.00523이었다. 적어도 이 생성모형과 관측 조건에서는 flexible size nuisance를 제대로 joint fit하는 단순한 해법이 효과적이다. Composition 보호가 원리적으로 항상 필요하다는 주장은 하지 않는다.

이것은 root 목표에 부합하는 해결 근거다. ‘평범한 GP이므로 의미 없다’고 버리지 않고, 어떤 size shape에서 기존 cubic이 부족했으며 어떤 교정이 충분했는지에 기여를 둔다. 반대로 GP를 추가했다는 사실 자체가 신규성 증명은 아니다.

### 강한 보호는 실제로 유용한 정보를 버렸다

Zero-size narrow 조건에서 posterior recommendation regret는 joint EB0.001973, cut0.021708이었다. Same-count regret도0.001688대0.019504였다. 같은 matched prior의 full joint도 이 조건에서는 강했다. 보호가 count 오지정의 경로를 끊으면서 유용한 between-count 정보를 버리는 대가다. 좋은 composition MSE나 불변성 성질만으로 실제 선택 효용을 대신할 수 없다.

### Curve flexibility는 지원 밖 정보를 만들지 않는다

| Narrow-support shape | SSPd3 total regret | Cut total regret | Joint EB total regret | Joint EB unsupported-level RMSE |
|---|---:|---:|---:|---:|
| Sigmoid | 0.825406 | 0.008083 | 0.008083 | 0.596652 |
| Cubic | 0.144687 | 0.424414 | 0.422019 | 4.681971 |
| Step | 0.068446 | 0.005719 | 0.005719 | 3.157592 |
| Nonmonotone wave | 0.282033 | 0.182167 | 0.182167 | 4.616494 |

**위 표는 고정 archive의 추천 지표다.** Stationary count GP는 관측되지 않은 영역의 cubic 증가나 비단조 극값을 자료 없이 알아내지 못한다. 특히 cubic에서는 원래 parametric shape 가정이 joint GP보다 유리했다. 특정 smooth stress의 이득으로 모든 비선형 count를 해결했다고 주장하지 않는다. 관측 지원을 확보하거나 정당한 shape 가정을 사용하는 문제가 남는다.

Count-level RMSE와 실제 선택 regret가 같은 척도도 아니다. Step의 지원 밖 RMSE가 큰데 추천 regret가 작은 것은 이 bank에서 잘못 예측한 모든 level이 최종 선택을 바꾼 것은 아니라는 뜻이다. Risk/uncertainty 진단을 실제 decision에 연결해야 한다.

## 정확성·감사와 발견한 한계

-8개 개발 검사 통과: analytic gradients, full parameter-space posterior, cut의 음의 cross covariance, 같은 count gap 소거, count-shift 시 stage1 scales까지 재적합한 불변성, singleton/rank 결손, scatter update, 기존 SSPd3와 zero-sum 재표현의 예측 동일성.
- 개발9001의120records를 전량 감사하고 실제CSV 저장도 검사했다. 초기 code review가 발견한 CSV schema 결함은 main 전에 고쳤고, 이전 개발 산출물은 development/에 보존했다.
- Main 원시 labels·truth·design·noise·모든 추천/지표를 전량 재계산했다. Metrics 감사 최대 오차0, Gaussian mean/covariance의 독립 precision 재구성 최대 오차3.58e−9. 최소 covariance eigenvalue2.26e−6. Raw24파일/source hash 전부 유지.
- 독립 결과 reviewer들도2880행 metrics를 별도 재계산했다. 자체 auditor의 통과만을 독립 검토로 대체하지 않았다.
- 식별되지 않는 composition 방향에 tied coefficients가 있는 archive에서는 shape별 floating-point 오차가 실제 ranking tie를 바꿀 수 있었다.6cases의 ordering 변동에서 최대 coefficient 차이는2.39e−15였다. 독립 reviewer가 해당 item들의 design columns가 같음을 확인했다. 이는 유의미한 size feedback 근거가 아니다. 등록한 stable-sort/exact-tie metric을 사후에 tolerance 변경해 다듬지 않았다. 불변성은 numerical coefficient/posterior 수준으로 표현하고, 실제 ranking의 수치 near-tie 한계를 함께 남긴다.

## 다음 연구 판단

1. **Cut를 기본 방법으로 채택하지 않는다.** 기존 simple SSP와 flexible joint를 강한 대조로 유지한다. 이번 bank에 feedback-strength나 lengthscale을 sweep하여 실패를 성공으로 바꾸지 않는다.
2. **표현 문제와 support 문제를 분리한다.** Wide shape misspecification은 flexible joint로 상당히 회복했다. Narrow extrapolation은 그 변경으로 항상 해결되지 않았다. 원래 목표의 size/composition 연결을 이 두 조건으로 더 정확하게 서술할 수 있다.
3. 다음 online 후보를 평가한다면 기존 SSPd3+probe와 flexible joint+같은 probe를 먼저 비교해야 한다. 새로운 acquisition을 동시에 바꾸기 전에 모델 효과를 분리하고, 실제 noisy-observed incumbent를 평가한다. Static sigmoid 이득이 online SSPd3의 빠른 회복 이후에도 남는다고 전제하지 않는다.
4. Query 개선은 새 count level과 같은 count의 composition contrast가 필요한 조건을 구분하는 방향으로만 정당화한다. Generic bandit/design와 직접 선행을 대조하고, 새 계획·리뷰·bank 없이 현재 bank에서 확인 성공을 주장하지 않는다.

이 cycle은 실행·감사·리뷰를 완료했다. 후속 online 실험을 실행 중이라고 뜻하지 않는다. 이번 결과는 새 kernel이나 cut의 독창성을 보여주는 결과가 아니라, **어떤 해결이 충분했고 어떤 복잡화가 필요하지 않았으며 무엇이 여전히 부족한지**를 좁힌 근거다.

## 산출물

- PLAN.md, REVISION_LOG.md, PLAN_REVIEW_*.md, CODE_REVIEW_*.md.
- model.py, benchmarks.py, run.py, analyze.py, verify.py, test_model.py 및 NUMERICAL_CHECKS.json.
- MANIFEST.json, EXECUTION.json, runs/00..23.json, RAW_HASHES.json.
- SUMMARY.json, per_record.csv, AUDIT_COMPLETE.json.
- POSTHOC_DIAGNOSTICS.json과 inspect_results.py는 결과 확인 후의 optimizer/수치 tie 설명용이다. 새 fit·primary 변경은 없다.
- RESULT_REVIEW_NOVELTY.md, RESULT_REVIEW_THEORY.md, RESULT_REVIEW_EMPIRICAL.md.
