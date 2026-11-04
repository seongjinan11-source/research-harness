# FP01 result review — mathematics / inference / implementation

완성된 `artifact_manifest.json`을 확인한 뒤 audit, summary, paired, report, budget, preflight checks와 세 seed의 운영 finished packet을 읽었다. 새 forward·학습·job 실행·코드 수정·branch 변경은 하지 않았다. 다른 current-result reviewer의 파일은 읽지 않았다. 아래 수치와 검증 범위를 확인한 뒤 마지막 절에 판정한다.

## 1. 수치와 provenance

[audit.json](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/results/audit.json)은 PASS다. 새 fit 12개, optimizer update 3,072회, task presentation 6,144개, target update 0회다. Synthetic prediction 16,704개, real 1,800개, 재사용 baseline 375개를 별도로 재채점했다. 새 prediction은 18,504개이며 baseline까지 합친 index는 18,879개다. Ensemble probability를 만든 결과가 아니다.

내 읽기전용 대조에서도 manifest가 가리키는 summary/index/checkpoint/evaluation-index/bank-manifest 등 **46개 항목의 존재와 SHA256이 모두 일치**했다. 앞서 확인한 source-lock `files` 및 inherited graph checkpoints의 불일치도 0이었다. 검토한 manifest의 SHA256은 `d93baa344eb2f3fe23abd89c6f0c67491f8446dc65b0cd6455efdedbc2c4ede1`, source-lock SHA256은 `50dd637bf4c193838757dbbad730a37299a4a3b18f4592cef8eb59b33dfbb605`다. 이는 감사 절차의 재현 가능한 근거이며, 모든 확률을 exact posterior라고 보증하는 검사는 아니다.

최종 감사는 own-feature-missing query와 context/source disjointness, feature arm·관측 개입 사이의 동일 graph-only role weights, model/frozen-graph state digests, input/target identity를 확인한다. `analysis.py`의 해당 검증 경로도 읽었다. 특히 다음 경계가 실제 저장 확률에서 확인됐다.

| 경계 | 최대 probability 차이 |
|---|---:|
| 동일 graph/context의 none, feature arms 간 | 0 |
| Source 0/1과 graph-matched none | 3.5763×10⁻⁷ |

## 2. 사전 synthetic primary와 보존할 긍정적 결과

Primary는 full graph / projected view / missing 50·75·90·95% / source≥2 / 공통 feature-missing query다. N×K의 12 strata에서 각 4개 base graph를 유지하고, 세 training seeds와 rates를 graph 안에서 평균한 사전 비교다.

| Arm | Accuracy % | NLL |
|---|---:|---:|
| A | 52.817 | 1.00329 |
| B | 52.800 | 1.00335 |
| C_global | 52.565 | 0.99801 |
| C | 53.752 | 0.97400 |

| Co-primary contrast | Accuracy %p, 97.5% CI | Mean NLL difference |
|---|---|---:|
| C−B | +0.952 [−0.585,+2.696] | −0.02935 |
| C−C_global | +1.187 [+0.213,+2.446] | −0.02400 |

C−B는 positive lower bound와 ≥1%p mean이라는 두 accuracy 요건을 모두 통과하지 못했다. C−C_global은 두 요건과 NLL guard를 통과했다. C−none accuracy mean은 +0.579%p이나, 보조 95% CI는 [−0.855,+1.929]%p다. 따라서 **전체 gate=False**라는 분석 결과가 사전 규칙과 맞는다. NLL 개선으로 accuracy gate를 바꾸거나 +0.952를 반올림해 1%p 조건을 통과시킬 수 없다.

동시에 C−C_global의 양의 결과를 지우면 안 된다. 이 finite-grid family의 동일 학습 budget 안에서는 adaptive hyperparameter encoder가 전역 세 계수보다 유리한 근거가 남았다. 세 seed의 primary accuracy도 C가 C_global보다 각각 높았다. NLL 차이 역시 C에 유리하다. 다만 probability-score intervals는 accuracy의 두 co-primary 판정과 별개의 보조 결과이며, 여러 score를 독립적인 성공 gate로 합산하지 않는다.

B−A primary accuracy는 약 −0.017%p이고 NLL도 거의 같다. “Projection의 수학적 non-closure가 존재한다”는 사실과 “그 interface를 바꾸면 이 구성·예산에서 성능이 회복된다”는 실험 가설을 구분해야 한다. 후자는 이번 결과에서 지지되지 않았다. 이를 projection mismatch가 원래 없었다는 반증으로 확대하는 것도 잘못이다.

Secondary edge panel도 숨길 수 없다. C의 accuracy는 42.561%, B는 43.182%, C_global은 43.121%다. C−B NLL은 **+0.46519**, C−C_global은 **+0.20866** nats로 악화한다. C−none accuracy는 +2.050%p, 보조 95% CI [+0.311,+3.867]로 실제 feature 효용의 신호가 있으나, 이는 더 강한 비교군에 대한 성공이나 원래 full primary의 대체가 아니다. Full에서 학습한 동일 operator의 edge-only evaluation이라는 범위도 유지한다.

## 3. 실제 그래프에서 무엇이 개선됐는가

아래는 C의 **95% observed−none accuracy**다. 세 반복의 seed-평균 기술 통계이며 모집단 CI가 아니다.

| Graph | Full %p | Edge %p |
|---|---:|---:|
| Cora | −0.024 | +0.245 |
| CiteSeer | +0.003 | −0.139 |
| PubMed induced2000 | −0.440 | −1.120 |
| CoauthorCS induced2000 | 약 +0.000 | +0.015 |
| CoauthorPhysics induced2000 | +0.205 | +0.095 |

C는 full CiteSeer의 missing-rate 평균 accuracy에서 B의 44.608%를 45.827%로 높인다. 그러나 95% observed−none은 B −3.613%p, C +0.003%p다. 즉 이 개선은 상당 부분 **관측 피처가 있던 B의 손해를 줄인 결과**와 양립하며, 강한 양의 feature recovery를 뜻하지 않는다. Cora도 같은 구분이 필요하다. 반대로 C의 full CiteSeer full−shuffle95 +0.998%p나 Physics의 full balanced-accuracy observed−none +0.975%p 같은 제한적인 정렬/소수-class 신호를 0으로 취급하면 안 된다. 이들 역시 작은 개발 반복에서의 기술 통계다.

None 확률이 공통이므로 C−B와 `(C−none)−(B−none)`는 같은 비교다. 이를 두 개의 독립적인 확인으로 세지 않는다. C가 확률적으로 피처를 완전히 무시한다고 단정할 수도 없다. Accuracy가 none과 같아도 확률이나 일부 argmax가 바뀔 수 있고, NLL·shuffle 결과는 따로 읽어야 한다.

큰 structural gain을 centroid의 새 성과로 돌리는 것도 피해야 한다. CoauthorPhysics에서 C의 full→edge accuracy는 46.387→86.240%지만, A도 46.520→86.255%다. 이 큰 차이는 기존 nonedge/graph-family 취약성과 공통 graph protocol이 지배하는 현상이다. CiteSeer에서는 같은 edge 선택이 C를 45.827→36.348%로 낮춘다. Dataset마다 유리한 graph panel을 고르는 방법으로 이 cycle을 바꿀 수 없다.

Minority/probability 품질도 전반적 개선이 아니다. Physics의 rate-평균 full balanced accuracy는 C 48.724%, B 49.476%이고, full NLL은 C 5.8459 대 B 5.8099다. Edge에서도 overall accuracy는 C 86.240%, B 86.258%로 비슷하지만 balanced accuracy는 C 58.111%, B 58.815%다. 높은 overall accuracy 하나로 안정적인 class discrimination·uncertainty quality를 주장할 수 없다.

Post-hoc source coverage는 합리적인 제한을 보여준다. CoauthorCS 95%는 source가 100개여도 15개 class 중 source에 나타난 class는 반복별 12/13/12개다. Physics는 같은 rate에서 5개 class를 모두 관측한다. Source coverage, 불균형, class-feature law, projection covariance, graph compatibility, finite inference는 여전히 여러 원인 후보이며 특정 하나를 유일한 원인으로 확정할 자료는 아니다. 이 정답 기반 통계는 forward 입력이 아니라 scoring-side 설명이다.

## 4. 수학적/계산적 타당성과 아직 입증하지 않은 것

잠긴 C의 정의는 유효한 **finite conditional structured model**이다. Class별 centroid latent는 같은 column prior를 공유하고, observed cell의 K개 gated pair factors 곱은 선택된 class의 Normal score 하나와 정확히 같다. GH masses는 normalized categorical prior이며 Gaussian prior density를 이중으로 곱하지 않는다. Old Potts Bethe estimator를 C feature factors에 잘못 재사용하지 않았고, feature 정보가 없는 graph-only weights를 모든 arm에 공통으로 둔다. 따라서 새 성과를 feature-conditioned role posterior 개선이라고 부를 수 없다.

Forward의 24회 synchronous log-damped BP는 각 local update가 sum-product인 근사 계산이다. 두 source 이상에서는 centroid factorization 자체가 4-cycles를 만든다. 실제 query에 feature가 없으면 centroid에서 query로 직접 이어지는 factor도 없다. 마지막 전달은 graph factors를 거친다. 코드와 graph-frozen gradient 검사에서 이 계산 경로를 확인한 기존 증거는 유효하다.

기존 correctness 결과를 결과 성공으로 돌리지 않아야 한다. Tree exact 오차 약 10⁻¹⁵, old-operator parity 0, finite directional-gradient 오차 약 6.6×10⁻¹²는 구현을 지지한다. 하지만 실제 tiny cyclic-grid 검사에서는 node marginal 오차가 최대 약 0.00848, centroid marginal 오차가 약 0.01531이었다. 이것은 discrete model 안의 BP approximation error다. Small consistency residual이 posterior 정확도를 보증하지 않는다는 예이기도 하다.

GH41의 continuous reference 진단은 별개다. 고정-label conjugate n=100 사례에서 log-evidence 오차 약 −2.919 nats, posterior-mean 오차 약 −0.130이 남았다. 이는 고정한 finite grid가 continuous Gaussian 적분과 같지 않음을 보여준다. 이번 결과를 continuous Bayes 또는 covariance-aware projected likelihood로 읽을 수 없다. 그렇다고 이 진단을 지금 발견한 구현 버그라고 부르거나, 최종 bank를 본 뒤 grid/iteration을 바꿔 FP01을 다시 평가할 근거로 사용할 수도 없다. 해당 차이는 lock 전에 측정·명시되어 있었다.

Singleton 경계는 오히려 중요한 성공적 sanity다. 동일 class prior와 role-independent cell law에서는 한 source의 centroid를 주변화한 값이 class/h와 무관하다. Source 0/1의 최종 probability 오차가 3.58×10⁻⁷ 이하인 결과는 이 제약과 일치한다. 경계에서 none과 같은 성능을 “available feature information을 학습하지 못했다”는 실패로 세지 않는다. 주요 source≥2 조건의 약한 feature utility와는 다른 문제다.

## 5. 중단·resume·메타데이터 cache 검토

세 wrapper finished packet이 모두 COMPLETE이며 동일 wrapper/runner/lock hash와 concurrency1을 기록한다. Seed11의 nondegenerate 완료 row replay는 최대 확률 차이 **5.9605×10⁻⁸**였다. 실제 replay row는 사전 orthogonal75 진단의 C/full row였으며, primary 결과를 고르는 작업이 아니었다. 원 worker 중단 후 남은 C_global/seed11 unpaired NPZ 한 개가 quarantine에 존재하고 기록된 SHA256과 일치하는 것도 별도로 확인했다.

Wrapper가 바꾼 runner attribute는 descriptive `_git_state` 하나다. 완성된 예측의 strict identity/hash resume 검사와 stage-end source/model 검증은 유지됐다. 앞선 operational review에서 replay 자체의 NPZ digest 사전 비교가 약하다고 지적했지만, 이후 unchanged runner의 strict resume와 최종 전체 audit가 성공해 해당 integrity 검증은 완료됐다. Scientific prediction을 바꾸는 post-lock 수정의 증거는 없다.

Git cache로 줄어든 실행시간은 모델 latency 성과가 아니다. `budget.json`의 fit seconds 합계 약 305.59초도 전체 elapsed wall time이 아니다. `discovered_failed_attempt_logs=[]`는 학습 failure log 부재로 읽어야 하며, 명시적으로 중단·재개한 평가 작업이 없었다는 뜻으로 쓰면 안 된다. 해당 운영 비용·증거는 runtime packets와 실행 amendment에 따로 남아 있다. Benchmark latency 주장은 여전히 성립하지 않는다.

## 6. 판정과 남길 문장

**FP01 구현·provenance 감사는 통과했고, 사전 adaptive-success 가설은 통과하지 못했다.** 이를 구현 버그 때문에 무효인 실험으로 분류할 근거는 발견하지 못했다. Full same-family synthetic에서 C가 C_global보다 유리한 accuracy·NLL 신호는 보존한다. 그러나 B보다 확실히 ≥1%p 낫다는 co-primary 조건, 그리고 실제 개발 그래프에서 일관된 양의 feature utility를 확보하지 못했다. Projection training interface만 바꾼 B도 이번 budget에서 회복을 보이지 않았다.

유지할 결론 예시:

> 같은 graph-only role protocol과 고정된 graph potentials 아래에서 adaptive centroid-grid operator는 전역 centroid operator보다 synthetic primary에서 개선됐으나, projection-trained preferred-class operator에 대한 사전 accuracy 성공 조건은 충족하지 못했다. 실제 개발 그래프에서는 관측 피처의 손해를 줄이는 경우가 있었지만 안정적인 양의 추가 가치는 확립하지 못했다.

피해야 할 결론 예시:

- “Projection mismatch를 해결해 실제 feature inference가 성공했다.”
- “GH41와 finite BP가 정확한 continuous centroid posterior를 복원했다.”
- “Coauthor edge-only의 큰 성능 회복은 새 centroid inference의 기여다.”
- “Feature-conditioned role posterior가 개선됐다.”
- “Source=1에서 none과 같으므로 centroid 구현에 버그가 있다.”

최종 bank를 기준으로 grid·BP 횟수·full/edge 선택을 바꾸는 rescue는 하지 않아야 한다. 새 operator/prior 가설을 시험한다면 FP02의 새 사전 계획과 평가 명세로 구분해야 한다. 현재 아티팩트에는 후속 code repair가 필수라는 수학적·구현상 결함은 확인되지 않았다.
