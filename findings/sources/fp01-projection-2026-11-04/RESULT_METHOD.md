# FP01 independent result review — method / interpretation

최종 `artifact_manifest.json`이 유효한 JSON으로 완성된 뒤에만 결과를 읽었다. Manifest가 지정한 summary/paired/audit/budget/report의 SHA256을 직접 대조해 일치를 확인했다. [summary](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/results/summary.json), [paired](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/results/paired.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/results/audit.json), [생성 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/results/REPORT_KO.md)를 독립적으로 검토했다. 다른 결과 reviewer의 결론을 읽지 않았고 소스·설정·결과 수정, 학습·평가 job, branch 변경을 하지 않았다.

## 1. Verdict

**사전 adaptive-success gate는 실패했다. 그러나 C_global 대비 C의 합성 성과는 보존할 부분 성공이다.** C−B accuracy의 97.5% CI가 0을 포함하고 평균 +0.952%p도 사전 1%p 기준에 못 미친다. 반면 C−C_global은 +1.187%p, 97.5% CI [+0.213,+2.446]이며 NLL도 개선된다. 그러므로 “adaptation이 아무 효과 없었다”도, “projection/centroid repair가 missing-feature 문제를 해결했다”도 맞지 않는다. Real graph에서는 C가 기존 feature 사용의 해를 줄이는 경우가 많지만 no-observed보다 추가되는 예측 가치는 대부분 작거나 불안정하다. 현재의 주된 결론은 **합성 centroid family 안에서 adaptive encoder의 제한된 증분 + 실제 utility 회복 미확인**이다. 이를 완성된 범용 방법의 성공으로 확대할 근거는 부족하다.

## 2. 잠긴 질문별 결과

Full/projected, 50/75/90/95%, source≥2의 합성 primary다. 48 base graph 안에서 rate와 고정 3 seed를 평균하고 N×K strata 안에서 bootstrap한 수치다.

| 비교 | Accuracy 차이 %p | NLL 차이 | 방법상 해석 |
|---|---:|---:|---|
| B−A | −0.017 | +0.000063 | Projection 노출 학습만의 이득은 현재 점추정에서 보이지 않는다. 이 보조 차이에 대한 동등성 결론은 아니다. |
| C_global−B | −0.234 | −0.005343 | 전역 계수 centroid 구성이 accuracy를 회복했다는 증거는 없다. 작은 probability-score 개선은 보존한다. |
| C−B | +0.952 [−0.585,+2.696] | −0.029345 | Co-primary accuracy 요건 실패. NLL 개선은 양성 보조 근거다. |
| C−C_global | +1.187 [+0.213,+2.446] | −0.024002 | 이 centroid family/예산에서 adaptive encoder의 증분을 지지하는 양성 결과다. |

마지막 두 accuracy CI는 각각 **97.5%**다. C−B NLL의 저장된 구간은 [−0.069182,−0.001444], C−C_global은 [−0.059758,−0.003516]로 0 아래다. 다만 확률 지표의 보조 분석을 accuracy co-primary gate의 대체물로 쓰지 않는다. C−B Brier의 구간은 0을 포함하므로 “모든 확률 지표에서 확실한 개선”이나 calibration 확보라고 쓸 수 없다.

**C의 절대 observed−none accuracy는 +0.579%p이나 95% CI [−0.855,+1.929]다.** NLL 차이 −0.002459의 95% CI도 [−0.015759,+0.012148]로 0을 포함한다. 사전 absolute gate가 양의 평균만 요구해 이 항목은 통과했지만, feature의 절대 효용을 통계적으로 확립했다는 뜻은 아니다. A/B/C_global의 observed−none 평균은 각각 −0.356/−0.373/−0.608%p다. 따라서 C의 비교 우위에는 불리한 feature 사용을 줄이는 효과가 섞였을 수 있다. 이는 수치와 양립하는 설명이며 learned hyperparameter나 causal mechanism을 직접 확인한 결과는 아니다.

None이 공통이므로 `(C−none)−(B−none)=C−B`다. 상대 accuracy 개선과 feature-utility 차이를 독립된 두 성공으로 세지 않는다.

Edge panel에서는 C accuracy 42.561%, B 43.182%, C_global 43.121%이며, C−B NLL은 **+0.465185**, C−C_global은 **+0.208659**로 나빠졌다. Full에서의 probability-quality 개선을 graph-factor 변경에도 안정적인 효과라고 일반화할 수 없다. Edge는 사전 secondary이며 이 결과로 primary를 교체해서는 안 된다.

## 3. Real feature utility와 비교 성능을 분리해야 한다

### 실제 관측값의 추가 가치

가장 많은 source가 있는 95% 조건의 C 결과다. 모두 고정 seed/mask 반복의 기술 통계이며 real 모집단 CI가 아니다.

| Graph | Full: observed−none %p | Full: observed−shuffle %p | Edge: observed−none %p | Edge: observed−shuffle %p |
|---|---:|---:|---:|---:|
| Cora | −0.024 | +0.128 | +0.245 | +0.467 |
| CiteSeer | +0.003 | +0.998 | −0.139 | −0.266 |
| PubMed | −0.440 | −0.206 | −1.120 | −0.395 |
| CoauthorCS | +0.00004 | +0.095 | +0.015 | +0.007 |
| CoauthorPhysics | +0.205 | +0.249 | +0.095 | −0.110 |

- **CiteSeer full의 +0.998%p shuffle 차이를 feature 추가 이득으로 바꾸면 안 된다.** 같은 C의 observed−none은 +0.003%p다. B의 −3.613%p를 C가 거의 0으로 줄인 것은 유용한 harm reduction이나, 새로운 관측 정보로 graph-only를 뚜렷하게 이겼다는 결과는 아니다.
- Cora edge의 +0.245%p observed−none도 NLL은 +0.004583으로 나빠졌다. C_global의 같은 accuracy 추가 이득은 +0.663%p로 C보다 크다. 이 설정을 adaptive encoder의 고유 성공 사례로 고르면 부정확하다.
- CoauthorPhysics full95의 +0.205%p는 반복별 약 [−0.088,−0.088,+0.792]%p다. 소수의 양성 신호는 보존하되 안정적 utility로 부르지 않는다. PubMed edge95는 accuracy −1.120%p, balanced accuracy −2.348%p, NLL +0.013783으로 명확히 불리한 방향이다.
- 95% source는 Cora 135, CiteSeer 166, 2000-node panels 100개다. Real95 utility 미확인을 모두 singleton 식별 불가능성으로 설명할 수 없다. 반대로 일부 극단 결측의 source0/1 경계는 feature method 실패로 세지 않는다.

### 외부 PFN/LP 대비 성과는 남지만 C의 공으로 전부 돌릴 수 없다

아래 accuracy는 95/99/99.5% 동등 평균이다. 두 graph panel을 나란히 유지했고 dataset별 winner를 고르지 않았다. Baseline은 동일 historical 입력/마스크의 재사용 예측이며 다른 입력 interface와 학습 recipe까지 통제한 새 재학습 비교는 아니다.

| Graph | C full | C edge | PCFI-PFN | FISF-PFN | LP harmonic | LP restart |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 59.735 | 60.159 | 52.898 | 55.481 | 38.246 | 59.056 |
| CiteSeer | 45.827 | 36.348 | 35.262 | 32.763 | 34.228 | 38.410 |
| PubMed | 61.798 | 66.034 | 60.831 | 57.205 | 51.248 | 73.690 |
| CoauthorCS | 61.750 | 72.673 | 68.761 | 69.601 | 64.822 | 68.891 |
| CoauthorPhysics | 46.387 | 86.240 | 75.743 | 76.900 | 85.411 | 84.111 |

Cora/CiteSeer full의 강한 PFN 대비 accuracy, Coauthors edge의 높은 overall accuracy는 실제 양성 근거다. 그러나 CoauthorCS edge에서 B도 72.573%, Physics edge에서는 B 86.258%로 C와 거의 같거나 높다. 큰 개선의 대부분을 새 centroid/adaptive inference의 성공으로 이전할 수 없다. CiteSeer는 full→edge에서 C 45.827→36.348%로 떨어져 nonedge 삭제를 보편적 해법으로 세울 수도 없다.

Probability/minority quality도 함께 제한한다. Cora full C NLL 1.6967은 FISF 1.3563보다 높다. PubMed는 C full/edge 모두 LP restart accuracy 73.690%보다 낮고, NLL도 PCFI/LP보다 높다. Physics edge의 C accuracy 86.240%가 높아도 balanced accuracy 58.111%는 LP restart 63.911%, FISF 63.472%보다 낮으며, C NLL .9307은 harmonic LP .5955보다 높다. 정확도 우위를 일반적인 class discrimination/uncertainty 우위와 동일시하지 않는다.

## 4. Gate 실패와 구현 오류

[Audit](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/results/audit.json)는 **PASS**다. 12 fits/3,072 updates/6,144 presentations/target updates 0, 합성 16,704 + real 1,800 + baseline 375 records가 완성됐다. Frozen graph, source/checkpoint/bank/input/query/target hashes와 saved-probability 재채점, common q, disjoint context/source가 확인됐다. None cross-arm 오차는 0이고 source0/1 대조 최대 오차는 약 3.58e−7이다.

이 근거와 앞선 독립 code-method 검토를 종합하면 **현재 gate 실패를 구현 오류로 분류할 근거는 없다.** Audit는 모든 posterior가 정확하다는 보증이 아니며 이번 analysis도 모든 prediction의 model forward를 독립 재실행한 것은 아니다. Finite BP/41-state grid, conditional prior, RP covariance 및 graph-family mismatch는 남지만, 그것들은 현재 명세의 알려진 통계·근사 한계다. 실제 오류 증거 없이 동일 final bank에서 수식·grid·head·loss·반복 수를 바꾸는 구제는 부적절하다.

## 5. 주장과 다음 판단

**유지할 주장:** “같은 graph-only mixture와 합성 budget에서 adaptive centroid encoder는 C_global 대비 합성 accuracy/NLL을 개선했으며, 일부 실제 graph에서 기존 feature 사용의 해를 줄였다. 다만 기존 preferred-class B 대비 사전 accuracy gate와 real observed-feature utility의 일관된 회복은 확보하지 못했다.”

**수정할 주장:** “Projection을 맞추고 Gaussian centroid를 추론하면 real missing-feature 문제가 해결된다”, “centroid+BP라는 새 원리”, “calibrated/exact Bayesian inference”, “feature-conditioned role posterior 개선”, “모든 PFN/imputer/LP보다 우월”은 현재 결과가 지지하지 않는다. [PRIOR_ART_SCOPE.md](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/PRIOR_ART_SCOPE.md)의 문헌 경계를 유지한다.

RP non-closure는 수학적으로 남지만, interface training B−A의 부재와 약한 absolute utility 때문에 그것이 기존 실패의 충분하거나 유일한 설명이었다고 할 수 없다. 이전 binary feature-law의 적응 실패를 FP01이 뒤집었다고 쓸 수도 없고, 이번 C_global 대비 양성 결과를 그 과거 실패 때문에 지워서도 안 된다.

FP01은 **사전 기준의 음성 결과와 제한된 적응 양성 결과를 함께 확정**하는 것이 맞다. 후속 cycle이 필요하면 “더 큰 head/추가 update면 해결될 것”이 아니라 관측 feature가 정확도 이득보다 해/과신을 만드는 조건, 구조 prior와 feature uncertainty의 충돌 등 새로운 검증 가능한 가설을 먼저 세워야 한다. 그때도 별도 계획·새 final bank가 필요하다. 이 검토는 비판적 과학 판단이며 루틴 사용자 허락이나 임의의 승인 gate를 추가하는 문서가 아니다.
