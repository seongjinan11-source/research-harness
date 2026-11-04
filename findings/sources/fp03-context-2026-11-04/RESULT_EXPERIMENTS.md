# FP03 결과의 독립 실험·통계 검토

2026-11-04. **판정: 저장 결과 감사 PASS, 사전 adaptive 개발 gate FAIL(10개 중5개 실패).** Context calibration은 약한 uniform-family prior를 크게 개선했고 일부 확률 점수도 개선했다. 그러나 fixed-edge/old-edge라는 더 강한 대조를 넘지 못했고 CiteSeer old-full의 장점을 보존하지 못했다. 추가 feature 효용도 확인하지 못했다. 이 고정15-family·balanced-CV·mean-score-scale1 설정을 성공 방법으로 확대하거나 같은 개발 bank에서 a/b/temperature를 조정해 구제하지 않는다.

실행·증거 완결성5/5, 사전 기준 달성2/5, 고유 실패 원인 분리3/5. 이 reviewer는 runner 작성자이기도 하므로 완전한 blind review가 아니다. 아래 검증은 run.py의 production/report 함수를 호출하지 않은 별도 CPU NumPy 코드로 수행했다. 현재 source·bank·저장 결과·branch를 변경하거나 GPU forward/추가 학습을 수행하지 않았다. 다음 endpoint-family 제안은 미실행이며 성공을 예고하지 않는다.

## 1. 독립 재검증

원 artifact manifest에 열거된248개 파일의 실제 SHA256을 재계산했고 mismatch는0개였다. 저장 calibration 확률에서 class-macro log scores와 weights를 재구성한 최대 오차는 각각3.5527e−15/1.6653e−16, final candidate probabilities에서5 controls를 재구성한 최대 오차는2.9802e−8이다. None/95 weights는 bit-identical이다.

120개 aggregate scores(새5 controls×10episode+재사용70 refs) 및150개 candidate의 accuracy/BAcc/macro-F1/NLL/Brier를 혼동행렬 기반 별도 scorer로 재채점했고 원 summary와의 최대 차이는 **0**이다. 열 가지 gate 차이도 별도로 계산해 전부 일치했다. 저장 raw 확률은 변경하지 않았고 scoring에서만 고정 clip[1e−8,1]+row normalization을 썼다.

Source lock `e7af4175bab33bc0ef47dfd05314fd0531ab63f318d1e8c1ab4f3b431fbffb17`, bank manifest `e6fae744c7e565e275ec23e9e7da7ed8f58a76268ba807782b50efe006f2fcbe`, checks `024c5c6e6ea2305f4f4cb6c1068e08faa2a01ad2a8dd37805b5b0df3a456d063`를 확인했다. 139 original context→126 held points/19 balanced folds→285 calibration branches+150 final branches가 실제 기록과 맞고 신경 optimizer/target-gradient update는0이다. 이는139/435/1890개의 독립 통계 표본이 아니다. 별도 audit JSON은 쓰지 않고 이 문서에 독립 재점검을 기록했다.

## 2. 큰 prior 대비 gain과 강한 baseline 대비 실패를 함께 보존

다음은 none 조건, source graph5개 동일 가중 평균이다.

| 방법 | Accuracy % | BAcc % | NLL |
|---|---:|---:|---:|
| Adaptive mixture | 60.910 | 53.272 | 1.33772 |
| Uniform prior mixture | 35.269 | 25.635 | 1.38693 |
| MAP diagnostic | 52.425 | 45.724 | 1.42016 |
| Fixed edge | 62.523 | 56.611 | 1.62348 |
| Old full C_global | 55.691 | 54.599 | 2.53411 |
| Old edge C_global | 63.750 | 57.432 | 1.62087 |
| Restart LP | 66.319 | 59.848 | 1.63741 |
| Harmonic LP | 50.689 | 43.137 | 1.23455 |

Adaptive−prior accuracy는 **+25.641%p**다. Context predictive scores가 이15-family에서 무의미했던 것은 아니다. 동시에 adaptive−fixed-edge는 **−1.6125%p**, adaptive−old-edge는 **−2.8399%p**, adaptive−restart LP는 약−5.409%p다. Uniform prior는 고정 homophilic edge보다 약27.25%p 낮아 매우 약한 출발점이었다. Prior 개선만을 핵심 효과로 내세우면 강한 단일 규칙보다 못한 사실이 사라진다.

NLL은 adaptive가 fixed-edge/old-edge보다 각각0.28576/0.28316 nats 낮다. 이는 보존할 확률 점수의 이득이며 accuracy/BAcc의 미달을 대체하지 않는다. Harmonic LP는 NLL1.23455로 더 낮지만 accuracy50.689%이므로 그것도 단일 지표로 전체 최강이라고 할 수 없다. Categorical ties 때문에 graph-off argmax accuracy는1/K와 같지 않을 수 있다. Uniform probabilities의 확률 sanity check와 hard-label tie 점수를 구분한다.

| None graph | Adaptive accuracy % | Fixed edge % | Old edge % | Old full % |
|---|---:|---:|---:|---:|
| Cora | 60.231 | 63.169 | 64.218 | 59.601 |
| CiteSeer | 40.674 | 40.991 | 41.728 | 52.898 |
| PubMed | 45.694 | 46.483 | 46.680 | 48.389 |
| CoauthorCS | 72.050 | 73.171 | 77.324 | 69.018 |
| CoauthorPhysics | 85.903 | 88.801 | 88.801 | 48.551 |

Physics−old-full +37.352%p는 사전 +20%p 조건을 통과한다. 그러나 fixed/old-edge보다2.898%p 낮고 BAcc62.465%도 old-edge71.026% 및 restart LP75.383%보다 낮다. CS는 old-full보다+3.032%p라서 +5%p 기준에 못 미친다. CiteSeer는 old-full보다−12.223%p/NLL+.23278로 보존 조건을 둘 다 실패했다. Mean BAcc−old-edge는−4.160%p, mean accuracy−fixed-edge는−1.6125%p다. 이 다섯 조건이 FAIL이며 mean NLL guards와 prior accuracy gain, Physics gain의 다섯 PASS가 이를 대체하지 않는다.

## 3. Calibration macro log-score와 query accuracy는 같은 목표가 아니다

저장15개 후보를 모두 확인했다. 아래 query class-macro NLL은 이 결과 검토에서 추가 계산한 **사후 진단**이며 primary나 weights를 변경하지 않았다. Cal-MAP는 balanced held labels의 macro NLL 최소값이다. Final query는 더 많은 원 context와 실제 query class 구성을 사용한다.

| Graph | Cal-MAP index | Cal macro NLL | 해당 후보 query macro NLL | 해당 후보 query accuracy % | Query accuracy-best index/accuracy % |
|---|---:|---:|---:|---:|---|
| Cora | 9 | 1.75200 | 1.87524 | 46.380 | 12 / 64.428 |
| CiteSeer | 7 | 1.67341 | 1.67343 | 40.042 | 13 / 41.834 |
| PubMed | 1 | 1.04512 | 1.22663 | 43.590 | 2 / 48.389 |
| CoauthorCS | 6 | 2.52230 | 2.18652 | 45.221 | 13 / 76.928 |
| CoauthorPhysics | 7 | 1.43435 | 1.16637 | 86.891 | 10 / 88.801 |

특히 CS fixed-edge index10은 cal macro NLL3.70135로 MAP6보다 나쁘지만 query macro NLL2.13422/accuracy73.171%로 MAP6의2.18652/45.221%보다 좋다. Cora도 index9가 context에서는 좋았지만 final query에서는 fixed10이나 accuracy-best12를 놓쳤다. Balanced calibration의 m−1 anchors/class와 final full-context 사이의 조건 차이, 적은 label 수, 구조적 family mismatch가 가능한 원인이다. 어느 하나로 인과를 확정하지 않는다.

반면 Physics에서는 MAP7이 query macro NLL도 최상1.16637이다. Accuracy-best10은 query macro NLL1.64175/ordinary NLL.70690으로 더 나쁘지만 accuracy는88.801%로 더 높다. 즉 여기서는 score가 완전히 엉뚱한 후보를 고른 것이 아니라 **proper-score와 hard accuracy의 목적 차이**도 작용한다. Cal-MAP7의 ordinary NLL.49357은 adaptive.78230보다도 낮다. Mean-score mixing이 항상 MAP보다 나쁜 것도 아니다. 전체 adaptive는 MAP보다 accuracy와 NLL이 모두 좋고, 다른 graph들의 개선을 보존해야 한다.

15개 후보의 cal macro NLL과 query macro NLL 순위는 대체로 같은 방향이다. 사후 Spearman 기술 통계는 Cora.811/CiteSeer.979/PubMed.950/CS.850/Physics.982지만, cal score와 query accuracy 순위는 각각.412/.217/.135/.177/.130이다. 후보15개는 공유 graph/계산으로 강하게 의존하므로 p-value나 표본수15의 일반화 검정은 붙이지 않는다. 이 결과는 “calibration에 아무 정보가 없다”도, “score가 좋으므로 accuracy 적응이 성공했다”도 지지하지 않는다.

Weights는 평균 log-score scale1의 보수적 predictive 평균이다. Physics에서조차 MAP edge index7 weight28.864%에 더해 graph-off24.228%와 negative-edge index1 22.719%가 남는다. Cora/CS의 ESS는8.38/8.22, Physics는4.78이다. 이를 true homophily의 posterior 확률이나 calibrated uncertainty로 해석할 수 없다. Mean-score scale을 높여 MAP 쪽으로 이동시키면 낫다는 결론은 이 결과를 본 뒤의 tuning 가설일 뿐이다.

## 4. CS의 대표성 제한과 feature effect

CS의 원 context55개는 class3에3개, class9에0개, 나머지 class에4개다. Balanced calibration은 represented class14개에서3개씩42개를 쓰고 extra13개를 모든 fold에서 숨겼다. Final에는55개를 모두 복구했다. 이 설계는 individual LOO의 class-count cue를 제거했지만 final distribution을 그대로 검증하지는 않는다. Class9 query4개는 평가에 남아 있다.

CS none adaptive는 class9에 query 전체의4.021%를 예측했으나 해당 class recall은0/4다. MAP/fixed-edge/prior의 class9 recall도0이다. Calibrated risk가 이 class를 검증했다고 할 수 없다. CS macro NLL을 represented14개만으로 다시 계산해도 MAP6은2.11246, fixed10은2.08029다. 따라서 앞의 ranking 역전이 unrepresented class9 하나 때문에 발생한 것만은 아니다.

95−none의 adaptive accuracy 차이는 Cora−1.364/CiteSeer+.105/PubMed−.263/CS+.264/Physics+.725%p다. 평균은 **−.107%p**, BAcc−.337%p, NLL+.00039이다. 일부 양의 graph 효과는 보존하지만 안정적인 추가 feature 효용은 확인하지 못했다. Fixed-edge feature harm−1.176%p보다 adaptive 손상이 작다는 것은 harm reduction의 증거이며 observed가 none보다 유용해졌다는 뜻은 아니다.

None/95 weights는 동일하므로 feature-driven mechanism adaptation의 결과도 아니다. 새 shuffle 대조가 없고 한 rep/seed의 재사용 graph이므로 alignment 또는 독립 feature 효과 CI를 주장할 수 없다. Cora/CiteSeer full과 나머지 induced2000의 scope도 유지한다.

95% 평균 accuracy는 adaptive60.804%, PCFI+public PFN59.330%, FISF+PFN60.744%, restart LP66.319%다. Adaptive가 PCFI/FISF보다 약간 높다는 비교만으로 우월성을 쓰면 안 된다. Adaptive BAcc52.935%는 PCFI54.259%/FISF55.687%보다 낮다. NLL은 adaptive1.33811이 둘보다 낮다. Physics FISF accuracy90.909%와 LP의 더 높은 BAcc도 그대로 보존한다. Strong external baselines와 minority/proper score의 tradeoff를 함께 읽어야 한다.

## 5. Best single-candidate oracle는 mixture의 상한이 아니다

ANALYSIS_SPEC은 모든15개 single-candidate query metric과 best-single accuracy/NLL을 family headroom 진단으로 미리 고정했다. 이 값은 query truth를 사용하므로 deploy 입력·weight·primary 대체·같은 bank 재탐색에 쓰면 안 된다.

또한 **best single-candidate 성능은 후보들의 확률 mixture에 대한 upper bound가 아니다.** Global probability averaging도 node마다 다른 confidence/error를 결합해 모든 단일 후보보다 나아질 수 있다. 실제 PubMed none adaptive NLL1.07710은 single-candidate minimum1.09861보다 낮고, Cora1.49042도 single minimum1.49200보다 낮다. 따라서 “CiteSeer single best가41.834%이므로 어떤 mixture도 그 이상은 불가능하다”는 결론은 성립하지 않는다. Accuracy-best와 NLL-best가 다른 후보인 표 역시 한 predictor가 달성한 점수 조합이 아니다.

CiteSeer에서 관측된 single-candidate best41.834%가 old-full52.898%보다 훨씬 낮다는 것은 현재 후보 endpoint들이 충분한지를 의심할 근거다. 그러나 그 숫자 하나로 모든 mixture의 최선이나 conditional contrast 적응 일반을 기각하지 않는다.

## 6. 어떤 원인까지 확인했고 다음에 무엇을 바꿀 수 있나

직접 확인한 것은 (a) 고정 family의 uniform 평균은 약하고 context weighting은 이를 개선한다, (b) 같은 family의 사전 fixed-edge보다 accuracy/BAcc가 낮다, (c) 기존 full의 CiteSeer 장점을 회수하지 못한다, (d) 추가 feature utility는 보이지 않는다는 것이다. 올바른 split/수치 감사가 성능 성공을 대신하지 않는다. FP02의 자기 예측 Gamma update 실패와 FP03의 held-context weighting 실패는 다른 실패이며 한 가지 원인으로 합치지 않는다.

**현재 family는 inherited old-full/old-edge를 정확히 포함하지 않는다.** Old encoder의 pair별 learned coupling과 모든 nonedge cavity 연산을, constant edge a+degree-product b와 nonedge node-product MF로 동시에 바꿨다. 따라서 old-full→FP03 하락은 calibration만의 통제된 개입이 아니다. Parametric candidate support와 MF 대 old cavity 근사 변경이 함께 다르다. Fixed-edge도 old-edge와 전 graph에서 같지 않으며 Physics accuracy 일치만으로 동일 endpoint라고 할 수 없다.

후속을 검토한다면 **기존 full/edge endpoint를 확률까지 그대로 보존한 family에서 context adaptation만 바꾸는 작은 새 cycle**이 더 명확한 인과 대조다. Full-context reference를 재현한 다음, 새 calibrated folds에서도 같은 inherited graph operator/checkpoint/frozen feature를 쓰고 label-dependent state를 새로 초기화해야 한다. 같은 balanced calibration과 uniform-prior/fixed-endpoint 대조, 강한 LP/PCFI/FISF 기준을 유지한다. 이것은 현재 grid의 a/b/temperature를 튜닝하는 제안이 아니다.

다만 endpoint를 포함하면 calibration이 성공한다는 근거는 아직 없다. 소량 context의 식별성, macro-NLL 대 accuracy 목적 차이, trimming/full-context shift는 여전히 남는다. 이전 reliability-selection 실패도 이런 회수의 어려움을 이미 보였다. 이 제안은 **family와 inference confound를 줄이는 다음 검증 조건**이지 성공 방법의 추천 확정이나 새 PFN 원리가 아니다. 다른 posterior head/loss나 학습량을 덧붙인다고 해결된다고 예측하지 않는다. 현재 gate FAIL이므로 계획에 있던 fresh synthetic final 또는 neural training을 자동 실행할 조건도 충족하지 않았다.

## 7. 실제 비용과 마무리

저장 budget의 calibration inference는.72999s, final inference는.65411s로 합계 약1.3841s다. 이는 전체 사용자 대기 시간이나 순수 kernel time과 같지 않다. Calibration setup.04111s/score.00678s, final preparation.03037s/CPU aggregation.00512s, model load.18321s, 준비.88990s, report 내부 audit/rescore1.88273s가 별도로 기록됐다.

Run 내부 attempt wall은3.89077s이고 그 앞의 source-lock validation1.40975s는 별도다. Root가 제공한 [TOOL_EXECUTION.json](C:/Users/seongjin/Desktop/code/gts_pfn/factor_context_cycle/TOOL_EXECUTION.json)의 실제 command-tool wall은 **8.1552139s(exit0)**다. 이 보충 기록은 원 scientific artifact manifest/source lock 밖의 orchestration 자료이며 SHA256은 `64830674c7975b842e4fab51d4c66beda4406a871ee32987d5d86a85a1fdeef2`다. 남는 차이는 launcher/tool/process startup 등을 포함한 미배분 overhead다. 전부 Python import 시간으로 귀속하거나 warmed per-graph latency로 비교하지 않는다. 준비와 report는 별도 명령이다.

단일 CUDA worker/chunk16, peak399.53MiB/incremental394.11MiB, 기록된 attempt1개 COMPLETE/실패0이다. 새 신경 fit/optimizer update는0이지만 inherited feature checkpoint와 외부 reference의 역사적 비용은0이 아니다. 435 branch 계산을1.3초 GPU시간만으로 저비용 학습 성과처럼 보고해서도 안 된다.

FP03의 양의 결과는 **약한 fixed-family prior 대비 context weighting의 실제 개선과 일부 NLL/harm reduction**이다. 닫는 범위는 이15-family와 현재 calibration을 메인 missing-feature 방법으로 확대하는 주장이다. 같은 개발 graph의 설정 탐색으로 FAIL을 구제하지 않고 결과·제한·비용을 knowledge에 보존하는 것이 타당하다.

원자료: [REPORT_KO](C:/Users/seongjin/Desktop/code/gts_pfn/factor_context_cycle/results/REPORT_KO.md), [summary](C:/Users/seongjin/Desktop/code/gts_pfn/factor_context_cycle/results/summary.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/factor_context_cycle/results/audit.json), [budget](C:/Users/seongjin/Desktop/code/gts_pfn/factor_context_cycle/results/budget.json), [artifact manifest](C:/Users/seongjin/Desktop/code/gts_pfn/factor_context_cycle/results/artifact_manifest.json).
