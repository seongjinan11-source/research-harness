# FP03 독립 추론 결과 검토

2026-11-04. PLAN/protocol, 현재 operators/checks/runner, REPORT·summary·audit·manifest 및 calibration/prediction NPZ를 읽었다. 다른 현재 결과 review는 읽지 않았다. CPU에서 저장 확률·fold metadata·통계만 재계산했으며 forward 재실행·weight fitting·GPU·코드/설정/원결과 변경을 하지 않았다.

**판정: 정상 실행/audit PASS, 사전 개발 gate FAIL.** Context 가중치는 uniform prior mixture보다 크게 개선했지만 fixed-edge의 accuracy/BAcc를 이기지 못했고 CiteSeer의 old-full 장점을 보존하지 못했다. 현재 결과는 inherited learned graph model의 full/edge를 context로 적응시켰다는 실험과도 다르다. **새 후보 potential과 nonedge MF를 함께 도입한 효과가 아직 분리되지 않았다.**

## 1. 독립 재구성과 fault 판정

- Manifest의248개 파일 SHA를 전부 독립 확인했다. 검사에 연결된 operators/checks/runner와 source lock이 그대로다.
- 저장 raw held probabilities를 float64로 변환하고 clip[1e-8,1]·row normalization·represented-class macro log score·softmax를 별도 NumPy로 계산했다. 최대 score 오차3.55e-15, weight 오차1.67e-16이었다.
- 저장 candidate probabilities로50개 control 확률을 재구성한 최대오차는2.98e-8이며, 이 control들의 accuracy/BAcc/macroF1/NLL/Brier는 summary와 정확히 같았다. None/95 weights와 scores도 bit-identical이었다.
- Calibration metadata의19fold/126held points를 직접 확인했다. 각 fold의 clamped IDs는 selected−held였고, extra13 CS labels는 제외됐다. Represented class별 남은 count가 m−1이며 held-position/label mapping도 일치했다. CS의 missing class9를 prediction에서 제거하지 않았다.
- Runner의 실제 calibration은 F=0/R=1 graph-only 입력, fold별 masked context, fresh-zero BP state다. Held labels는 branch prediction 뒤 score에만 들어간다. 현재 저장 자료/소스에서 query-label calibration, full-context warm-start, fold-cache 혼합, sign/normalization 오류의 증거는 없다.

독립 tiny checks는 마지막 사용 MF field, reverse cavity, source0/1, hidden values, chunk/순열/gradient를 검증했다. 이 PASS는 finite loopy BP나 mean-field가 실제 posterior를 정확히 근사한다는 보장이 아니다. 현재 성능 실패를 implementation fault로 바꾸어 같은 bank의 grid·temperature를 수정할 근거는 없다.

## 2. 가중치 효과: 부분 성공은 보존하되 fixed control과 구분한다

None의 equal-five-graph 평균이다.

| 방법 | Accuracy % | BAcc % | NLL | Brier |
|---|---:|---:|---:|---:|
| adaptive | 60.910 | 53.272 | 1.33772 | .62142 |
| prior mixture | 35.269 | 25.635 | 1.38693 | .66661 |
| fixed edge | 62.523 | 56.611 | 1.62348 | .56913 |
| old edge | 63.750 | 57.432 | 1.62087 | .54456 |
| LP restart | 66.319 | 59.848 | 1.63741 | .76178 |
| LP harmonic | 50.689 | 43.137 | 1.23455 | .57337 |

Adaptive−prior accuracy는+25.641pp다. **동일 candidate predictions 위에서 context-derived weighting이 uniform weighting보다 유용했다**는 효과는 직접 비교로 보존한다. 그러나 prior가 약한 family average라는 점도 함께 적어야 한다. Adaptive−fixed-edge는 accuracy−1.612pp/BAcc−3.339pp이고 old-edge 대비 BAcc−4.160pp다. NLL은 fixed-edge보다.28576 낮지만 Brier는.05229 높다. 이것을 accuracy·minority 회복이나 전반적인 calibration 성공으로 통합하지 않는다.

Physics는 old_full48.551→adaptive85.903%로 큰 회복을 보였지만, fixed-edge/old-edge88.801%보다2.898pp 낮다. CS도 old_full69.018→72.050%지만 사전+5pp를 달성하지 못했고 fixed-edge73.171%보다 낮다. CiteSeer는 adaptive40.674%/NLL1.66729 대 old_full52.898%/1.43452로 accuracy−12.223pp, NLL+.23278이다. 열 gate 중 다섯 개가 실패했으므로 MAP이나 individual candidate의 수치를 primary 성공으로 대체할 수 없다.

Physics의 adaptive major-class hard mass73.979%는 실제70.026% 및 fixed-edge71.080%보다 크다. Major recall은 adaptive97.648%, fixed97.366%로 비슷하지만 BAcc는62.465% 대71.026%다. 즉 이번 accuracy 손해를 balanced minority 회복을 위한 대가로 설명할 수도 없다.

## 3. 실제 calibration weights가 뜻하는 것

NPZ에서 직접 합한 b별 weight mass다.

| Graph | b=-2 % | b=0 % | b=+2 % | fixed-edge weight % | MAP index |
|---|---:|---:|---:|---:|---:|
| Cora | 43.873 | 55.191 | .936 | 10.320 | 9 |
| CiteSeer | 36.740 | 58.552 | 4.708 | 14.290 | 7 |
| PubMed | 8.612 | 90.162 | 1.226 | 7.595 | 1 |
| CS | 54.882 | 44.950 | .167 | 4.732 | 6 |
| Physics | 8.449 | 91.533 | .018 | 9.656 | 7 |

Weighting은 Physics에서 b0를 대부분 지지했지만, 그 질량은 edge strength a에 걸쳐 퍼졌다. Physics에서는 a<0 weight24.151%, a=0 weight28.449%이고 고정 edge(a=log4,b=0)는9.656%뿐이다. 따라서 “nonedge를 버리기로 알아냈으니 fixed-edge 성능을 자동 회수한다”는 설명은 틀리다. 단위 **mean** score의 넓은 mixture가 약한/graph-off branches까지 평균한 실제 결과다. 이를 사후 score scale을 키워 구제하면 새로운 방법이다.

MAP도 query accuracy oracle와 일치하지 않는다. Cora MAP9의 none accuracy46.380% 대 fixed-edge63.169%, Physics MAP7의86.891% 대 fixed88.801%다. Held-context **macro log score**와 final query accuracy는 다른 목적이며 calibration에는 더 적은 anchors가 있었다. Sparse context·coverage 및 trimmed→full-context shift가 가중치의 실효성을 제한했다는 해석은 가능하지만, 어느 하나가 고유 원인인지 분리된 개입은 없다.

ESS4.347–8.809/15와 entropy는 이 유한 predictive mixture의 기술량이다. Bayesian mechanism posterior·정확한 epistemic uncertainty가 아니며, 그 값이 크다는 이유로 유용한 ambiguity를 보존했다고 판단하지 않는다. Homophily scalar나 query-truth homophily로 weights를 정한 실험도 아니다.

## 4. 핵심 미검증 confound: inherited full endpoint가 family에 없다

기존 `factor_projection_cycle/common.py`는 learned graph encoder가 만든 pair-specific J_ij를 사용한다. Full branch는 모든 유효한 nonedge를 포함해 **방향별 cavity messages**를 유지한다. 이것은 dense all-pair loopy BP이며, exact joint marginal inference라는 뜻은 아니다.

FP03는 이를 다음 두 방향으로 동시에 바꿨다.

1. **Potential family:** observed edges는 scalar a, nonedges는 rank-one opportunity 형태 `b theta_i theta_j`다. 기존 learned Bernoulli same/different contrast가 일반적으로 이 형태와 같지 않다. 평균 edge strength가 비슷해도 동일 potential이 아니다.
2. **Inference:** nonedge의 pairwise sum-product message를 `b theta_i sum_non-neighbor theta_j p_j(c)`라는 node-product MF field로 바꿨다. 동일 J라도 `log[1+(exp J−1)r_{i→j}(c)]`와 `J p_i(c)`는 유한 J/cavity에서 다른 연산이다. 약한 coupling 한계의 근사를 exact equality로 취급할 수 없다.

그러므로 a/b의15개 선택 또는 그 확률 mixture가 old_full을 endpoint로 포함하지 않는다. b=-2는 “기존 full”, b=0은 “기존 edge”의 정확한 재현이 아니다. Fixed-edge와 old-edge도 Physics accuracy는 같지만 다른 graph에서는 다르다. CiteSeer에서 유용했던 learned full negative evidence를 이 family가 그대로 보존했다고 가정할 수 없다.

**CiteSeer gap을 adaptive weighting 실패 하나나 MF 오류 하나로 귀속할 수 없다.** 동일 potential의 dense-vs-MF 비교, 동일 inference의 learned-J-vs-new-J 비교가 없기 때문이다. 이 점은 계획의 method change가 잘못 구현됐다는 뜻이 아니라, 현재 결과가 어떤 인과 질문까지 답하는지에 대한 제한이다. 이번이 inherited learned graph family의 context adaptation 전체를 반증한 실험도 아니다.

## 5. Best single candidate는 mixture 성능의 상한이 아니다

CiteSeer none의 best single candidate는 index13/41.834%이며 old_full52.898%보다 낮다. **어느 tested single branch도 old_full을 보존하지 못했다**는 관찰은 타당하다. 그러나 “family의 모든 mixture도41.834%를 넘을 수 없다”는 결론은 틀리다.

예컨대 true class0에 대해 두 candidate가 `[.4,.5,.1]`, `[.4,.1,.5]`면 각 argmax는 틀리지만 half mixture `[.4,.3,.3]`는 맞는다. Accuracy는 mixture weights의 선형 함수가 아니다. NLL도 best single branch가 mixture의 하한이 아니다. 실제 저장 PubMed none에서 adaptive NLL1.07710은 best-single NLL1.09861보다 낮다.

따라서 report의 query-truth best candidate는 **best-single 선택 headroom 진단**으로 제한해야 한다. 모든 convex mixtures의 달성 가능 accuracy/NLL를 최적화하거나 엄밀히 bound한 결과가 아니다. 본 review는 query labels로 새로운 weights를 fitting하거나 family upper bound를 주장하지 않았다.

## 6. Feature 효과와 MAP 민감도

Adaptive95−none 평균은 accuracy−.107pp/BAcc−.337pp/NLL+.00039다. 기존 fixed-edge의 feature 손상−1.176pp보다 작지만 **추가 feature utility가 회복됐다는 결과는 아니다.** Weights는 graph-only이고 none/95 공통이므로 feature-conditioned mechanism adaptation 주장도 불가하다. 신규 shuffle은 없다.

CiteSeer MAP accuracy40.042→29.715%의 큰 변동도 단독 code-fault 근거는 아니다. 저장 candidate7의 none/95 비교에서 argmax27.713%가 바뀌지만 평균 row-max probability 변화는.00485, none에서 top-two margin<=1e-6인 query가36.881%다. Near-tie 민감성과 일치하는 진단이며, 모든 변화의 원인을 증명한 것은 아니다. Primary mixture의 실제 feature 효과는 별도로 보존한다.

## 7. 닫을 범위와 다음에 분리할 계산

현 FP03은 **사전 고정15-family/mean-score mixture의 개발 실패로 닫는다.** Grid/temperature/weights를 같은 결과에 맞춰 변경하지 않는다. 새로운 neural pretraining이나 PFN novelty의 증거도 아니다. 사용된 학습 요소는 과거 frozen C_global이며 relational family와 predictive averaging은 고전 조건부 추론이다.

다음에 한 가지를 확인한다면 **inherited learned graph potential과 dense all-pair BP의 full/edge endpoints를 그대로 포함하는 matched-endpoint context calibration**을 권한다. 같은 frozen C_global, 같은 balanced masks/score와 사전 규칙을 쓰되, learned J와 inference를 보존한 두 branch에 대한 calibration 자체가 도움이 되는지 묻는다. MF를 계속 쓰려면 먼저 동일 frozen J에 dense BP/MF를 대조해 그 오차를 따로 측정해야 한다. 이는 현재15-candidate 결과를 다시 채점해 성공으로 바꾸는 작업이 아니라 별도 계획·검토·source lock이 필요한 후속 실험이다. 여기서는 실행하지 않았다.

이렇게 해야 “context weights가 잘못됐다”, “degree opportunity family가 맞지 않았다”, “nonedge MF가 손실을 만들었다”를 구분할 수 있다. Strong LP와 기존 fixed full/edge를 계속 유지하고, 새 endpoint mixture가 이들을 실제로 넘는지 및 observed−none이 양수인지로 판단해야 한다.
