# FP02 Stage 0 실험·통계 결과 검토

2026-11-04. **판정: 구현·출력 감사 PASS, 사전 개발 gate FAIL. 이 설정의 DC/SBM operator는 다음 학습 단계의 성공 후보로 채택하지 않는다.** DC와 SBM 모두 일곱 gate를 전부 실패했다. Eta/shape/damping/반복 수를 이 다섯 graph에서 바꾸어 판정을 구제하지 않는다. 기존 Factor의 보존 효과와 강한 LP 결과는 남기며, 모든 DC-SBM/BP 또는 합성 사전학습이 불가능하다는 결론은 내리지 않는다.

실행·증거 완결성 5/5, 사전 성능 기준 달성 1/5, 실패 원인 분리의 완결성 3/5. 원인 분리보다 이 고정 operator가 이 개발 조건에서 크게 실패했다는 결론의 확신이 높다. 이 reviewer는 Stage 0 runner 작성자이기도 하므로 완전한 blind reviewer는 아니다. 아래 검증은 runner의 metric 함수를 재사용하지 않은 별도 CPU 혼동행렬 scorer로 수행했다. 현재 source·bank·결과를 수정하거나 GPU/추가 실험을 실행하지 않았다. 결과 검토 이후 새 학습 제안은 **미실행**이다.

## 저장 결과의 독립 점검

`results/artifact_manifest.json`의 260개 파일을 실제로 읽어 SHA256을 다시 계산했고 mismatch는 0개였다. 새 30개와 재사용 70개를 합한 100개 NPZ의 확률을 별도 코드로 재채점했다. Accuracy·BAcc·macro-F1·NLL·Brier의 원 summary 대비 최대 절대 차이는 **0**이다. Confusion matrix에서 직접 BAcc/F1을 계산했고, NLL/Brier는 사전에 정한 [1e-8,1] scoring clip과 재정규화를 사용했다. 저장 확률은 바꾸지 않았다. DC/SBM의 일곱 gate 차이도 독립 산출과 정확히 일치했다.

검토한 source lock은 `a95965f73010f8225b9f66138a426829c0e5ce7d46ba4698a1249a7782a40fda`, 개발 manifest는 `2076f9dfe996a45fe40f926b55cbf175a6999e4e498b7de95420f8f36b8a3208`이다. 수치 검사 PASS, fixed/updated의 nuisance update 0/3/3, 신경 optimizer/target-gradient update 0, query own-feature-missing, 동일 context/query/truth 및 historical homophilic references가 보존됐다. Audit PASS가 성능 gate PASS를 뜻하지 않는다. 별도 audit JSON은 작성하지 않았으며 이 문서가 이번 독립 재점검 기록이다.

## 실패는 피처를 넣기 전부터 크다

아래는 rep0/seed11, none 조건의 source graph 5개 동일 가중 평균이다. 서로 다른 node 수로 pooled accuracy를 만들지 않았다.

| 방법 | Accuracy % | BAcc % | NLL |
|---|---:|---:|---:|
| Poisson fixed | 47.226 | 51.961 | 4.07794 |
| SBM update | 42.150 | 49.246 | 5.79886 |
| DC update | 43.805 | 48.989 | 4.21414 |
| Old full | 55.691 | 54.599 | 2.53411 |
| Old edge | 63.750 | 57.432 | 1.62087 |
| Restart LP | 66.319 | 59.848 | 1.63741 |
| Harmonic LP | 50.689 | 43.137 | 1.23455 |

| Graph | DC accuracy % | Old full % | Old edge % | Restart LP % | DC NLL |
|---|---:|---:|---:|---:|---:|
| Cora | 59.077 | 59.601 | 64.218 | 58.972 | 2.59998 |
| CiteSeer | 35.090 | 52.898 | 41.728 | 40.042 | 2.89470 |
| PubMed | 42.275 | 48.389 | 46.680 | 71.598 | 4.05819 |
| CoauthorCS | 62.294 | 69.018 | 77.324 | 73.764 | 3.72223 |
| CoauthorPhysics | 20.290 | 48.551 | 88.801 | 87.220 | 7.79559 |

DC−old-full은 Physics **−28.261%p**, CS **−6.724%p**, CiteSeer **−17.808%p**다. CiteSeer NLL은 +1.46019 nats, 5-graph BAcc는 old-full보다 −5.610%p/old-edge보다 −8.443%p, NLL은 old-edge보다 +2.59327이다. 목표였던 Physics +20%p/CS +5%p 회복 및 CiteSeer 보존과 모두 반대다. SBM도 일곱 조건을 전부 실패해 더 간단한 update로 이번 gate를 대체할 수 없다.

양의 결과도 분리한다. Cora DC BAcc 64.299%는 old-full 60.022%와 old-edge 59.494%보다 높다. PubMed/CS DC accuracy는 Poisson fixed보다 각각 약 +3.81/+1.65%p다. 전체 DC는 SBM보다 accuracy가 높고 NLL이 낮다. 그러나 강한 비교군을 넘지 못했고, Physics에서는 DC accuracy가 fixed 32.740%보다 12.45%p 낮다. Physics NLL은 fixed 9.26854보다 DC 7.79559가 낮으므로 “모든 지표가 모든 update에서 나빠졌다”고 쓰지는 않는다.

## Joint update가 잘못된 분할을 강화한 양상

Physics none에서 실제 query 최대 class는 index2, 비율 **70.026%**다. 같은 class의 hard predicted mass/recall은 다음과 같다.

| 방법 | 해당 class 예측 비율 % | 해당 class recall % | 평균 top confidence % | confidence>99%인 오답 query 비율 % |
|---|---:|---:|---:|---:|
| Poisson fixed | 13.241 | 18.909 | 96.352 | 50.527 |
| SBM update | 4.743 | 6.679 | 98.075 | 66.864 |
| DC update | 3.426 | 4.798 | 98.031 | 67.325 |

DC의 보고 population mean에서 이 class는 **1.410%**이고 다른 class index1은 53.644%다. Population을 자유롭게 갱신한 것만으로 semantic class의 실제 큰 비율을 회수하지 못했다. Query 비율과 unlabeled 전체 population은 모집단이 완전히 같지 않지만 이 크기의 역전과 낮은 recall은 분명한 실패 양상이다. PubMed도 true dominant query class48.389%에 DC hard prediction8.810%/population9.232%로 비슷한 방향의 왜곡이 보인다.

이는 현재 초기화·finite BP·mean-field·Poisson score 아래에서 잘못된 분할이 nuisance 업데이트와 함께 강화되는 관찰이다. “Class가 graph community와 다르다”는 가설과 양립하지만 그것만을 고유 원인으로 확정하지 않는다. Loopy 근사, conditional semantic model의 부적합, point/분포 갱신의 scale 및 식별성 등을 여기서 분해하지 않았다. Tiny exact/tree oracle가 맞았다는 사실은 실제 graph에서 올바른 latent partition을 찾는다는 보장이 아니다.

DC none의 class-conditional max pair rate는 Cora10.77/Physics5.59로 binary-Poisson working approximation의 포화 한계를 드러낸다. 다만 관측 edge의 q-weighted mean rate>1 비율은 각각 약0.0379%/0.0556%이고, Physics Poisson fixed는 max rate0.01834에서도 이미 accuracy32.74%다. **최대값>1만으로 전체 실패를 설명하거나 rate clipping을 즉시 처방하지 않는다.**

## 95% 피처와 baseline의 올바른 해석

DC의 95−none accuracy 차이는 Cora −0.210/CiteSeer +1.370/PubMed +0.131/CS +0.132/Physics −0.198%p, 동일 가중 평균 **+0.245%p**다. BAcc 평균은 +0.036%p, NLL은 −0.00507이다. 작은 관측 변화가 구조적 실패를 회복하지 못했다. 95% source 수는 Cora135/CiteSeer166/나머지각100이므로 source0/1 불식별성으로 이 전체 결과를 설명할 수 없다. 새 shuffle arm이 없으므로 feature alignment 성공을 말할 수 없다. Nuisance도 mode별로 재추정했으므로 이 차이는 graph parameter를 고정한 feature likelihood 효과가 아니라 joint inference의 전체 효과다.

PCFI+public PFN의 95−none 평균 +17.875%p와 FISF+public PFN +1.526%p도 보존한다. 그러나 PCFI none의 약한 출발점을 기준으로 한 gain이 graph-only보다 유용한 추가 정보를 뜻하지는 않는다. 95% absolute accuracy는 DC44.050%, PCFI+PFN59.330%, FISF+PFN60.744%, restart LP66.319%다. Physics 95 FISF accuracy90.909%는 restart LP87.220%보다 높지만 BAcc74.315%는 LP75.383%보다 낮다. Cora/CiteSeer에는 old-full/edge의 별도 장점이 남는다. 단일 accuracy 또는 사후 최고 panel로 전체 우월성을 선언하지 않는다.

이는 기존 개발 graph 다섯 개의 한 rep/seed 결과다. Node bootstrap, seed를 독립 graph로 세는 CI, graph별 full/edge/DC oracle 선택은 정당하지 않다. Cora/CiteSeer full과 나머지 induced2000의 범위도 다르다. 과거 3rep×3seed 평균과 이 rep0 한 행의 수치를 바꾸어 비교하지 않는다.

## Homophily에 맞춘 적응: 현재 데이터가 허용하는 답

**Homophily에 적응할 필요는 있지만, 전체 homophily 비율 하나로 nonedge 삭제 여부를 정할 수는 없다.** 이미 가장 homophilic한 Physics에서 old nonedge 삭제가 가장 큰 회복을 줬다.

- Physics 전체 graph의 저장 metadata h=0.9314451408, Stage 0 induced rep0 h=**0.9581710258**다. 나머지 induced rep1/2는0.9690273501/0.9634317862다.
- CiteSeer의 기존 raw `ctxedit/data/planetoid_pyg/CiteSeer/processed/data.pt`를 CPU로 읽어 사후 확인했다. 저장 manifest의 SHA `fd9a4d3c20865c250e602f2b346eae0a9cfc9ed74493f68d2979f61631d25b57`와 실제 bytes가 같다. Undirected edge4552개 중 같은 label3348개로 h=**0.7355008787**다. CiteSeer에서는 old full이 edge보다 훨씬 좋았다.

이 h는 전체 label을 읽은 **audit 전용 metadata**다. 배포 입력/분기/학습 목표로 사용하면 query label 누수다. 현재 보이는 context끼리의 undirected edge는 Physics20context에서 **2개(동일 label2)**, CiteSeer24context에서는 **0개**다. 2/2를 “확실히 homophilic”이라고 point estimate로 넣거나 0개를 heterophilic이라고 읽으면 안 된다. 단순 iid Beta(1,1) 모형의 예시조차 Physics 2/2 후95% 구간이 [0.2924,0.9916]이다. 이것은 graph dependence와 class-balanced context 선택 편향을 보정한 정식 global-h CI가 아니다. Context class당4개 선택은 실제 population 비율을 대표하지도 않는다.

따라서 적응의 대상은 적어도 **edge-label agreement의 강도/부호**와 **semantic class가 graph partition을 어떻게 묶는가/nonedge의 조건부 의미**를 분리해야 한다. Tiny context에서는 이를 식별하지 못할 수 있음을 posterior uncertainty로 남겨야 한다. Query truth로 얻은 h, FP02의 잘못된 pseudo-label에서 얻은 높은 확신, graph 이름에 기반한 hard full/edge 선택은 대체 추정기가 아니다.

## 누적 기록에서 닫는 것과 다음 학습 후보 하나

FP01은 Gaussian/RP mean-family 제약을 넓혔지만 C−B 주평가 gate와 안정적인 C−none feature 효용을 통과하지 못했다. 이번 FP02는 feature가 없는 상태에서도 더 강한 손상을 보인다. 따라서 다음을 새 centroid head/loss/temperature 또는 같은 DC score의 eta sweep으로 진행하지 않는다. 이전 RATIO의 arbitrary full−neutral 보정과 reliability selector도 성공하지 못했으므로 LP/PFN 출력을 새 gate로 혼합하는 것을 해결책으로 이름 붙이지 않는다. 반면 old-edge의 Physics/CS 회복과 Cora/CiteSeer old-full 장점, FP01 C−C_global 부분 효과는 그대로 남는다.

**다음 후보 제안: graph를 조건으로 두는 conditional Factor에 semantic-partition 불확실성과 homophily 불확실성을 함께 두고, 새 semantic prior와 단일-community control에서 실제로 학습한다. 미실행 제안이며 아직 성공/novelty 근거는 없다.**

핵심 dependency는 adjacency를 모든 semantic label pair의 반복 독립 관측으로 fitting하는 데서, 관측 graph A 위의 label field `p_theta(Y | A,m)`를 학습하는 쪽으로 옮긴다. P(A,Y)의 factorization 순서를 바꾸기만 하면 Bayes 항등식이므로 아무 변화가 없다. **실제 변화는 semantic class당 하나의 exchangeable graph block이라는 제한을 버린 prior와 conditional evidence normalization**이어야 한다.

구체적인 첫 operator는 기존 학습 graph-potential encoder와 predictive query CE를 재사용하고, latent mechanism m 아래의 graph-conditional label score를 정규화한다. m은 homophily role와 semantic full/nonedge-free mechanism의 소형 사전 고정 집합이며 새 neural selector가 아니다. 혼합 posterior의 context evidence는

    q(m | A,Y_C,X_O) ∝ p(m) * Z_m(A,Y_C,X_O) / Z_m(A).

`Z_m(A)`는 context/feature를 clamp하지 않은 같은 label model의 normalizer다. 기존 joint-A evidence 또는 arbitrary PFN full−neutral probability ratio를 이 값으로 대체했다고 주장할 수 없다. 같은 factor 아래의 numerator/denominator를 tiny graph exact enumeration으로 먼저 검증해야 한다. 실제 graph의 loopy Bethe ratio는 근사이며 tiny-context uncertainty를 자동 해결한다고 보장하지 않는다. Point homophily를 입력하는 대신 fixed prior에서 marginalize하고, context가 식별하지 못하면 불확실성을 유지한다. Conditional random fields 자체와 graph-conditioned partition normalization은 기존 원리다. [Lafferty·McCallum·Pereira 원 논문](https://www.cis.upenn.edu/~danroth/Teaching/CS598-05/Papers/crf.pdf).

가장 작은 실행 명세 제안은 다음과 같다. 이 문서는 source lock이 아니며 root의 다음 계획·세 검토에서 수치/API를 확정해야 한다.

1. **Synthetic dependency를 실제로 교차한다.** N128, K3/5에서 latent microcommunity U를 먼저 만들고 graph A를 생성한다. Control은 semantic class당 microcommunity1개, 새 prior는 class당1개 또는3개를 random many-to-one으로 묶는 mixture다. 여러 graph community가 같은 semantic class에 들어가면 그 사이의 absent edge를 class 불일치 증거로 일괄 세기 어렵다. Degree/skew supports·expected edge count·signal·feature law·mask/context budget은 두 arm에 공통으로 둔다. 동일 class histogram/density에서 pooled edge-homophily와 pooled nonedge frequency는 수학적으로 연결되므로, “둘을 완전히 독립 조절했다”고 쓰지 않는다. 독립으로 교차할 것은 **목표 h와 semantic coarsening/nonexchangeability**다.
2. **Homophily를 coarsening과 교차한다.** 목표 raw h={.35,.65,.90}, class skew={balanced,dominant .70}를 양쪽 microcommunity mechanism에서 모두 만든다. Degree-preserving/label-aware 생성 단계 rewiring으로 h를 맞춘다면 generator audit에 tolerance·도달 실패·시도 한도를 고정하고, 실패 graph를 결과를 보고 버리지 않는다. 실제 h, degree-adjusted/random-class baseline, semantic fragmentation을 audit-only로 저장한다. Query labels에서 h를 추정해 model에 넘기지 않는다. Real Physics/CiteSeer h를 맞추는 최적화는 하지 않는다.
3. **실제 6 fits.** 같은 conditional operator를 단일-community control prior와 새 mixed semantic prior에서 각각 seeds11/12/13,256updates×2tasks로 학습한다. 각512 task, 총1,536 updates/3,072 presentations다. 동일 init/architecture/optimizer/CE를 사용하고 새 posterior head·auxiliary loss를 붙이지 않는다. 이 control은 matched 단일-community prior이며 역사적 원래512-task bank의 완전 replay라고 부르지 않는다. C_global/GH41 feature path를 양쪽에 공통 고정하면 graph dependency 학습의 대조가 된다. 학습 parameter·conditional normalizer gradient·role prior와 계산 budget을 사전에 명시한다. 후보가 classical finite-grid보다 왜 필요한지도 결과에서 따로 판단한다.
4. **개발과 새 final을 분리한다.** 새 dev24 graph를 먼저 고정해 correctness/resource만 점검한다. Final은 h3×coarsening2×skew2×K2×4=96 base graphs를 source lock 후 새 seed로 생성하는 범위면 충분하다. Context1/4 per class와 none/95/shuffle95는 같은 base graph의 반복 조건이며 독립 graph로 세지 않는다. Own-feature-missing 공통 query를 고정한다. FP01/FP02 final/dev를 새 최종 검증으로 다시 쓰지 않는다.
5. **Primary와 강한 비교군.** 학습한 mixed-prior Factor−matched control 및 mixed-prior Factor−사전 고정 restart LP를 co-primary로 놓을 수 있다. Class-skew를 다루므로 BAcc를 primary, 각97.5% paired graph-bootstrap CI 하한>0/평균+1%p, accuracy −1%p/NLL +.01 guard를 미리 정한다. Graph별 seed/context/rate를 먼저 평균하고 strata 안에서 base graph를 resample한다. Same-operator fixed conditional mechanism grid, normalized/harmonic LP, 기존 학습 full/edge, PCFI/FISF+public PFN도 mandatory다. Conditional grid가 신경 모델을 따라잡으면 learned 필요성은 미확인이다. Real은 재사용 개발 표와 향후 untouched graph 별도 표로 남긴다.

위 범위의 6 fits와 N128/작은 latent grid는 실제 실행 가능한 pilot이며, 단일 GPU 자원 smoke 후 예컨대60분 ceiling을 source lock 전에 확정할 수 있다. Normalizer 계산·추론 iteration·실패 attempt를 포함해 비용을 기록한다. 새 final 결과를 보고 grid/h 범위/온도를 바꾸지 않는다. Feature observed−none 및 observed−shuffle 이득은 구조 BAcc 개선과 별도로 판정한다. LP 근처로 수축한 효과만으로 새 feature 효용을 선언하지 않는다. 이 제안이 tiny exact conditional-evidence 검사를 통과하지 못하면 대형 학습으로 넘어가지 않고 구현/모형 정의부터 고친다.

## 비용 및 cycle 마무리

실제 Stage 0은 새 training0/target gradient0, forward30회다. Prediction 합계는 fixed0.8981s/SBM0.5457s/DC0.5241s이며 첫 arm의 cold allocator 등의 차이가 있어 이 작은 차이를 알고리즘 효율 우위로 해석하지 않는다. Shared feature table0.0958s, 입력 읽기0.0254s, 준비0.3282s, model load0.1911s, pre-run lock validation0.4263s, 실행 attempt wall5.3738s, 저장 결과 재채점1.1150s다. CUDA peak52.40MiB/incremental38.76MiB다. 이 기록에는 numerical-check suite 및 역사적 학습/외부 baseline 비용이 포함되지 않는다. 기록된 execution attempt는1개 COMPLETE/실패0이며, FP01의 별도 metadata-cache 중단·orphan 재계산을 FP02 학습 비용으로 합치거나 여기서 발생한 것처럼 쓰지 않는다.

FP02는 실제로 계산을 수정하고 실행했으며 사전 조건에 크게 실패했다. 이 negative 결과를 knowledge에 보존하고, 실패한 operator의 추가 학습을 자동 집행하지 않는 것이 타당하다. 위 graph-conditioned semantic-prior 후보는 새 계획과 별도 검증이 필요한 후속 학습 실험이지, 이번 FAIL의 다른 이름이나 결과 재해석이 아니다.

원본: [저장 재채점 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/factor_nuisance_cycle/results/REPORT_KO.md), [summary](C:/Users/seongjin/Desktop/code/gts_pfn/factor_nuisance_cycle/results/summary.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/factor_nuisance_cycle/results/audit.json), [budget](C:/Users/seongjin/Desktop/code/gts_pfn/factor_nuisance_cycle/results/budget.json), [artifact manifest](C:/Users/seongjin/Desktop/code/gts_pfn/factor_nuisance_cycle/results/artifact_manifest.json).
