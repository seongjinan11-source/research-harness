# FP04 최종 실험·통계 검토

2026-11-04. **판정: 실행·저장 결과 감사 PASS, 사전 feature-utility/arm-superiority 복합 gate FAIL. 이 후보는 현재 고정 예산에서 종료하는 것이 타당하다.** 관측 전용 연산자의 구현 경로는 검증됐지만, 관측값을 실제로 유용하게 활용하도록 학습됐다는 결과는 얻지 못했다. 결과를 숨은 구현 실패나 모든 missing-robust PFN의 불가능성으로 바꾸어 해석할 근거도 없다.

검토자는 data.py 작성 담당이었다. 따라서 데이터 구현 자체에 대한 완전히 독립적인 저자 심사라고 주장하지 않는다. 이번에는 production `analyze.py`/`data.py`의 통계·채점 helper를 호출하지 않는 별도 CPU NumPy 계산으로 저장 결과와 paired estimand를 검증했다. GPU forward, 새 학습, 추가 예측, source/bank/checkpoint/원결과 변경은 하지 않았다. 새로 작성한 파일은 이 문서뿐이다.

## 1. 저장 결과의 식별·재채점

다음 항목을 실제 파일에서 확인했다.

- Scientific lock의 모든 source SHA, 네 split manifest SHA/row count 및 모든 bank NPZ SHA가 일치한다. Lock SHA256은 `ac86695ede9281d97208bb00ccf2846c439860c8fdf8ef27d578efdbdf2ee49c`다.
- 9 fits 모두 19,748개 parameter가 trainable이며, 각 768 optimizer updates/3072 task presentations다. 총 6912 updates/27,648 presentations와 일치한다. 실제 schedule indices를 protocol seed에서 재생성해 array digest와 대조했고, seed별 세 arm의 schedule/initial digest가 같다. Checkpoint bytes SHA, 내부 lock/initial identity와 저장 state의 final digest도 연결된다.
- 7776 synthetic predictions(864 episodes×3 arms×3 seeds) 및 405 real predictions(45×3×3), 합계 **8181개**를 모두 직접 열어 SHA/receipt association, 순서가 같은 query/truth/context, 확률 shape·유한성·정규화 및 다섯 metric을 재검증했다. 이번 독립 metric 계산과 runner에 저장된 metric의 최대 차이는 **0.0**이었다. Root analyzer의 별도 계산은 3.33e-16이다.
- 모든 final/real base의 rates/interventions가 실제 graph edge arrays·context·query/truth를 공유한다. Query는 전부 own-feature-missing이며 context와 겹치지 않는다. Disjoint 설계에서는 context도 관측 source가 아니다. Metadata의 source 수와 observed mask에서 계산한 수가 같다.
- Real 45 cache의 native feature 수(Cora1433/CiteSeer3703/PubMed500), 관측값 PCFI clamp, none의 allzero PCFI 및 보고된 clipping fraction/maxabs를 원 NPZ에서 확인했다. External 225개도 prediction/sidecar/legacy bank hash와 ordered query/truth, context-label identity를 확인하고 직접 재채점했다.

초기 prelock bridge 검사 실패는 서로 다른 seed embedding을 가진 context 노드까지 같은 logit이어야 한다는 잘못된 assertion이었다. 실패 artifact와 수정 기록이 남아 있고, 이후 실제 학습 전 checks/source lock이 완료됐다. 이를 이번 성능 실패의 원인으로 재분류하거나, 반대로 tiny bridge derivative PASS를 predictive utility 증거로 사용하는 것은 모두 부적절하다.

## 2. 사전 통계와 네 gate

Final96은 family2×shape2×K3의 **12 strata에 각8개 독립 base graphs**다. Primary는 각 graph에서 고정된 3 training seeds와 node95/node995 두 rates를 먼저 평균한 뒤 arm 차이를 만든다. 5000회 graph bootstrap은 해당 strata 안에서 graph를 재추출한다. Node·rate·training seed를 독립 표본으로 부풀리지 않았다. 독립 RNG/bootstrap 재구성으로 아래 네 CI가 1e-15 이내에서 모두 재현됐다.

| 사전 비교 | Accuracy 차이 %p | 명목 97.5% CI %p | 독립 graph 수 | 판정 |
|---|---:|---:|---:|---|
| observed − mask | +0.4232 | [−0.2930, +1.1340] | 96 | FAIL |
| observed − pcfi | +0.4232 | [−0.2713, +1.1176] | 96 | FAIL |
| observed − none | +0.0434 | [−0.6293, +0.6293] | 48 | FAIL |
| observed − shuffle | −0.3255 | [−0.8247, +0.0868] | 48 | FAIL |

앞 두 비교의 primary accuracy는 observed48.2530%, mask47.8299%, pcfi47.8299%다. Observed의 BAcc46.9988%, NLL1.02142, Brier0.61350도 mask/pcfi의 BAcc약46.55%, NLL약1.02359, Brier약0.61510보다 점추정상 좋다. 이 작은 긍정적 차이는 보존해야 한다. 그러나 사전 accuracy CI lower>0을 만족하지 못했으며, NLL이나 특정 subgroup으로 primary gate를 교체할 수 없다. 모든 CI가 0을 포함한다는 사실은 정확한 효과0 또는 동등성을 증명한 것도 아니다.

Utility는 별도 조건부 estimand다. 예정 disjoint48 graphs 모두 node95에서 source>=2라 eligible graph 수는48이며, 이 중20 graphs만 node995도 eligible이다. 즉 **48 node95 + 20 node995 = 68 graph-rate cases**, graph별 eligible rates를 먼저 평균하므로 28개 one-rate graph와20개 two-rate graph는 각각 동일한 graph weight를 갖는다. 68 cases를 하나로 pooled 평균하거나 primary96과 같은 분모로 부르면 안 된다. Utility에는 12 strata당4 graphs가 남는다. 네 명목97.5% intervals를 전체 네 비교의 동시95% coverage라고 부르지 않은 RESPONSE의 해석도 유지해야 한다.

## 3. Sparse support와 family/shape가 말하는 범위

Node95의96 cases는 모두 source>=2다. Node995는 source0=22, source1=25, source>=2=49다. 따라서 99.5% panel의 **47/96(49.0%)**는 source0/1이지만, primary에는 이들을 그대로 포함했다. 두 primary rates 합계192 graph-rate cases의 별도 support 성능은 다음과 같다. 이는 고정 seeds를 평균한 **case-weighted descriptive subset**이지 primary graph-weighted 추정량이나 새 gate가 아니다.

| Source bucket | Graph-rate cases | Observed acc % | Mask acc % | PCFI acc % |
|---|---:|---:|---:|---:|
| 0 | 22 | 48.011 | 47.443 | 47.538 |
| 1 | 25 | 46.583 | 45.667 | 45.667 |
| 2+ | 145 | 48.578 | 48.261 | 48.247 |

Source0에서 observed가 더 높은 점추정은 observation inference의 증거가 될 수 없다. 같은 데이터로 훈련한 서로 다른 계산 방식의 graph/context prior 차이도 포함한다. Source1에서는 이 혼합 synthetic prior 아래 불변성을 강요하지 않았고, 현재 결과로 값의 식별 가능성이 입증됐다고 말하지도 않는다. Source0/1을 사후 삭제해 primary를 구제하면 안 된다.

Primary family means는 centroid observed58.681% 대 mask58.116%/pcfi58.138%, field37.826% 대37.543%/37.522%다. Heldout shape에서는51.259% 대50.586%/50.608%, matched에서는45.247% 대45.074%/45.052%다. 다만 조건부 utility observed−none의 descriptive means는 centroid+0.0868%p/field0.0000%p, heldout+0.1736%p/matched−0.0868%p에 그친다. 어느 family/shape도 utility에 대한 명확한 구제 증거가 아니다. Heldout는 N/F=(384,12)라는 shape holdout이지 새로운 실제 의미 schema에 대한 독립 증거가 아니다.

추가 episodes도 같은 경향이다. 모든96 graphs의 observed standard accuracy는 full48.199%, node90 47.852%, node95 48.199%, node995 48.307%, cell95 48.134%, cell995 48.069%, none48.155%다. 여기서 full은 모든 eligible nonquery feature를 관측한다는 뜻이며 query 자신의 feature는 여전히 없다. Full−none도 +0.0434%p여서 실패를 단지 99.5%의 source0/1 비중으로만 설명하기 어렵다. 반대로 이것만으로 feature route가 원천적으로 무용하다거나 충분한 학습에서도 실패한다고 증명한 것은 아니다. 원인(예산·최적화·통계 축약·prior/과제 식별성)은 분리되지 않았다.

## 4. Real: 높은 arm 점수와 feature utility를 구분해야 한다

Cora/CiteSeer는 전체 graph, PubMed는 기존2000-node induced graph다. 각 새 arm의 graph/mode 셀은 **3 trained seeds×3 고정 context/mask repetitions=9개**를 평균했다. 새 trained seeds가 추가됐어도 새 independent real graph9개가 되는 것은 아니다. 이들은 이미 반복 검토한 개발 graph이며 node/seed/repetition bootstrap으로 새 real generalization CI를 만들지 않았다. 모든405 predictions가 synthetic gate 실패와 무관하게 실행됐다.

| Graph | Obs95 acc % | Mask95 % | PCFI95 % | Obs none % | Obs95−none %p | Obs95−shuffle95 %p |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 52.876 | 47.819 | 47.924 | 52.923 | −0.0476 | −0.0466 |
| CiteSeer | 29.977 | 25.840 | 25.828 | 30.119 | −0.1419 | −0.0242 |
| PubMed | 64.002 | 55.331 | 55.097 | 64.264 | −0.2626 | −0.0214 |

Observed−mask의95 gains +5.057/+4.137/+8.671%p는 실제 결과다. 그러나 none에서도 observed−mask가 각각+3.924/+3.553/+8.084%p이고, observed95는 자기 none/shuffle보다 더 좋지 않다. 세 dataset의 observed99/995 또한 대체로 none 근처다(Cora995의 none 대비 +0.00025%p는 사실상 변화 없음). 따라서 native feature를 관측해서 얻은 의미 있는 개선으로 이 arm 차이를 해석할 수 없다. Mask/PCFI도95−none이 모두 음수다. Main learned PCFI와 mask의 가까운 결과는 관측 효용이 입증되지 않은 현재 학습을 보여줄 뿐, 강한 PCFI estimator 전반의 무효화 증거가 아니다.

Minority-sensitive metric도 유지해야 한다. Observed95의 BAcc는 Cora56.903%, CiteSeer34.054%, PubMed62.016%, macro-F1은53.030/30.352/58.863%다. 특히 PubMed accuracy64.002%만을 이용해 class imbalance를 해결했다고 할 수 없다. NLL은 각각1.87092/1.76660/1.07095이며, 아래 외부 모델과의 ranking은 accuracy ranking과 같지 않다.

## 5. 강한 외부 기준선 225개는 실제로 보존·재채점됐다

5 arms×3 datasets×3 repetitions×5 modes의 **225개** 모두 identity/rescore를 통과했다. 외부 모델은 training seed를 새로 세 개 만든 것이 아니므로 셀당3 repetitions다. 이를 새 모델의9개처럼 독립 복제하지 않았다. 아래는95 mode이고, all-rate/none/shuffle table은 원 `results/REPORT_KO.md`/`summary.json`에 남아 있다.

| Graph | Obs % | Public PFN % | PCFI+Public % | FISF+Public % | Restart LP % | Harmonic LP % |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 52.876 | 17.023 | 54.796 | 54.347 | 59.056 | 38.246 |
| CiteSeer | 29.977 | 14.614 | 34.078 | 34.189 | 38.410 | 34.228 |
| PubMed | 64.002 | 40.629 | 55.888 | 56.091 | 73.690 | 51.248 |

Restart LP accuracy가 observed보다 각각6.180/8.433/9.688%p 높다. PubMed95에서 observed가 public feature PFNs보다 좋은 점추정은 유효하나, feature-free none이64.264%라는 사실도 함께 보여야 한다. Cora/CiteSeer의 PCFI/FISF+public 결과를 plain public의 약한17.0/14.6%만 강조하며 가리면 안 된다. Cora995 FISF56.902%, CiteSeer995 PCFI+public39.074%, PubMed995 PCFI+public64.284% 같은 강한 rate별 결과도 유지돼 있다.

NLL을 포함하면 단일 전면 winner를 선언할 수 없다. 예를 들어 Cora95 PCFI+public NLL1.34872는 observed1.87092보다 좋고, PubMed95 PCFI+public은 accuracy가 낮지만 NLL1.01269로 observed보다 좋다. Restart LP는 PubMed accuracy/BAcc에서 더 높지만 NLL1.07187은 observed1.07095보다 소폭 높다. Harmonic LP 역시 PubMed accuracy51.248%와 NLL0.87244처럼 metric tradeoff가 있다.

물리적 관측값과 graph/context/query/truth의 동일성은 확인됐지만, public의 원래 scaling/fill/PCA/ensemble과 새 native RMS/19k model의 preprocessing·capacity·pretraining budget은 다르다. 그래서 세 matched learned arms의 비교와 외부 강한 estimator 비교는 목적이 다르다. FISF의 none에서도 그래프 및 내부 인공 anchor로 결과가 나올 수 있으므로 그 점수를 실제 관측값의 효용으로 부르지 않는다.

## 6. Native representation과 실제 비용

Full-native caches는 바뀌지 않았고, 모든 arm에 공통인 represented-value clamp[-10,10]은 prelock에 고정됐다. 이 때문에 main PCFI arm은 **literal PCFI cache를 계산한 뒤 bounded representation으로 읽는 학습 모델**이지 unclipped PCFI 전체 정보를 보존하는 연산자는 아니다. 모든 real cache를 직접 읽어 다음 값을 확인했다. Fraction의 분모는 native matrix의 모든 cells이며 학습에서 관측된 source 비율과 다른 수치다.

| Graph | Raw PCFI non-none 최대 abs의 rep/mode 범위 | 절댓값10 초과 cell fraction 범위 |
|---|---:|---:|
| Cora | 11.004…32.704 | 0.000283%…0.095708% |
| CiteSeer | 26.534…59.553 | 0.019237%…0.339963% |
| PubMed | 9.046…24.880 | 0%…0.238100% |

Fraction은 작지만 그것만으로 clipping의 predictive 영향이 0이라고 할 수 없다. 그렇다고 결과를 본 뒤 unclipped rescue나 다른 clamp를 이 cycle의 확인 실험으로 선택할 수도 없다.

실제 budget 기록은 training448.413s, saved evaluation 전체282.178s(그중 forward+probability CPU transfer147.033s), split 준비 wall 합173.667s다. 준비에는 실제 PCFI107.953s가 포함되므로 이를 다시 더해 이중 계산하면 안 된다. Real native 준비127.714s 중 PCFI95.935s, raw native load4.081s다. 이는 부분별 내부 timer 합이고 launcher/startup·최종 감사·중복되지 않는 전체 연구 wallclock 총량은 아니다. Preflight의 discarded dev updates9회는 scientific fits6912 updates와 따로 남아 있다. 과거 external 모델을 만드는 비용은 이 새 fit 예산에 포함되지 않는다.

세 seed training 합은 observed187.468s, mask129.120s, pcfi131.825s다. Native real forward 평균은 각각약0.11485/0.07806/0.07913s per row이며, 관측 sparse arm이 이 구현에서 더 빠르다는 증거는 없다. 반면 real inference의 기록된 최대 additional allocated peak는141.157/446.835/493.833MiB로 observed가 작았다. Training allocated peak는475.578/220.719/220.750MiB다. 서로 다른 timing/memory scope를 섞거나 equal updates/parameters를 equal compute라고 하면 안 된다. PCFI를 CPU cache로 한 번 계산한 뒤 세 seeds에서 재사용한 비용도 명시돼 있다.

## 최종 해석

이번 실험은 architecture·input support·literal PCFI·fresh factorial synthetic bank·native transfer·외부 LP/public references를 연결한 실제 end-to-end 9-fit 검증이다. 이전 고정 factor/operator 실험만 보고 모든 missing-data PFN을 포기한 것은 아니다. 그럼에도 **19,748 parameters, rank4,4 rounds, 384 training base graphs/768 updates per fit**의 이 학습 후보에서는 관측 값을 유용하게 읽는다는 사전 기준을 충족하지 못했다. Raw native input을 복원했다는 사실이나 graph-only 수준으로 가까워진 성능을 새 feature inference 성공으로 부르지 않는 것이 맞다.

현재 자료는 작은 실제 task/parameter/훈련 예산과 두 synthetic families에 조건부다. PFN 일반의 불가능성, 더 큰 예산에서도 불가능, 비graph 사례에서도 실패 또는 특정 유일한 실패 원인을 주장하지 않는다. 다음 방법을 검토한다면 이 실패와 full/none/shuffle 경향을 출발점으로 새 가설·독립 계획·새 확인 bank를 정해야 하며, 이 final96을 다시 튜닝 bank로 써서 현재 후보를 구제하면 안 된다.

검토한 frozen results SHA256: summary `c3d931308c21e58337bb22885552f7327fc9a92b4a2453e172bae6311a5a9989`; audit `cc77a21a72c7c74f77f3eb9746112e6f9e03e180e663ca54dd5d45bd741ee3d0`; budget `5a36f8d3bb4296d70b24e03a37030d2dde399f118c0e40d604114a50f7e27f52`; paired `e1c0643cd100b7fa2ba1b7c81196a9fbd8a824c15ad19574f04de6631b039fff`.
