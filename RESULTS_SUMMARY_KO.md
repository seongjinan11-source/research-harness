# Missing feature GraphPFN 실험 결과 정리

기준일은2026-11-02다. 목표는 **새 그래프에서 가중치를 다시 학습하지 않고, 불완전한 node feature와 context labels로 분류하는 PFN**이다. Heterogeneous schema로의 확장은 장기 목표다.

**불완전한 관측값과 떨어진 context label을 연결하는 추론 구조는 확보했다. 그 구조를 학습하는 것이 강한 통계 모형보다 언제 유리한지는 아직 입증해야 한다.** Signed likelihood BP, 같은 potential의 자기 role-update, circuit의 neural 대조 대비 개선, Factor의 teacher 분포 근사 개선은 보존한다. 현재 독립적인 메인 메소드로 확정한 후보는 없으며, 이를 실제 분류·품질과 비용·graph/schema 전이의 이점으로 연결하는 것이 남은 문제다.

기록은34개 프로젝트 묶음의144항목이다. 실패·중단42개, 보존32개, 교훈48개, 미실행22개이며 **144번의 독립 실험을 했다는 뜻은 아니다.** 상세 관찰과 재개 조건은 [전체 기록](out/research_log.md), 다른 연구자에게 전달할 설명은 [연구 인계서](RESEARCH_HANDOFF_KO.md)에 있다.

## 결과를 비교할 때 유지할 구분

- 실제 ACM/IMDB 등의 frozen-model 비교와 작은 known-family synthetic 실험은 다른 검증이다. Synthetic shape 변경이나 relation-role reversal을 새로운 semantic schema의 검증으로 부르지 않는다.
- Accuracy, NLL/Brier, teacher predictive KL, posterior KL, 비용은 각각 다른 결과다. KL 감소를 accuracy 개선이나 calibration 개선으로 바꾸지 않는다.
- 동일 bank에서 짝지은 차이로 학습·구조 효과를 판단한다. 아래 서로 다른 bank의 절대 정확도를 빼서 전후 개선으로 쓰지 않는다.
- 최근 circuit/factor/teacher의 context1/2/4는 **총 label 수**다. 실제 graph의 k20은 **클래스당20개**다.
- 원인 미상은 그대로 남긴다. CI가0을 포함하면 우위가 확인되지 않은 것이며, 무효·동등성·비열등성을 증명한 것은 아니다.

## 재독 후 미실행 학습 후보

56편의 정리 항목을 선별하고 SSGP/GGP·CSBM·Spatial Context·PFN separability·PEMix 원문을 다시 대조했다. 새 후보는 missing rate 대신 관측 feature의 기대 Bayes 분류 가치로 사전학습 observation prior를 구성하는 것이다. 기존 regret 재가중·mechanism crossing·subgraph 선택과 차이를 명시했고, value-of-information 자체의 선행연구 중복도 남겼다. **새 학습·성능·novelty 입증이 없는 미실행 가설**이다. [재독 메모와 수식](ideas/decision-value-observation-prior.md), [아이디어 기록](ideas/decision-value-observation-prior.yaml).

## Gaussian/probit의 비교 범위와 선행연구 기록

SSGP(IJCAI2007)와 GGP(NeurIPS2018)의 원문·인용·가정·강점·출처별 한계를 추가했다. 현재 Gaussian/probit는 두 논문의 재현이 아니며, 논문 코드로 실행한 대조는 아직 없다. [SSGP](papers/SSGP.yaml), [GGP](papers/GGP.yaml), [문헌과 결과 대조 보고서](findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md).

기존 primary의 보조 paired 비교에서 직접 Gaussian/probit56.154827%는 비특권적 대조11개보다 accuracy·NLL·Brier가 좋았다. PCFI 대비accuracy +4.589585 [95% CI +2.569802, +6.587119]%p,적응 LP 대비 +3.107972 [95% CI +0.741358, +5.489381]%p다. CI는 개별95%이며 multiplicity 미보정이다. 모든 세부 조건·privileged reference·다른 family와의 승리로 넓히지 않는다. 현재 구현은 강한 고정 통계 기준선/추론 자산으로 보존하며 메인 메소드 확정은 보류한다. 기존 frozen 입력 선택기 No-Go와 직접 추론의 긍정 결과를 구분한다. 새학습0.

## 참 법칙 없는 적응 전달과 직접 통계 추론

새64graph·2,048episodes에서0 training으로 비교했다. Primary는 sources8,k4/class,네 label/feature-law 조합 평균이다. 8/192관측=95.83% whole-node missing이며99.5% Bernoulli 조건이 아니다.

| 경로 | Missing-query accuracy % | NLL |
|---|---:|---:|
| 기존 PCFI Plain·일반 복원 | 51.565242 | 0.715365 |
| 같은 PCFI Plain·50:50 두 복원 평균 | 52.354767 | 0.702633 |
| 같은 PCFI Plain·피처만 사용해 적응 | 52.461961 | 0.704598 |
| 같은 PCFI Plain·피처와 context로 적응 | 52.555985 | 0.705381 |
| Context 적응 LP (graph-only) | 53.046855 | 0.694709 |
| 직접 Gaussian/probit 추론 | 56.154827 | 0.672120 |

Context 적응−50:50 accuracy +0.201218 [95% CI -0.127618, +0.537932]%p, context 적응−feature-only +0.094024 [95% CI -0.066909, +0.271023]%p. Primary=False다. 직접 Gaussian/probit의 동일-query feature-removal 차이는 +2.466964 [95% CI +0.884420, +4.166715]%p로 보존한다. 새 neural 학습이나 novelty의 성과가 아니며 frozen 후보 선택기를 확대하지 않는다. [보고서](findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md), [고정 프로토콜](findings/sources/adaptive-transport-2026-11-02/protocol.json), [구조화 판정](findings/adaptive-transport.yaml).

## 부호 전달과 기존 frozen GraphPFN 진단

기존 matched192-update Mask/PCFI Plain checkpoint6개를 새32graph sign-gauged family에서 추가 학습 없이 비교했다. Label heterophily와 feature heterophily를 교차했으며, graph/label/context/mask를 유지하고 feature 부호 패턴을 바꿨다.

| Primary: t1/g1,k4/class,sources8 | Missing-query accuracy % | NLL |
|---|---:|---:|
| 기존 Mask Plain | 52.405028 | 0.709999 |
| 기존 PCFI Plain | 50.190226 | 0.715966 |
| 같은 PCFI Plain + known-law signed 복원 | 56.026584 | 0.707501 |
| 고정 양의 LP | 46.683098 | 0.796353 |
| 고정 signed LP | 56.260834 | 0.687817 |

Known-law signed−PCFI accuracy +5.836358 [95% CI +2.291311, +9.440175]%p. 같은 PCFI→PFN checkpoint에서 부호 법칙을 반영한 복원의 예측적 가치는 확인했다. 그러나 강한 직접 예측·graph-only 대조를 모두 넘는 기준은 통과하지 못해, 새 학습 모델이 필요하다는 근거로 확대하지 않는다. Known-law g는 privileged diagnostic 정보이며 학습된 adaptation이나 새 메소드가 아니다. [보고서](findings/sources/sign-transport-2026-11-02/REPORT_KO.md), [계산 정의](findings/sources/sign-transport-2026-11-02/METHOD_KO.md), [구조화 판정](findings/sign-transport.yaml).

## 내부 고정 PCFI의 새 graph 재검증

기존 checkpoint6개를 새128graph bank에서 추가 학습 없이 확인했다. 원64graph bank를 보고 선택한 후보이며, 생성 family와 학습 seeds11/12/13은 같다.

| 방법 | Missing-query accuracy % | NLL |
|---|---:|---:|
| external | 49.821540 | 0.856885 |
| internal_fixed | 50.041098 | 0.856498 |

Fixed−external accuracy +0.219558, 95% CI [-0.331926, +0.796665]%p. 새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다. NLL/Brier와 비용은 별도로 보고한다. [보고서](findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md), [사전 프로토콜](findings/sources/internal-pcfi-confirmation-2026-11-02/protocol.json), [구조화 판정](findings/internal-pcfi-confirmation.yaml).

## 최신 실제 PCFI 내부화 실험

공개 GraphPFN에서 실제 PCFI를100회 전파+1회 상관 보정으로 구현하고, 외부 처리·내부 고정·context router·strict label-blind router를 각각3seeds/256 updates로 새 학습했다. 이전 generic MLP refiner 재실행이 아니다. Fresh64 final graphs의 uniform/clustered90/99.5%에서 자기 피처 없는 query가 primary다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| external | 49.24327 | 0.867128 |
| internal_fixed | 49.90070 | 0.863062 |
| context | 49.41351 | 0.863255 |
| label_blind | 49.69055 | 0.863001 |

context-external accuracy%p +0.170247 CI[-0.557668,+0.866045], NLL -0.003872 CI[-0.010121,+0.001957]; context-internal_fixed accuracy%p -0.487184 CI[-0.861821,-0.130108], NLL +0.000193 CI[-0.001404,+0.001615]; context-label_blind accuracy%p -0.277040 CI[-0.704964,+0.174764], NLL +0.000254 CI[-0.001276,+0.001713]. 사전 primary=False, full2%p guard=True, 확대 기준=False. 이 구현의 확대는 보류하며, 고유 실패 원인은 미상이다. Context 대조는 labels-only가 아니라 PFN feedback 경로 전체다. [보고서](findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md), [계산 정의](findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md), [구조화 판정](findings/internal-pcfi.yaml).

## 최신 feature-law Factor 실험

새15개 모델을 동일2048 tasks/256 updates/3 seeds/query CE로 학습했다. Gaussian 평균·heavy-tail 평균·분산 차이 law를 모두 학습하고, 새 graph와 미학습 parameter tuple 조합에서 cell만 바꿨다. Graph encoder/factor는 한 역사적 checkpoint로 고정했다. Law 이름 자체의 미관측 전이나 공개 GraphPFN 개선 모듈은 아니다.

| Primary: heldout96 graphs, 자기 피처 없는 query | Accuracy % | NLL | Reference KL |
|---|---:|---:|---:|
| 기존 부호 제약 Factor | 64.6281 | 0.643004 | 0.022515 |
| Flexible-static | 64.4125 | 0.643406 | 0.021705 |
| Flexible-adaptive | 64.4675 | 0.650797 | 0.027830 |
| 고정 law 주변화 exact | 63.4065 | 0.631744 | 0.004696 |
| 고정 law 주변화 BP | 64.0811 | 0.646232 | 0.023510 |
| 새로 학습한 Mask PFN | 53.4883 | 0.693767 | 0.068593 |
| 새로 학습한 PCFI-PFN | 52.9697 | 0.693928 | 0.068545 |

Adaptive−static accuracy +0.054977 CI[-0.976294,+1.037881]%p, adaptive−original -0.160659 CI[-1.202481,+0.821535]%p다. 두 CI에0을 포함해 적응 primary gate를 통과하지 못했다. 이 cell 교체만으로 메인 메소드가 된다는 가설은 닫고 새 loss/head/학습량으로 구제하지 않는다. 원인은 미상이며 부호 제약 제거 자체나 모든 feature-law 학습을 기각하지 않는다. 모든 seed/episode 평균 뒤 **base graph**를 bootstrap했다.

Adaptive 6.759ms, fixed exact 8.861ms, fixed BP 10.077ms. Reference KL은 broad known-family exact reference에 대한 값으로 참 OOD Bayes KL이 아니다. [보고서](findings/sources/feature-law-2026-11-02/REPORT_KO.md), [수식](findings/sources/feature-law-2026-11-02/METHOD.md), [구조화 판정](findings/feature-law-factor.yaml).

## 실제 그래프에서 확보한 강한 비교

ACM/IMDB는 target node feature의99.5% whole-node missing, labels20/class, 전체 public test의 평균 accuracy다. PCFI/FISF는 원본 좌표에서 대치한 뒤 이 실험의 SVD·typed paths·고정 공개 GraphPFN을 사용했다. 원 논문의 downstream classifier 성과나 새로운 데이터셋 holdout은 아니다.

| 방법 | ACM accuracy % | IMDB accuracy % |
|---|---:|---:|
| Missing zero + typed paths + GraphPFN | 72.49 | 40.41 |
| PCFI + typed paths + GraphPFN | 75.45 | 42.61 |
| FISF + typed paths + GraphPFN | 74.36 | 44.55 |
| Feature 없는 metapath LP 평균 | 85.55 | 35.41 |
| 실제 관측값을 제거한 FISF + GraphPFN | 74.14 | 44.51 |

ACM에서는 구조·라벨만 사용하는 LP가 매우 강했다. IMDB의 PFN 우위는 실제 관측값을 제거해도 거의 유지됐으므로, 그 우위를 physical feature의 추가 정보 때문이라고 설명할 근거는 약하다. 이것이 아래 Gaussian synthetic family에서도 feature가 무용하다는 뜻은 아니다.

DBLP의 native source-block 비교에서는 k5 full의 cross-type paths89.96%, adapted HetGFD89.92%였고, k20 full은 둘 다93.55%였다. 기존 path 입력을 넘는 adapted diffusion의 독립 이득이 작았다. Feature가 원래 없는 Freebase에서는 RDB-PFN DFS5 accuracy56.60%, 구조 입력 GraphPFN56.13%였으며, 이는 feature-missing 강건성 검증과 구분한다.

근거: [원본 좌표 대치](findings/sources/graphpfn-2026-11-01/raw.md), [관측값 개입](findings/sources/graphpfn-2026-11-01/feature_intervention.md), [LP·relational 비교](findings/sources/graphpfn-2026-11-01/history.md), [native block](findings/sources/graphpfn-2026-11-01/native_blocks.md). 결과를 보고 새 그래프의 predictor를 자동 선택하는 규칙을 얻은 것은 아니다.

## 최근 여섯 학습 실험의 실행 범위

| 실험 | 새 학습 모델 | 모델당 성공 updates | 평가 bank와 주 조건 |
|---|---:|---:|---|
| Feature memory 분리 | 4 arms × 3 seeds =12 | 512 | 새64 test graphs, uniform/clustered node90·99.5%, accuracy primary |
| Conditional tree circuit | 3 arms × 3 seeds =9 | 256 | 새192 graphs 중 matched48, node uniform/clustered50·90%, NLL primary |
| Disjoint Factor PFN | 3 arms × 3 seeds =9 | 256 | 새96 graphs 중 matched48, observed nodes3/6, context와 feature source 분리, accuracy primary |
| Exact Grid6 teacher | 2 arms × 3 seeds =6 | 256 | 기존 train bank 재사용, 새96 graphs 중 matched48, Mask teacher KL primary; 기존 CE9개 재평가 |
| Feature-law cell Factor | 5 arms × 3 seeds =15 | 256 | 새192 graphs 중 heldout96, 자기 피처 없는 query accuracy primary,3개 law 모두 학습 |
| 실제 PCFI 내부화 | 4 arms × 3 seeds =12 | 256 | 새64 final graphs, node90/99.5% 자기 피처 없는 query accuracy primary, 외부/내부 고정/context/label-blind |

합계는 **새 모델63개와 성공 updates19,200회**다. Teacher 실험의 기존 CE9개 재평가를 새 학습으로 더하지 않았다. 이 합계는 최근 여섯 실험만 포함하며 전체 연구의 총 학습량이 아니다. 기존 circuit·Factor·teacher는12 nodes, binary classes,2 Gaussian features,2 relations의 family다. Cell 실험은 같은 graph 규모에서 feature law3종을 포함한다. Factor·teacher·cell 실험의3/6 observed nodes는75/50% node missing이며99.5% 조건이 아니다. 이 네 소형 실험에서 원래 graph의 edge/nonedge는 완전히 관측된다. 최신 PCFI 내부화는 별도 official graph/SCM family와 공개 GraphPFN이며12-node Factor의 절대 정확도와 직접 비교하지 않는다.

### Feature memory 분리

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| 기존 label-context memory | 52.3336 | 0.687776 |
| 전체 node feature memory | 51.9552 | 0.689573 |
| 관측 feature-group memory | 52.2888 | 0.692156 |
| 실제 PCFI 입력으로 학습 | 53.9246 | 0.683446 |

Observed−context accuracy−0.0448%p의95%CI는[−1.5015,+1.4608]였다. NLL은 오히려+0.004380,CI[+0.000906,+0.007896] 높았다. **현재 routing 변경의 독립 이득은 미확인**이다. 두 좌표의 native feature group과 frozen attention projection을 유지한 비교이며, full sparse cell-token 모델을 학습한 결과는 아니다. [보고서](findings/sources/feature-memory-2026-11-02/REPORT_KO.md)

### 직접 주변화하는 conditional circuit

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| 학습된 circuit | 61.3839 | 0.605128 |
| 공개 Mask GraphPFN | 53.4846 | 0.687080 |
| Direct graph predictor | 51.1512 | 0.693134 |
| Grid6-BP | 69.5602 | 0.540755 |

Circuit−Mask accuracy+7.8993%p,CI[+3.8890,+11.7908], NLL−0.081952,CI[−0.101972,−0.062402]였다. **Neural 대조 대비 개선은 확인했지만 Grid6-BP보다 accuracy·NLL이 나빴다.** Circuit teacher KL0.064716과 median12.369ms vs Grid6-BP13.443ms는 사전 .01/2배 기준을 통과하지 못했다.

모델 내부의 합 규칙·정규화·결측 leaf 적분은 검증했다. 이것이 cyclic SBM의 참 posterior를 정확히 계산한다는 뜻은 아니다. GSPN/MaM에서 착안한 별도 conditional tree mixture이며 두 논문의 재현도 아니다. [보고서](findings/sources/direct-marginal-2026-11-02/REPORT_KO.md)

### Disjoint Factor PFN

Node-class latent와 feature-column latent를 관측 cell factor로 연결하고, graph의 edge/nonedge factors와 context clamp를 함께 BP로 처리했다. 같은 potential의 Bethe score로 role weights를 계산하며 별도 posterior head는 없다. Numeric completion을 먼저 만들지 않는다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| 학습된 Factor PFN | 65.4004 | 0.628722 |
| 공개 Mask PFN | 57.0974 | 0.676368 |
| PCFI 입력으로 학습한 PFN | 55.2304 | 0.677401 |
| Fixed-mode factor BP | 66.0053 | 0.621022 |
| Grid6-BP | 65.9557 | 0.555143 |

Factor−Mask accuracy+8.3030%p,CI[+2.7750,+14.0753], Factor−PCFI+10.1700%p,CI[+4.4253,+16.3513]였다. **Neural accuracy 개선은 보존한다.** Fixed-mode factor 대비 accuracy/NLL의 추가 이득은 확인하지 못했고, Grid6-BP 대비 NLL+0.073580,CI[+0.019579,+0.150308]로 나빴다.

이 bank의 사전 feature 제거는 accuracy−6.1343%p,CI[−12.9630,+0.9259]로 제거 gate 미통과였다. Feature shuffle과 graph permutation은 성능을 낮췄다. Query 자신도 feature가 없는 secondary에서는 Factor가 두 neural 대조보다 accuracy와 NLL에서 좋았지만 fixed factor의 추가 가치는 확인하지 못했다. [보고서](findings/sources/disjoint-factor-2026-11-02/REPORT_KO.md), [수식](findings/sources/disjoint-factor-2026-11-02/METHOD.md)

### Exact Grid6 teacher로 감독만 교체

기존2048 train tasks와 원래 초기화·학습 순서·예산을 유지하고, sampled-label CE를 exact Grid6 query predictive KL로 바꿨다. Teacher는 actual task parameter나 sampled query/helper label을 받지 않는다.

| 학생 | 감독 | Accuracy % | NLL | Grid6 exact KL |
|---|---|---:|---:|---:|
| 공개 Mask PFN | CE | 55.3475 | 0.678973 | 0.104788 |
| 공개 Mask PFN | Exact teacher | 54.6255 | 0.679253 | 0.104134 |
| Factor PFN | CE | 69.8468 | 0.575272 | 0.067257 |
| Factor PFN | Exact teacher | 69.5877 | 0.568642 | 0.049731 |
| Grid6-BP | Model-based 추론 | 68.3284 | 0.551872 | 0.000010 |

**Mask primary는 미통과**다. Teacher KL 차이−0.000654,CI[−0.002034,+0.000583]였고 accuracy/NLL 차이도0을 포함했다. Mask median59.735ms, Grid6-BP12.425ms로 latency 이점도 없었다.

Factor secondary의 teacher KL은−0.017526,CI[−0.026031,−0.010538]로 약26.1% 감소했다. 이 확률분포 근사 개선은 보존한다. 그러나 accuracy−0.2590%p와 NLL−0.006630의 CI는0을 포함했다. **Query 자신도 feature가 없는 secondary에서는 accuracy가64.7164%→63.5556%, −1.1609%p,CI[−2.4190,−0.0482]로 하락했다.** 이 보조 비교는 multiplicity 미보정이다.

새 bank의 사전 context2/observed6에서 physical feature 제거는 distilled Factor−11.9213%p,CI[−19.0972,−4.8611]였고 기존 CE Factor도−12.0370%p,CI[−19.6759,−4.2824]였다. **관측값과 떨어진 label context를 연결하는 기능은 보존하지만, 그 기능이 distillation으로 새로 생겼다는 뜻은 아니다.** 이전 bank의 제거 미통과 결과도 그대로 유지한다.

Teacher2048개 offline 준비는15.313s였다. Factor median8.764ms는 Grid6-BP보다 약1.42배 빠른 측정값이며 teacher KL0.049731은 .01을 넘었다. 사전 압축 기준은 미통과했다. [최종 판정](findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md), [전체 보고서](findings/sources/grid6-teacher-2026-11-02/REPORT_KO.md), [학습 수식](findings/sources/grid6-teacher-2026-11-02/METHOD.md)

## 이전 시도에서 무엇을 남겼나

아래는 서로 다른 조건의 요약이다. 자세한 원인 수준·닫은 범위·재개 조건은 연결된 구조화 기록에 있다.

| 시도 | 결과와 현재 판정 | 기록 |
|---|---|---|
| Missingness prior 확대 | Label-dependent에서 작은 개선이 있었으나 강한 공식 전처리 대조에 범용 우위가 없었다. 현재 갈래 중단. | [기록](findings/missingness-prior.yaml) |
| PCFI/FISF input views | Cora/CiteSeer 평균69.65→69.73%로 작았다. PubMed73.00% vs FISF71.50% 신호는 secondary로 보존. | [기록](findings/input-views.yaml) |
| Source-overlap 보정 | Covariance proxy는 반응했지만 accuracy 증분+0.032%p의 CI는0 포함. 현재 보정 모듈의 독립 가치 미확인. | [기록](findings/source-error.yaml) |
| Feature↔label 양방향 feedback | Prototype matched72.66% vs 일방향72.72%, 차이 CI는0 포함. 반복 확대 보류. | [기록](findings/feature-label-coupling.yaml) |
| LP/PFN 선택·50:50 혼합 | Oracle 보완성은 있으나 IMDB Local41.92% vs PFN44.55%, 혼합42.54%. 사후 선택·고정 혼합 중단. | [선택](findings/reliability-selector.yaml), [혼합](findings/path-selection-fusion.yaml) |
| Frozen full/neutral ratio | IMDB34.45%로 LP35.41%, PFN44.55%보다 낮았다. Neutral은 유효한 주변화 분포가 아니었다. 중단. | [기록](findings/counterfactual-ratio.yaml) |
| LP offset·내부 conditioning | 내부 conditioning의 offset 대비 증분은 작고 Plain 대비 안정적 우위가 없었다. 튜닝·확대 중단. | [기록](findings/lp-conditioning.yaml) |
| Relation evidence head | Conflict Separate86.09% vs Mean85.34%, Plain88.96%. 출처 보존의 Mean 대비 이득은 있으나 Plain보다 낮아 종료. | [기록](findings/relation-evidence.yaml) |
| CRR 전역 router 제안 | Plain이 이미 context에 따른 role adaptation을 했다. 새 router 학습 실패가 아니라 중복 제안 철회. | [기록](findings/plain-role-reversal.yaml) |
| Same-context witness 곱셈 | IMDB에서 공통 seed support가 약96.1%의 test rows에 없었다. 고정 product가 evidence를 없애 현재 규칙 중단. | [기록](findings/binding-product.yaml) |
| Relational compiler | Unseen compositions61.59% vs relational features+TabPFN85.18%, ProgramGrid87.73%. 현재 compiler 확대 보류. | [기록](findings/relational-program.yaml) |
| Reference-regret curriculum | 같은 update의 Random보다 작은 NLL 이득은 있었으나 비용을 맞춘 Random·Loss sampling을 넘지 못했다. | [기록](findings/reference-regret.yaml) |
| 관측-aware 부분그래프 정책 | Uniform NLL1.07193 vs 경로 규칙1.05353, context prior0.85241. Mask 효과는 있으나 강한 대조 대비 확대 보류. | [기록](findings/observation-budget.yaml) |
| Task-trained numeric imputer | PFN 앞/중간 refiner를 실제 학습했으나 같은 예산 Mask/PCFI와 사전 대치를 안정적으로 넘지 못했다. | [기록](findings/joint-imputation-pfn.yaml) |
| Gaussian completion 여러 개의 예측 평균 | Single mean보다 NLL은 개선됐으나 variance를 맞춘 random-fill을 안정적으로 못 넘었다. 현재 conditional completion의 고유 가치 미입증. | [Sampling](findings/feature-marginal-pfn.yaml), [Null 대조](findings/random-fill-controls.yaml) |
| Mechanism-crossed prior | 원래 NLL gate는 실패. Off-diagonal clustered accuracy+3.434%p,CI[+0.960,+6.152]는 사후 신호로 보존하며 독립 재현은 없다. | [기록](findings/mechanism-crossed-prior.yaml) |
| Native hetero diffusion | DBLP adapted HetGFD의 cross-type paths 대비 이득이 작았다. 원래 HetGFD/HGT 전체나 모든 heterograph를 기각한 결과가 아니다. | [기록](findings/hetero-baselines.yaml) |

## JR와 BP와 EAJ에서 남은 연산

이 갈래는 초기 feature-free12-node SBM의 결과다. 최근 Gaussian feature 실험과 절대 수치를 비교하지 않는다.

| 단계 | 확인한 결과 | 남기는 판정 |
|---|---|---|
| Joint role hypotheses | Exclusive에서 Large Plain 대비+3.20%p, factorized 대비+1.76%p. Learned posterior−known prior는+0.14%p. | Role 제약 보존 효과는 있으나 학습 posterior의 일반적 우월성은 아님. |
| Exact Bayes 진단 | Exclusive context4에서 역할 정보0.12447 nats, exact role update 기대 NLL 이득0.02278. | Context 정보는 존재했지만 당시 conditional decoder 병목도 컸음. |
| Role-conditional distillation | Conditional KL은 낮아졌지만 최종 predictive 증분의 CI는0 포함. Posterior 감독 추가는 세 prior 모두 predictive KL을 악화시킴. | 당시 decoder에서 loss만 추가하는 확대는 중단. |
| Positive diffusion 교체 | All-context-A일 때 p(A)≥.5인 함수족 제한을 signed edge/nonedge decoder가 넘음. | 표현력 제한은 해결한 중간 병목이며 현재도 미해결인 것처럼 반복 제안하지 않음. |
| Learned likelihood BP | Exclusive matched predictive KL0.052586→0.011013, predictive-only 학습. | 실제 작동한 연산. 추가 likelihood auxiliary loss의 고유 이득은 없음. |
| Coherent role/parameter update | Coupled high-density에서 frozen-grid0.078075→joint-grid0.017615. Same-marginal product 대비 차이는−0.001434. | 전체 개선을 joint dependence 하나에 귀속하지 않음. |
| EAJ 학습 | Independent heads보다 좋았지만 exact·강한 constant 기준선을 넓게 넘지 못함. 직접 predictive 학습도 posterior 복원과 다른 결과였음. | 현재 EAJ 확장 중단. |
| EAJ를 직접 predictive 학습 | 여섯 조건의 grid predictive KL을 약15.7~31.9% 개선했으나 posterior KL은 악화. Signal-high에서는 strong constant+role-update보다 NLL·Brier가 나빴음. | 학습 목표 변경의 실제 증분은 보존하되 메소드 구제로 해석하지 않음. |
| R1 자기-update | R1−Constant KL−0.001744,CI[−0.002072,−0.001425]. 그러나 R1 평균KL0.020488 vs Grid6 0.011939. | 자기-update 개선은 보존하고 독립 메인 메소드 확대는 중단. |
| R1 비용 | 단일 graph5.17ms vs Grid6 4.66ms. Batch16 추가 작업 메모리1.44 vs64.04MiB. | 메모리 절감은 실제이나 큰 latency·실제 처리 규모 우위는 미확보. |

근거: [초기 JR](findings/joint-role-initial.yaml), [BP와 EAJ 전체 기록](findings/llbp-eaj.yaml). Nonedge는 “관측해 연결 없음”이고 unknown pair와 다르다. Edge observation missing과 node-feature missing을 섞어 해석하지 않는다.

## 확인된 병목과 아직 모르는 원인

| 확인 수준 | 내용 | 해석 |
|---|---|---|
| 당시 함수족에서 확인 | Positive-only diffusion은 특정 반대-class 예측을 표현할 수 없었음. | Loss만 바꿔 해결할 수 없었고 이후 decoder를 교체했음. |
| 현재 구현에서 확인 | 기본 sample attention의 직접 KV는 label-context 중심이며 graph/residual 경로는 별도로 있음. | Unlabeled feature에 대한 접근 제한은 사실이나 전체 실패 원인이라는 증명은 아님. 단순 memory routing 실험은 개선되지 않았음. |
| 진단이 지지 | 고정 common-witness product는 sparse support를 크게 없앴음. | 모든 binding·logical executor가 실패한다는 뜻은 아님. |
| 원인 미상 | 최신 Mask distillation·feature memory·numeric imputation의 미개선. | Frozen projection, 표현력, input 경로, prior, 최적화, budget을 단일 원인으로 확정하지 않음. |
| 원인 미상 | Factor의 Grid6 predictive gap. | Potential fit·함수족·finite BP/Bethe 근사·학습량을 분리하지 않았으므로 BP만 원인이라고 단정하지 않음. |

## 실제로 아직 하지 않은 것

주요 미실행 범위는 아래와 같다. 설계·문헌 검토까지 한 항목을 학습 실패로 세지 않는다.

| 미실행 범위 | 가까운 기존 실행과의 차이 |
|---|---|
| Full sparse observed-cell GraphPFN와 projection까지 학습 | Feature memory 실험은 frozen projection의 두 좌표 group routing이었다. |
| Context label에 조건부인 joint feature posterior와 공유 encoding | Gaussian completion은 label-blind surrogate였고, coherence 설계는 산술 예시까지였다. |
| 가변 feature/schema의 tractable circuit과 독립 MaM 학습 | 현재 circuit은 고정2-feature·2-relation의 별도 tree mixture였다. |
| Observation-aware walk 모델과 원래 Wander의 예산 일치 비교 | 부분그래프 선택 뒤 PCFI+고정 PFN과 다른 계산이다. |
| Node-feature missing과 구분한 MNAR edge-observation BP | 현재 feature 실험의 graph pairs는 모두 관측됐다. |
| Unseen feature law·semantic schema·실제 graph·큰 graph의 teacher/Factor 전이 | 새 cell 실험은 law3종을 모두 학습한 parameter tuple 전이이며 이 범위의 검증은 아니다. |

미실행이라는 사실이 유망함이나 재학습의 근거는 아니다. 새 후보는 어떤 정보·함수족·추론 연산을 바꾸는지 먼저 명시하고, PCFI/FISF-trained PFN·LP·fixed factors·해당 family의 명시적 추론과 비교해야 한다. 이미 끝난 구성에 이름·head·loss만 추가한 반복과 구분한다.

## 다음 후보가 설명해야 할 학습의 역할

분리된 feature/label source 연결, signed edge/nonedge evidence, 같은 potential의 자기 role-update는 이미 구현한 기능이다. 다음 후보는 현재 함수족으로 부족한 **feature–relation 의존성**을 명시하고, 무엇을 어떻게 학습해 새 graph에서 재학습 없이 활용하는지 설명해야 한다.

현재 Factor 학생도 neural potential과 관측값 계수를 학습한다. 고정된 것은 모든 likelihood 숫자가 아니라 binary class·binary feature-column orientation·pairwise Ising형 factor라는 주요 가정이다. 따라서 기존 결과의 “피처 의미 추론”은 이 작은 방향 잠재변수 범위로 읽어야 한다. 더 복잡한 feature 법칙을 추가했다는 사실만으로 학습의 필요성이나 novelty가 생기지는 않으며, 그 법칙을 다루는 강한 통계 추론과 비교해야 한다.

CE Factor에도 이미 feature 제거 효과가 있었다. Distillation은 teacher predictive KL을 더 낮췄지만 연결 연산 자체의 발명이나 실제 분류의 범용 이득으로 해석하지 않는다. [연구 판정 원문](findings/sources/factor-position-2026-11-02/user_review.md), [구현 범위](findings/sources/disjoint-factor-2026-11-02/METHOD.md), [teacher 효과](findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md)를 구분한다.

## 자료를 읽거나 전달하는 순서

1. 이 요약으로 전체 결과와 범위를 확인한다.
2. [연구 인계서](RESEARCH_HANDOFF_KO.md)의 요청문으로 독립 검토를 요청한다.
3. 필요한 수치는 연결된 보고서와 [144개 전체 기록](out/research_log.md), `findings/*.yaml`에서 확인한다.
4. [최신 그래프](out/graph.html)와 [엑셀](out/research_harness.xlsx)에서 논문과 실패 병목의 검색 연결을 본다. 연결은 해결 판정이 아니다.

공유용 `graphpfn_review_packet_2026-11-02_v11.zip`에는 이 요약, 인계서, 모든 papers/findings와 보존 source snapshots를 넣는다. Checkpoint와 raw prediction 실행 환경은 제외한다. 과거 v1/v2/v3/v4/v5/v6/v7/v8/v9/v10 ZIP은 이전 snapshot으로 유지한다. 외부 모델은 로컬 C 드라이브를 읽을 수 없으므로 ZIP이나 문서를 실제로 첨부해야 한다.
