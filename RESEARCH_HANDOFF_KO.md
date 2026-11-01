# Missing-feature GraphPFN 연구 인계서

기준일: 2026-11-02. 실험 기록 기준 commit: `06d58de`.

## 검토받고 싶은 문제

우리는 **노드 피처가 심하게 결측된 새 그래프에서, target graph 재학습 없이 예측하는 GraphPFN**을 만들고 싶다. 장기적으로 다른 feature 차원과 heterogeneous schema까지 다루되, 현재 실험은 주로 작은 synthetic graph와 일부 실제 그래프에서 수행했다. 여기서 feature missing과 edge observation missing은 다른 문제다.

새로운 이름이나 모듈을 늘리는 것보다, **기존 강한 예측기가 놓치는 정보를 회수하는 학습·추론 원리**를 찾고 싶다. 현재 독립적인 메인 메소드로 확정한 후보는 없다. 그렇다고 모든 시도가 무효인 것은 아니다. Signed likelihood BP, 같은 potential에서의 role update, 최근 conditional circuit의 neural baseline 대비 개선은 실제로 확인했다. 이 효과를 강한 model-based baseline보다 경쟁력 있는 방법으로 연결하지 못했다.

이번 검토는 새로운 실험을 즉시 실행하라는 요청이 아니다. 아래 사실과 실패 범위를 검토하고, 설득력 있는 후속 메소드가 있는지 독립적으로 판단해 달라.

## 먼저 읽을 자료

이 파일만 읽어도 핵심 판단은 가능하다. 세부 확인은 다음 순서가 좋다.

1. [최근 circuit 보고서](findings/sources/direct-marginal-2026-11-02/REPORT_KO.md)
2. [최근 feature-memory 보고서](findings/sources/feature-memory-2026-11-02/REPORT_KO.md)
3. [누적 실패·보존·미실행 기록](out/research_log.md): 필요한 갈래만 검색한다.
4. `findings/*.yaml`: 각 항목의 `closed_scope`, `not_closed`, `reason_status`, `comparison_floor`, `reopen_when`을 확인한다.
5. `papers/*.yaml`: 확인한 논문의 장치·가정·한계·원문 링크가 있다. 자동 매칭은 해결 근거나 novelty 판정이 아니다.

저장소에는 논문 32편, 연구 기록 118항목, 아이디어 8개가 있다. 118항목 전부가 실패가 아니다. `failure`, `preserved`, `lesson`, `untested`를 구분한다. ZIP에는 본문·구조화 기록·핵심 보고서를 담았고 대형 raw prediction, checkpoint와 모든 원문 snapshot은 넣지 않았다. 일부 `source_refs`는 전체 저장소에서 확인해야 한다.

## 이전 실험에서 남긴 판단

아래는 서로 다른 과제와 예산의 누적 결과다. 표의 갈래끼리 절대 성능을 직접 비교하면 안 된다. 원인이 분리되지 않은 항목은 원인 미상으로 남겼다.

| 갈래 | 실제로 확인한 것 | 닫은 범위와 남은 범위 | 기록 |
|---|---|---|---|
| PCFI/FISF → GraphPFN 및 입력 결합 | 일부 데이터의 소폭 개선이 전체 조건에서 안정적으로 이어지지 않았다. | 단순 입력 view 결합을 범용 메인 기여로 삼기 어렵다. 모든 imputation 방법을 기각한 것은 아니다. | `input-views.yaml`, `missingness-prior.yaml` |
| LP/PFN 혼합·confidence·Brier selector | Oracle 보완성은 컸지만 실제 context-only selector가 최강 단일 모델을 안정적으로 넘지 못했다. | 이 설정의 사후 선택·평균화는 중단했다. Brier와 accuracy의 최적 predictor가 다를 수 있다. | `reliability-selector.yaml`, `path-selection-fusion.yaml` |
| 관측 feature intervention | 과거 IMDB, 99.5% node missing에서 FISF+PFN은 실제 feature를 제거해도 44.55%→44.51%였다. | 그 조건의 PFN 우위를 실측 feature 때문이라고 설명할 수 없다. 모든 그래프·노드의 feature가 무용하다는 뜻은 아니다. | `feature-intervention.yaml` |
| Frozen counterfactual ratio | LP에 PFN full/neutral 확률 비율을 곱하는 방법은 실패했다. | Neutral PFN은 feature를 정확히 주변화한 분포가 아니다. 비율을 새 feature evidence로 간주하는 근거가 부족했다. | `counterfactual-ratio.yaml` |
| LP conditioning·relation evidence head | 작은 내부 conditioning 이득, Separate>Mean은 있었지만 강한 Plain을 안정적으로 넘지 못했다. | 출처 보존이나 conditioning 일반의 무효가 아니다. 해당 head·offset을 메인 방법으로 확대하지 않는다. | `lp-conditioning.yaml`, `relation-evidence.yaml` |
| 전역 relation router | Plain이 이미 context에 따라 relation role reversal에 적응했다. | 새 전역 router 제안은 중복 때문에 철회했다. 이를 새 모델을 학습해서 실패한 결과로 세지 않는다. | `plain-role-reversal.yaml` |
| Feature↔label feedback·source overlap 보정 | 추가 효과가 없거나 극히 작았다. | 해당 연산의 독립적인 가치가 확인되지 않았다. 정확한 단일 원인은 확정하지 않았다. | `feature-label-coupling.yaml`, `source-error.yaml` |
| Joint relation hypotheses | Exclusive role 제약에서 강한 Plain과 factorized 모델을 이긴 설정이 나왔다. | Joint 제약 효과는 보존한다. 모든 prior에서의 우위나 학습된 posterior 자체의 이득은 아니었다. | `joint-role-initial.yaml` |
| Learned-likelihood BP | Positive diffusion의 표현력 제약을 signed edge/nonedge likelihood로 완화했고 예측이 크게 개선됐다. Unknown pair를 nonedge로 간주하면 안 됐다. | 연산의 효과는 보존한다. Learned potential은 constant potential보다 항상 좋지 않았다. | `llbp-eaj.yaml` |
| EAJ·joint role–parameter evidence | Coherent update는 유효했지만 learned evidence의 오차가 컸다. 별도 posterior head를 제거한 R1도 실제 개선이었다. | 작은 known-family Joint Grid6가 품질·비용에서 강했다. 현재 EAJ/R1을 독립 메인 메소드로 확대하지 않는다. BP 일반을 기각한 것은 아니다. | `llbp-eaj.yaml` |
| 부분그래프 선택 | Mask-aware 학습 정책은 일부 약한 대조보다 좋았지만 관측 source 경로 규칙과 class-prior NLL을 넘지 못했다. | 선택 이후 PCFI·고정 GraphPFN은 그대로였던 실험이다. PCFI 자체를 바꾼 실험과 구분한다. | `observation-budget.yaml` |
| PFN-aware joint imputation | 실제 공개 PFN의 query CE로 imputer를 앞단/중간에서 학습했지만 강한 대조 대비 독립적 성과는 부족했다. | 현재 numeric imputer 구성의 결과다. 완전한 context-conditioned feature posterior는 미실행이다. | `joint-imputation-pfn.yaml` |
| 여러 completion 후 PFN 평균 | Point fill 대비 개선은 있었으나 matched random-fill 대조를 넘는 조건부 completion의 고유 가치는 확인하지 못했다. | 여러 숫자를 넣어 생긴 효과를 posterior inference의 성과로 부를 수 없다. | `feature-marginal-pfn.yaml`, `random-fill-controls.yaml` |
| Mechanism-crossed prior | 원래 NLL gate는 실패했지만 off-diagonal clustered accuracy +3.434%p 신호를 보존했다. | 사후 보조 결과이며 독립 재현·다중 비교 보정이 없다. 이후 memory 실험은 이 효과의 재현이 아니다. | `mechanism-crossed-prior.yaml`, `MCP-P02` |

일반적으로 **추가 학습이 부족해서인지, 정보가 없는지, 표현력·최적화·prior 중 무엇이 원인인지 분리하지 않은 경우가 많다.** 결과가 나쁘다는 이유만으로 그럴듯한 원인을 확정해서는 안 된다.

## 최근 완료한 feature memory 실험

공개 GraphPFN 1.3/LimiX의 기본 sample attention은 label-context 노드를 key/value로 읽는다. Query 자신의 feature residual과 graph adapter 경로는 있으므로 unlabeled node의 feature가 완전히 차단됐다는 뜻은 아니다. Feature memory와 label context를 분리하면 나아지는지 실제 학습으로 비교했다.

- 새 64 train/64 test base graphs, 3 training seeds, 4 arms, 각 512 성공 updates.
- 같은 초기화·과제 순서·파라미터·query CE. 원래 transformer는 고정했다.
- 공개 모델의 두 feature 좌표 묶음을 memory 단위로 사용했다. 완전한 sparse cell-token 모델은 아니다.
- `context`: 기존 방식. `all_nodes`: 모든 노드의 feature memory. `observed`: 관측값이 있는 group만 memory. `pcfi`: 실제 PCFI 입력으로 학습한 기존 방식.
- Primary는 공식/constructed off-diagonal family의 whole-node uniform/clustered 90/99.5% 조건 평균 accuracy다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| Context | 52.3336 | 0.687776 |
| All nodes | 51.9552 | 0.689573 |
| Observed memory | 52.2888 | 0.692156 |
| PCFI | 53.9246 | 0.683446 |
| LP | 53.3203 | 0.777961 |

Observed−Context accuracy는 −0.044759%p, graph 단위 paired 95% CI [−1.501516, +1.460826]이었다. NLL은 오히려 +0.004380 [ +0.000906, +0.007896 ] 높았다. 현재 routing 변경만으로 개선된다는 가설은 지지되지 않았다. Frozen projection, grouping, 학습량과 task 정보량 중 고유 원인은 분리하지 않았다.

## 최근 완료한 직접 주변화 circuit 실험

### 모델과 입력

소형 typed-SBM에서 **normalized graph-conditioned tree mixture**를 새로 구현했다. GSPN/MaM에서 착안했지만 두 논문의 재현은 아니다. MaM을 별도로 학습한 결과도 아니다.

- 12 nodes: target 8/helper 4, relation 2, binary label, 연속 Gaussian feature 2.
- 세 role-prior family인 Independent/Exclusive/Coupled의 descriptor는 모든 학습 모델에 제공했다.
- Graph의 edge/nonedge는 완전히 관측된다. 이 실험에서 결측은 node feature다.
- 구조 encoder는 G와 family만 보고 파라미터를 만든다. 실제 feature와 context label은 leaf evidence로 넣는다.
- 원본 변수 scope가 겹치지 않는 정규화된 tree mixture이고, 결측 Gaussian leaf는 적분값 1로 처리한다.
- 같은 conditional model 안의 정규화·합 규칙·연속 적분을 약 10^-15 오차로 검증했다. 원래 cyclic SBM posterior가 정확하다는 뜻은 아니다.
- Circuit 56,385 parameters. 직접 graph predictor와 공개 Mask GraphPFN을 비교했다. 공개 PFN은 4,012,259 trainable parameters이며 typed context support도 받는다. 파라미터 수는 같지 않다.
- 모든 학습 arm에 같은 2,048 task exposures/256 updates, query CE만 사용했고 3 seeds를 평가했다. Exact predictive distillation이나 새 auxiliary loss를 쓴 실험이 아니다.
- Primary는 새 matched 48 base graphs의 whole-node uniform/clustered 50/90%, 총 context 1/2/4 평균이다. OOD 포함 전체는 192 graphs다.

### 성능과 강한 대조

| 방법 | Accuracy % | NLL | Brier |
|---|---:|---:|---:|
| 공개 Mask PFN | 53.4846 | 0.687080 | 0.493828 |
| Direct graph predictor | 51.1512 | 0.693134 | 0.499986 |
| **Circuit** | **61.3839** | **0.605128** | **0.430740** |
| Grid6-BP | 69.5602 | 0.540755 | 0.375136 |
| Grid6-exact | 69.3680 | 0.541211 | 0.375627 |

Circuit−Mask PFN accuracy는 +7.899306%p [ +3.889009, +11.790761 ], NLL은 −0.081952 [−0.101972, −0.062402]였다. **Neural baseline 대비 개선은 실제 결과로 보존한다.**

반면 Grid6-BP 대비 accuracy는 −8.176257%p [−13.111376, −3.230252], NLL은 +0.064373 [ +0.024247, +0.105302 ]였다. 단일 graph median은 circuit 12.369 ms, Grid6-BP 13.443 ms로 사전 2배 속도 기준도 충족하지 못했다. 단일 GPU 측정이며 외부 GPU 부하는 통제하지 않았다.

Grid6는 알려진 생성 family와 사전에 정한 여섯 parameter support를 이용하는 model-based baseline이다. 실제 test parameter를 제공하는 별도 oracle과 구분한다. 큰 실제 graph의 일반 해법은 아니지만, 현재 실험에서 이겨야 할 강한 비교다.

### 실제 feature 이용과 graph-specific 이득은 구분해야 한다

Context 2/clustered 50%의 보조 intervention에서 관측 feature 제거 시 accuracy −15.45%p, 노드 간 shuffle 시 −13.69%p였다. 이 Gaussian family에서는 실제 feature와 node 대응을 활용한다. Primary와 조건이 다른 보조 결과다.

그런데 설계 고정 뒤 추가한 **known-family feature-only** 대조는 graph를 전혀 쓰지 않고도 NLL 0.605147을 얻었다. Circuit의 NLL 0.605128과의 차이는 −0.000019 [−0.013783, +0.013696]였다. Accuracy 차이 +3.251075%p의 CI [−0.142816, +6.531240]도 0을 포함했다. 동등성 검정은 아니지만 graph-specific 추가 이득을 확인한 결과도 아니다. 이 대조는 Gaussian magnitude 0.8/sd 0.6이라는 정확한 family 상수를 안다.

가장 구체적으로 남은 조건은 **관측 feature가 label context 바깥에만 있는 경우**다. Primary의 사후 분할에서 circuit은 52.69%/NLL 0.69955, Grid6-BP는 64.87%/0.62460이었다. 단순 feature-only는 이 조건에서 class orientation을 알 수 없다. Graph를 이용하면 회수 가능한 정보가 있지만 현재 학습 모델이 충분히 회수하지 못한 조건이다. 사후 기술적 분석이므로 새 primary나 독립 확인으로 쓰지 않는다. Tree 표현력만의 탓으로도 확정하지 않는다.

## 아직 실행하지 않은 범위

다음은 남아 있는 가설이지 검증된 해결책이 아니다.

- 관측 cell을 직접 token으로 만들고 attention까지 학습하는 full sparse architecture. 최근 실행은 frozen backbone의 native group routing이었다.
- 원래 Wander 및 관측 경로를 읽는 observation-aware walk 모델. 이전 부분그래프 선택+PCFI와는 계산 대상이 다르지만, 긴 uniform walk나 기존 label-filtered Wander보다 나은지는 모른다.
- Context labels가 feature 가설의 구성부터 바꾸는 neural feature posterior, 관측 encoding을 공유하는 효율적 다중 가설 처리, 타당한 refinement law를 이용한 marginal consistency.
- PEMix의 장치를 활용한 확률적 첫 feature 변환. Expected token을 통과시킨 Transformer가 전체 posterior predictive와 같지는 않다.
- 실제 MaM 구현, 가변 feature 폭/새 semantic schema/큰 cyclic graph의 circuit 학습.
- Edge observation process가 role/label에 의존하는 MNAR BP. 원래 node feature missing 문제와 구분해야 한다.
- 보존한 mechanism-crossing accuracy 신호의 사전 고정 독립 재현.

## 관련 문헌에서 가져올 수 있는 것

정확한 metadata와 원문 링크는 아래 YAML에 있다. 이 목록이 현재 문헌 전체를 포괄하지는 않는다. 새 후보의 novelty는 추가 원문 검색으로 확인해야 한다.

| 논문 기록 | 이미 알려진 장치와 검토할 차이 |
|---|---|
| [GraphPFN](papers/GraphPFN.yaml) | Synthetic graph prior와 고정 backbone/graph adapter. 기존 모델의 강한 입력·학습 대조를 유지한다. |
| [Ephris](papers/Ephris.yaml) | Context/query feature를 함께 gather하고 압축하는 sparse graph ICL. 단순 feature-memory 분리와 겹침이 크다. |
| [Wander](papers/Wander.yaml) | Walk 기반 graph/task 추론. Label 방문 필터나 긴 walk 자체를 새 기여로 삼을 수 없다. |
| [GSPN](papers/GSPN.yaml) | Missing evidence를 내부에서 주변화하는 probabilistic circuit. 유효한 내부 계산과 원래 graph joint의 정확성은 별개다. |
| [MaM](papers/MaM.yaml) | Conditional/marginal의 합 규칙으로 모델을 구속한다. Discrete 설정을 연속·가변 graph PFN으로 옮기는 문제가 남는다. |
| [PEMix](papers/PEMix.yaml) | 구조 조건부 mixture와 첫 GNN layer에서의 해석적 주변화. PFN 전체에 같은 등식이 성립하지는 않는다. |
| [StructuralMissingPFN](papers/StructuralMissingPFN.yaml) | 불완전 입력에서 직접 예측하며 보조 posterior를 학습하는 PFN. Completion 없이 예측한다는 주장만으로 차별화하기 어렵다. |
| [MCSSL](papers/MCSSL.yaml) | 정보 증가에 따른 martingale consistency. 적절한 조건부 refinement 분포 없이 같은 식만 쓰면 안 된다. |
| [PFNSeparability](papers/PFNSeparability.yaml) | Feature law와 label mechanism의 분리. 우리의 crossing pilot은 원래 gate 실패와 보조 accuracy 신호를 함께 보존했다. |
| [TabImpute](papers/TabImpute.yaml), [GNNmim](papers/GNNmim.yaml) | 각각 query attention 접근과 강한 missing-feature 입력 대조를 검토할 자료다. 두 논문의 task/protocol과 우리 zero-shot 설정을 구분한다. |

## 독립 검토자에게 요청하는 답변

1. **문제 정의를 먼저 비판해 달라.** 현재 관측 정보에서 풀 수 있는 예측 문제인지, known-family oracle gap을 새로운 모델의 필요성으로 잘못 연결하고 있지는 않은지 확인해 달라.
2. **메소드 후보는 최대 두 개만 제안해 달라.** 후보마다 기존 실패 중 어떤 정보 손실·연산 제한을 바꾸는지 명시해 달라. 기존 실패 원인이 미상이면 그 가정을 표시해 달라.
3. 후보의 입력, 내부 추론 연산, 학습 목적, 새 graph에서 gradient-free로 하는 일을 수식이나 의사코드로 적어 달라. 새로운 head/loss를 붙이는 것만으로 끝내지 말고 필요한 이유를 설명해 달라.
4. 가장 가까운 선행연구와 이미 실행한 방법을 함께 비교해 달라. 단순 attention, Bayesian averaging, distillation, BP, circuit 또는 PFN 이름 자체를 novelty로 간주하지 말아 달라.
5. 새 모델 학습 전에 점검할 강한 비교와, **한 번의 제한된 실험으로 기각 가능한 기준**을 제시해 달라. 그 뒤 끝없이 진단을 늘리기보다 실제 모델 비교로 이어져야 한다.
6. Accuracy, NLL/Brier, calibration, 계산비용, 실제 graph/schema 전이를 따로 평가해 달라. 개선이 유의하지 않다는 것과 유의하지만 강한 baseline에 진다는 것을 구분해 달라.

설득력 있는 후보가 없으면 없다고 답해도 좋다. 목표를 사후에 바꾸거나, 미실행 범위를 실패로 세거나, 모델 용량·학습량만 늘려 기존 구성을 살리는 제안은 피하고 싶다.

## 바로 전달할 요청문

```text
첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.
실제 실행한 결과, 보존한 개선, 미실행 제안을 구분하고, 필요하면 함께 첨부한 보고서와 findings를 확인해줘.

목표는 새 그래프에서 재학습 없이 심한 node-feature missing을 처리하는 GraphPFN이야.
최근 circuit은 학습된 neural baseline은 이겼지만, known-family Grid6보다 낮고 feature-only 대비 추가 NLL 이득도 확인하지 못했어.

이 실패 범위를 반복하지 않는 메소드 후보를 최대 2개 제안해줘.
후보마다 바꾸는 정보/추론 연산, 학습 목적, 가장 가까운 선행연구와의 차이,
강한 비교 대상, 한 번의 작은 결정적 실험을 적어줘.
특히 unlabeled node의 관측 feature와 떨어진 context label을 graph로 연결하는 문제를 검토하되,
이 방향이 답이라고 가정하지 말고 다른 설명도 비판해줘.

논문 novelty를 단정하지 말고 확인된 사실과 추론을 구분해줘.
새 실험을 바로 실행하거나 저장소를 수정하지 말고, 우선 연구 제안과 비판을 답해줘.
설득력 있는 후보가 없다면 없다고 말해줘.
```
