# FP05 result-review response

2026-11-05. 세 reviewer는 같은 최종 자료를 method/prior-art, mathematics/code, experiments/statistics의 관점에서 검토했다. 각자 준비·구현·감사에 참여한 사실을 공개했으며 외부 blind review로 부르지 않는다. Root는 기존 리뷰 문서를 덮어쓰지 않고 이 응답에 판단을 남긴다.

| 지적/구분 | Root의 조치 |
|---|---|
| 감사 PASS와 효능 PASS는 다르다 | Main 및 두 독립 감사는 PASS로 보존하고, 네 efficacy gate FAIL/full guard PASS와 composite FAIL을 별도 명시했다. |
| PG−Fixed가 음수여도 유의한 열세를 확정할 수 없다 | −0.497%p의 nominal97.5% CI가0을 포함함을 표기했다. 우위 미확인과 열세 증명을 혼용하지 않는다. |
| Greedy000을 정책분포 붕괴라고 부르면 부정확하다 | 모든6fits의 final/real greedy000과6606개 replay 확률 배열 exact equality를 기록했다. 학습 sampling 다양성/entropy가 남았으며 최종 evaluation entropy를 재측정하지 않았다는 한계를 남겼다. |
| 동일θ routing 효과와 θ 학습 효과를 분리해야 한다 | Main−own000은0이다. PG−Fixed나 PG−Direct 차이는 서로 다르게 학습한 adapter를 포함하며 routing의 증분으로 귀속하지 않는다. |
| Cost penalty만 또는 PG variance만 원인이라고 할 수 없다 | Dev CE+0.01cost 최선도 non000이고 Direct도 배포000이다. 확인한 현상과 실패의 최종 원인을 구분해 reason_status는unknown으로 남긴다. |
| Synthetic feature gate와 real의 양의 feature response를 함께 남겨야 한다 | Conditional48graphs/58cases의 utility FAIL과 real95의 observed−none/shuffle 점 추정을 모두 보존했다. 후자는 PCFI→PFN 경로·학습된θ에서도 생기는 효과이며 RL 고유 성과가 아니다. |
| Strong baselines와 동일 estimand가 중요하다 | Frozen NaN/PCFI, LP/prior, Uniform8와 단일 global을 primary-same-subset 감사에 포함했다. Uniform8의 최고 point accuracy와 별도 비용을 남기고 유의한 승리로 승격하지 않는다. |
| 실제 panel은 k4/class·RP16·재사용 개발 데이터다 | Cora7/28labels, CiteSeer6/24labels, PubMed3/12labels와 varying query counts를 확인했다. Native k20 결과와 직접 비교하거나3seed×3rep를9graphs로 세지 않는다. Cora/CiteSeer class 수 변화도 전이 조건이며 단독 실패 원인은 아니다. |
| Training objective/deployment와 Direct의 정확성 범위를 밝혀야 한다 | Detached hidden-state에서의 정확 정책 기대 목적, adapter 조건부 CE gradient, group-normalized REINFORCE 및 sequential greedy를 각각 명시했다. 본체 Transformer RL이나 새 GRPO 알고리즘 주장을 하지 않는다. |
| 예산·재현성 범위를 과장하면 안 된다 |9,600 실제 train forwards/backwards,2,688 calibration,51,747 평가 forward와 버린 개발10groups/steps를 분리했다. CPU/hash overhead·미기록LP 시간 및 prelock old utility bytes 미보존 제한을 남겼다. |

추가 scientific rerun이 필요한 numerical/leakage/source 결함은 세 결과 관점에서 발견하지 못했다. 따라서 이번 결과에 대한 code repair나 추가 fit은 하지 않는다. 새로운 시도는 이 bank를 독립 최종 자료로 다시 쓰지 않는 별도 cycle에서 검토해야 한다.

남는 좁은 질문은 학습의 stochastic 경로 분포와 greedy 배포 사이 차이다. 학습된 정책 확률로 전체8경로를 가중한 예측은 이번에 실행하지 않았다. 이는 calibrated Bayesian posterior가 아닌 계산 정책의 분포이며, 실현 비용8회·Uniform8·dev-global·Direct 대조와 새 bank가 필요한 미실행 분석이다. 현재 실패를 구제했다고 세거나 자동 재개하지 않는다. 한편 PG dev-global110의 primary48.795%도 greedy48.793%와 거의 같으므로 고정 경로 하나를 바꾸면 accuracy가 해결된다는 근거도 없다.

최종 판정은 현재 연산·예산·목적·greedy 배포의 **No-Go**다. 일반 fine-tuning의 작은 point gain, real의 관측 피처 효용과 작동하는 public-adapter 학습 연산은 지운 결과가 아니다. 실행·리뷰·실패 범위와 미실행 질문을 원본 snapshot 및 구조화 finding으로 각각 보존한다.
