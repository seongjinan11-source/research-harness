# 접목 후보 (2026-11-18)

A의 한계 하나와 B의 강점이 같은 태그로 걸린 경우예요. 점수는 한계 출처 신뢰도(후속 논문 1.0, 리뷰 0.8, 저자 0.6, 조사자 0.4) × 겹친 태그의 희소성이고, 분야가 다르면 0.3을 더해요. 여러 논문에 흔한 태그일수록 가중치가 낮아요.
태그 매칭은 후보를 제안할 뿐이에요. B의 장치가 A의 한계를 실제로 메우는지는 직접 판단해야 합니다.

## 1. GraphPFN ← NodeGround [feature-heterogeneity, graph-structure-use, label-imbalance, label-space] (점수 1.2)
- **GraphPFN 한계** (후속 논문 지적 — NodeGround: A Node Classification Benchmark in the Graph Foundation Model Era (2026)): 대규모 node classification benchmark에서 GraphPFN은 가장 강한 GFM이었지만, 데이터셋 조건별로 보면 강한 homophily, 고차원 feature, 심하게 불균형한 target에서 튜닝된 GNN(GCNII, GPRGNN)에 뒤졌다. 10개 초과 class task도 불리했다. 라벨을 50%까지 늘려도 고차원 feature·불균형 target·다중 class 약점은 남았다. 저자들은 사전학습 표현을 새 그래프에 적응시키는 능력의 한계로 해석한다.
- **NodeGround 강점**: 7개 데이터셋 성질로 22개 부분집합을 만들어 '어떤 조건에서 GFM이 지는가'를 진단한다. 예를 들어 GraphPFN은 binary task, high-degree 그래프, 중간 수준 homophily에서는 튜닝된 GCNII·GPRGNN보다 앞서지만, 강한 homophily, 고차원 feature, 심한 불균형 target, 10개 초과 class에서는 뒤진다. 평균 순위 하나로는 보이지 않는 약점을 드러낸다.

## 2. Time-MoE ← Ephris [context-scale, tokenization] (점수 0.87 · 분야 교차)
- **Time-MoE 한계** (후속 논문 지적 — TimeSqueeze: Dynamic Patching for Efficient Time Series Forecasting (2026)): Time-MoE 같은 point-wise encoding은 sequence가 길어질수록 확장성이 떨어지며, 이것이 긴 context 사전학습의 병목이라고 지적한다. 이 논문은 Time-MoE의 backbone, Time-300B, 손실을 그대로 두고 tokenizer만 SSM 기반 dynamic patching으로 바꿨다. 그 결과 학습이 최대 20배 빠르고 사전학습 데이터 효율이 8배 높다고 보고한다.
- **Ephris 강점**: Section 3.1과 Appendix B의 feature별 Gather는 context와 query 노드를 모두 사용한다. Context-label embedding도 압축 전 scalar token에 제공하므로, feature memory를 label-context 노드에만 한정하지 않는 직접적인 선행 대조다.

## 3. Moirai ← TabImpute [channel-dependency, tokenization] (점수 0.86 · 분야 교차)
- **Moirai 한계** (후속 논문 지적 — Moirai 2.0: When Less Is More for Time Series Forecasting (2025)): 같은 Salesforce 그룹의 후속작이 Moirai의 핵심 설계를 대부분 되돌렸다. masked encoder는 sample 하나당 loss를 한 번만 내고 가려진 일부 token만 loss에 참여해서 데이터 활용이 비효율적이고, 반복 예측 때마다 key-value를 처음부터 다시 계산해야 한다고 지적했다. multi-patch 구성은 frequency 간 학습을 오히려 제약했다고 보고 single patch로 바꿨다. mixture distribution 출력은 실제로는 효과가 덜하고 설계와 최적화를 복잡하게 만들며, NLL이 outlier에서 variance collapse나 불안정한 gradient를 겪을 수 있다며 quantile loss로 대체했다. multivariate 예측과 covariate 지원은 이득이 거의 없었다며 아예 뺐고, 그런 성질을 가진 고품질 데이터가 부족한 탓으로 추정했다.
- **TabImpute 강점**: Section 3의 architecture는 원래 TabPFN의 query 사이 attention 제한을 제거한다. Missing-entry query가 공유 잠재 구조를 통해 의존할 수 있다는 이유이며, 우리 graph PFN의 성능 개선을 입증한 결과는 아니다.

## 4. NodePFN ← Chronos-2 [pretraining-data, prior-mismatch] (점수 0.83 · 분야 교차)
- **NodePFN 한계** (후속 논문 지적 — A Fair Evaluation of Graph Foundation Models for Node Property Prediction (2026)): NodePFN이 약한 이유로 세 가지를 가설로 든다. 첫째, sample 하나를 token 하나로 보는 TabPFN v1 시절 구조를 쓴다. 둘째, 강한 tabular FM에서 초기화하지 않고 처음부터 학습한다. 셋째, GraphPFN보다 prior가 단순하다. 이 이유로 자신들의 공정 비교 평가에서는 NodePFN을 아예 제외했다. 검증된 원인이 아니라 저자들의 추정이라는 점은 감안해야 한다.
- **Chronos-2 강점**: 실제 다변량 데이터 없이 synthetic multivariatizer만으로 다변량·covariate 능력을 얻었다. synthetic 데이터만으로 학습한 변형도 GIFT-Eval과 Chronos Benchmark II에서 본 모델에 거의 근접해, synthetic prior 설계의 효과를 보여준다.

## 5. NodePFN ← TimesFM [pretraining-data, prior-mismatch] (점수 0.83 · 분야 교차)
- **NodePFN 한계** (후속 논문 지적 — A Fair Evaluation of Graph Foundation Models for Node Property Prediction (2026)): NodePFN이 약한 이유로 세 가지를 가설로 든다. 첫째, sample 하나를 token 하나로 보는 TabPFN v1 시절 구조를 쓴다. 둘째, 강한 tabular FM에서 초기화하지 않고 처음부터 학습한다. 셋째, GraphPFN보다 prior가 단순하다. 이 이유로 자신들의 공정 비교 평가에서는 NodePFN을 아예 제외했다. 검증된 원인이 아니라 저자들의 추정이라는 점은 감안해야 한다.
- **TimesFM 강점**: synthetic 데이터를 빼면 실제 데이터에 드문 granularity(quarterly, yearly, 10분 등)와 15분 ETT에서 성능이 떨어진다는 ablation을 보여, 실제 corpus의 빈 곳을 synthetic prior로 메우는 효과를 실증했다.

## 6. Time-MoE ← LoGIC [context-scale, inference-cost] (점수 0.81 · 분야 교차)
- **Time-MoE 한계** (후속 논문 지적 — TimeSqueeze: Dynamic Patching for Efficient Time Series Forecasting (2026)): Time-MoE 같은 point-wise encoding은 sequence가 길어질수록 확장성이 떨어지며, 이것이 긴 context 사전학습의 병목이라고 지적한다. 이 논문은 Time-MoE의 backbone, Time-300B, 손실을 그대로 두고 tokenizer만 SSM 기반 dynamic patching으로 바꿨다. 그 결과 학습이 최대 20배 빠르고 사전학습 데이터 효율이 8배 높다고 보고한다.
- **LoGIC 강점**: 예산을 정한 context만으로 full-context 정확도를 유지하면서 peak 메모리를 크게 줄인다. GraphPFN 기준 artnet-exp에서 full graph 추론은 51.86 AP에 19.59GB였는데, LoGIC은 52.06 AP에 8.24GB였다. TabPFNv2 기준 tolokers-2에서도 4.10GB가 2.20GB로 줄었고 AP는 오히려 조금 올랐다.

## 7. Moirai ← TSFM-Survey [channel-dependency, tokenization, uncertainty] (점수 0.79)
- **Moirai 한계** (후속 논문 지적 — Moirai 2.0: When Less Is More for Time Series Forecasting (2025)): 같은 Salesforce 그룹의 후속작이 Moirai의 핵심 설계를 대부분 되돌렸다. masked encoder는 sample 하나당 loss를 한 번만 내고 가려진 일부 token만 loss에 참여해서 데이터 활용이 비효율적이고, 반복 예측 때마다 key-value를 처음부터 다시 계산해야 한다고 지적했다. multi-patch 구성은 frequency 간 학습을 오히려 제약했다고 보고 single patch로 바꿨다. mixture distribution 출력은 실제로는 효과가 덜하고 설계와 최적화를 복잡하게 만들며, NLL이 outlier에서 variance collapse나 불안정한 gradient를 겪을 수 있다며 quantile loss로 대체했다. multivariate 예측과 covariate 지원은 이득이 거의 없었다며 아예 뺐고, 그런 성질을 가진 고품질 데이터가 부족한 탓으로 추정했다.
- **TSFM-Survey 강점**: Table 2가 15개 모델을 같은 열로 정리해서 모델 선택용 참고표로 쓰기 좋다. 예를 들어 MOMENT와 TimesFM은 결정적 출력, Chronos와 MOIRAI는 확률적 출력, Time-MoE는 point forecast만 낸다는 점을 바로 비교할 수 있다.

## 8. TSB-AD ← WavelengthBOCS [eval-protocol, robustness] (점수 0.79 · 분야 교차)
- **TSB-AD 한계** (후속 논문 지적 — Did We Actually Fix It? An Independent Adversarial Stress-Test of Post-Point-Adjustment Evaluation Metrics for Time-Series Anomaly Detection (2026)): TSB-AD가 권고한 VUS-PR을 무작위 점수와 best-of-N seed 선택으로 공격해 봤다. UCR, SMD, SMAP, MSL에서는 대체로 버텼지만, NAB에서는 시도 횟수를 늘리면 부풀려지는 시계열 비율이 25%에 이르렀다. 한 benchmark에서 안전한 지표가 다른 benchmark에서도 안전하다는 보장은 없다는 결론이다. TSB-AD의 지표 분석은 독립적이거나 adversarial한 감사가 아니었다고 분류했다.
- **WavelengthBOCS 강점**: 반복 실행의 선택 일관성과 one-bit removal 후 예측오차를 보고한다.

## 9. NodePFN ← FoundTS [adaptation, zero-shot-gap] (점수 0.77 · 분야 교차)
- **NodePFN 한계** (후속 논문 지적 — GraphPFN: A Prior-Data Fitted Graph Foundation Model (2026)): 공식 코드는 training sample을 1,024개 이하로 쓰라고 권장하는데, 표준 split의 benchmark들은 모두 이를 넘는다. 이 조건에서 돌린 NodePFN은 모든 데이터셋에서 GraphPFN에 크게 졌고, GCN·GraphSAGE·GAT·GT보다도 평균 순위가 낮았다. 공식 구현에 fine-tuning 경로가 없어 ICL로만 평가할 수 있었다는 점도 함께 지적됐다.
- **FoundTS 강점**: zero-shot, few-shot, full-shot을 같은 protocol로 평가해서 적응 단계에 따라 순위가 어떻게 바뀌는지 볼 수 있다. 시계열로 사전학습한 TSFM의 zero-shot이 dataset별 모델의 few-shot·full-shot보다 나은 데이터셋이 많았다. 반대로 큰 사전학습 모델 상당수는 데이터를 더 주고 학습시키면 zero-shot보다 오히려 나빠졌다.

## 10. NodePFN ← MOMENT [adaptation, zero-shot-gap] (점수 0.77 · 분야 교차)
- **NodePFN 한계** (후속 논문 지적 — GraphPFN: A Prior-Data Fitted Graph Foundation Model (2026)): 공식 코드는 training sample을 1,024개 이하로 쓰라고 권장하는데, 표준 split의 benchmark들은 모두 이를 넘는다. 이 조건에서 돌린 NodePFN은 모든 데이터셋에서 GraphPFN에 크게 졌고, GCN·GraphSAGE·GAT·GT보다도 평균 순위가 낮았다. 공식 구현에 fine-tuning 경로가 없어 ICL로만 평가할 수 있었다는 점도 함께 지적됐다.
- **MOMENT 강점**: supervision을 거의 쓰지 않아도 성능이 강하다. 고정 embedding 위 SVM만으로 UCR classification에서 데이터셋별로 학습한 방법 대부분을 앞섰고, linear probing만으로 장기 예측에서 PatchTST 다음 수준에 도달했으며, anomaly detection과 imputation에서도 TimesNet·GPT4TS보다 나았다.

## 11. NodePFN ← Time-MoE [adaptation, zero-shot-gap] (점수 0.77 · 분야 교차)
- **NodePFN 한계** (후속 논문 지적 — GraphPFN: A Prior-Data Fitted Graph Foundation Model (2026)): 공식 코드는 training sample을 1,024개 이하로 쓰라고 권장하는데, 표준 split의 benchmark들은 모두 이를 넘는다. 이 조건에서 돌린 NodePFN은 모든 데이터셋에서 GraphPFN에 크게 졌고, GCN·GraphSAGE·GAT·GT보다도 평균 순위가 낮았다. 공식 구현에 fine-tuning 경로가 없어 ICL로만 평가할 수 있었다는 점도 함께 지적됐다.
- **Time-MoE 강점**: TSLib 장기 예측 6개 데이터셋의 zero-shot 평균 MSE에서 ultra가 0.322로 Moirai-large(0.359), TimesFM(0.396), Chronos-large(0.416), MOMENT(0.461)보다 낮았다. 저자들은 각각 대비 11%, 23%, 30% 감소라고 보고한다. 같은 데이터셋의 train split으로 한 epoch만 fine-tuning해도 iTransformer, PatchTST 같은 full-shot 모델보다 평균 MSE가 24% 낮았다.

## 12. MOMENT ← Wander [task-coverage, zero-shot-gap] (점수 0.76 · 분야 교차)
- **MOMENT 한계** (후속 논문 지적 — When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection (ICLR 2026)): MOMENT를 포함한 다섯 TSFM 계열을 reconstruction·forecasting error 기반 anomaly detector로 평가했다. 모델 크기와 context 길이를 바꿔도 moving-window variance 같은 한 줄짜리 baseline과 유의미한 차이가 없었다. 이상 구간이 정상 구간보다 꼭 더 복원하기 어렵지 않기 때문에, 복원 오차로 이상을 잡는다는 전제 자체가 이 모델들에서 성립하지 않는다는 진단이다.
- **Wander 강점**: 하나의 checkpoint가 zero-shot으로 세 task 모두에서 최고이거나 그에 가까운 성능을 낸다. Knowledge graph 54개의 link prediction에서 MRR 0.399로 전용 모델 FLOCK(0.391)과 비슷하거나 조금 높았다. Homogeneous link prediction 10개 데이터셋에서는 평균 Recall@20 43.39%로 데이터셋별로 튜닝한 NBFNet(38.82%)을 앞섰다. Node classification 26개 그래프에서는 평균 정확도 68.20%로 GraphPFN(66.55%)과 튜닝된 GCN(60.64%)보다 높았다.

## 13. Time-MoE ← Wander [task-coverage, zero-shot-gap] (점수 0.76 · 분야 교차)
- **Time-MoE 한계** (후속 논문 지적 — When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection (ICLR 2026)): MOMENT, Chronos, TimesFM, Time-MoE, TSPulse 다섯 계열을 reconstruction·forecasting error 기반 anomaly detector로 평가했다. 모델 크기와 context 길이를 바꿔도 moving-window variance 같은 한 줄짜리 baseline과 유의미한 차이가 없었다. 예측 오차로 이상을 잡는다는 전제가 Time-MoE 같은 forecasting TSFM에서도 성립하지 않는다는 진단이다.
- **Wander 강점**: 하나의 checkpoint가 zero-shot으로 세 task 모두에서 최고이거나 그에 가까운 성능을 낸다. Knowledge graph 54개의 link prediction에서 MRR 0.399로 전용 모델 FLOCK(0.391)과 비슷하거나 조금 높았다. Homogeneous link prediction 10개 데이터셋에서는 평균 Recall@20 43.39%로 데이터셋별로 튜닝한 NBFNet(38.82%)을 앞섰다. Node classification 26개 그래프에서는 평균 정확도 68.20%로 GraphPFN(66.55%)과 튜닝된 GCN(60.64%)보다 높았다.

## 14. NodePFN ← TimesFM [inference-cost, zero-shot-gap] (점수 0.75 · 분야 교차)
- **NodePFN 한계** (후속 논문 지적 — NodeGround: A Node Classification Benchmark in the Graph Foundation Model Era (2026)): labeled node 사이의 dense self-attention 때문에 큰 그래프에서 추론 비용이 크다고 지적한다. 성능-비용 비교에서 NodePFN은 더 싸면서 더 정확한 방법에 dominate되어 Pareto frontier에 들지 못했고, 잘 튜닝된 supervised 방법보다 성능이 낮았다.
- **TimesFM 강점**: 200M 규모의 비교적 작은 모델로 Monash, Darts, ETT에서 zero-shot 성능이 dataset별 supervised 모델과 비슷한 수준에 도달했다. GPT-3 기반 LLMTime보다 Monash에서 10% 이상 좋으면서 추론 비용은 훨씬 작다.

## 15. SSGP ← StructuralMissingPFN [amortized-inference, uncertainty] (점수 0.65 · 분야 교차)
- **SSGP 한계** (저자 인정): §2의 label posterior와 §3.2의 geometry conditioning에 Gaussian 근사를 사용한다.
- **StructuralMissingPFN 강점**: Mask를 포함한 불완전 입력에서 직접 예측하며 explicit posterior sampling과 분류 경로를 구분한다.

## 16. GraphPFN ← Wander [feature-heterogeneity, label-space] (점수 0.64)
- **GraphPFN 한계** (후속 논문 지적 — NodeGround: A Node Classification Benchmark in the Graph Foundation Model Era (2026)): 대규모 node classification benchmark에서 GraphPFN은 가장 강한 GFM이었지만, 데이터셋 조건별로 보면 강한 homophily, 고차원 feature, 심하게 불균형한 target에서 튜닝된 GNN(GCNII, GPRGNN)에 뒤졌다. 10개 초과 class task도 불리했다. 라벨을 50%까지 늘려도 고차원 feature·불균형 target·다중 class 약점은 남았다. 저자들은 사전학습 표현을 새 그래프에 적응시키는 능력의 한계로 해석한다.
- **Wander 강점**: Feature와 class를 channel별 token으로 다루고 같은 종류의 channel끼리 모듈을 공유하므로, feature 차원 수와 class 수에 제한 없이 차원 축소 없이 처리한다. 같은 표에서 NodePFN은 지원 class 수를 넘는 데이터셋 계열(Cit. Full, HM Cat.)에서 아예 빠졌다.

## 17. NodePFN ← NodeGround [feature-heterogeneity, label-space] (점수 0.64)
- **NodePFN 한계** (후속 논문 지적 — To Learn is to Wander: Learning Across Graphs and Tasks with Random Walks (2026)): node마다 representation 하나만 유지하고 feature 차원 상한이 고정돼 있다고 지적한다. 지원 class 수를 넘는 데이터셋 열에서는 NodePFN 결과를 아예 보고할 수 없었다.
- **NodeGround 강점**: 7개 데이터셋 성질로 22개 부분집합을 만들어 '어떤 조건에서 GFM이 지는가'를 진단한다. 예를 들어 GraphPFN은 binary task, high-degree 그래프, 중간 수준 homophily에서는 튜닝된 GCNII·GPRGNN보다 앞서지만, 강한 homophily, 고차원 feature, 심한 불균형 target, 10개 초과 class에서는 뒤진다. 평균 순위 하나로는 보이지 않는 약점을 드러낸다.

## 18. NodePFN ← Wander [feature-heterogeneity, label-space] (점수 0.64)
- **NodePFN 한계** (후속 논문 지적 — To Learn is to Wander: Learning Across Graphs and Tasks with Random Walks (2026)): node마다 representation 하나만 유지하고 feature 차원 상한이 고정돼 있다고 지적한다. 지원 class 수를 넘는 데이터셋 열에서는 NodePFN 결과를 아예 보고할 수 없었다.
- **Wander 강점**: Feature와 class를 channel별 token으로 다루고 같은 종류의 channel끼리 모듈을 공유하므로, feature 차원 수와 class 수에 제한 없이 차원 축소 없이 처리한다. 같은 표에서 NodePFN은 지원 class 수를 넘는 데이터셋 계열(Cit. Full, HM Cat.)에서 아예 빠졌다.

## 19. Chronos-2 ← GraphPFN [pretraining-data, prior-mismatch] (점수 0.62 · 분야 교차)
- **Chronos-2 한계** (저자 인정): 단변량을 넘어서는 능력은 전적으로 synthetic 데이터에 기대고 있다고 저자들이 명시한다. synthetic만으로 학습한 변형은 fev-bench skill score가 45.9%로 본 모델(47.3%)보다 낮아, 실제 데이터가 여전히 기여한다는 점도 저자들 ablation에 나타난다.
- **GraphPFN 강점**: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.

## 20. Chronos-2 ← NodePFN [pretraining-data, prior-mismatch] (점수 0.62 · 분야 교차)
- **Chronos-2 한계** (저자 인정): 단변량을 넘어서는 능력은 전적으로 synthetic 데이터에 기대고 있다고 저자들이 명시한다. synthetic만으로 학습한 변형은 fev-bench skill score가 45.9%로 본 모델(47.3%)보다 낮아, 실제 데이터가 여전히 기여한다는 점도 저자들 ablation에 나타난다.
- **NodePFN 강점**: 합성 그래프 수를 늘릴수록 성능이 오르는 prior-scale 분석과, ER·cSBM·BA 구성을 바꾼 prior ablation을 함께 제시해 어떤 prior 성분이 도움이 되는지 일부 보여준다.

## 21. Chronos-2 ← PFNSeparability [pretraining-data, prior-mismatch] (점수 0.62 · 분야 교차)
- **Chronos-2 한계** (저자 인정): 단변량을 넘어서는 능력은 전적으로 synthetic 데이터에 기대고 있다고 저자들이 명시한다. synthetic만으로 학습한 변형은 fev-bench skill score가 45.9%로 본 모델(47.3%)보다 낮아, 실제 데이터가 여전히 기여한다는 점도 저자들 ablation에 나타난다.
- **PFNSeparability 강점**: Feature law와 conditional label law의 의존성을 prior 설계의 명시적 선택으로 다룬다.

## 22. Chronos-2 ← Wander [pretraining-data, prior-mismatch] (점수 0.62 · 분야 교차)
- **Chronos-2 한계** (저자 인정): 단변량을 넘어서는 능력은 전적으로 synthetic 데이터에 기대고 있다고 저자들이 명시한다. synthetic만으로 학습한 변형은 fev-bench skill score가 45.9%로 본 모델(47.3%)보다 낮아, 실제 데이터가 여전히 기여한다는 점도 저자들 ablation에 나타난다.
- **Wander 강점**: Synthetic prior가 NodePFN식과 GraphPFN식 생성기에 Watts–Strogatz, random geometric, graphon 기반 구조를 더해 섞는다. 그래서 GraphPFN prior가 덮지 못한다고 인정한 공간·기하 구조 그래프를 사전학습 분포에 명시적으로 포함한다.

## 23. EDiS ← TSAD-OneLiners [nonstationarity, task-coverage] (점수 0.62 · 분야 교차)
- **EDiS 한계** (저자 인정): Task-aware score, dynamic temporal composition, graph-/link-level prediction 확장을 향후 과제로 남겼다.
- **TSAD-OneLiners 강점**: 예측 기반 점수의 실패 원인도 설명했다. sliding window가 sequence anomaly 안으로 들어가면 context가 이미 이상한 값을 담고 있어서, 모델이 다음 이상값을 잘 맞혀 버리고 오차가 작아진다. 한 사례에서 TimesFM-1.0-512의 VUS-PR은 sequence anomaly에서 horizon 1일 때 0.04, 128일 때 0.68이었고, point anomaly에서는 반대로 1.0에서 0.02로 떨어졌다. TSB-AD가 보고한 'point anomaly에는 강하고 sequence anomaly에는 약하다'는 현상을 설명하는 근거다.

## 24. GGP ← Ephris [graph-scale, inference-cost] (점수 0.62 · 분야 교차)
- **GGP 한계** (저자 인정): §3.3의 추론 비용은 O(L M^2)이며 covariance 계산에 최대 차수의 제곱 비용이 있다.
- **Ephris 강점**: 논문은 51 datasets에서 품질과 실행 시간의 Pareto 개선을 보고하며 sparse graph ICL의 강한 비교 대상을 제시한다.

## 25. GraphPFN ← Chronos-2 [pretraining-data, prior-mismatch] (점수 0.62 · 분야 교차)
- **GraphPFN 한계** (저자 인정): 부록의 prior ablation에서 단순 degree-corrected SBM으로 사전학습해도 제안한 multi-level SBM + PA prior와 비슷하거나 일부 데이터셋에서 더 좋았다. 저자들도 이전 revision과 결과가 달라졌음을 인정하고, 제안 prior는 일관성과 다양성 때문에 유지했다고 설명한다. 즉 새 graph prior의 성능상 기여는 최종본에서 뒷받침되지 않는다.
- **Chronos-2 강점**: 실제 다변량 데이터 없이 synthetic multivariatizer만으로 다변량·covariate 능력을 얻었다. synthetic 데이터만으로 학습한 변형도 GIFT-Eval과 Chronos Benchmark II에서 본 모델에 거의 근접해, synthetic prior 설계의 효과를 보여준다.

# 빈칸: 한계로만 지적되고, 하네스 안에 해결한 논문이 없는 축

