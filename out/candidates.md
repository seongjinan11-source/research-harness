# 접목 후보 (2026-10-30)

A의 한계 하나와 B의 강점이 같은 태그로 걸린 경우예요. 점수는 한계 출처 신뢰도(후속 논문 1.0, 리뷰 0.8, 저자 0.6, 조사자 0.4) × 겹친 태그의 희소성이고, 분야가 다르면 0.3을 더해요. 여러 논문에 흔한 태그일수록 가중치가 낮아요.
태그 매칭은 후보를 제안할 뿐이에요. B의 장치가 A의 한계를 실제로 메우는지는 직접 판단해야 합니다.

## 1. GraphPFN ← TimesFM [pretraining-data, prior-mismatch] (점수 0.94 · 분야 교차)
- **GraphPFN 한계** (저자 인정): 부록의 prior ablation에서 단순 degree-corrected SBM으로 사전학습해도 제안한 multi-level SBM + PA prior와 비슷하거나 일부 데이터셋에서 더 좋았다. 저자들도 이전 revision과 결과가 달라졌음을 인정하고, 제안 prior는 일관성과 다양성 때문에 유지했다고 설명한다. 즉 새 graph prior의 성능상 기여는 최종본에서 뒷받침되지 않는다.
- **TimesFM 강점**: synthetic 데이터를 빼면 실제 데이터에 드문 granularity(quarterly, yearly, 10분 등)와 15분 ETT에서 성능이 떨어진다는 ablation을 보여, 실제 corpus의 빈 곳을 synthetic prior로 메우는 효과를 실증했다.

## 2. GraphPFN ← MOMENT [feature-heterogeneity] (점수 0.89 · 분야 교차)
- **GraphPFN 한계** (후속 논문 지적 — NodeGround: A Node Classification Benchmark in the Graph Foundation Model Era (2026)): 대규모 node classification benchmark에서 GraphPFN은 가장 강한 GFM이었지만, 데이터셋 조건별로 보면 강한 homophily, 고차원 feature, 심하게 불균형한 target에서 튜닝된 GNN(GCNII, GPRGNN)에 뒤졌다. 10개 초과 class task도 불리했다. 라벨을 50%까지 늘려도 고차원 feature·불균형 target·다중 class 약점은 남았다. 저자들은 사전학습 표현을 새 그래프에 적응시키는 능력의 한계로 해석한다.
- **MOMENT 강점**: patching, RevIN, 좌측 zero-padding, channel 단위 처리라는 단순한 조합으로 길이·스케일·채널 수가 다른 데이터셋을 한 모델에 넣는다. 데이터셋마다 구조를 바꿀 필요가 없다.

## 3. MOMENT ← GraphPFN [task-coverage] (점수 0.89 · 분야 교차)
- **MOMENT 한계** (후속 논문 지적 — When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection (ICLR 2026)): MOMENT를 포함한 다섯 TSFM 계열을 reconstruction·forecasting error 기반 anomaly detector로 평가했다. 모델 크기와 context 길이를 바꿔도 moving-window variance 같은 한 줄짜리 baseline과 유의미한 차이가 없었다. 이상 구간이 정상 구간보다 꼭 더 복원하기 어렵지 않기 때문에, 복원 오차로 이상을 잡는다는 전제 자체가 이 모델들에서 성립하지 않는다는 진단이다.
- **GraphPFN 강점**: LimiX의 per-feature tokenization을 그대로 물려받아, 데이터셋마다 feature 개수가 달라도 차원 축소 없이 하나의 모델로 처리한다. 같은 모델이 node classification과 node regression을 모두 다룬다.

## 4. GraphPFN ← TimesFM [inference-cost] (점수 0.78 · 분야 교차)
- **GraphPFN 한계** (후속 논문 지적 — A Fair Evaluation of Graph Foundation Models for Node Property Prediction (2026)): 후속 재평가에서 GraphPFN은 finetuning 시 10개 GraphLand 데이터셋 모두 1위였지만, 추론 비용과 메모리가 튜닝된 GNN보다 훨씬 컸다. GNN 추론은 1초 미만인데 GraphPFN은 수 초에서 수 분이 걸렸고, 일부 데이터셋에서는 학습 peak 메모리가 80GB GPU 한 장을 넘어 model parallelism이 필요했다. 수백만 노드 규모 데이터셋은 평가 대상 GFM 모두가 처리하지 못해 제외되었다. (GraphPFN 저자 일부가 쓴 논문이다.)
- **TimesFM 강점**: 출력 patch를 입력 patch보다 길게 잡은 설계 덕분에 긴 horizon을 적은 decoding step으로 예측한다. 512 step 예측 ablation에서 출력 patch 128이 32보다 오차가 작았는데, step 수가 16에서 4로 줄어 누적 오차가 줄어든 효과로 해석된다.
- **TimesFM 강점**: 200M 규모의 비교적 작은 모델로 Monash, Darts, ETT에서 zero-shot 성능이 dataset별 supervised 모델과 비슷한 수준에 도달했다. GPT-3 기반 LLMTime보다 Monash에서 10% 이상 좋으면서 추론 비용은 훨씬 작다.

## 5. GraphPFN ← TimesFM [inference-cost] (점수 0.78 · 분야 교차)
- **GraphPFN 한계** (후속 논문 지적 — LoGIC: Budgeted Context Construction for Node-Level Graph In-Context Learning with Tabular Foundation Models (2026)): GraphPFN의 adapter 방식은 graph attention이 message passing을 하도록 전체 그래프를 가속기 메모리에 올려 두고, 학습 table 전체를 context로 쓴다고 지적한다. 그래서 peak 메모리가 노드 수뿐 아니라 그래프 크기와 feature 수에 따라 커진다. 이 논문은 GraphPFN checkpoint를 그대로 둔 채 예산을 정한 subgraph context만 구성해도 정확도가 비슷하거나 조금 높게 나온다고 보고한다. 예를 들어 artnet-exp에서 peak 메모리가 약 19.6GB에서 8.2GB로 줄었다.
- **TimesFM 강점**: 출력 patch를 입력 patch보다 길게 잡은 설계 덕분에 긴 horizon을 적은 decoding step으로 예측한다. 512 step 예측 ablation에서 출력 patch 128이 32보다 오차가 작았는데, step 수가 16에서 4로 줄어 누적 오차가 줄어든 효과로 해석된다.
- **TimesFM 강점**: 200M 규모의 비교적 작은 모델로 Monash, Darts, ETT에서 zero-shot 성능이 dataset별 supervised 모델과 비슷한 수준에 도달했다. GPT-3 기반 LLMTime보다 Monash에서 10% 이상 좋으면서 추론 비용은 훨씬 작다.

## 6. MOMENT ← GraphPFN [inference-cost] (점수 0.78 · 분야 교차)
- **MOMENT 한계** (후속 논문 지적 — FoundTS: Comprehensive and Unified Benchmarking of Foundation Models for Time Series Forecasting (2024)): 통합 forecasting 벤치마크에서 MOMENT를 channel independence를 가정하는 대표 모델로 분류했다. 변수 간 상관이 강한 데이터셋에서는 channel dependence를 다루는 foundation model보다 뒤처졌고, 특히 Traffic에서 고전했다고 지적했다. runtime과 정확도 양쪽에서 dataset-specific 모델에 밀린다는 평가도 함께 내렸다.
- **GraphPFN 강점**: LimiX에서 warm-start하고 adapter만 학습하므로 사전학습이 A100 8장으로 약 36시간에 끝난다. ICL 추론 시간은 GCN을 처음부터 학습하는 것과 같은 자릿수이고, 하이퍼파라미터 탐색이 필요 없다.

## 7. MOMENT ← GraphPFN [zero-shot-gap] (점수 0.78 · 분야 교차)
- **MOMENT 한계** (후속 논문 지적 — Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts (ICLR 2025)): MOMENT가 고정된 입력 context 길이에 묶여 있다는 점을 범용성의 제약으로 꼽았다. 여섯 개 벤치마크 평균의 zero-shot 장기 예측 비교에서 MOMENT는 MSE와 MAE가 가장 나쁜 baseline이었다.
- **GraphPFN 강점**: ICL과 finetuning을 모두 지원하며, finetuning 시 13개 데이터셋 중 11개에서 최고 성능을 낸다. 같은 구조에 random adapter를 붙여 finetuning하면 성능이 크게 떨어지므로, 이득이 구조가 아니라 사전학습에서 온다는 점을 보였다.

## 8. MOMENT ← GraphPFN [inference-cost] (점수 0.78 · 분야 교차)
- **MOMENT 한계** (후속 논문 지적 — Exploring Representations and Interventions in Time Series Foundation Models (ICML 2025)): MOMENT-Large의 layer 간 CKA 유사도를 보면 여러 layer가 거의 같은 표현을 내는 block 구조가 나타났다. 이 중복 block을 잘라내도 장기 예측 성능은 거의 유지되면서 메모리가 크게 줄었다. 385M 파라미터 중 상당 부분이 효율적으로 쓰이지 않는다는 뜻이다. MOMENT 저자 일부가 참여한 후속 분석이다.
- **GraphPFN 강점**: LimiX에서 warm-start하고 adapter만 학습하므로 사전학습이 A100 8장으로 약 36시간에 끝난다. ICL 추론 시간은 GCN을 처음부터 학습하는 것과 같은 자릿수이고, 하이퍼파라미터 탐색이 필요 없다.

## 9. TimesFM ← GraphPFN [pretraining-data] (점수 0.78 · 분야 교차)
- **TimesFM 한계** (후속 논문 지적 — Unified Training of Universal Time Series Forecasting Transformers (Moirai, 2024)): 사전학습 데이터가 Google Trends와 Wiki pageview에 크게 의존해서 시계열이 나온 도메인의 다양성이 부족하다고 지적한다. 비교표에서도 TimesFM을 any-variate zero-shot과 probabilistic forecasting을 지원하지 않는 모델로 분류한다.
- **GraphPFN 강점**: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.

## 10. MOMENT ← GraphPFN [adaptation, zero-shot-gap] (점수 0.68 · 분야 교차)
- **MOMENT 한계** (조사자 판단): [analyst] 사전학습만으로 바로 쓸 수 있는 zero-shot forecasting 경로가 없다. 장기 예측은 horizon마다 forecasting head를 linear probing으로 학습해야 하고, 단기 예측의 zero-shot도 source 데이터셋에서 fine-tuning한 뒤 target으로 옮기는 transfer 설정이다. 그래서 Chronos나 TimesFM의 zero-shot forecasting과 같은 기준으로 비교하기 어렵다.
- **GraphPFN 강점**: ICL과 finetuning을 모두 지원하며, finetuning 시 13개 데이터셋 중 11개에서 최고 성능을 낸다. 같은 구조에 random adapter를 붙여 finetuning하면 성능이 크게 떨어지므로, 이득이 구조가 아니라 사전학습에서 온다는 점을 보였다.

## 11. GraphPFN ← MOMENT [task-coverage] (점수 0.65 · 분야 교차)
- **GraphPFN 한계** (저자 인정): Node-level task 전용이다. Masked graph modeling head를 사전학습에 넣었지만 예측은 여전히 node label에 크게 의존하므로, link prediction과 graph-level task에는 바로 쓸 수 없다.
- **MOMENT 강점**: 하나의 사전학습 모델로 forecasting, classification, anomaly detection, imputation 네 가지 태스크를 다룬다. 같은 reconstruction head를 그대로 써서 imputation과 anomaly detection은 별도 학습 없이 수행할 수 있다.

## 12. GraphPFN ← MOMENT [feature-heterogeneity] (점수 0.65 · 분야 교차)
- **GraphPFN 한계** (저자 인정): Feature 차원이 높은 text-embedding 데이터셋(amazon-ratings, questions)은 finetuning이 불가능해 PCA로 64차원까지 줄여야 했다. PCA를 쓰면 성능이 크게 떨어지는 pubmed(500차원)는 finetuning 중 메모리 부족이 나서 ICL 결과로 대신 보고했다.
- **MOMENT 강점**: patching, RevIN, 좌측 zero-padding, channel 단위 처리라는 단순한 조합으로 길이·스케일·채널 수가 다른 데이터셋을 한 모델에 넣는다. 데이터셋마다 구조를 바꿀 필요가 없다.

## 13. GraphPFN ← TimesFM [prior-mismatch] (점수 0.65 · 분야 교차)
- **GraphPFN 한계** (저자 인정): Graph prior가 social·information network에 맞춰져 있어 traffic network 같은 공간·기하 구조 그래프는 덮지 못한다. 저자들은 geometric graph 같은 random graph model 추가를 향후 과제로 남긴다.
- **TimesFM 강점**: synthetic 데이터를 빼면 실제 데이터에 드문 granularity(quarterly, yearly, 10분 등)와 15분 ETT에서 성능이 떨어진다는 ablation을 보여, 실제 corpus의 빈 곳을 synthetic prior로 메우는 효과를 실증했다.

## 14. MOMENT ← GraphPFN [task-coverage] (점수 0.65 · 분야 교차)
- **MOMENT 한계** (저자 인정): zero-shot 단기 예측이 가장 약한 태스크였다. M3·M4 부분집합에서 Theta, ETS 같은 통계 기법에 대체로 밀렸고 저자들도 개선 여지가 가장 크다고 인정했다. 향후 과제로 causal attention과 forecasting objective로 다시 사전학습하는 방향을 제시했다.
- **GraphPFN 강점**: LimiX의 per-feature tokenization을 그대로 물려받아, 데이터셋마다 feature 개수가 달라도 차원 축소 없이 하나의 모델로 처리한다. 같은 모델이 node classification과 node regression을 모두 다룬다.

## 15. GraphPFN ← MOMENT [pretraining-data] (점수 0.59 · 분야 교차)
- **GraphPFN 한계** (저자 인정): 부록의 prior ablation에서 단순 degree-corrected SBM으로 사전학습해도 제안한 multi-level SBM + PA prior와 비슷하거나 일부 데이터셋에서 더 좋았다. 저자들도 이전 revision과 결과가 달라졌음을 인정하고, 제안 prior는 일관성과 다양성 때문에 유지했다고 설명한다. 즉 새 graph prior의 성능상 기여는 최종본에서 뒷받침되지 않는다.
- **MOMENT 강점**: 모델 가중치, 사전학습 데이터(Time-series Pile), 코드를 모두 공개했다. 당시 TSFM 다수가 데이터나 가중치를 닫아 두었던 상황에서 재현과 후속 분석의 기반이 되었다.

## 16. MOMENT ← GraphPFN [adaptation] (점수 0.59 · 분야 교차)
- **MOMENT 한계** (저자 인정): harm과 trustworthiness 평가가 부족하다고 인정했다. high-stakes 응용에 쓰기 전에는 in-domain 데이터로 fine-tuning하고 검증하라고 권고했다.
- **GraphPFN 강점**: ICL과 finetuning을 모두 지원하며, finetuning 시 13개 데이터셋 중 11개에서 최고 성능을 낸다. 같은 구조에 random adapter를 붙여 finetuning하면 성능이 크게 떨어지므로, 이득이 구조가 아니라 사전학습에서 온다는 점을 보였다.

## 17. MOMENT ← TimesFM [context-horizon] (점수 0.59)
- **MOMENT 한계** (후속 논문 지적 — Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts (ICLR 2025)): MOMENT가 고정된 입력 context 길이에 묶여 있다는 점을 범용성의 제약으로 꼽았다. 여섯 개 벤치마크 평균의 zero-shot 장기 예측 비교에서 MOMENT는 MSE와 MAE가 가장 나쁜 baseline이었다.
- **TimesFM 강점**: 출력 patch를 입력 patch보다 길게 잡은 설계 덕분에 긴 horizon을 적은 decoding step으로 예측한다. 512 step 예측 ablation에서 출력 patch 128이 32보다 오차가 작았는데, step 수가 16에서 4로 줄어 누적 오차가 줄어든 효과로 해석된다.
- **TimesFM 강점**: 첫 patch를 무작위 길이로 masking하는 학습 트릭 하나로, patch 길이의 배수가 아닌 context까지 포함해 임의 길이 입력을 하나의 모델로 처리한다.

## 18. TimesFM ← MOMENT [tokenization] (점수 0.59)
- **TimesFM 한계** (후속 논문 지적 — Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts (2025)): TimesFM이 frequency embedding으로 데이터를 frequency 단위로 구분하는 방식을 사람이 정한 heuristic이라고 비판한다. frequency가 실제 패턴을 잘 대변하지 못하고, non-stationary 시계열에는 frequency 단위 구분이 너무 거칠다는 것이 근거다. 평가에서도 TimesFM의 Monash와 Electricity 결과에 사전학습 데이터 중복 표시를 달았다.
- **MOMENT 강점**: patching, RevIN, 좌측 zero-padding, channel 단위 처리라는 단순한 조합으로 길이·스케일·채널 수가 다른 데이터셋을 한 모델에 넣는다. 데이터셋마다 구조를 바꿀 필요가 없다.

## 19. GraphPFN ← TimesFM [prior-mismatch] (점수 0.54 · 분야 교차)
- **GraphPFN 한계** (조사자 판단): [analyst] 부록 통계에서 synthetic graph의 homophily는 중앙값 0.01, 상위 25% 경계가 0.08로 0 근처에 몰려 있다. 반면 city-reviews(0.69) 같은 강한 homophily 그래프는 prior 분포의 꼬리에 있다. NodeGround가 보고한 강한 homophily 조건에서의 열세와 맞아떨어지며, prior의 라벨-구조 결합 분포가 한쪽으로 치우쳐 있을 가능성을 시사한다. Class 비율도 SCM target 이산화에 맡겨져 있어 명시적으로 통제되지 않는다.
- **TimesFM 강점**: synthetic 데이터를 빼면 실제 데이터에 드문 granularity(quarterly, yearly, 10분 등)와 15분 ETT에서 성능이 떨어진다는 ablation을 보여, 실제 corpus의 빈 곳을 synthetic prior로 메우는 효과를 실증했다.

## 20. GraphPFN ← MOMENT [zero-shot-gap] (점수 0.49 · 분야 교차)
- **GraphPFN 한계** (조사자 판단): [analyst] ICL만 놓고 보면 G2T-LimiX 대비 우위가 크지 않다. GraphLand 전체 평균 rank는 3.50 대 3.88이고, tolokers-2·avazu-ctr·city-roads-M에서는 G2T-LimiX가 앞선다. 헤드라인의 큰 격차는 대부분 finetuning에서 나온다. 또 twitch-views처럼 random feature 추가 여부로 성능이 크게 바뀌는 데이터셋이 있어, 일부 이득은 추론 시 trick에도 기대고 있다.
- **MOMENT 강점**: supervision을 거의 쓰지 않아도 성능이 강하다. 고정 embedding 위 SVM만으로 UCR classification에서 데이터셋별로 학습한 방법 대부분을 앞섰고, linear probing만으로 장기 예측에서 PatchTST 다음 수준에 도달했으며, anomaly detection과 imputation에서도 TimesNet·GPT4TS보다 나았다.

## 21. GraphPFN ← TimesFM [zero-shot-gap] (점수 0.49 · 분야 교차)
- **GraphPFN 한계** (조사자 판단): [analyst] ICL만 놓고 보면 G2T-LimiX 대비 우위가 크지 않다. GraphLand 전체 평균 rank는 3.50 대 3.88이고, tolokers-2·avazu-ctr·city-roads-M에서는 G2T-LimiX가 앞선다. 헤드라인의 큰 격차는 대부분 finetuning에서 나온다. 또 twitch-views처럼 random feature 추가 여부로 성능이 크게 바뀌는 데이터셋이 있어, 일부 이득은 추론 시 trick에도 기대고 있다.
- **TimesFM 강점**: 200M 규모의 비교적 작은 모델로 Monash, Darts, ETT에서 zero-shot 성능이 dataset별 supervised 모델과 비슷한 수준에 도달했다. GPT-3 기반 LLMTime보다 Monash에서 10% 이상 좋으면서 추론 비용은 훨씬 작다.

## 22. MOMENT ← GraphPFN [pretraining-data] (점수 0.49 · 분야 교차)
- **MOMENT 한계** (조사자 판단): [analyst] Time-series Pile에는 평가에 쓰인 ETT, Electricity, Traffic, Weather, UCR 데이터셋의 train split이 그대로 들어 있다. test split은 분리했으므로 엄밀한 누수는 아니다. 그래도 장기 예측, classification, imputation 결과는 같은 데이터셋의 다른 구간을 본 in-distribution 성능에 가깝고, 처음 보는 도메인으로의 일반화를 보여주지는 않는다.
- **GraphPFN 강점**: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.

## 23. TimesFM ← GraphPFN [pretraining-data] (점수 0.49 · 분야 교차)
- **TimesFM 한계** (조사자 판단): [analyst] 실제 데이터의 time point 대부분이 Wiki pageview와 Google Trends라는 웹 관심도 시계열이다. 교통·전력 같은 물리 센서 시계열(Traffic, LibCity, Electricity)은 비중이 아주 작고 바로 그 데이터들이 흔한 벤치마크와 겹친다. 따라서 센서 네트워크 데이터에서의 zero-shot 성능이 일반화 능력인지 작은 중복 데이터의 기억인지 이 논문만으로는 구분하기 어렵다.
- **GraphPFN 강점**: Graph prior가 구조(multi-level SBM + preferential attachment)와 attribute(MLP·GNN neuron 혼합 SCM)를 분리해 설계되어 있고, 그래프 의존 정도를 mixing 확률로 조절할 수 있다. 부록 분석에서 synthetic graph의 평균 차수·clustering·power-law 적합도 분포가 실제 GraphLand 그래프 통계를 대체로 덮는다.

## 24. MOMENT ← TimesFM [inference-cost] (점수 0.48)
- **MOMENT 한계** (후속 논문 지적 — FoundTS: Comprehensive and Unified Benchmarking of Foundation Models for Time Series Forecasting (2024)): 통합 forecasting 벤치마크에서 MOMENT를 channel independence를 가정하는 대표 모델로 분류했다. 변수 간 상관이 강한 데이터셋에서는 channel dependence를 다루는 foundation model보다 뒤처졌고, 특히 Traffic에서 고전했다고 지적했다. runtime과 정확도 양쪽에서 dataset-specific 모델에 밀린다는 평가도 함께 내렸다.
- **TimesFM 강점**: 출력 patch를 입력 patch보다 길게 잡은 설계 덕분에 긴 horizon을 적은 decoding step으로 예측한다. 512 step 예측 ablation에서 출력 patch 128이 32보다 오차가 작았는데, step 수가 16에서 4로 줄어 누적 오차가 줄어든 효과로 해석된다.
- **TimesFM 강점**: 200M 규모의 비교적 작은 모델로 Monash, Darts, ETT에서 zero-shot 성능이 dataset별 supervised 모델과 비슷한 수준에 도달했다. GPT-3 기반 LLMTime보다 Monash에서 10% 이상 좋으면서 추론 비용은 훨씬 작다.

## 25. MOMENT ← TimesFM [zero-shot-gap] (점수 0.48)
- **MOMENT 한계** (후속 논문 지적 — When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection (ICLR 2026)): MOMENT를 포함한 다섯 TSFM 계열을 reconstruction·forecasting error 기반 anomaly detector로 평가했다. 모델 크기와 context 길이를 바꿔도 moving-window variance 같은 한 줄짜리 baseline과 유의미한 차이가 없었다. 이상 구간이 정상 구간보다 꼭 더 복원하기 어렵지 않기 때문에, 복원 오차로 이상을 잡는다는 전제 자체가 이 모델들에서 성립하지 않는다는 진단이다.
- **TimesFM 강점**: 200M 규모의 비교적 작은 모델로 Monash, Darts, ETT에서 zero-shot 성능이 dataset별 supervised 모델과 비슷한 수준에 도달했다. GPT-3 기반 LLMTime보다 Monash에서 10% 이상 좋으면서 추론 비용은 훨씬 작다.

# 빈칸: 한계로만 지적되고, 하네스 안에 해결한 논문이 없는 축

- **eval-protocol** (벤치마크 누수(사전학습 데이터와 평가셋 중복), 평가 protocol·baseline 공정성): GraphPFN, MOMENT, TimesFM
- **channel-dependency** (변수 간 의존성 모델링 (channel-independent 가정 포함)): MOMENT, TimesFM
- **uncertainty** (확률적 예측, calibration): MOMENT, TimesFM
- **context-scale** (context(학습 샘플 수, 노드 수, 시계열 길이)가 커질 때의 메모리·성능 한계): GraphPFN
- **covariates** (외생 변수·covariate 활용): TimesFM
- **graph-scale** (대규모 그래프 처리): GraphPFN
- **label-imbalance** (context 라벨 분포 불균형): GraphPFN
- **label-space** (지원 클래스 수·라벨 공간 제약): GraphPFN
- **nonstationarity** (분포 변화, regime shift, 오래된 history 처리): TimesFM
- **theory-gap** (왜 작동하는지에 대한 이론적 설명): TimesFM
- **training-stability** (사전학습·fine-tuning의 안정성, seed 간 변동): GraphPFN
