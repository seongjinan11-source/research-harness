# Diffusion + Factor correction + Rényi-2/ESS — 실제 실행 결과

2026-11-02. 기존 Factor3개를 고정하고8,476-parameter denoiser3개를 실제로 학습했다. 새 학습3,072updates,seed당1024updates/batch16이다. 새96base graph/576episodes,primary는 matched48graph의6disjoint settings와3fixed seeds 평균이다. 같은 potential의 정확한 분포·실제 query 정답·Grid6 reference를 구분한다. Missing50/75%의 작은 known-family다.

## 사전 primary

| 방법 | Accuracy % | NLL | Brier | Exact-Factor KL ↓ | Grid6 KL ↓ |
|---|---:|---:|---:|---:|---:|
| mask_pfn | 57.6320 | 0.672858 | 0.480278 | 0.106673 | 0.120610 |
| pcfi_pfn | 56.7681 | 0.671670 | 0.479045 | 0.107023 | 0.121114 |
| constant_bp | 71.0813 | 0.598383 | 0.387143 | 0.173074 | 0.142698 |
| factor_bp | 71.4038 | 0.595856 | 0.390603 | 0.085031 | 0.091114 |
| factor_exact | 71.6449 | 0.561181 | 0.382866 | 0.000000 | 0.011446 |
| smc_gibbs1 | 69.7655 | 0.574529 | 0.395202 | 0.008698 | 0.020647 |
| smc_gibbs_cost | 68.8533 | 0.572085 | 0.392851 | 0.007171 | 0.017499 |
| diffusion_only | 65.6732 | 0.612413 | 0.424654 | 0.047351 | 0.060687 |
| diffusion_smc_fixed | 65.1510 | 0.632424 | 0.433833 | 0.079318 | 0.090732 |
| diffusion_smc_ess | 64.9526 | 0.626241 | 0.433258 | 0.071562 | 0.085439 |
| grid6_exact | 71.7055 | 0.548307 | 0.375268 | 0.011624 | 0.000000 |
| grid6_bp | 71.6642 | 0.548310 | 0.375277 | 0.011734 | 0.000026 |

**Gate: inference=False,classification=False,ESS increment=False.** KL 개선을 accuracy 성공으로 바꾸지 않는다. 아래는64particles/8stages 구현의 pointwise95%CI이며,세 학습 seed를 고정한48base-graph paired bootstrap5000회다. 다중비교 보정은 하지 않았다.

| 비교 (왼쪽−오른쪽) | Accuracy %p [95% CI] | Exact-Factor KL 차이 [95% CI] |
|---|---:|---:|
| diffusion_smc_fixed - diffusion_only | -0.5222 [-2.7434,+1.7817] | +0.031967 [+0.012991,+0.050540] |
| diffusion_smc_fixed - smc_gibbs_cost | -3.7023 [-6.3466,-1.2343] | +0.072148 [+0.060790,+0.084608] |
| diffusion_smc_fixed - factor_bp | -6.2528 [-9.1795,-3.6968] | -0.005713 [-0.035208,+0.020485] |
| diffusion_smc_ess - diffusion_smc_fixed | -0.1984 [-1.9664,+1.6011] | -0.007756 [-0.019864,+0.006172] |
| diffusion_smc_ess - smc_gibbs_cost | -3.9007 [-6.2983,-1.5625] | +0.064392 [+0.053542,+0.076688] |
| diffusion_only - smc_gibbs_cost | -3.1801 [-6.3317,-0.0244] | +0.040181 [+0.027731,+0.054193] |
| factor_exact - factor_bp | +0.2411 [-0.8474,+1.3200] | -0.085031 [-0.117743,-0.056384] |
| diffusion_smc_fixed - constant_bp | -5.9303 [-9.1008,-2.7998] | -0.093756 [-0.152182,-0.042761] |
| diffusion_smc_fixed - grid6_bp | -6.5132 [-9.7774,-3.4749] | +0.067585 [+0.055054,+0.081085] |

## 판정

**이번 learned diffusion proposal + MH/SMC + ESS controller를 메인 메소드로 확대하지 않는다.** 추론·분류·ESS 증분 gate가 모두 실패했다. Diffusion+fixed는 비용 대조 SMC보다 accuracy가 유의하게 낮았고, 같은 frozen-model marginal의 KL도 더 컸다. 더 저렴한1sweep SMC도 평균적으로 diffusion보다 좋았다. Pure diffusion보다 보정 arm의 finite-particle KL이 악화한 결과도 숨기지 않는다.

이는 diffusion 전체, RNP objective 또는 모든 Rényi 접근의 실패가 아니다. 닫는 범위는 이번512task/1024update·width24·64particle/8stage·기존 Factor 함수족과 fixed/ESS 스케줄이다. 학습량·proposal fit·초기 particle coverage·mutation mixing·finite-particle error의 고유 원인은 분리하지 않았다. 계산적으로 유효한 kernel이라는 것과 작은 budget에서 좋은 예측을 만든다는 것은 다르다.

## 무엇이 남았나

동일한 frozen Factor를 정확히 합산하면 Grid6 KL이0.091114에서0.011446으로 줄었다. 기존 potential이 참 생성 posterior인 것은 아니지만, 그 조건부 분포를 더 정확히 계산할 예측적 여지가 있었다. 반면 accuracy는71.4038%→71.6449%,차이+0.2411%p이고 CI는0을 포함했다. 따라서 현재 bank에서 “추론을 정확히 하면 classification이 크게 좋아진다”는 근거는 약하다. Exact 합산은 지수적 비용의 소형 진단이며 배포 메소드로 선택하지 않는다.

## ESS와 particle 검사

아래는 전체576episode의 세 seed 평균으로 primary48graph만의 값과 다르다. Final ESS는 가중치의 균등함을 나타낼 뿐, 초기 모드 손실·particle 간 의존성·예측 정확도를 측정하지 않는다. Pure diffusion의 ESS64는 uniform weights라서 정의상 나오는 값이다. Acceptance에는 self-transition도 포함될 수 있다.

| Sampler | Final ESS /64 | MH acceptance | 고유 final state 수 /64 | Role KL |
|---|---:|---:|---:|---:|
| smc_gibbs1 | 61.074 | 해당 없음 | 38.683 | 0.024296 |
| smc_gibbs_cost | 61.100 | 해당 없음 | 38.239 | 0.020796 |
| diffusion_only | 64.000 | 해당 없음 | 52.530 | 0.151594 |
| diffusion_smc_fixed | 60.946 | 0.6513 | 38.217 | 0.089703 |
| diffusion_smc_ess | 60.593 | 0.6156 | 37.962 | 0.104579 |

고유 state 수가 많아도 posterior mass를 올바르게 배분했다는 뜻은 아니다. 큰 correction 오차를 “weight collapse 한 가지 원인”으로 설명하지 않는다. Production 후 별도4-spin tree/4096particle 수치 감사에서는 세 corrected sampler가 정확한 node marginal과 최대오차0.01미만으로 일치했다. 이는 production의64particles 정확성이나 일반적인 수렴 증명이 아니다.

## 비용

개발 graph의 latency만 보고 ordinary SMC 비용 대조를2Gibbs sweeps로 잠갔다. 실제 단일 graph 측정에서도 fixed diffusion45.82ms와2sweep SMC50.44ms는 비슷했으나,1sweep SMC33.67ms·원 Factor BP4.73ms·Grid6-BP15.67ms가 더 빨랐다. 동일 particle/stage를 같은 wall time으로 부르지 않는다. 아래는 seed11의 사전 primary형 입력1개,3warmup/15repeat,CPU visible input→CPU probabilities이며 factor 추출·public PCFI 재계산을 포함한다. 모델과 reference state-table loading/I/O는 제외했고 외부GPU 부하는 통제하지 않았다.

| 방법 | Median ms | p10–p90 ms |
|---|---:|---:|
| smc_gibbs1 | 33.671 | 26.603–43.059 |
| smc_gibbs_cost | 50.436 | 46.199–55.461 |
| diffusion_only | 19.862 | 17.357–24.207 |
| diffusion_smc_fixed | 45.821 | 42.614–51.541 |
| diffusion_smc_ess | 64.272 | 57.345–73.474 |
| factor_bp | 4.725 | 3.923–5.103 |
| constant_bp | 3.088 | 2.925–4.616 |
| grid6_exact | 10.661 | 10.054–11.516 |
| grid6_bp | 15.670 | 14.550–18.256 |
| mask_pfn | 44.585 | 41.686–49.303 |
| pcfi_pfn | 52.267 | 48.294–59.164 |

실제 denoiser 학습 시간 합계15.665s,오프라인 train-joint teacher 준비 합계1.026s였다. Teacher의 full enumeration은 학습 시 추가 비용이며 추론에는 없다. 원 console의 마지막 PREPARED 시간은 마지막 seed부터 센 timer여서 전체 준비 비용으로 사용하지 않았다. Batch16 cached-potential 비용은 statistics.json에 따로 보존했다.

## 전체 조건·seed

원 sampled-label accuracy/NLL/Brier,exact-model KL,Grid6 KL은 모든 condition·seed에서 별도로 남겼다. 표는accuracy이며 subgroup으로 primary를 바꾸지 않는다.

| Arm | seed11 | seed12 | seed13 | 자기 피처 없는 query | Density-low | Signal-high |
|---|---:|---:|---:|---:|---:|---:|
| mask_pfn | 57.4363 | 57.9448 | 57.5149 | 57.7615 | 55.3158 | 55.7016 |
| pcfi_pfn | 57.2255 | 57.3123 | 55.7664 | 56.8550 | 52.9982 | 56.9169 |
| constant_bp | 71.0813 | 71.0813 | 71.0813 | 68.1257 | 57.9282 | 73.7765 |
| factor_bp | 71.4286 | 71.4327 | 71.3500 | 66.8823 | 58.5290 | 71.8171 |
| factor_exact | 71.6642 | 71.6642 | 71.6063 | 67.9140 | 58.8735 | 71.0483 |
| smc_gibbs1 | 70.1058 | 68.8327 | 70.3580 | 66.7106 | 56.7763 | 69.1744 |
| smc_gibbs_cost | 68.2002 | 69.8785 | 68.4813 | 65.3149 | 58.4987 | 70.3097 |
| diffusion_only | 65.7821 | 65.1538 | 66.0838 | 62.1913 | 58.0688 | 69.0394 |
| diffusion_smc_fixed | 65.9929 | 65.8606 | 63.5995 | 62.6429 | 55.2965 | 65.8317 |
| diffusion_smc_ess | 64.7859 | 65.1703 | 64.9016 | 61.7499 | 56.0847 | 65.4652 |
| grid6_exact | 71.7055 | 71.7055 | 71.7055 | 68.4590 | 55.7126 | 73.7434 |
| grid6_bp | 71.6642 | 71.6642 | 71.6642 | 68.4590 | 56.5228 | 73.6276 |

## 재현 및 범위

- 모든8640sampler episode output을 다시 실행해 original probability와 최대오차0을 확인했다. 기존 Mask/PCFI6개도 각1회 replay 최대오차0,source/checkpoint/bank hash 및3,072실제 updates를 확인했다.
- 기존 Factor/Mask/PCFI checkpoint9개와 원148개 findings는 바꾸지 않는다. 새 학습은denoiser3개뿐이다. Original Factor encoder·likelihood constants와 role prior는 동일하다.
- 원래 node/feature latent joint의 정확한 합산으로 inference fit을 측정했다. 실제 hidden query/helper labels를 joint teacher나 sampler가 입력받지 않는다. Query 정답은 평가에서만 사용했다.
- ESS는 Rényi-2 entropy identity를 사용한 기존 SMC 제어다. Rényi Neural Processes의 학습 목적, DPM-SNC/ReDiSC/RDSMC 원본 구현은 실행하지 않았다. 이를 해당 논문들의 실패로 기록하지 않는다.
- 작은2-relation/2-feature/binary schema와50/75%feature missing이다. Source/context는 disjoint지만 query/source는 겹칠 수 있다. 자기 피처 없는 query 결과는 별도 secondary이며99.5%missing·실제 graph·unseen schema/feature-law의 성과가 아니다.
- Density/signal OOD의 Grid6는 알려진 parameter support의 model-based reference이며 true-parameter Bayes가 아니다. 모든 CI는 pointwise,training seeds 고정이다.

정확한 연산은 METHOD_KO.md와protocol.json,원본 수치는results/statistics.json·benchmark.json·audit.json·convergence_audit.json에 있다. Raw bank·teacher joint draws·checkpoint·모든 probabilities는 workspace에 보존하고 공유 snapshot에는 제외한다.
