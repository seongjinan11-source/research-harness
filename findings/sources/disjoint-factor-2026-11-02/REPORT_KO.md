# Disjoint feature–label factor PFN 결과

**새 모델 학습을 포함한 소형 known-family pilot.** 최종 판정은 아래 gate와 별도 비용 결과에 한정한다.

Primary: 새 matched48 graphs, feature 노드와 label context의 교집합0, context1/2/4·관측노드3/6을 graph 내 평균, fixed3 seeds 평균. Accuracy가 primary이며 NLL/Brier와 Grid6 KL은 별도다.

| 방법 | Accuracy % | NLL | Brier | Grid6 exact KL |
|---|---:|---:|---:|---:|
| factor_pfn | 65.4004 | 0.628722 | 0.414710 | 0.056253 |
| mask_pfn | 57.0974 | 0.676368 | 0.483465 | 0.128746 |
| pcfi_pfn | 55.2304 | 0.677401 | 0.484152 | 0.129725 |
| constant_factor | 66.0053 | 0.621022 | 0.404660 | 0.083170 |
| grid6_exact | 66.2946 | 0.555341 | 0.386365 | 0.000000 |
| grid6_bp | 65.9557 | 0.555143 | 0.386196 | 0.000008 |
| feature_only | 50.3514 | 0.693147 | 0.500000 | 0.137822 |
| lp | 55.7044 | 1.215995 | 0.644138 | 0.665098 |
| class_prior | 49.8677 | 0.760336 | 0.558752 | 0.207920 |
| uniform | 50.2646 | 0.693147 | 0.500000 | 0.137822 |

## Paired 차이

| Factor PFN − 대조 | Accuracy %p [95% CI] | NLL [95% CI] |
|---|---|---|
| mask_pfn | +8.3030 [+2.7750, +14.0753] | -0.047645 [-0.141633, +0.050681] |
| pcfi_pfn | +10.1700 [+4.4253, +16.3513] | -0.048678 [-0.142740, +0.049630] |
| constant_factor | -0.6049 [-3.4860, +1.9721] | +0.007700 [-0.056868, +0.098511] |
| grid6_bp | -0.5553 [-2.8137, +1.6452] | +0.073580 [+0.019579, +0.150308] |

## 판정과 범위

Neural accuracy gate=True; feature-removal gate=False; mechanism gate=False; learned potential accuracy increment=False.

Feature latent state와 node class 사이의 관측값 factor를 통해 두 evidence source를 연결한다. PCFI처럼 완성 feature 행렬을 만들지 않으며 별도 role posterior head가 없다. 다만 CSBM/attributed-graph BP 자체는 선행연구가 있고, 고정2-feature·2-relation·binary family의 결과를 새 schema나 실제 graph 전이로 확대하지 않는다.

Factor 모델은 finite loopy BP와 Bethe score를 사용하는 conditional structured model이다. 원래 데이터 posterior의 exact 추론이나 연속 X의 정규화된 생성 likelihood를 학습한 것은 아니다. Feature-return 차단은 고정된 계산을 깨는 보조 개입이며 새 방법으로 선택하지 않는다.

실제 9 models, 2304 successful updates, 11952 saved predictions 재채점. Query 정답은 query CE 및 평가에만 사용한다. Test-side parameter tuning 없음.

세부: protocol.json, results/summary.json, paired.json, interventions.json, query_without_features.json, audit.json. 비용·competitive gate는 별도 benchmark 후 확정한다.

## 완료 후 비용

Factor median 6.837ms, Grid6-BP 11.874ms. BP-only reference 예측 일치 최대오차 0. Competitive gate=False. 외부 GPU 부하는 통제하지 않았다.

## 최종 해석

**정확도 개선은 보존한다. 현재 구성을 독립 메인 메소드로 확대하는 것은 보류한다.**

새 test bank의 disjoint primary에서 Factor PFN은 공개 Mask 대비 +8.3030%p, 실제 PCFI 입력으로 학습한 PFN 대비 +10.1700%p 높았다. 각각 paired graph CI가0을 제외했다. 다만 두 대조에 대한 NLL 차이의 CI는0을 포함했다.

고정 potential을 쓰는 같은 연산은66.0053%로 learned65.4004%보다 수치상 높고, learned−constant accuracy CI는0을 포함한다. Grid6-BP와의 accuracy 차이도0을 포함하지만, 비열등성·동등성을 증명한 것은 아니다. Grid6-BP 대비 NLL은 +0.073580 [+0.019579,+0.150308] 높았다. Grid6-exact predictive KL0.056253은 목표.01을 넘고, 측정 latency1.74배 차이는 사전2배 기준에 못 미쳤다. 따라서 현재 adaptive potential이나 신경망 자체의 필요성을 입증하지 못했다.

사전 지정한 context2/observed6의48 graph intervention에서는 feature shuffle−11.2269%p CI[−20.3704,−2.0833], graph permutation−21.7593%p CI[−34.3779,−8.7963]였다. 올바른 입력 정렬에 대한 의존성은 보존한다. 그러나 no-observed−standard는−6.1343%p CI[−12.9630,+0.9259]로 원래 feature-removal gate를 통과하지 않았다. 이를 feature 효용이0이라는 증명으로도, shuffle 결과로 원래 gate를 통과한 것처럼도 쓰지 않는다.

### Query 자체에도 feature가 없는 경우

이 분할은 사전 지정한 secondary이며 primary 판정을 바꾸지 않는다. 전체96 graphs 중 matched48을 유지한다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| Factor PFN | 64.6030 | 0.600055 |
| Mask PFN | 56.9842 | 0.679388 |
| PCFI PFN | 54.4846 | 0.684093 |
| Constant factor | 63.3009 | 0.599397 |
| Grid6-BP | 65.7650 | 0.585340 |

Factor−Mask accuracy+7.6188%p CI[+1.7946,+13.6193], NLL−0.079333 CI[−0.137580,−0.023154]였다. Factor−PCFI accuracy+10.1184%p CI[+3.5331,+16.8865], NLL−0.084038 CI[−0.142503,−0.027776]였다. 두 neural 대조보다 좋은 신호지만 Constant 대비 accuracy/NLL CI는0을 포함한다. Grid6-BP 대비 NLL 차이도 이 분할에서는0을 포함하며, 동등성을 증명한 것은 아니다.

Context에 두 class가 모두 있는43 graphs의 사후 진단에서는 Factor69.7513%, Mask64.5510%, PCFI63.0975%였다. Accuracy 차이 CI는0을 포함했고 NLL 차이는 각각−0.118258 [−0.201652,−0.035343], −0.117606 [−0.200205,−0.035256]였다. 이 진단을 새 primary나 별도 holdout으로 해석하지 않는다. 세부는 results/secondary_partitions.json이다.

### 판정 범위

- Context1/2/4는 **클래스당 개수가 아니라 총 label 수**다. 관측 node3/6은12개 중25/50% 관측, 즉75/50% whole-node missing이다. 이전99.5% 프로토콜의 성과로 쓰지 않는다.
- 두 typed relations와 binary class/linear Gaussian feature-law를 가진 작은 known-family다. Unseen semantic schema, 실제 graph, variable feature-law, 대형 sparse graph는 미실행이다.
- BP·Bethe 근사, potential fit, model assumptions와 학습량의 기여는 분리하지 않았다. NLL gap의 고유 원인을 확정하지 않는다.
- Old circuit을 이 bank에서 재학습하지 않았다. 이전 circuit의 절대 수치와 이번 성능을 직접 빼지 않는다.
- Feature-only는 disjoint에서 대칭성 때문에 확률이0.5다. Floating-point tie에 따른 accuracy 차이는 의미 있는 feature-only 분류 능력으로 해석하지 않는다.

원래 sources/bank lock, all3200 bank hashes,9 checkpoint finite/frozen state, 저장11952 prediction 재채점과27개 재실행을 확인했다. Supplementary partition 분석은 추가 학습·gate 변경 없이 저장 예측만 사용했다.
