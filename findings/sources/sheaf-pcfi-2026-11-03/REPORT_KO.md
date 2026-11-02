# Sheaf-PCFI GraphPFN — synthetic 학습·실제 graph 고정 평가

2026-11-03. 판정: **현재 구현 확대 No-Go**.

**추가 학습은 synthetic task에서만 했고 실제 graph optimizer update는0회다.** 공개 GraphPFN 초기화에서6arm×3seed를 실제 학습했다. 이 보고서는 특정2D transport·RP16·384updates pilot의 판정이며 모든 sheaf 방법의 실패를 뜻하지 않는다.

## 주 비교: 원래의 양의 PCFI 전파를2D transport로 바꾸면 좋아지는가?

Confidence·관측 source clamp·마지막1회 inter-channel correction은 유지했다. 학습된 공간 operator를 PFN 앞의 미분 가능한 계산에 넣고 query CE로 함께 학습했다. Router459개 파라미터를 포함한 전체 trainable container는4,012,718개이며 static arm의 미사용 front weight는 gradient를 받지 않는다. 유효 용량이 같다는 주장은 하지 않는다.

주 지표는 자기 피처가 없는 query accuracy다. 90/99.5% whole-node missing, uniform/clustered4조건을 평균했다. 아래 실제 graph 수치는3context/mask 반복×3고정학습seed 평균이다. Context는기존 train mask에서class당최대4개이며 validation label은사용하지않았다.

| 방법 | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|
| mask | 39.05 | 27.17 | 19.57 | 23.18 |
| pcfi100 | 50.78 | 32.40 | 28.81 | 22.69 |
| pcfi200 | 50.23 | 33.16 | 29.55 | 22.61 |
| weighted | 50.76 | 33.46 | 29.50 | 22.59 |
| signed | 48.28 | 30.17 | 28.80 | 23.03 |
| sheaf | 50.46 | 33.64 | 28.32 | 22.40 |
| native_pcfi100 | 52.68 | 33.92 | 32.42 | 24.92 |
| public_native_pcfi100 | 50.12 | 32.15 | 32.19 | 20.49 |
| lp | 56.28 | 34.43 | 13.71 | 26.53 |
| negative_lp | 32.03 | 24.76 | 63.54 | 42.73 |
| context_prior | 12.69 | 7.76 | 17.26 | 3.17 |

단위는%. native_pcfi100은 **원본 고차원 피처에서PCFI100→같은고정RP16**을 사용하고 pcfi100과같은 synthetic-trained PFN으로 예측한 강한입력대조다. public_native_pcfi100은 같은wrapper·RP16·singleforward에서 public 초기가중치를 유지한 대조로, 공식 full전처리·ensemble 재현 점수는아니다. lp/negative_lp는고정100step 양/음전파이며 CLP/CPGNN의 재현이아니다. context_prior는context 빈도이며 동률이면첫class를선택해class불균형query에서매우낮을수있다.

Heterophily 두 graph 평균에서 sheaf−pcfi100은-0.392%p, sheaf−signed는-0.554%p, sheaf−원본피처PCFI는-3.308%p다. Homophily 두 graph 평균차이는+0.460%p다. 이4개는개발용graph이며 dataset-population 유의성을주장하지않는다.

강한고정대조도남긴다. Feature를쓰지않는negative_lp는Texas63.54%,Wisconsin42.73%였다. 현재PFN구성이이구조/context기준선도넘지못했다. 이것은signed label전파와node-feature전파가동일한연산이라는뜻이나새PFN의성과라는뜻이아니며,관측된성능격차만보존한다.

## 새 synthetic graph에서의 결과

64train graphs×6episodes=384task를seed당1회노출했다. 새32test basegraphs,128nodes,16features,2~7classes다. Label-SBM의homo/hetero와feature filter의+/−law를교차했다. Class수는독립RNG추출이다. 이작은bank가class별정확한factorial matching인것은아니다.

| 방법 | Accuracy % | NLL | Brier |
|---|---|---|---|
| mask | 47.250 | 1.26855 | 0.61712 |
| pcfi100 | 45.159 | 1.31004 | 0.64037 |
| pcfi200 | 45.021 | 1.30652 | 0.63712 |
| weighted | 45.176 | 1.30170 | 0.63346 |
| signed | 43.596 | 1.34178 | 0.65541 |
| sheaf | 45.291 | 1.30224 | 0.63456 |

| Sheaf − 비교군 | Accuracy 차이 %p | 95% graph-paired CI |
|---|---|---|
| mask | -1.959 | [-3.159, -0.637] |
| pcfi100 | +0.132 | [-0.585, +0.968] |
| pcfi200 | +0.271 | [-0.207, +0.772] |
| weighted | +0.116 | [-0.303, +0.520] |
| signed | +1.695 | [+0.774, +2.663] |

CI는32basegraph paired bootstrap5000회다. 같은graph의4조건과3고정학습seed를먼저평균했다. 노드나seed를독립graph처럼세지않았다. PCFI100/200/weighted대비accuracy CI가0을포함하므로증분우위는미확인이다. Signed보다높다는것만으로cross-stalk메소드의성공으로판정하지않는다. Mask보다낮은결과도함께보존한다.

| Label mode / feature filter | mask | pcfi100 | pcfi200 | weighted | signed | sheaf |
|---|---|---|---|---|---|---|
| homo_smooth | 63.46 | 62.21 | 62.15 | 62.99 | 58.68 | 61.85 |
| homo_rough | 63.06 | 57.57 | 58.15 | 58.69 | 55.93 | 58.78 |
| hetero_smooth | 35.78 | 34.70 | 33.97 | 33.86 | 33.59 | 34.96 |
| hetero_rough | 26.70 | 26.15 | 25.81 | 25.17 | 26.18 | 25.57 |

Feature filter 이름smooth/rough는생성계수의부호를뜻하며 달성된featureenergy와labelagreement는bank에별도기록했다. Label heterophily를피처이웃불일치와동일시하지않는다.

학습 후 transport 진단에서는weighted/signed/sheaf 모두synthetic missing test의첫대각성분 a<0비율이0이었다. Sheaf의평균절대cross-stalk는0.05523였다. **첫좌표의직접적인scalar부호전환은관측되지않았다. 이는2D행렬의모든성분이나고유값이양수라는뜻은아니다.** 다른좌표와반복경로의signed효과까지없었다고단정하지않으며현재결과를음의전파나표준NSD원리의반증으로쓰지않는다.

## 원본 피처에서 전파한 추가 진단

주 비교결과를읽기전에추가범위를정했다. Seed11·첫context/mask반복·4결측조건만사용한다. 같은합성학습router를각physicalchannel에공유하고,spatial propagation을128channel씩처리한뒤전체channel의correlation을한번계산하고RP16으로압축한다. 추가학습·실제정답tuning은없다. 공간계산chunking검증오차는4.2e−7이하다. 이진단은원본피처에서학습한모델이아니며primary대체나최고조합선택에쓰지않는다.

| 입력/방법 | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|
| pcfi100 | 48.70 | 33.64 | 32.49 | 26.51 |
| native_pcfi100 | 51.11 | 32.08 | 32.01 | 26.44 |
| sheaf | 46.67 | 33.78 | 33.75 | 28.94 |
| native_pcfi200 | 50.56 | 34.11 | 36.24 | 28.12 |
| native_weighted | 49.74 | 34.49 | 34.66 | 29.10 |
| native_signed | 52.03 | 33.08 | 38.96 | 27.06 |
| native_sheaf | 50.06 | 33.04 | 34.03 | 30.18 |

첫반복에서native_sheaf는Texas/Wisconsin의native_pcfi100보다높지만,Cora/CiteSeer에서는일관되게높지않고Texas에서는native_signed가더높다. 이는입력좌표순서에따른개발용신호이며3seed×3반복검증이나원본피처에서의학습성과로승격하지않는다.

## 피처 효용과 관측 source 수

아래는seed11의3반복·4조건평균이다. 피처제거·source-row shuffle에도원래결측query집합을그대로유지했다. 관측수0/1이면shuffle이아무변화를못주는경우가있으므로분리해서해석해야한다.

| Graph | 방법 | Full 입력 % | No-feature % | Shuffled % |
|---|---|---|---|---|
| Cora | pcfi100 | 51.55 | 25.14 | 48.88 |
| Cora | sheaf | 50.11 | 25.11 | 49.20 |
| CiteSeer | pcfi100 | 32.92 | 20.48 | 31.33 |
| CiteSeer | sheaf | 33.88 | 24.51 | 31.61 |
| texas | pcfi100 | 28.92 | 16.32 | 28.00 |
| texas | sheaf | 29.42 | 20.68 | 28.13 |
| wisconsin | pcfi100 | 22.29 | 24.48 | 23.23 |
| wisconsin | sheaf | 23.29 | 31.40 | 24.19 |

| Graph | 결측조건 | 관측 노드 수(3반복) |
|---|---|---|
| Cora | uniform90 | 259,286,294 |
| Cora | uniform995 | 8,12,21 |
| Cora | cluster90 | 271,271,271 |
| Cora | cluster995 | 14,14,14 |
| CiteSeer | uniform90 | 326,322,345 |
| CiteSeer | uniform995 | 20,13,24 |
| CiteSeer | cluster90 | 333,333,333 |
| CiteSeer | cluster995 | 17,17,17 |
| texas | uniform90 | 17,17,20 |
| texas | uniform995 | 2,0,0 |
| texas | cluster90 | 18,18,18 |
| texas | cluster995 | 1,1,1 |
| wisconsin | uniform90 | 31,34,27 |
| wisconsin | uniform995 | 1,0,1 |
| wisconsin | cluster90 | 25,25,25 |
| wisconsin | cluster995 | 1,1,1 |

관측수가0인episode도재추출하지않았다. 모든missing을복구할정보가있다고가정하지않는다. Reconstruction은hidden truth를예측후에만읽은RP16좌표의MSE다. 원래raw-feature공간의MSE가아니며정확도성공기준을대체하지않는다.

| RP16 missing MSE | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|
| mask | 1.069 | 1.082 | 3.773 | 1225344457477.720 |
| pcfi100 | 1.119 | 1.203 | 3.868 | 1225344453623.051 |
| pcfi200 | 1.132 | 1.241 | 3.897 | 1225344448155.443 |
| weighted | 1.132 | 1.241 | 3.896 | 1225344448170.885 |
| signed | 1.119 | 1.206 | 3.850 | 1225344450299.524 |
| sheaf | 1.132 | 1.242 | 3.897 | 1225344448235.543 |

## Full-feature reference와 확률 점수

| Full-feature accuracy % | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|
| mask | 38.47 | 28.11 | 33.32 | 20.33 |
| pcfi100 | 36.66 | 26.56 | 30.86 | 24.10 |
| pcfi200 | 35.94 | 26.99 | 31.12 | 23.87 |
| weighted | 35.76 | 26.16 | 30.83 | 24.54 |
| signed | 35.54 | 27.08 | 29.32 | 24.75 |
| sheaf | 35.27 | 26.70 | 30.53 | 25.63 |

Full-feature에서는모든cell을clamp하므로동일checkpoint내부에서front의전파효과가사라진다. 모델별차이는학습경로가달랐던결과다. Missing primary와query집합도달라직접적인missing손실차이의분해로사용하지않는다.

| Graph | 방법 | NLL | Brier | Macro-F1 % |
|---|---|---|---|---|
| Cora | pcfi100 | 1.50082 | 0.65389 | 50.12 |
| Cora | weighted | 1.48161 | 0.64996 | 50.24 |
| Cora | signed | 1.50085 | 0.67054 | 47.87 |
| Cora | sheaf | 1.48096 | 0.64903 | 49.83 |
| Cora | native_pcfi100 | 1.47131 | 0.63705 | 52.35 |
| CiteSeer | pcfi100 | 1.74514 | 0.79977 | 28.55 |
| CiteSeer | weighted | 1.72980 | 0.79175 | 29.66 |
| CiteSeer | signed | 1.74241 | 0.80285 | 26.40 |
| CiteSeer | sheaf | 1.74120 | 0.79381 | 29.92 |
| CiteSeer | native_pcfi100 | 1.79455 | 0.80762 | 30.01 |
| texas | pcfi100 | 2.42474 | 0.93566 | 16.58 |
| texas | weighted | 2.33083 | 0.90930 | 16.46 |
| texas | signed | 2.29289 | 0.92773 | 16.41 |
| texas | sheaf | 2.32575 | 0.91011 | 16.24 |
| texas | native_pcfi100 | 2.60593 | 0.95720 | 18.95 |
| wisconsin | pcfi100 | 2.17169 | 0.95446 | 17.92 |
| wisconsin | weighted | 2.01913 | 0.91617 | 18.11 |
| wisconsin | signed | 2.19686 | 0.96185 | 17.76 |
| wisconsin | sheaf | 2.01596 | 0.91569 | 17.49 |
| wisconsin | native_pcfi100 | 2.41785 | 1.00079 | 18.96 |

## 비용·검증·설계 교정

유효학습은18checkpoints/6912updates,학습시간합1894.7초다(3process 병렬실행의elapsed time과다름). 초기생성기의class수–featurelaw결합을성능표판독전에발견해mask3개의1152updates를폐기하고모든arm을public초기화에서다시학습했다. 원본은별도폴더에보존했다. Smoke optimizer step8회도폐기했다. [교정기록](DESIGN_CORRECTION_KO.md)을참조한다.

| 방법 | Graph | CPU 입력→확률 median ms | p10/p90 ms | 추가 GPU MiB |
|---|---|---|---|---|
| mask | Cora | 118.30 | 115.01/120.26 | 184.08 |
| mask | texas | 57.10 | 50.58/59.44 | 20.13 |
| pcfi100 | Cora | 120.58 | 116.52/124.76 | 184.08 |
| pcfi100 | texas | 54.78 | 48.29/63.32 | 20.13 |
| pcfi200 | Cora | 121.75 | 118.47/126.27 | 184.08 |
| pcfi200 | texas | 57.70 | 54.03/61.62 | 20.13 |
| weighted | Cora | 134.62 | 129.59/141.83 | 183.95 |
| weighted | texas | 68.16 | 63.13/73.75 | 20.13 |
| signed | Cora | 143.53 | 131.98/146.11 | 183.95 |
| signed | texas | 66.58 | 62.59/71.06 | 20.13 |
| sheaf | Cora | 135.35 | 128.90/140.57 | 183.95 |
| sheaf | texas | 70.66 | 64.98/79.86 | 20.13 |

비용은RTX3090·현재공통구현에서warmup3/repeat10으로측정했다. 매번RP16입력으로geometry·warm100을다시계산하고이후전파와PFN1회를포함한다. Mask에도불필요한공통geometry비용이포함된다. Checkpoint loading·최초RP·원본피처PCFI비용은제외되어native-PCFI와속도비교한표가아니다. 다른사용자의GPU활동은통제하지못했다.

Audit PASS_WITH_REPLAY_VARIATION: bank712개와저장예측4756개hash,18checkpoint,동일초기화·schedule·고정가중치,12forecast재실행을확인했다. 최대checkpoint재실행확률차이는0.0074972,같은session반복차이는0.0052063다. 초기.005재현기준은weighted/test에서.00749716으로실패했고,뒤의audit는차이를숨기지않고argmax변경과평균차이까지기록했다. .1은손상검사용상한일뿐엄격재현PASS기준이아니다. 표는원래저장예측을모두다시scoring한값이며재학습·예측교체는없다. PCFI/SciPy,identity환원,source clamp,zero-source,hidden값·querylabel poison,class-name permutation 및router gradient preflight는통과했다. 구현/정보경계검증과엄격수치재현을구분한다.

재실행한episode의argmax변경은최대2.48%였으며,같은session반복에서도차이가관측됐다. Front수치차이는최대2.9e-06였다. 작은입력차이의증폭가능성은있지만원인을단정하지않는다. Bootstrap CI는저장된예측에조건부이며이수치변동까지포함한CI가아니다. 특히+0.13%p같은작은차이를재현된방법우위로해석하지않는다.

## 판정과 닫는 범위

사전에정한gate: synthetic=False,real-development=False,native-PCFI floor=False,확대=False.

**현재의2D symmetric-contraction Sheaf-PCFI를메인메소드로확대하지않는다.** PCFI100/200/양수weighted보다합성accuracy증분이확인되지않았고실제heterophily평균에서도개선하지못했다. 일부CiteSeer증분이나signed대비개선을전체성공으로묶지않는다. 지금결과를살리기위한loss·head·step증가는실행하지않았다.

실패원인의고유한인과분해는미확인이다. Warm positive100,RP16,제한된transport함수족,384updates,synthetic–real차이는설정상의제한이며확인된원인이라고쓰지않는다. 이실험은원본NSD·FSD-CAP·CLP/CPGNN재현,원본피처공간합성학습,일반적sheaf표현력,heterogeneous graph의실패를입증하지않는다. 재개하려면현재한계중무엇을어떤연산으로바꾸며왜강한native-PCFI와양수weighted를넘을지를새근거로명시해야한다.

계산정의는[METHOD_KO.md](METHOD_KO.md),원점수는[statistics.json](results/statistics.json),추가진단은[supplement.json](results/supplement.json),정보경계/재실행은[audit.json](results/audit.json)에있다.

문헌: [PCFI](https://arxiv.org/abs/2305.16618)의confidence/채널전파와 [Neural Sheaf Diffusion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html)의학습가능transport를출발점으로삼았다. 원래논문의heterophily학습결과를이번의synthetic-only전이성과로간주하지않는다.
