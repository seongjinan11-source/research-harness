# NSD token GraphPFN — synthetic 학습·실제 고정 평가

2026-11-03. 판정: **현재 pilot 확대 기준 미통과**.

관측값·mask·context에서 바로 시작하고 PFN hidden state와 restriction map을 함께 갱신하는 모델을 실제 학습했다. PCFI로 먼저 채우거나 scalar completion으로 되돌리지 않는다. 추가 학습은 synthetic query CE만 사용했고 실제 graph optimizer update는0회다. 이 구성은 NSD의 연산을 PFN에 넣은 SO(2) adapter이며, 원본 NSD의 전체 아키텍처·target-graph 학습 재현이나 새로운 원리라고 주장하지 않는다.

핵심결과:synthetic dynamic-PCFI는+1.872%p,95%CI[0.959, 2.84]다. 그러나identity대비-0.083%p/static대비+0.221%p의CI는0을포함한다. 실제heterophily평균dynamic-PCFI는-3.252%p다. 따라서합성에서의개선과learned/refreshedgeometry의독립가치를구분한다.

## 계산과 비교

PFN의192차원 feature/label token을2×96으로 해석한다.0/3/6/9층에서 ordered endpoint restriction F_i,e/F_j,e를 만들고, weighted augmented-normalized sheaf Laplacian과 학습된 좌우 mixing·ELU residual update를 적용한다. Dynamic은 현재 token에서 map을4번 계산하고 Static은 첫 map을4층에서 재사용한다. Identity는 map만I다. 세 모델의 학습 가능한 구조는 같으며, 관측 feature-pair token/context label token은 NSD 단계에서 직전 상태로 고정한다. 일반 PFN 층은 이후 latent state를 갱신할 수 있다.

Mask/PCFI100도 같은 public initialization·동일 common modules·synthetic task 순서·1,024updates로 새로 학습했다. Plain arm의 미사용 NSD container가 있다는 사실을 동일 유효 용량으로 해석하지 않는다. **이전384update pilot과 절대 점수를 빼서 구조의 효과라고 쓰지 않는다.**

64개 기존 corrected train graph에8조건을 부여한512episode를2pass 사용했다. 원래6조건의 관측값·mask·context·query 배열은 그대로 보존했고, PCFI cache는 원래100step+channel correction을 완료했다. Source2/8을 추가한 것은 informative geometry 조건도 학습하기 위한 사전 결정이며, 원래의 zero-source99.5% 조건을 제거하지 않았다.

## 새로운 synthetic graph: primary

새64base graph,128nodes,16features,2~7classes,기존과같은 label-SBM×feature-filter family다. Primary는90/99.5%·uniform/cluster4조건의 자기 피처가 없는 query accuracy다. 조건·3고정학습seed를graph내평균하고base graph paired bootstrap5000회를계산했다.

| 방법 | Accuracy % | NLL | Brier |
|---|---|---|---|
| mask | 48.608 | 1.20643 | 0.59048 |
| pcfi100 | 47.735 | 1.22741 | 0.60197 |
| identity | 49.691 | 1.18657 | 0.57936 |
| static | 49.387 | 1.19357 | 0.58383 |
| dynamic | 49.608 | 1.19134 | 0.58245 |

| Dynamic − 비교군 | Accuracy 차이 %p | 95% graph-paired CI |
|---|---|---|
| mask | +1.000 | [+0.350, +1.625] |
| pcfi100 | +1.872 | [+0.959, +2.840] |
| identity | -0.083 | [-0.564, +0.407] |
| static | +0.221 | [-0.280, +0.751] |

## 실제 graph: 개발용 고정 평가

이전4dataset/3context-mask 반복을 재사용했다. 기존 train mask에서class당최대4개context를선택하고공식test mask에서평가한다. 나머지label과validationlabel은입력/선택/학습에사용하지않았다. 모든학습된arm은3고정trainingseed평균이다. 실제graph는선정된developmentbank이며새독립realtest로부르지않는다.

| 방법 | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|
| mask | 40.05 | 25.01 | 21.28 | 23.86 |
| pcfi100 | 50.15 | 30.84 | 27.77 | 21.61 |
| identity | 37.16 | 24.24 | 21.43 | 23.28 |
| static | 36.35 | 25.43 | 19.64 | 24.47 |
| dynamic | 38.07 | 22.35 | 19.06 | 23.81 |
| native_pcfi100 | 52.03 | 33.87 | 33.26 | 24.44 |
| public_native_pcfi100 | 50.12 | 32.15 | 32.19 | 20.49 |
| lp | 56.28 | 34.43 | 13.71 | 26.53 |
| negative_lp | 32.03 | 24.76 | 63.54 | 42.73 |
| context_prior | 12.69 | 7.76 | 17.26 | 3.17 |

단위%. 주 입력은 fixed RP16·관측값만의RMS/clip이다. Native PCFI는원본physicalfeature에서전파한뒤같은RP16을적용해새pcfi100checkpoint로예측했다. Native는전파때원본고차원관측정보를사용하고주모델은미리압축된RP16을받으므로동일표현정보의geometry-only대조가아니라강한전체pipelinefloor다. Public은동일wrapper의singleforward초기가중치대조이며공식full전처리/ensemble재현이아니다. 고정positive/negativeLP는동일입력의이전확률artifact를hash확인해재사용했고CLP/CPGNN의재현이아니다. Context prior동률은첫class선택으로낮은accuracy가발생할수있다.

| Heterophily mean: Dynamic − 비교군 | %p |
|---|---|
| mask | -1.136 |
| pcfi100 | -3.252 |
| identity | -0.921 |
| static | -0.620 |
| native_pcfi100 | -7.416 |
| negative_lp | -31.695 |

Homophily mean dynamic−pcfi100은-10.288%p다.4dataset의노드를독립표본처럼bootstrap하거나전체graph모집단의유의성으로확대하지않는다.

## 실제로 geometry를 갱신했는가?

| 방법/평가 | stage별 cos-negative 비율 | stage별 첫map과의 차이 | 최대anchor변경 |
|---|---|---|---|
| identity/test | 0.0000,0.0000,0.0000,0.0000 | 0.0000,0.0000,0.0000,0.0000 | 0 |
| identity/real | 0.0000,0.0000,0.0000,0.0000 | 0.0000,0.0000,0.0000,0.0000 | 0 |
| static/test | 0.0354,0.0354,0.0354,0.0354 | 0.0000,0.0000,0.0000,0.0000 | 0 |
| static/real | 0.0378,0.0378,0.0378,0.0378 | 0.0000,0.0000,0.0000,0.0000 | 0 |
| dynamic/test | 0.0456,0.0292,0.0502,0.0287 | 0.0000,0.1573,0.2319,0.1876 | 0 |
| dynamic/real | 0.0562,0.0128,0.0386,0.0267 | 0.0000,0.1705,0.2355,0.1913 | 0 |

여기서negative는SO(2)transport의cos항(T00=T11)이음수인비율이다. 행렬모든성분이음수라는뜻도,그러한상태가정답geometry라는뜻도아니다. Refresh차이는첫단계map과의차이다. Map이달라졌다는사실은갱신연산이작동했다는검증이며그갱신의예측적유용성은static대비성능으로판단해야한다.

## 고정 source 수와 no-source: secondary

| 조건 | mask | pcfi100 | identity | static | dynamic |
|---|---|---|---|---|---|
| uniform90 | 50.14 | 49.86 | 51.43 | 51.25 | 51.02 |
| uniform995 | 46.73 | 44.70 | 47.91 | 47.44 | 48.04 |
| cluster90 | 49.50 | 50.30 | 50.34 | 50.18 | 51.01 |
| cluster995 | 48.05 | 46.07 | 49.08 | 48.68 | 48.36 |
| sources2 | 48.92 | 48.24 | 49.86 | 49.10 | 49.99 |
| sources8 | 49.29 | 48.83 | 50.37 | 49.79 | 49.93 |

| Primary partition | mask | pcfi100 | identity | static | dynamic |
|---|---|---|---|---|---|
| zero | 45.43 | 44.42 | 47.17 | 46.43 | 47.20 |
| nonzero | 49.09 | 48.24 | 50.08 | 49.84 | 49.98 |

Source2/8은별도mask/context/query설정이며다른case의절대차이를source수하나의인과효과로쓰지않는다. No-source여도label token과graph로계산이진행되지만새로운관측feature정보를만들었다고해석하지않는다. Primary를secondary로바꾸거나좋은case만고르지않는다.

| 실제graph | 관측조건 | 관측노드수(3반복) |
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

관측수가0인경우도유지했다. 위횟수는3개의mask/context반복이며학습seed3개로복제해독립관측수로세지않는다.

## Feature/context 개입: seed11 진단

| Graph | 방법 | 원래입력 % | Feature제거 % | Feature셔플 % | Context셔플 % | 고정map전환 % |
|---|---|---|---|---|---|---|
| Cora | pcfi100 | 50.10 | 30.34 | 47.32 | 13.72 | — |
| Cora | dynamic | 38.89 | 32.12 | 37.82 | 13.31 | 38.16 |
| CiteSeer | pcfi100 | 30.00 | 24.74 | 28.50 | 14.77 | — |
| CiteSeer | dynamic | 22.85 | 20.97 | 24.55 | 17.60 | 23.97 |
| texas | pcfi100 | 28.48 | 16.74 | 27.81 | 22.63 | — |
| texas | dynamic | 15.24 | 14.19 | 14.46 | 21.69 | 19.38 |
| wisconsin | pcfi100 | 20.82 | 28.38 | 24.38 | 15.46 | — |
| wisconsin | dynamic | 22.36 | 26.04 | 21.41 | 13.37 | 20.40 |

각개입은원래missingquery집합을고정한다. Feature셔플은관측source내순열,context셔플은제공된label의대응만섞는다. No-feature에서PCFIcache도재계산/제거해잔존정보를차단했다. 고정map전환은같은dynamiccheckpoint의추론시freeze진단이며추가학습된static모델과구분한다. 가장좋은개입을새방법으로선택하지않았다.

## Full-feature와 확률 품질

| Full-feature accuracy % | Synthetic | Cora | CiteSeer | texas | wisconsin |
|---|---|---|---|---|---|
| mask | 96.61 | 36.63 | 26.49 | 30.55 | 21.45 |
| pcfi100 | 96.70 | 38.22 | 26.83 | 31.11 | 26.73 |
| identity | 96.37 | 36.64 | 25.83 | 28.40 | 24.55 |
| static | 96.65 | 37.10 | 26.37 | 29.34 | 22.99 |
| dynamic | 96.13 | 37.63 | 26.92 | 31.12 | 25.40 |

Full-feature는all-queryreference로missingprimary와query집합이다르다. Dynamic은fullfeature에서도querylabeltoken을갱신하므로원래scalarimputer실험과다르다. Syntheticfull과realfull의차이는전이의단서이며특정원인의증명은아니다.

| Graph | 방법 | NLL | Brier | Macro-F1 % |
|---|---|---|---|---|
| Cora | pcfi100 | 1.53131 | 0.66804 | 49.98 |
| Cora | identity | 1.67889 | 0.74824 | 37.23 |
| Cora | static | 1.68006 | 0.75344 | 36.63 |
| Cora | dynamic | 1.65589 | 0.74566 | 38.81 |
| Cora | native_pcfi100 | 1.55395 | 0.65865 | 51.98 |
| Cora | negative_lp | 2.20200 | 0.86373 | 31.99 |
| CiteSeer | pcfi100 | 1.87618 | 0.83783 | 27.91 |
| CiteSeer | identity | 1.85981 | 0.86475 | 21.09 |
| CiteSeer | static | 1.85823 | 0.86250 | 21.68 |
| CiteSeer | dynamic | 1.96974 | 0.90354 | 19.11 |
| CiteSeer | native_pcfi100 | 2.04233 | 0.88148 | 29.94 |
| CiteSeer | negative_lp | 1.87034 | 0.83601 | 25.64 |
| texas | pcfi100 | 2.53853 | 0.94260 | 16.96 |
| texas | identity | 2.40182 | 0.94595 | 15.63 |
| texas | static | 2.44603 | 0.98073 | 14.67 |
| texas | dynamic | 2.56458 | 1.02748 | 14.08 |
| texas | native_pcfi100 | 2.73681 | 0.96205 | 20.11 |
| texas | negative_lp | 1.45691 | 0.75119 | 40.75 |
| wisconsin | pcfi100 | 2.08644 | 0.94634 | 17.69 |
| wisconsin | identity | 1.94318 | 0.87835 | 19.32 |
| wisconsin | static | 1.99462 | 0.88440 | 19.69 |
| wisconsin | dynamic | 2.04402 | 0.90914 | 19.27 |
| wisconsin | native_pcfi100 | 2.48795 | 1.00717 | 19.77 |
| wisconsin | negative_lp | 1.64461 | 0.79909 | 28.78 |

## 비용과 검증

총15checkpoints,15360validupdates,학습시간합3597.8초다.3process병렬실행의벽시계시간과다르다. Devsmoke5steps는전부폐기했다. 실제학습0회다.

| 방법 | Graph | median ms | p10/p90 ms | 추가GPU MiB |
|---|---|---|---|---|
| mask | Cora | 73.85 | 72.93/74.96 | 149.72 |
| mask | texas | 49.29 | 44.14/54.82 | 17.20 |
| pcfi100 | Cora | 104.85 | 101.66/110.01 | 149.88 |
| pcfi100 | texas | 62.14 | 56.44/66.62 | 17.21 |
| identity | Cora | 120.89 | 119.02/123.03 | 235.33 |
| identity | texas | 58.22 | 52.69/60.65 | 21.03 |
| static | Cora | 118.74 | 115.85/121.35 | 235.49 |
| static | texas | 55.98 | 53.38/57.96 | 21.04 |
| dynamic | Cora | 118.14 | 117.05/119.88 | 235.81 |
| dynamic | texas | 56.53 | 53.75/64.54 | 21.05 |

CPU의visibleRP16입력에서CPU확률까지측정했다. PCFI는spatial100+channelcorrection을다시계산하고NSD는confidence/map/hiddenupdate를포함한다. Checkpointload·최초RP·원본physicalfeaturePCFI는제외한다. Warmup3/repeat10이며다른사용자GPU활동은통제하지못했다.

Audit상태:PASS_WITH_REPLAY_VARIATION. 입력1164개와예측8136개hash,15checkpoint와동일초기화/schedule/고정가중치를검사했다.10selectedforecast재실행최대확률차이는0.0120428,같은session반복차이는0.0126505,최대argmax변경은0.50%다. .1assertion은손상검사용일뿐엄격재현기준이아니다. CI는저장예측에조건부이며이수치변동전체를포함하지않는다.

명시적denseincidence일치·LaplacianPSD/spectrum·nodepermutation·공급된map하의gaugeidentity·mapgradientfinite-difference·negativecos표현가능성을검증했다. Gauge검증은학습mapencoder전체의gaugeequivariance주장이아니다. Anchor고정·static/dynamicrefresh·미사용PCFIcachepoison·mapgradientsmoke도통과했다.

## 판정 범위

사전기준: {"synthetic": false, "real_input": false, "negative_lp_floor": false, "mask_floor": true, "expand": false}

PCFI대비이득,추가hidden diffusion효과(identity),geometry효과(static),갱신효과(dynamic)를별도로읽는다. 일부accuracy/NLL/source조건의개선만으로모든gate성공으로묶지않는다. 기준미통과시새loss/head/추가update로현재구성을구제하지않는다. 결과의고유한실패원인은이비교만으로확정할수없다.

새syntheticgraph도동일knownfamily이며실제graph는RP16의기존developmentbank다. 원본NSD나모든sheaf의실패/성공,unseenheterogeneousschema일반화,완전한missingmarginalization을입증하지않는다. [METHOD_KO.md](METHOD_KO.md),[원점수](results/statistics.json),[audit](results/audit.json),[개입](results/control_summary.json)을함께보면된다.
