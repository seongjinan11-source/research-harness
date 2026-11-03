# 독립 문헌·메소드 신규성 심사

2026-11-03. 심사 대상: `DRAFT_KO.md`의 confidence-weighted flat SO(2) node-frame frontend. 초안, `../METHOD_KO.md`, 기존 좌표 정렬 검토와 핵심 코드의 transport/PFN freeze 부분을 읽고, 아래 선행연구의 원문을 제한적으로 대조했다. 수치는 초안의 개발 결과를 전제로 하며 실험을 재실행하거나 데이터 누수를 전면 감사한 평가는 아니다. 문헌 검색은 전체 문헌조사가 아니며, 직접 일치 논문을 발견하지 못했다는 사실에 신규성 점수를 주지 않았다.

**현재 판정은 weak reject다. 방법 자체의 신규성은 2/5, 이 판정에 대한 confidence는 4/5다. 다만 성능과 작동 원리가 충분히 입증된다면, 현재 구조 그대로도 제한적이지만 독립적인 메소드 기여로 인정할 수 있다.** “알려진 부품의 결합이므로 불가능하다”는 판정도, “같은 조합을 찾지 못했으므로 새롭다”는 판정도 타당하지 않다. 핵심은 이 결합이 재사용할 수 있는 계산 원리를 제공하는지다.

## 1. 실제로 남는 기여

가장 방어 가능한 기여 후보는 **관측값을 경계로 유지하는 confidence diffusion에 입력 조건부 source–query compatibility를 넣고, 고정된 PFN의 예측 목적에 맞춰 합성 과제에서 그 compatibility를 학습하는 작은 입력 연산**이다. “새로운 sheaf diffusion” 또는 “새로운 PFN 학습 원리”라는 설명보다 이 정의가 구현에 가깝다.

| 축 | 기존 원리와 겹치는 부분 | 현재 남는 증분 | 판단 |
|---|---|---|---|
| 계산 | 관측 경계 diffusion, 거리 confidence, 직교 transport와 학습된 map은 기존 원리다. | Column별 confidence 전파에 공유 node frame을 결합한 제약 있는 endpoint kernel. | 작지만 구체적인 설계 기여 후보. 일반 sheaf보다 표현력이 커진 것은 아니다. |
| 목적 | 불완전 입력을 downstream label loss로 학습하는 것은 기존이다. | 수치 추정기를 frozen PFN의 query CE로 최적화한다. | 목적함수 자체의 신규성은 낮고, PFN에 맞는 추론 연산 설계가 평가 대상이다. |
| 학습 원리 | 합성 과제의 query 예측으로 amortized predictor를 학습하는 PFN 원리가 이미 있다. | Backbone을 고정하고 737개 모수의 공유 frontend만 합성 과제에서 학습한다. | 유용할 수 있는 parameter-efficient 구성이나 새 학습 원리로 보기는 어렵다. 모수 수 자체는 신규성이 아니다. |
| 전이 설정 | 사전학습한 하나의 모델을 새 그래프에서 가중치 업데이트 없이 쓰는 설정은 기존이다. | 심한 numeric feature missing에 관측 경계 연산을 추가한 고정 가중치 전이. | 가장 실용적인 기여 후보지만, 현재 개발 그래프 결과는 독립적인 전이 증거가 아니다. |

초안의 공간 출력은 다음과 같다.

\[
\hat x^{sp}_{if}=\sum_{j\in O_f}H_{ij,f}K_{ij,f}x_{jf},\qquad
K_{ij,f}=\cos(\theta_{if}-\theta_{jf}).
\]

여기서 H는 같은 관측 경계와 confidence를 갖는 scalar diffusion의 source 계수다. **학습된 geometry가 H의 경로 선택이나 도달 가능성을 바꾸지는 않는다.** 경로를 따라 회전한 중간 node frame은 소거되고 endpoint의 각도 차이만 남는다. 따라서 “이웃마다 다른 좌표계로 메시지를 전달한다”는 실행상의 설명을 “관측 source와 query의 호환성을 학습한다”는 기능상의 설명과 함께 제시해야 한다.

K는 \((\cos\theta_i,\sin\theta_i)\)의 Gram kernel이므로 unit diagonal을 갖는 PSD 행렬이며 rank가 최대 2다. 이는 K에 대한 말이고, 실제 계수 행렬 \(H\odot K\)의 rank가 2라는 뜻은 아니다. Source의 기여를 줄이거나 부호를 바꿀 수 있지만 절댓값을 해당 H 계수보다 키우지는 않는다. 각도가 입력에서 계산되므로 전체 frontend가 입력에 대해 선형이라는 뜻도 아니다. 이 제약은 일관된 compatibility를 공유하는 정규화 효과가 될 수 있고, 임의의 source–query 관계를 표현하지 못하는 한계가 될 수도 있다.

두 scalar 채널 전파와 readout으로의 정확한 환원은 신규성을 **제한하지만 제거하지는 않는다.** 새로운 표현 능력보다, 이 제한된 함수족이 극심한 결측에서 좋은 추정 편향을 제공하는지로 평가해야 한다. 비자명한 cycle holonomy, 경로별 geometry, 일반 sheaf의 강한 표현력은 현재 방법의 성질이 아니다. NSD 원문은 경로 독립적이지 않은 transport를 명시하고, feature에서 restriction map을 학습하는 더 일반적인 모델을 다룬다. [Neural Sheaf Diffusion, §§2·5](https://arxiv.org/html/2202.04579v3).

## 2. 가장 가까운 선행연구 5개

단일 논문이 전부를 포괄한다기보다 계산, 의미 조건부 보정, 학습 및 전이의 인접 연구가 다르다. 아래 다섯 개를 우선 대조하겠다.

| 연구 | 정확히 겹치는 부분 | 남는 차이와 심사상 의미 |
|---|---|---|
| **PCFI, ICLR 2023** | Channel별 최근접 관측 source 거리의 지수 confidence, confidence를 쓰는 inter-node diffusion, inter-channel correlation 보정. 극단적 결측과 downstream prediction도 이미 문제 설정에 포함된다. | 현재 방법은 같은 기반에 입력 조건부 frame을 더하고 query loss로 이를 학습한다. 따라서 PCFI와의 차이는 confidence나 극단 결측 대응이 아니라 **학습된 compatibility의 추가 가치**다. 이 논문이 계산상 가장 직접적인 기반이다. [원문 §§3.4–3.6](https://arxiv.org/html/2305.16618). |
| **Neural Sheaf Diffusion, NeurIPS 2022** | Feature에서 map을 학습하고 직교 map을 사용해 diffusion에서 정렬·반전을 조절하며 heterophily를 다룬다. 2D 회전도 이미 실험한다. | 현재는 원래 numeric 값을 관측 경계로 보존하고, column마다 고정된 flat node frame을 사용하며, frozen PFN 앞에서 합성 과제로 학습한다. 새 transport 원리라기보다 **기존 geometry를 더 제약해 특정 결측 추론 연산으로 만든 것**이다. NSD 전체를 그대로 실행하는 것과 동일하지도 않다. [원문 §5 및 Appendix F](https://arxiv.org/html/2202.04579v3). |
| **FSD-CAP, ICLR 2026** | 관측 node에서 시작하는 diffusion, 극심한 결측, distance를 고려한 propagation, label 유래 정보 및 neighborhood entropy로 feature 보정. 초안의 “신뢰도와 의미를 함께 써 결측 전파를 조절한다”는 넓은 동기는 이미 상당히 겹친다. | FSD-CAP은 fractional weighting·progressive subgraph expansion과 GNN pseudo-label 기반 class anchor를 쓴다. 현재는 context-only 통계로 SO(2) compatibility를 만들고 frozen PFN query CE에 amortize한다. **Label-aware imputation 자체를 차별점으로 삼을 수 없고**, target에서 학습하지 않는 compatibility의 가치가 남는다. 초안에서 빠진 중요한 가까운 연구다. [원문 §§2.2–2.4, Algorithm 2](https://arxiv.org/html/2601.18938v1). |
| **FAGCN, AAAI 2021** | 학습한 self-gating으로 양·음 성분 및 저·고주파 정보를 조절해 homophily를 미리 알지 않고 여러 관계를 처리한다. “부호 반전을 허용하는 학습된 전파”만으로는 새로운 계산 원리가 아니다. | FAGCN의 local message gate와 현재의 flat endpoint kernel은 동치가 아니다. 현재는 관측 경계와 confidence를 보존하고 source–query compatibility를 전파 전체에 일관되게 묶는다. 이 제약이 좋은지 확인할 **강한 scalar/adaptive filter 대조의 선행 근거**다. [원문](https://ojs.aaai.org/index.php/AAAI/article/download/16514/16321). |
| **GraphPFN, ICML 2026** | Synthetic graph prior에서 예측을 학습하고 실제 graph에 ICL로 적용한다. Graph-aware aggregation과 node-level tasks도 이미 있다. | 현재는 이미 학습된 PFN을 고정한 채 결측 frontend만 별도로 학습한다. 따라서 “합성 학습으로 새 graph에서 gradient-free prediction”은 기존 설정이고, **그 설정에서 결측 입력을 다루는 추가 연산**이 기여다. 현재 wrapper가 공식 GraphPFN 전체 pipeline의 재현이 아니라는 제한도 그대로 유지해야 한다. [공식 논문](https://proceedings.mlr.press/v306/eremeev26a.html). |

GRAPE는 위 다섯 개보다 공간 연산상 거리가 있지만 목적함수 주장에는 직접적인 선행이다. 원문 식 (4)–(5)는 GNN의 edge prediction으로 얻은 imputed dataset을 downstream predictor에 넣는다. Unseen observations와 동일 feature schema의 외부 dataset에서의 전이도 논의한다. 따라서 “imputation을 task loss로 학습”이나 “재학습 없이 새 sample에 적용”을 새 원리로 주장하면 안 된다. 현재의 차이는 주어진 원 graph에서의 경계 diffusion, schema에 덜 묶인 공유 descriptor, 합성 task→새 graph의 frozen PFN 설정에 있다. [GRAPE, §3.3](https://papers.neurips.cc/paper_files/paper/2020/file/dc36f18a9a0a776671d4879cae69b551-Paper.pdf).

초안의 FP·FISF·Wander 설명은 원문의 큰 범위와 부합한다. 다만 역할을 구분해야 한다. FP는 이미 diffusion과 관측값 재고정을 제시하고, feature-dependent learnable diffusion을 향후 방향으로 명시했다. 이는 현재 구현의 선취는 아니지만 동기의 새로움을 낮춘다. [FP, §5–6](https://proceedings.mlr.press/v198/rossi22a/rossi22a.pdf). FISF는 저분산 문제를 synthetic feature injection으로 해결하므로 강한 결측 기준선이며, 현재 방법과 계산이 같지는 않다. [FISF, ICML 2025](https://proceedings.mlr.press/v267/um25a.html). Wander는 2026-10-06 v2에서 부분 관측 graph completion과 하나의 고정 checkpoint를 통한 graph/task 전이를 명시한다. 포괄적인 문제 설정의 신규성을 제한하지만, 현재 numeric frontend의 가장 직접적인 계산 경쟁자는 아니다. [Wander, abstract·§4](https://arxiv.org/html/2610.06694v2).

## 3. 성능이 충분히 좋아져도 메소드로 인정할 수 있는가

**조건부로 그렇다. 새 loss나 non-flat connection을 반드시 추가할 필요는 없다.** 관측 경계의 안정성을 유지하면서 signed compatibility를 극소수 공유 모수로 학습하고, 새 graph에 고정 가중치로 전달하는 연산이 일관된 이득을 준다면, 제약 있는 함수족을 선택한 것 자체가 메소드 기여가 될 수 있다. 두 채널로 환원된다는 사실은 오히려 단순한 구현과 명확한 제약을 제공한다.

그러나 “PCFI보다 수치가 크게 좋아졌다”는 사실만으로 강한 알고리즘 신규성이 생기지는 않는다. 같은 descriptor·목적·예산의 scalar gate나 일반적인 learned encoder–diffuse–decoder가 같은 결과를 내면, SO(2) 설명의 독립 가치는 작다. 이때도 유용한 PFN adaptation 연구는 가능하지만 기여는 설계·분석·전이 증거 쪽에 있다. 메소드 논문으로 인정할 기준은 **이 특정한 compatibility 제약을 선택할 이유와 그 결과가 독립적으로 확인되는지**이지, 새로운 수학적 부품의 개수는 아니다.

현재 결과는 이 판단을 뒷받침하기에 부족하다. Node−PCFI +0.080%p의 CI는 0을 포함하고 NLL/Brier는 약간 낮은 품질이다. Label 제거 후 재학습의 차이는 label-derived 입력의 훈련상 기여 가능성을 시사하지만, geometry-only shuffle에 반응하지 않으므로 올바른 label 묶음의 의미 사용까지 입증하지 않는다. Signed scalar의 clipping은 현재 비교의 음성 결과를 해석하기 어렵게 한다. 이것들은 신규성이 없다는 증거가 아니라 **기여 후보가 실제로 작동하는지 아직 판별하지 못한 증거**다.

## 4. 독립 기여에 필요한 최소 증거 3개

1. **현재 함수족의 이득을 판별하는 공정한 강한 대조.** Signed scalar의 hard clipping을 부드러운 bounded parameterization으로 바꾸고, 같은 descriptor·frozen PFN·학습 노출에서 adaptive signed filter와 비교한다. 학습하지 않은 random frame 또는 입력 조건부를 없앤 frame도 포함한다. 두 scalar diffusion의 동치 구현은 별도 경쟁 모델로 세지 않는다. 목적은 “2D가 scalar보다 본질적으로 우월”을 증명하는 것이 아니라, 현재 제약이 최적화 결함이나 추가 입력만으로 설명되지 않는 이득을 주는지 확인하는 것이다.

2. **Compatibility가 무엇을 선택하는지 보여 주는 작은 진단.** 같은 graph·source·confidence에서 feature 관계 또는 여러 context의 label 묶음만 바꾸고, 유효 source 계수 H·cos(Δθ)와 예측 변화를 함께 보여 준다. Label class 이름 순열에 대한 불변성과 label assignment 변경에 대한 반응을 구분한다. 한두 개 명확한 통제 조건이면 충분하며 일반 sheaf 표현력 정리는 필요 없다. 올바른 label 묶음이 중요하지 않다면 “context semantics에 적응”을 내려놓고, 관측 feature/mask 기반 compatibility로 주장을 좁히면 된다.

3. **선택 후 독립 전이와 실제 강한 baseline.** 구조·선택 기준을 고정한 뒤 새로운 실제 graph 또는 학습 family와 다른 graph/feature 법칙에서 평가한다. 동일 표현의 PCFI/FISF+PFN과 전체 pipeline의 Native-PCFI를 구분해 비교하고, 적은 context 및 결측 패턴 변화에서도 이득이 유지되는지 확인한다. FSD-CAP처럼 target classifier를 학습하는 모델은 학습 예산이 다른 별도 setting으로 표기할 수 있다. 512개 train episode의 반복과 기존 네 실제 개발 graph만으로 foundation-level transfer를 주장할 수는 없다.

위 세 가지는 대규모 새 architecture 탐색을 뜻하지 않는다. 현재 구조를 유지하고 독립성을 가르는 증거를 얻는 경로다. Non-flat edge map, 새 loss, joint PFN 학습을 추가하면 신규성이 자동으로 해결된다는 권고는 하지 않는다.

## 5. 현재 쓸 수 있는 주장과 과장인 주장

| 정당하게 쓸 수 있는 주장 | 현재 피해야 할 주장 |
|---|---|
| “PCFI의 관측 경계와 confidence를 유지하며 입력 조건부 SO(2) frame으로 source–query compatibility를 학습한다.” | “신뢰도를 학습·보정한다.” Confidence 값 자체는 PCFI의 고정 pseudo-confidence다. |
| “공간 연산은 두 scalar confidence diffusion과 cosine readout으로 정확히 재표현된다.” | “경로별 geometry나 비자명한 cycle 효과를 추론한다.” Flat factorization 때문에 현재는 성립하지 않는다. |
| “Synthetic query classification loss로 frontend만 학습하며 target graph의 optimizer update는 0회다.” | “학습이 필요 없는 방법”, “새로운 PFN 원리”, 또는 “처음으로 graph 간 gradient-free 전이를 한다.” |
| “분류 목적에 맞춘 numeric surrogate를 생성한다.” | “실제 결측값 또는 posterior uncertainty를 복원한다.” CE만으로 값의 충실성이나 posterior 해석이 보장되지 않는다. |
| “개발 bank에서 일부 ablation 차이가 관측됐다.” | “PCFI보다 일관되게 우수”, “heterophily 전반을 해결”, “2D transport의 우월성 입증”, “독립적인 unseen real graph 검증 완료”. |

현재 명칭은 **Confidence-weighted Coordinate Transport for Missing-Feature PFN** 정도가 가장 정확하다. “Sheaf”를 제목의 중심에 두면 일반 sheaf의 표현력과 위상적 효과를 기대하게 하므로, flat node-frame이라는 제한을 같은 위치에서 밝혀야 한다.

## 6. 최종 평점과 판정

| 항목 | 1–5 평가 | 근거 |
|---|---:|---|
| 현재 메소드 신규성 | **2/5** | 계산상의 작은 구체적 증분은 있으나, 목적·학습·전이의 큰 원리는 기존이며 flat endpoint kernel의 선택 이유가 아직 약하다. |
| 증분이 독립적인 메소드가 될 가능성 | **3/5** | 경계 보존·제약 있는 compatibility·frozen PFN 전이가 강한 통제 비교에서 함께 작동하면 인정 가능하다. 이는 성능을 가정한 전망이지 현재 달성 점수가 아니다. |
| 심사 confidence | **4/5** | 핵심 계산과 주요 인접 원문을 확인했다. 다만 제한된 검색이며 전체 최근 imputation·reprogramming 문헌과 공식 GraphPFN 전처리 전체를 감사하지 않았다. |

**현재 제출물 기준 weak reject; 신규성만 따로 보면 낮은 쪽의 borderline이다.** 낮은 성능과 낮은 신규성을 하나로 섞어 기각할 필요는 없지만, 지금은 효과도 특정 설계의 필요성도 충분히 확인되지 않아 독립 기여를 받아들이기 어렵다. 강한 성능을 가정하면 자동 accept는 아니지만, 위의 구별되는 작동 증거가 더해질 경우 현재의 단순한 구조도 메소드로 정당화할 수 있다.
