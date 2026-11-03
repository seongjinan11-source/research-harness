# 연산·학습 원리 중심 독립 novelty 심사

2026-11-03. 검토 범위는 현재 static Node-frame 방법이며, 향후 가능한 일반 sheaf 방법을 대신 평가하지 않았다. 먼저 `DRAFT_KO.md`를 읽고 `pcfi_orthogonal_pilot/core.py`, `reviewer_development/aligned_operator.py`, `reviewer_development/REVIEW_KO.md`, `pcfi_reviewer_ablation/REPORT_KO.md` 및 scalar 대조 구현을 대조했다. 새 학습·체크포인트 변경·추가 수치 실험은 수행하지 않았다.

**판정: novelty 2/5, confidence 4/5. 현재의 메소드 novelty 주장은 weak reject 수준이다.** 이는 낮은 정확도만을 근거로 한 판단이 아니다. 정확도가 충분히 높다고 가정해도, 핵심 연산은 학습된 flat-bundle diffusion의 관측 경계 버전이며, 학습 목적은 기존 downstream task-aware imputation과 synthetic prior 학습을 따른다. 다만 이 좁은 결합이 실용적 메소드 기여가 될 가능성까지 부정하지는 않는다. 문헌상 동일한 전체 조합이 이미 발표됐다고 확인한 것은 아니다.

표기: **[확정·코드]** 구현에서 확인, **[확정·유도]** 명시한 조건에서 수학적 귀결, **[문헌]** 연결한 원문에서 확인, **[추측·평가]** 아직 입증되지 않은 해석 또는 심사 판단이다. Confidence 4는 연산 환원과 가까운 선행연구에 대한 확신이며, 완전한 문헌 포괄성을 뜻하지 않는다.

## 1. novelty를 가장 강하게 제한하는 수학적 반론

### 1.1 학습된 connection은 항상 gauge로 제거된다

**[확정·코드]** 각 column에서 737-parameter MLP는 관측 숫자·mask·거리 confidence·이웃 요약·context 통계로 노드 각도 하나를 만든다. 각도는 100번 반복 중 바뀌지 않는다. `kind='node'`에서 정확히 \(T_{ij}=Q_i^TQ_j\)이다. 관측 상태는 매번 \((x_i,0)\)으로 고정하고 첫 좌표를 읽은 뒤 기존 channel correction을 한 번 적용한다. PFN parameter는 `requires_grad_(False)`이며 query CE는 frontend만 갱신한다.

**[확정·유도]** column을 생략하고 \(B=\operatorname{diag}(Q_1,\ldots,Q_n)\)라 두면 공간 연산자는

\[
S=B^T(P\otimes I_2)B.
\]

따라서 \(u=Bz\)로 좌표를 바꾸면 보통의 두 scalar diffusion이 된다. 관측 경계는 함께 \(u_i=Q_i(x_i,0)^T\)로 돌려 고정해야 한다. 단순히 transport를 지우고 경계를 원래대로 두는 것은 동치가 아니다. 코드의 `aligned_forward`는 이 조건을 올바르게 구현한다.

관측 집합 O와 결측 집합 U에 대해, 0 초기화와 T번 반복의 source 영향 행렬은

\[
H_{UO}^{(T)}=\sum_{k=0}^{T-1}P_{UU}^{k}P_{UO},\qquad H_{OO}^{(T)}=I.
\]

이에 따라 channel correction 전의 숫자 출력은 정확히

\[
\hat x_i=\sum_{j\in O}H_{ij}^{(T)}x_j\cos(\theta_j-\theta_i).
\]

즉 **고정 PCFI의 source 영향에 입력 의존 endpoint cosine kernel을 곱하는 것**이 현재의 남는 계산이다. 같은 입력·mask·P·각도·경계·readout이라면 세 표현은 같은 estimator다. 100단계가 수렴했을 필요도 없고 graph가 연결됐을 필요도 없다. 최종 nonlinear channel correction도 세 표현의 공통 후처리이므로 전체 구현 동치는 유지된다. 다만 위 단순 source 합 식 자체는 correction 이전의 식이다.

### 1.2 spectrum과 경로 선택을 새로 학습하는 방법은 아니다

**[확정·유도]** U의 homogeneous update도

\[
S_{UU}=B_U^T(P_{UU}\otimes I_2)B_U
\]

이므로 고정 입력에서 eigenvalue는 \(P_{UU}\)의 두 복사본이다. 각도로 diffusion 자체의 spectrum이나 homogeneous vector update의 spectral 수렴률을 바꿀 수 없다. 특정 입력과 scalar readout에서 느린 mode가 소거되어 관측되는 수렴 양상이 달라질 가능성까지 배제하는 말은 아니다. 이것은 값의 출력과 예측이 바뀌지 않는다는 뜻도 아니다. 서로 다른 경로의 transport 곱은 모두 \(Q_i^TQ_j\)로 소거되며 cycle holonomy는 identity다. 경로별 모순이나 cycle별 상호작용을 표현하는 새 geometry는 없다.

적용 조건은 **고정된 node frame, scalar P, 두 좌표에 같은 P, 완전한 관측 벡터 clamp, 반복 안의 추가 비선형성 부재**다. 반복마다 frame을 재계산하거나, edge map을 독립적으로 학습하거나, 좌표별 다른 갱신을 넣으면 이 환원을 그대로 적용할 수 없다. 이 범위 밖 모델을 현재 모델의 표현력으로 주장해서도 안 된다.

### 1.3 표현력은 rank-2 correlation kernel로 제약된다

**[확정·유도]** \(v_i=(\cos\theta_i,\sin\theta_i)\), \(K_{ij}=v_i^Tv_j\)라 두면 \(K=VV^T\)는 PSD, rank≤2, diagonal=1이다. 실제 source coefficient는 \(H\odot K\)다. **H는 일반적으로 비대칭이므로 H⊙K까지 PSD라고 주장하면 틀린다.** Source-query 부분 행렬만 보아도 cosine factor의 rank는 최대 2다. 이는 column별 공간 재가중 factor의 제약이며, **H⊙K의 rank가 2 이하라는 뜻은 아니다.** 기존 channel correction이나 비선형 PFN 전체의 rank 또는 분류 가능한 클래스 수의 한계로 확대할 수도 없다. 임의 source-query signed affinity나 임의 edge별 관계를 독립적으로 정하는 함수족은 아니라는 제한만 따른다.

비음수 substochastic P와 0 결측 초기화에서는 \(H\ge0\), 각 행의 source mass가 1 이하이고

\[
|\hat x_i|\le\sum_{j\in O}H_{ij}|x_j|
\]

다. 공간 연산이 할 수 있는 것은 기존 source 기여의 감쇠와 부호 반전이다. 새로운 source mass 증폭, 관측 숫자가 없는 성분에서 숫자 생성, confidence 경로의 재선택은 하지 않는다. 이 경계는 최종 channel correction 이후 출력의 일반적 bound로 확장할 수 없다. 또한 각도가 x에서 계산되므로 전체 estimator가 x에 대해 선형이라는 결론도 나오지 않는다.

**가장 강한 반론:** “이것은 새로운 diffusion 법칙이나 일반 sheaf 표현력이 아니라, 알려진 flat-bundle 좌표 정렬을 이용한 rank-2 source reweighting이다.” 이 반론은 현재 코드에 대해 성립한다.

## 2. 선행연구와 겹치는 원리, 남는 차이

**[문헌] 현재 초안에서 빠진 가장 가까운 연산 선행연구는 BuNN이다.** Bamberger et al., *Bundle Neural Networks for message diffusion on graphs*는 node별 학습 orthogonal map과 \(O_v^TO_u\) connection을 사용하고, §3에서 bundle Laplacian의 conjugation과 scalar heat kernel으로의 환원, Lemma 3.1에서 endpoint transport factorization을 명시한다. 같은 절의 식 (1)은 frame을 신경망으로 계산하고, 2D에서는 직접 orthogonal parameterization도 허용한다. 따라서 node frame 학습과 “회전–보통 diffusion–역회전” 자체는 현재 방법의 독립 발명으로 내세울 수 없다. [BuNN 원문 §3, Lemma 3.1](https://arxiv.org/html/2405.15540v2#S3)

**[확정·코드 + 문헌 비교]** BuNN과 현재 방법의 전체 동일성은 주장하지 않는다. 현재의 차이는 column별 PCFI confidence, 반복적 관측 경계 고정, scalar 숫자의 2D lift와 첫 좌표 readout, 결측/context descriptor, 고정 PFN의 query CE를 통한 frontend 학습 및 target에서 optimizer update가 없는 적용에 있다. BuNN 원문에서 이 전체 imputation/PFN 프로토콜이 같다는 근거는 확인하지 않았다.

| 원리 | 선행연구에서 확인되는 것 | 현재 방법에 대한 정확한 제한 |
|---|---|---|
| Orthogonal connection diffusion | Singer & Wu의 weighted orthogonal transport와 connection Laplacian | Orthogonal map으로 이웃을 정렬해 평균한다는 계산은 기존 원리다. 현재 학습·결측·PFN 전체가 같다는 뜻은 아니다. [원문](https://arxiv.org/abs/1102.0075) |
| 학습 sheaf/heterophily | NSD의 learned restriction map과 sheaf diffusion | “sheaf이므로 heterophily에 새 표현력이 생긴다”는 포괄적 novelty는 사용할 수 없다. 현재 flatness에 맞춰 제한해야 한다. [원문](https://arxiv.org/abs/2202.04579) |
| 관측 경계 기반 결측 전파 | FP의 Dirichlet-energy 기반 feature diffusion | 관측값을 보존하며 결측을 확산하는 원리는 기존이다. [원문](https://proceedings.mlr.press/v198/rossi22a.html) |
| 거리 confidence와 channel correction | PCFI의 shortest-path channel confidence, inter-node/inter-channel propagation | confidence 자체와 이를 이용한 극심한 missing 처리도 기존이다. [원문](https://arxiv.org/abs/2305.16618) |
| Adaptive/signed graph filter | GPR-GNN은 hop별 양·음 계수를 학습하는 polynomial filter다. | 학습된 부호로 heterophily를 다룬다는 발상은 기존이다. 다만 hop kernel과 현재의 입력 의존 endpoint kernel은 같은 함수족이라고 입증되지 않았다. [원문 §1·3](https://arxiv.org/html/2006.07988v6#S3) |
| Downstream task-aware imputation | GRAPE는 관측 sample-feature bipartite graph와 imputation/label prediction을 구성하며 downstream end-to-end 학습을 분석한다. | CE로 결측처리 모듈을 학습한다는 사실만으로 새 학습 원리를 주장하기 어렵다. 원 graph의 PCFI transport나 frozen PFN 설정까지 같지는 않다. [원문 §4.6](https://proceedings.nips.cc/paper/2020/file/dc36f18a9a0a776671d4879cae69b551-Paper.pdf) |
| Synthetic prior와 graph ICL | GraphPFN은 synthetic attributed graph prior에서 pretrain하고 graph node prediction에 사용한다. | synthetic-to-real와 context conditioning만으로 새 원리가 되지 않는다. 현재 실험은 공식 GraphPFN 전체 학습 재현이 아니라 별도 wrapper라는 초안의 제한이 적절하다. [원문](https://proceedings.mlr.press/v306/eremeev26a.html) |

**[추측·평가]** 이 목록은 서로 다른 원리가 기존이라는 근거이지, 현재의 정확한 전체 결합이 어디선가 이미 발표됐다는 증거는 아니다. 기존 부품을 조합했다는 이유만으로 novelty가 자동으로 0이 되지도 않는다. 그러나 가장 가까운 BuNN을 누락한 채 NSD만 비교하면 실제 증분을 과대평가하게 된다.

## 3. 반론은 기여를 없애는가, 주장을 좁히는가

**[확정·유도] 완전한 무효화는 아니다.** 좌표를 바꿀 때 경계와 readout도 바뀌므로 일반적으로 PCFI와 다른 숫자가 나온다. 하나의 source x=1에서 query로 오는 source mass가 h일 때, endpoint 각도 차이가 π이면 현재 출력은 −h, PCFI는 +h다. 따라서 “pure gauge이므로 아무 효과가 없다”는 반론은 잘못이다. 고정 원래 좌표에서 읽는 task에 대해 학습된 embedding/readout은 유의미할 수 있다.

**[추측·평가] 사라지는 것은 새 geometry/diffusion 원리라는 강한 주장이고, 남는 후보는 특정 문제에서의 제약된 estimator 설계다.** 적절한 주장은 “confidence diffusion의 source와 query에 관측/context 조건의 unit-circle embedding을 주고, frozen graph predictor의 task loss로 전이 가능한 숫자 보간을 학습한다” 정도다. 두 scalar 채널 동치는 별도 경쟁 모델이 아니라 본 방법의 더 간단한 구현으로 제시해야 한다.

결합 전체가 유의미한 기여가 되려면 다음이 필요하다. **[추측·평가]**

- 왜 이 문제에는 자유로운 signed filter보다 bounded rank-2 endpoint 구조가 유용한지 설명해야 한다. 예컨대 제한된 source에서 낮은 자유도가 과적합을 줄이거나, feature별 latent phase로 정렬하면 smooth해지는 생성 조건이 있어야 한다.
- frozen PFN을 위한 task conditioning이 범용 imputation/adaptive filtering의 단순 재사용보다 어떤 반복 가능한 기능을 제공하는지 보여야 한다. 정확한 class 묶음이 불필요하다면 context semantics 주장은 줄여도 된다.
- 고정 예산과 동일 입력에서 강한 대조에 대해 이점이 있고, 구조를 정한 뒤 unseen graph/family에서도 그 이점 또는 예측 가능한 적용 조건이 남아야 한다. 큰 성능 향상은 강한 실용 기여의 근거가 될 수 있으나, 이미 알려진 연산을 새 수학이라고 바꾸지는 않는다.

## 4. 성능 부족과 별개로 현재 설계에 빠진 고유 원리

**[추측·평가] 핵심 결손은 모듈 개수가 아니라 결측 숫자와 frame을 연결하는 문제 고유의 식별·추론 원리다.** 왜 관측 x가 회전된 smooth latent field에서 왔는지, 왜 context 통계로 그 phase를 복원할 수 있는지, 왜 unit-circle rank 2가 이 결측 문제의 적절한 제약인지 현재는 제시하지 않는다. 학습 가능한 737개 parameter와 CE만으로는 이 선택이 adaptive diffusion의 유용한 변형인지, 새 결측 추론 원리인지 구분되지 않는다. 새 정리가 반드시 필요하다는 요구는 아니다. 명확한 생성 가설과 그것이 맞고 틀릴 때의 예측도 원리가 될 수 있다.

다음은 실제 설계의 제한과 아직 검증되지 않은 의미를 구분한 것이다.

- **[확정·코드/유도]** 관측은 scalar 하나인데, lift한 상태에서는 두 좌표 모두 \((x,0)\)으로 clamp한다. 두 번째 0은 추가 latent 관측이 아니라 설계자가 선택한 embedding이다. 따라서 현재 방법을 일반적인 “부분 관측 vector field의 posterior reconstruction”으로 읽을 수 없다. 이 embedding이 성능에 유리할 가능성은 있지만 **[추측]** 근거는 아직 없다.
- **[확정·유도]** 모든 각도가 같으면 PCFI로 돌아가며 각도에 대한 공간 출력의 1차 미분은 0이다. 작은 차이에서 \(\cos(\theta_j-\theta_i)=1-\tfrac12(\theta_j-\theta_i)^2+O(\Delta\theta^4)\)이므로 PCFI 주변의 변화는 2차다. **[확정·코드]** 작은 무작위 마지막 layer 초기화는 완전한 동일각은 피한다. **[추측]** 이것이 실제 부호 적응을 어렵게 했는지는 학습 궤적을 보지 않고 단정할 수 없다.
- **[확정·코드]** confidence는 0.9의 거리 거듭제곱이며 학습된 posterior uncertainty가 아니다. 별도 확률 head·reconstruction loss·missingness likelihood는 없다. **[확정·유도]** query CE의 최적화만으로 올바른 feature conditional mean이나 uncertainty calibration은 보장되지 않는다. 성능이 높아도 “Bayesian imputation” 또는 “불확실성 추론” 기여는 별도 원리 없이는 생기지 않는다.
- **[확정·코드]** 추론 중 optimizer update가 0인 것은 weight adaptation의 부재이지 task 입력의 부재가 아니다. frame은 입력에서 매번 계산되므로 amortized task-conditioned estimator다. **[추측·평가]** 이 점은 유효한 설정이지만 새로운 meta-learning 알고리즘의 증거가 되지는 않는다.

현재 설계를 복잡하게 만들어야만 novelty가 생긴다고 보지는 않는다. Non-flat edge map이나 PFN joint training을 추가해도, 필요성을 설명하지 못하면 또 다른 기존 부품의 추가일 뿐이다.

## 5. 현재 실험이 지지하는 것과 지지하지 않는 것

**[확정·기록]** 아래 수치는 기존 보고서의 재해석이며 이번 검토에서 재실행한 결과가 아니다.

| 기록 | 지지하는 범위 | 지지하지 않는 주장 |
|---|---|---|
| Node−PCFI +0.080%p, CI [−0.137,+0.303] | 현재 known-family bank의 평균이 비슷함 | 동등성/비열등성을 위한 margin 검정도 없으며, 우위도 확보하지 못함 |
| Node−label 제거 재학습 +0.269%p, CI [+0.042,+0.506] | 이 학습 프로토콜에서 descriptor 내 label 통계가 있는 arm의 점수 차이 | 올바른 class 묶음의 의미를 읽어 phase를 정했다는 증거 |
| Geometry shuffle −0.035%p, CI [−0.122,+0.048] | 이 개입에서 뚜렷한 손실이 검출되지 않음 | context가 무가치하다는 동등성 결론; 반대로 context semantic adaptation이 확인됐다는 결론도 불가 |
| Confidence×Node interaction +0.371%p, CI [+0.055,+0.697] | 제한된 2×2에서 양의 interaction | PCFI보다 새로운 신호를 확보했다는 결론; uniform Node의 악화가 interaction에 포함됨 |
| Node−signed +0.253%p, CI [−0.021,+0.531] | 이번 직접 edge scalar arm과의 점수 차이가 미결정 | rank-2가 일반 signed/adaptive 함수족보다 필요하다는 결론 |
| Scalar 음수 0%, +1 clipping 약 67% | 이 실행에서 음의 계수를 실제 사용하지 않음 | signed 표현력 부족 또는 Node의 우월한 부호 학습 입증 |
| 세 구현의 수치·gradient 동치 | 환원식과 더 단순한 구현의 정확성 | 새로운 함수족 또는 별도의 세 방법이 존재한다는 주장 |
| Real 4개 개발 graph와 native-PCFI | 구현 적용 가능성과 현재 강한 pipeline 대비 위치 | unseen real graph 전이와 heterophily 전반의 우위 |

**[확정·유도]** edge scalar의 경로별 gain 곱과 Node의 endpoint cosine은 다르다. 따라서 scalar가 clip 없이 잘 학습돼도 Node의 정확한 동치 대조는 아니다. 동시에 현재 scalar가 학습되지 않았다는 이유로 Node 구조가 필요하다고 결론내려서도 안 된다.

**[추측·평가]** 현재 실험은 “이 조합의 각 구성에서 무엇이 실제로 달라졌는가”를 개발 단계에서 조사한 것으로는 성실하다. 하지만 새 계산·학습 원리를 확인하는 증거는 아니다. 알려진 family 안의 32 base graph, 세 고정 학습 seed, 미보정 다중 비교 및 CI 밖 GPU replay 변동 때문에 작은 차이를 강한 기능 발견으로 읽기 어렵다. 이 한계는 novelty의 수학적 제한과는 별개다.

## 6. 심사 질문과 결정적인 다음 단계: 최대 세 가지

1. **동일 연산을 새 방법으로 이름 붙인 부분을 제거한 뒤 무엇이 남는가?** **[문헌 + 확정·유도]** BuNN §3와 현재 estimator를 나란히 놓고 PCFI, clamp, column sharing, descriptor, frozen decoder objective만을 증분으로 명시하라. 정확히 동치인 aligned 구현은 대조군으로 세지 않는다. **[추측·평가]** “관측으로 식별 가능한 latent phase가 존재하는 feature law”와 그런 phase가 없는 law를 최소한의 생성 가설로 구분하면, 설계가 언제 필요한지 검증 가능한 주장을 만들 수 있다. 독립 edge rotation을 추가하는 것보다 먼저 할 작업이다.

2. **최적화가 정상인 adaptive/signed scalar와 비교해도 endpoint 제약의 가치가 남는가?** **[확정·기록]** 현재 hard clip arm은 이 질문을 끝내지 못했다. **[추측·제안]** smooth bounded coefficient, 실제 초기 전달함수와 saturation을 통제한 대조를 같은 descriptor·예산·PFN에 맞추고, 음수 사용 여부와 최종 loss를 확인하라. 비교 목적은 clipping 문제가 사라진 뒤에도 두 채널의 endpoint 일관성이 유리한지 판단하는 것이다. 모든 scalar 함수족의 우열을 주장할 필요는 없다. 새로운 학습을 이번 리뷰에서 실행하라는 뜻은 아니다.

3. **예측 목적과 context가 frame을 과제에 맞게 정한다는 고유 주장에 직접 증거가 있는가?** **[추측·제안]** graph·mask·관측 숫자를 고정하고 k≥4 context 묶음을 바꿔, geometry만 변경한 출력과 정상 PFN context의 조합을 비교하라. phase 정렬이 유리한 feature law에서 learned/source kernel 변화와 query 이득이 함께 나타나는지 보고, mask/degree/confidence만 쓴 재학습 frame 또는 고정 random frame으로도 같은 결과가 나는지 확인하라. 구조 선택 후의 새 family/real graph에서 이 예측이 유지되면 단순 조합을 넘어 문제에 맞춘 inductive bias의 근거가 된다. 유지되지 않으면 context-conditioned 주장부터 줄이고 더 단순한 estimator로 정리하는 것이 합리적이다.

최종 정성 verdict: **새 sheaf/connection diffusion 또는 새 synthetic meta-learning 원리로는 현재 인정하기 어렵다. 알려진 flat-bundle 연산을 PCFI 경계 보간과 frozen PFN task objective에 맞춘 제한적 메소드 후보로는 남는다.** 성능 향상을 가정하더라도 BuNN과의 연산 중복은 사라지지 않는다. 반대로 위 제한을 정직하게 명시하고 이 제약이 결측 전이에 왜 적절한지를 설명·검증하면, 부품 조합이라는 이유만으로 유의미한 기여가 불가능한 것은 아니다.
