# FP04 method review — observed-cell GraphPFN

작성일: 2026-11-04. 검토 대상은 `observed_cell_cycle/PLAN_v1.md`와 root가 전달한 구현 구상이다. 현재 주기의 다른 reviewer 문서나 구현은 읽지 않았다. 이 문서는 구현·학습·실험 결과를 주장하지 않는다.

**판정: 수정 조건부 진행.** 새로운 완전 학습형 observed-cell 모델을 동일 초기값의 dense-mask·실제 PCFI와 비교하는 실험은 지금까지의 frozen predictor/centroid/graph-potential 수정 실패가 닫지 못한 질문을 직접 다룬다. 다만 관측 정보 자체가 더 많아지는 것은 아니며, sparse routing의 귀납 편향과 입력 표현의 효과를 검증하는 제한된 실험이다. 수학적으로 구현을 막는 모순은 발견하지 않았다. 아래 정의를 학습 전에 고정해야 결과의 해석이 성립한다.

점수(1–5): 수학적 실현 가능성 4, 비교 설계 4, 현재 명세의 완결성 3, 기존 연구와 구별되는 기여의 확실성 2. 성공 가능성에 대한 수치 예측은 아니다.

## 보존할 강점

- 값·열·그래프·분류 경로를 모두 다시 학습하며, 과거 고정 predictor의 표현 접근 변경과 명확히 다르다. 세 arm의 같은 파라미터 모양·초기값·task exposure는 중요한 통제다.
- 실제 observed bit, hidden-value poison, source 0/1 포함, 실제 PCFI 재계산, feature-free query의 사전 선택을 명시했다. query 선택으로 유리한 mask 결과만 남기지 않는 점도 적절하다.
- observed–mask와 observed–PCFI뿐 아니라 none/shuffle 대비도 요구하는 결합 성공 조건은 유용하다. 단순한 baseline 손상이나 우연한 정규화 이득을 feature evidence 이용 성공으로 읽을 위험을 줄인다.
- label/node/column 공유 파라미터와 class별 공유 one-vs-rest score는 가변 클래스·열에 맞는 설계다. 다만 이것만으로 실세계 열의 의미를 이해하거나 임의의 posterior를 표현하는 것은 아니다.

## 주요 이슈 — 최대 5개

### 1. observed와 mask의 실제 차이를 routing과 정규화로 고정해야 한다

모든 arm은 같은 가시 정보 `(G, M, M⊙X, context)`를 받는다. PCFI도 이 입력에서 계산되는 결정적 표현이다. 따라서 observed의 장점이 생기면 추가 정보 획득이 아니라 계산 구조·정규화·최적화 차이의 증거다.

제안된 차이는 포함 계수 `A_if=M_if`(observed), `A_if=1`(mask/PCFI)로 명시하는 것이 가장 명확하다. 예를 들어 열 aggregate와 node read는 각각 `sum_i A_if W_rif P_r(h_ci) / max(sum_i A_if,1)` 및 `sum_f A_if W_rif V_r(z_cf) / max(sum_f A_if,1)` 형태다. 실제 구현에서는 round의 상태 사용 순서와 비선형 함수의 위치까지 고정해야 한다. dense-mask도 입력의 observed bit를 이용해 missing message를 억제할 수 있어야 한다. 반대로 모든 arm의 feature message에 공통으로 hard `M`을 곱하면 관측-only 비교가 소실된다.

관측-only에서는 observed bit가 포함 셀에서 항상 1이므로 일부 parameter 방향의 유효 학습 조건은 dense와 다르다. 같은 parameter 수가 같은 유효 함수군이나 FLOPs를 뜻하지 않는다. 파라미터·업데이트 수 외에 실제 시간·peak memory·PCFI 전처리 비용을 함께 보고해야 한다.

### 2. class-conditioned evidence의 경로가 존재한다는 사실과 그 경로로 성공한다는 주장을 분리해야 한다

class별 seed와 공유 score는 label permutation equivariance에 적합하다. 그러나 value basis 4개·width 32·4 round의 저차원 메모리는 표현력에 제약을 둔다. 가변 열 공유는 task 내 패턴을 이용하는 장치이며, 임의의 열 간 투영이나 복잡한 class-conditional density를 보장하지 않는다. dense-mask와 PCFI도 같은 semantic evidence 경로를 가져야 하며, observed만 class-conditioned 경로를 추가하면 arm의 의미가 바뀐다.

초기 v1의 동적 global context pooling은 멀리 떨어진 class evidence를 직접 broadcast할 수 있어, 주장한 `context→graph→source1→column→source2→graph→query` 경로를 우회한다. root는 이를 제거하고 **label-count의 정적 seed summary만 허용**한다고 회신했다. 이 수정은 적절하다. 고정 context·label count에서 disconnected 두 component 사이의 source 값이 query에 미치는 미분이 column 경로에 의존하는지 확인해야 한다. column read를 끄면 그 미분이 0이어야 한다. 정적 count summary도 whole-task class prior의 입력임을 명시해야 한다.

### 3. generator와 primary estimand의 남은 자유도를 학습 전에 닫아야 한다

v1의 Family A/B는 방향은 분명하지만 label rule, nonlinear transform, noise·Student-t·centroid 분포, 그래프·degree 분포와 family 내 혼합 비율이 아직 실행 가능한 정의로 완결되지 않았다. 이를 “final 생성 전 코드에 기록”하는 것만으로는 train/dev 관찰 후 prior를 바꿀 여지가 남는다. **train 시작 전에** generator와 task schedule·primary mask/shape/context 패널·집계 가중치까지 protocol에 고정해야 한다. 수정이 필요하면 새 revision과 변경 이유를 기록해야 한다.

특히 graph-first Family B의 label 규칙이 실제 observed cell로부터 추론 가능한지는 설계 가정이다. 이번 학습 이전에 실세계 결과를 보고 규칙을 고르면 안 된다. N/F shape holdout은 의미나 메커니즘 OOD의 증거가 아니며, 실제 효과를 family별로 함께 보여야 pooled 결과를 과해석하지 않는다. absent class를 vocabulary에 남기는 규칙은 유지할 가치가 있다.

### 4. zero-support와 intervention의 계산 정의가 missing robustness의 일부다

source 0/1, 열 support 0/1에서 count·mean·variance·RMS·정규화와 denominator의 처리를 명시해야 한다. observed arm의 빈 aggregate는 안전한 zero이고, bias가 “가짜 관측 메시지”처럼 빈 셀별로 반복되어서는 안 된다. PCFI의 초기값·clamp·반복 수와 source-zero fallback도 고정해야 한다.

hidden 숫자는 `abs`, square, moment 계산이나 PCFI 입력 전에 제거해야 한다. none/shuffle마다 관측 통계·정규화·PCFI를 다시 계산하고 cache identity도 intervention을 포함해야 한다. whole-node row shuffle과 cell-mask의 column별 shuffle은 보존하는 joint 정보가 다르므로 별도 panel로 보고해야 한다. utility subset의 source≥2가 관측 행 수인지 셀 수인지, disjoint의 정확한 정의, eligible base graph가 없는 경우의 집계를 사전에 정해야 한다. 관측 행 수가 많아도 각 열의 support가 1이면 class contrast는 약할 수 있으므로 열 support 분포도 보조 진단으로 유용하다.

### 5. 성공은 작은 예산의 관측-routing 효용이며, 새 PFN 원리의 확립은 아니다

이미 Partial VAE/EDDI는 임의 관측 subset을 인코딩했고, TabImpute는 synthetic prior로 incomplete table을 처리한다. 이들은 이번 graph label predictor와 목적이 다르지만 “관측값만 인코딩” 또는 “synthetic 학습으로 missing을 처리” 자체는 새 원리가 아니다. [EDDI](https://proceedings.mlr.press/v97/ma19c.html), [TabImpute](https://openreview.net/pdf?id=pqgAeeCCT8).

가장 가까운 graph-PFN 선행은 Ephris다. 공유 scalar encoding, 열·노드 정보 교환, 크기와 무관한 node compression, synthetic graph ICL이 이미 결합되어 있다. 이번의 명확한 검증 질문은 **extreme missing 하에서 class-conditioned observed-cell 경로를 완전 학습하면 같은 모델의 dense-mask·PCFI 표현보다 유리한가**이다. Ephris와 동일 모델이라는 뜻은 아니며, 새 알고리즘이라고 선점할 근거도 아직 없다. [Ephris](https://arxiv.org/html/2609.37057v1).

Structural Missingness PFN의 undefined/MNAR 설정이나 MA-PFN의 real-source에서 학습한 value/mask prior와 이번 MCAR형 mask·synthetic-only 설정도 구분해야 한다. 해당 논문의 보장이나 전이 결과를 이번 모델로 옮길 수 없다. [Structural Missingness PFN](https://arxiv.org/abs/2601.18500), [MA-PFN](https://arxiv.org/abs/2611.01851).

4개 일차 조건을 모두 만족하면 고정한 prior와 작은 학습 예산에서 routing 선택의 효용을 입증한다. 실패하면 이 구조·prior·예산을 기각하며 observed-set PFN 일반을 기각하지 않는다. real 세 그래프는 개발 전이 결과로 보고하고, native F와 N이 훈련 범위 밖이라는 한계를 유지해야 한다. secondary NLL/Brier나 한 dataset의 개선으로 primary 실패를 번복해서는 안 된다.

## 최소 사전 검증 — 최대 3개

1. **동일 상태·완전 관측 parity와 arm 분리:** 모든 셀이 관측된 작은 입력에서 sparse/dense forward와 parameter gradient가 일치하는지 검사한다. 일부 셀을 숨긴 공유 상태에서는 정의한 routing·분모 차이가 실제 계산에 남는지, masked 숫자 poison·열/노드/클래스 permutation·empty/singleton에서 finite한지도 확인한다.
2. **경로와 intervention 독립성:** 두 disconnected component의 고정 context/count에서 cross-component source-value→query 미분을 확인하고 column 경로 제거 시 0인지 검사한다. none/shuffle의 통계·PCFI 재계산과 hidden-value 미참조도 작은 예제로 검사한다. 이는 구조 검사이며 학습 후 성능 성공을 대신하지 않는다.
3. **학습 전 protocol 잠금과 독립 집계:** generator의 모든 확률 분포, arm 식·round 순서·빈 support 처리, 정확한 primary panel과 utility subset, schedule·seed·bootstrap 단위를 잠근다. 독립 scorer가 toy prediction으로 네 결합 조건과 eligible-basegraph 집계를 재현해야 한다. 동일 parameter 초기 hash와 실제 연산·메모리 비용도 기록한다.

## 구현 API 의견 및 검토 종료

모델 forward는 visible values, observed mask, graph edges, context labels, vocabulary K만 받아야 한다. query label·mask 생성용 truth·평가 subset metadata는 넘기지 않는다. arm은 생성 시 고정하고 PCFI 값은 명시적으로 분리된 입력 또는 순수 전처리 결과로 전달한다. 모든 arm의 parameter 이름·모양을 동일하게 유지하고, class/feature chunk는 수학적 결과를 바꾸지 않는 실행 옵션으로 둔다. source 0/1 경계와 전처리 통계의 책임 위치를 API 문서에 남기는 편이 좋다.

root가 알린 global dynamic context pool 제거는 본 리뷰의 2번 우려를 처리하는 올바른 수정이다. 나머지 구현·protocol 명세가 v2에서 고정되면 진행할 수 있다. 이 review는 v1 심사 기록으로 보존하며 현재 구현이나 실험을 승인·검증한 결과로 읽지 않는다.
