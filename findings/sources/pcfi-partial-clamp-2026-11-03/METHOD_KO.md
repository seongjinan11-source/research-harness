# 관측 좌표만 고정하는 Confidence Sheaf PFN

2026-11-03. 기존의 관측 cell 전체 clamp와, 실제 관측된 첫 좌표만 clamp하는 계산을 비교한다. Frame·confidence·PFN·입력 descriptor·query CE·예산을 유지한다. 새 구조가 더 좋다는 결론을 미리 두지 않는다.

## 계산

기존은 관측 cell의 2차원 상태를 매번 (x,0)으로 고정했다. 새 계산은 첫 좌표 x만 보존하고, 관측하지 않은 두 번째 좌표는 모든 노드에서 전파한다.

\[
\bar z_{if}^{t+1}=\sum_jP_{ij,f}Q_{if}^{\top}Q_{jf}z_{jf}^{t},\qquad
z_{if}^{t+1}=\bar z_{if}^{t+1}+M_{if}e_1(x_{if}-e_1^\top\bar z_{if}^{t+1}).
\]

PCFI의 원래 거리 confidence와 row weights, self loop, 100회 공간 전파, 한 번의 channel correction과 최종 관측값 고정을 유지한다. 두 번째 좌표는 0에서 시작한다. Frame은 기존 737-parameter MLP가 관측값·mask·graph·context 통계로 계산하고 100회 동안 고정한다. 숨은 feature·query 정답은 읽지 않는다. 정의는 유한 반복 계산이며, 수렴한 유일한 잠재 상태나 참 posterior를 구했다고 주장하지 않는다.

공통 좌표 h=Qz에서 관측은 (Qe1)^T h=x라는 한 방향 제약이다. 기존처럼 잠재 벡터 전체를 경계로 지정하지 않는다. 모든 frame이 같으면 0 초기화의 두 번째 성분이 계속 0이므로 PCFI 공간 연산으로 돌아간다. Source0에서는 새 숫자를 생성하지 않는다. Full 입력에서는 PFN에 원래 관측값을 제공한다.

이를 구체적으로 쓰면 a_if=Q_if e1에 대해 공통 좌표의 한 step은 다음과 같다.

\[
\bar h_{if}=\sum_jP_{ij,f}h_{jf},\qquad
h_{if}^{+}=(I-M_{if}a_{if}a_{if}^{\top})\bar h_{if}+M_{if}a_{if}x_{if}.
\]

따라서 frame은 transport의 좌표 변환뿐 아니라 관측 방향의 projection에도 들어간다. 기존 전체 clamp의 고정 source 영향 H에 endpoint cosine을 곱하는 식이 새 계산에도 그대로 성립한다고 가정할 수 없다. 여전히 flat connection이며 비자명한 cycle holonomy를 새로 얻은 것은 아니다. 이것은 연산 차이에 대한 유도이지 신규성 또는 분류 이득의 증거가 아니다. 독립 aligned 구현이 위 식을 사용해 forward와 gradient를 대조한다.

## 비교

- PCFI와 기존 학습 Node 전체 clamp.
- 기존 Node checkpoint의 추론 clamp만 변경한 Partial-frozen.
- 같은 초기화·train bank·schedule·1,024 update로 새로 학습한 Partial-trained.
- 같은 초기화에서 학습하지 않은 Partial-initial: near-identity random-network 대조.
- 동일한 고정 uniform random angle을 공유하는 Full-random과 Partial-random.
- 기존 clipped scalar 및 tanh로 clipping을 제거한 Signed-smooth. Smooth scalar는 원래 .99 offset과 같은 출력 위치를 택하지만 초기 전달함수는 Node와 같지 않다. 음수 계수·saturation·gradient를 보고한다.

새 학습은 Partial-trained와 Signed-smooth 각각 3 seed, 총 6개 frontend와 6,144 update다. 기존 synthetic-only PFN은 frontend 학습 중에도 고정한다. PFN과 실제 graph update는 모두 0회다. 별도 loss·posterior head·teacher·학습 가능한 fallback은 없다. 기존 결과를 보며 angle range나 추가 regularizer를 선택하지 않는다.

Downstream은 공개 코드 기반의 기존 synthetic-trained 실험용 PCFI-PFN wrapper다. 공식 GraphPFN의 전체 사전학습과 full pipeline을 재현한 실험은 아니다. 이 pilot은 같은 고정 predictor 앞에서 숫자 추론 연산의 변경을 분리한다.

## 평가

새 same-family base graph 32개, label homo/hetero × feature smooth/rough 네 cell에 8개씩 사용한다. Context 1/4개 per class와 uniform/clustered whole-node missing 90/99.5%를 primary로 두고 full 및 source2/8도 보고한다. 기존 네 실제 graph/60개 입력은 개발 비교로 재사용한다. 입력과 PFN은 각 seed 안에서 모든 arm이 공유한다.

Primary는 자기 피처가 없는 query accuracy다. NLL·Brier와 source가 있는 조건을 별도로 보고한다. 세 고정 학습 seed·두 context·네 mask를 base graph 안에서 평균하고 graph 단위 paired bootstrap 5,000회로 차이의 CI를 계산한다. CI는 다중비교 미보정이며 저장된 예측에 조건부다. GPU replay 변동은 별도 기록한다.

핵심 비교는 Partial-trained−Node, Partial-trained−PCFI, Partial-frozen−Node 및 Partial-trained−Partial-frozen이다. 이들이 각각 경계 변경 전체, 강한 기준선 대비, 추론 시 변경, 새 학습의 증분을 구분한다. Random 및 initial 대조는 frame 학습의 독립 가치에 대한 질문이다. 결과가 약해도 전체 sheaf/PFN 연구를 자동 종료하지 않으며, 변경한 계산과 주장 범위를 판정한다.

코드·protocol과 bank를 학습 전에 hash로 잠그고, 초기 frontend·고정 PFN·task schedule을 기존 실행과 대조한다. 소형 독립 좌표계 reference의 forward/gradient, 관측값 보존, identity 환원, source0/full 및 gradient 전달을 학습 전에 검사한다.
