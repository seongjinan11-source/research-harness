# Confidence-weighted Coordinate Transport — novelty 심사 요약

2026-11-03. 사용자의 요청에 따라 Intro·Related Work·Method·Experiments의 짧은 Markdown 초안을 만들고, 두 subagent에게 독립적으로 심사를 요청했다. 아래는 **내부 모의 심사**이며 실제 학회 심사나 완전한 문헌조사가 아니다. 새 학습·실험은 실행하지 않았다.

## 문서

- [읽기용 수정 초안](DRAFT_REVISED_KO.md): 빠졌던 가까운 선행연구와 주장 범위를 보완했다. 재심사는 하지 않았다.
- [원본 심사 대상](DRAFT_KO.md): 심사자가 읽은 버전을 그대로 보존했다.
- [문헌 중심 심사 원문](NOVELTY_REVIEW_LITERATURE_KO.md)
- [연산·학습 원리 중심 심사 원문](NOVELTY_REVIEW_OPERATOR_KO.md)
- [심사 요청과 원본 hash](REVIEW_REQUEST.json)

## 두 심사자의 공통 판정

| 심사 | 현재 novelty | 심사 confidence | 현재 제출물 판정 |
|---|---:|---:|---|
| 문헌 중심 | 2/5 | 4/5 | weak reject |
| 연산·학습 원리 중심 | 2/5 | 4/5 | weak reject |

두 심사자는 다른 검토를 맡았지만 같은 코드·실험 기록을 사용했다. 점수를 독립적인 통계적 증거 두 개처럼 취급하지 않는다. Confidence는 제한된 검토 범위에 대한 자기평가다.

공통 결론은 **새 sheaf diffusion 또는 새 PFN 학습 원리라는 주장은 어렵지만, 현재 결합 전체가 메소드가 될 가능성은 남아 있다**는 것이다. 알려진 부품의 결합이라는 이유만으로 기각하지 않았고, 새 loss나 non-flat map을 추가해야만 통과한다는 요구도 없었다.

## 가장 중요한 문헌 보완

**연산상 가장 가까운 것은 BuNN이다.** 학습한 node별 orthogonal frame, flat connection, scalar diffusion으로의 gauge 환원이 직접 겹친다. 현재 방법의 관측 clamp·column별 PCFI confidence·숫자 lift/readout·고정 PFN용 synthetic 학습까지 동일하다고 확인한 것은 아니다. 따라서 차이를 이 구체적인 경계 추정기 설계로 한정해야 한다. [BuNN §3와 Lemma 3.1](https://arxiv.org/html/2405.15540v2#S3).

**Label-aware feature imputation도 새 범주는 아니다.** FSD-CAP은 거리 기반 확장·fractional diffusion 뒤 pseudo-label과 neighborhood entropy로 class-aware refinement를 수행한다. ICLR 2026 게재는 공식 proceedings에서 확인했다. 현재의 context-only 입력과 target 재학습 없는 PFN 설정은 구분할 수 있지만, label 정보를 써 전파를 보정한다는 넓은 주장으로 novelty를 잡기는 어렵다. [FSD-CAP 공식 논문](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8063ef83cf43c341bc124b6175f2729d-Abstract-Conference.html).

FAGCN과 GPR-GNN도 adaptive/signed filtering의 비교 대상으로 추가했다. 부호를 바꾸는 전파 자체의 최초성은 주장하지 않는다. [FAGCN](https://ojs.aaai.org/index.php/AAAI/article/download/16514/16321), [GPR-GNN](https://arxiv.org/abs/2006.07988).

## 현재 방법에 남는 계산과 기여 후보

관측 source j가 query i에 주는 공간 기여는 정확히 다음과 같다.

\[
\hat x_i^{sp}=\sum_{j\in O}H_{ij}\cos(\theta_j-\theta_i)x_j.
\]

H는 PCFI confidence와 관측 clamp를 사용하는 scalar diffusion의 source 영향이다. 학습하는 것은 endpoint compatibility다. 이것은 PCFI와 다른 값을 낼 수 있지만, 경로를 새로 고르거나 일반 sheaf의 cycle 효과를 표현하지는 않는다. Cosine 행렬 K의 rank≤2는 **H⊙K나 최종 predictor 전체의 rank 제약이 아니다.**

따라서 현재 방어할 수 있는 기여 후보는 다음이다.

> 관측 경계와 confidence를 보존하는 diffusion에 입력 조건부 source–query compatibility를 결합하고, frozen PFN의 예측 목적에 맞춰 작은 공유 추정기를 synthetic task에서 학습하여 새 그래프에 적용한다.

이 정의는 쓸 수 있는 **메소드 가설**이다. 이 제약이 왜 결측 전이에 적절하고 강한 대안보다 언제 유리한지까지는 아직 입증되지 않았다. 큰 성능 향상을 가정해도 BuNN과의 연산 중복은 남고, 반대로 그 중복만으로 특정 결합의 실용적 기여가 자동으로 사라지지도 않는다.

## 현재 증거의 빈틈

- Node−PCFI는 +0.080%p, 95% CI [−0.137,+0.303]으로 현재 bank에서 우위를 확인하지 못했다. 이 수치의 약함과 연산 신규성은 별개의 문제다.
- Label descriptor 제거 재학습은 낮았지만, geometry-only label shuffle의 명확한 손해는 없었다. 올바른 class 의미에서 frame을 추론한다는 주장은 아직 이르다.
- Signed scalar는 음수 gain을 쓰지 않았고 +1 clipping이 약 67%였다. 이 대조로 2D 또는 endpoint 구조의 필요성을 판정할 수 없다.
- 실제 네 graph는 반복 사용한 개발 graph다. 독립적인 unseen-real 전이 근거로 사용할 수 없다.

## 다음에 가를 질문: 세 가지로 제한

1. **Endpoint 제약의 가치:** 최적화가 정상인 signed/adaptive 대조 및 random/fixed frame과 동일 입력·예산으로 비교해도 이 구조의 이득이 남는가?
2. **문제에 맞는 원리:** 관측에서 식별 가능한 phase 정렬이 있는 조건과 없는 조건을 구분할 수 있고, compatibility 변화가 예측 이득으로 이어지는가? Context semantics가 불필요하다면 주장을 좁힌다.
3. **독립적인 전이:** 구조를 고정한 뒤 새 family/실제 graph에서 강한 PCFI/FISF+PFN 기준을 넘어서는가? 동일 표현 비교와 Native-PCFI 전체 pipeline 비교를 구분한다.

이는 자동 실행 계획이나 새 실패 기록이 아니다. 이번 심사만으로 PCFI 변형 전체 또는 sheaf 기반 결측 연구 전체를 종료하지 않는다. 현재 초안의 넓은 novelty 주장은 낮추고, 남는 estimator의 필요성을 검증할 기준을 분명히 한 것이다.

원본 초안 SHA256: `f59c85cc5d6d0da8c2973524e9b11bc7b374c54f03adbca14f7dadc753bebc40`.
