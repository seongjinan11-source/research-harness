# Graph-correlated feature completion → frozen GraphPFN

이전의 네 단계 numeric refiner를 더 키우지 않고, **대치값 하나로 확정하는 연산**을 바꾼 후보 구현이다. 독립적인 논문 메소드라고 주장하지 않는다.

핵심은 아래 두 계산의 차이다.

\[
p_{\rm mean}(y_q)=p_\theta(y_q\mid G,\mathbb E[X_{\rm mis}\mid X_{\rm obs},G],M,Y_C)
\]

\[
p_{\rm MC}(y_q)=\frac1S\sum_{s=1}^S p_\theta(y_q\mid G,X_{\rm obs}\cup X_{\rm mis}^{(s)},M,Y_C).
\]

이번 구현에서 sampling law는 GraphPFN의 참 생성 prior가 아니라 명시한 Gaussian-process surrogate다. 따라서 결과를 정확한 Bayesian PFN marginalization이라고 부르지 않는다. Context labels는 PFN 분류에만 들어가며 피처 분포에는 사용하지 않는다. 일반적인 완전 Bayesian 계산은 completion law도 Y_C에 조건화해야 하므로, 이번 연산은 그 계산과도 구별한다.

## 조건부 feature law

Self loop 없는 symmetric normalized adjacency를 \(S_G\)라 하고, \(\rho\in\{-0.9,0,0.9,0.99\}\)에 대해

\[
R_\rho=(I-\rho S_G)^{-1},\quad
\widetilde R_{\rho,ij}=R_{\rho,ij}/\sqrt{R_{\rho,ii}R_{\rho,jj}},\quad
K_\rho=0.5\mathbf 1\mathbf 1^\top+0.5\widetilde R_\rho.
\]

모든 rho에서 covariance는 양의 정부호다. 채널은 rho 조건부 독립이고 하나의 rho는 graph 전체에 공유된다. Equal rho prior를 관측값의 Gaussian marginal density로 업데이트한다. Synthetic SCM의 분포를 복원했다고 가정하지 않는다. 기존 pilot과 같은 observed-only RMS 좌표에서 정의한 제한된 surrogate다.

각 채널의 관측 위치 O와 결측 위치 U에서

\[
\mu_U=K_{UO}K_{OO}^{-1}x_O,\qquad
\Sigma_U=K_{UU}-K_{UO}K_{OO}^{-1}K_{OU}.
\]

관측이 0개이면 해당 rho의 prior를 사용한다. Padding channel은 상수 0이며 evidence에 넣지 않는다. 실제 관측값은 모든 arm에서 그대로 유지한다. 결측 샘플은 공통 입력 범위 ±10으로 clip하고 비율을 기록한다.

주변화를 해볼 이유는 단지 샘플 수를 늘리기 위해서가 아니다. Smooth predictor F를 평균 근처에서 전개하면 이차항은 대략 `0.5 * tr(H_F * Sigma)`다. 따라서 node covariance가 달라지면 각 cell의 평균·분산이 같아도 결과가 달라질 수 있다. 이것은 작은 변동에서의 설명이며 이번 full-variance 계산의 정확한 등식이나 성능 개선 보장은 아니다. 기존 방법과의 novelty 주장도 아니다.

## 대조군과 판정

- `zero`: 0 + original mask.
- `pcfi`: 기존 PCFI 대치 + original mask.
- `native_nan`: 공통 입력 interface에서 native `encoder_x` missing vector 사용. 공식 전처리 전체나 해당 입력으로 다시 학습한 대조는 아니다.
- `posterior_mean`: 위 mixture의 조건부 평균.
- `joint_samples`: 노드 간 조건부 covariance를 유지한 4개 completion.
- `diagonal_samples`: 같은 rho draw·조건부 평균·대각 분산을 유지하고 rho 내 off-diagonal covariance만 제거. Rho 공유에 의한 의존성까지 없앤 대조는 아니다.

모든 arm은 동일한 `joint_imputation_pfn/mask_plain_s11,s12,s13` checkpoint를 사용한다. Wrapper의 `pcfi_plain` switch는 입력 배열을 읽기 위한 선택일 뿐, PCFI 학습 checkpoint를 쓰는 뜻이 아니다. Feature-position RNG 4개와 PFN forward 수가 같다. 점 대치도 4번 실행해 probability ensemble을 구성한다. 평균 logit 대신 평균 확률을 사용한다.

`protocol.json`에 실행 전 설정과 gate를 고정했다. 새 32개 graph의 uniform/clustered 각 90/99.5% 조건에서, joint가 mean·diagonal·zero·PCFI를 모두 이겨야 다음 학습 후보로 검토한다. 단순 context prior와 native NaN도 보고하며, 이것들에 뒤지는 결과를 성공으로 포장하지 않는다. Accuracy/NLL/Brier를 각각 보고한다. Full과 k1은 보조 조건이다.

실패해도 닫는 범위는 이 frozen replacement와 surrogate다. 학습 가능한 joint feature law, cross-feature dependency, context-conditioned density, 실제 그래프·새 schema는 평가하지 않았다. 같은 official prior family의 새 seed이며 별도 real-world 검증이 아니다.

반대로 성공해도 실제 관측값의 추가 정보를 활용했다고 바로 결론 내릴 수 없다. Graph-correlated random input이 구조 feature처럼 작동할 수도 있다. 이후 검증에는 FISF·비조건부 구조 random-feature 대조와 별도의 PCFI-trained 모델이 필요하다. 이 추가 대조를 이번에 이미 실행했다고 기록하지 않는다.

## 관련 연구와 novelty 경계

GCNmf는 missing feature의 GMM과 첫 layer expected activation을 이미 다뤘고, GSPN은 probabilistic graph representation과 missing-data marginalization을 다뤘다. 따라서 분포·샘플·평균 자체는 새 기여가 아니다. [GCNmf](https://arxiv.org/abs/2007.04583), [GSPN](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6be46e85b1e3ff727c107f2673ce8027-Abstract-Conference.html).

최근 PEMix는 구조 조건부 feature mixture를 제시한다. 이번 surrogate는 그 구현을 재현하지 않는다. PFN에서 joint covariance 보존이 실제 유용한지부터 검사하며, 개선돼도 PEMix/GSPN 및 같은 예산 앙상블보다 독립적인 기여가 있는지는 남는다. [PEMix](https://arxiv.org/abs/2610.18113).

## 재현

Windows GraphPFN 환경의 Python으로 `python -u experiment.py all`을 실행한다. `prepare`, `preflight`, `run`, `analyze`를 따로 실행할 수도 있다. 현재 model/reference source와 frozen checkpoints는 변경하지 않는다. 새 bank는 hidden feature를 저장하지 않으며 query labels는 평가 target으로만 별도 보관한다.

`results/lock.json`은 source·bank·checkpoint hash를 고정한다. 저장된 forward별 확률을 독립 재채점하며, full-observed에서 모든 arm의 같은 view가 정확히 일치해야 한다. `preflight.json`은 관측 clamp·label 분리·moment 순열·sample covariance·반복 결정성과 weight 보존을 검사한다. 이 검사는 메소드 성능의 증거가 아니다.

실행 도중 checkpoint별 독립 process를 선택하는 `run --seed 11` 옵션만 추가했다. 변경 전 source와 lock, 보존한 완료 확률 hash는 `runtime/parallel_restart.json`에 기록했다. 중단 당시 미완료 episode의 파일은 보존용 폴더로 옮기고 episode 전체를 다시 실행했다. 프로토콜·수식·입력·모델은 그대로다. 동시 process의 기록 시간을 단일 모델 latency 측정으로 사용하지 않는다.
