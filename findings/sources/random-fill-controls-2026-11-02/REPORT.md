# Frozen sampling — random-fill 대조 결과

실행 완료: 2026-11-02. **새 neural PFN 학습이 아닌, 설계안 첫 단계의 실제 frozen inference 실험이다.**

판정: **Frozen Gaussian conditional advantage over random-fill nulls not established; no automatic scaling of this operator**

기존32개 개발 graph·3 frozen checkpoints·4 probability views를 유지했다. Uniform/Clustered는 각각 node missing90/99.5 평균, context 최대4/class다.

| Input | Uniform NLL ↓ | Clustered NLL ↓ | Uniform acc | Clustered acc | 보조 k1 NLL ↓ |
|---|---:|---:|---:|---:|---:|
| conditional_joint | 0.91589 | 0.91352 | 49.95% | 49.82% | 0.95789 |
| iid_variance_matched | 0.91367 | 0.91728 | 50.02% | 49.30% | 0.96151 |
| graph_prior_variance_matched | 0.91325 | 0.91166 | 50.00% | 50.28% | 0.95722 |
| iid_unit | 0.91305 | 0.91666 | 49.40% | 49.64% | 0.96257 |
| wrong_sign_mean | 0.91511 | 0.91460 | 49.97% | 49.74% | 0.95667 |

| Conditional − null NLL | Uniform [95% CI] | Clustered [95% CI] |
|---|---:|---:|
| iid_variance_matched | +0.00222 [-0.00277, +0.00718] | -0.00376 [-0.00973, +0.00184] |
| graph_prior_variance_matched | +0.00265 [-0.00076, +0.00626] | +0.00186 [-0.00105, +0.00512] |
| iid_unit | +0.00284 [-0.00131, +0.00747] | -0.00313 [-0.00824, +0.00173] |
| wrong_sign_mean | +0.00078 [-0.00114, +0.00275] | -0.00107 [-0.00427, +0.00243] |

주 gate 통과: **False**. 두 variance-matched null과 두 주 group의 네 비교를 모두 만족해야 통과다. k1이나 개별 seed로 gate를 바꾸지 않는다.

## 무엇을 통제했는가

- 모두 같은 관측값·mask·graph·context labels·query·checkpoint·feature-position view schedule을 사용한다.
- 두 variance-matched null은 원래 mixture의 analytical per-cell conditional variance를 받는다. 관측값에 의존한 scale 정보가 남으므로 완전 unconditional control이라고 주장하지 않는다.
- Unit-noise의 missing proposal에는 실제 관측값을 쓰지 않는다. 실제 관측 위치에는 모든 방법이 원래 값을 그대로 넣는다.
- Wrong-sign은 missing mean에만 개입하고 conditional covariance/rho/noise를 유지한 진단이다. 새 hybrid로 선택하지 않는다.
- Primary conditional은 원래 probability 파일을 바로 붙이지 않고 같은 worker에서 다시 실행했다. Full-observed에서는 모든 입력이 같아 같은 probabilities를 공유했다.

## 기존 point baseline과 실행 감사

| 이전 동일 bank 입력 대조 | Uniform NLL | Clustered NLL |
|---|---:|---:|
| zero | 0.92888 | 0.93084 |
| pcfi | 0.93539 | 0.93279 |
| posterior_mean | 0.94890 | 0.94698 |

위 point 수치는 이전 실행의 저장 결과이며 새로운 null 주 gate와 구분한다. PCFI도 같은 zero-input-trained checkpoint의 입력 대조다. 별도 PCFI/FISF-trained 모델 비교로 확대하지 않는다.

- 재채점한 확률 파일 2880개, scored views11520개, optimizer update0회.
- Original conditional replay 최대 확률 차이 0.00000000, 평균 NLL 차이 +0.00000000.
- Source·input bank·completion bank·checkpoint hashes와 frozen weights를 검증했다. 원래 실험 파일을 변경하지 않았다.
- Bootstrap은 graph32개를 paired resampling하며 rates와 fixed3 checkpoint seeds를 먼저 평균했다. Completion RNG와 checkpoint seeds 모집단의 불확실성은 포함하지 않고, 다중 비교 보정도 하지 않았다.

## 해석의 경계

Null과 구분되지 않는다면 이전 sampling 개선을 유효한 observed-value conditional completion이나 참 posterior 추론의 성과라고 해석할 근거가 부족하다. 이것이 모든 observed feature의 무용함을 의미하지 않는다. 실제 값의 직접 PFN 입력 경로는 유지돼 있다.

이 결과는 context labels를 feature law에 사용하는 모델의 성공·실패 판정이 아니다. Shared encoder·hypothesis suffix와 coherence training도 아직 실행하지 않았다. 이 frozen operator의 gate를 통과하지 못하면 자동으로 추가 head/loss/규모를 붙여 확장하지 않는다. 새로운 conditional-law 학습은 별도 가설과 강한 대조를 가져야 한다.

전체 condition·seed·accuracy/Brier CI는 `results/summary.json`, `paired.json`에 보관한다. `protocol.json`은 사전 설정, `experiment.py`는 실제 실행 source다.
