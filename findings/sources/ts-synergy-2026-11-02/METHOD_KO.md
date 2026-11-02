# 공동 정보 보존·방해 변수 안정성: frozen TSFM 비교

2026-11-02. 이 작업은 GraphPFN/BP/EAJ를 수정하지 않는 별도 시계열 후보의 첫 실행이다. [사전 프로토콜](protocol.json)은 bank 생성 시 hash를 기록했고, 결과를 본 뒤 gate를 변경하지 않았다. TSFM은 재학습하지 않았다.

## 과제와 시간 정렬

독립적인 base task96개를 새로 생성했다. Product64개가 primary이고 additive32개는 입력 경로의 양성 대조다. 모든 채널은 iid 균형 Rademacher이고, 유용한 두 채널·각 부호·지연0/1/2/3·채널 순서는 task별로 무작위다. 방해 채널8개도 같은 모집단 주변분포를 가진다. 유한 표본의 양수 개수를 억지로 일치시키지는 않는다.

Product는 `y[t+1]=a*x[i,t-l1]*b*x[j,t-l2]+N(0,.15²)`, additive는 두 항의 합을 sqrt(2)로 나눈 값에 같은 잡음을 더한다. 모델은 채널 역할·참 지연·참 부호를 받지 않는다.

각 task에서 history512, origin16개(stride4), horizon1을 사용한다. origin t의 입력은 y와 x의 `[t-511:t+1]`이며 y[t+1]은 입력에 없다. Chronos에는 future covariate를 전달하지 않는다. 회귀 학습에서 행 t의 피처는 x[t-lag], y[t-lag], label은 y[t+1]이다. 첫 평가 label은512이고, fit/validation/calibration label은 최대511이다. 이후 origin에서는 이미 관측한 과거 test outcome을 history로 쓰는 정상적인 rolling forecast다.

입력 버전은 history/u/v/uv/uvN3/uvN8. u/v/uv는 참 유용 변수에 대한 **입력 개입 진단**이며, 이 선택에 oracle 지식이 사용된다. 배포 상황을 대표하는 uvN8에는 익명 순서의 모든10채널이 들어가며, 유용한 쌍의 표시는 없다.

## 고정 모델과 강한 비교군

- 로컬 공개 Chronos-2, 기존147,456-parameter group-Q/K rank4 null-invariance LoRA, 동일 frozen Chronos-2 앞의 과거 상관 screening을 비교한다. 이전 LoRA는 Alibaba에서 teacher 예측 보존·null 불변성을 학습한 checkpoint로, 이번 synthetic bank에서 학습하거나 선택하지 않았다.
- Screening은 첫 origin까지 알려진 label로 lag0..3 중 최대 절대 상관을 계산하고 상위2채널을 남긴다. 미래 label이나 유용 변수 정체를 쓰지 않는다.
- Raw Ridge, 모든 후보 interaction Ridge, 모든 후보 interaction Lasso를 각 task의 과거 데이터로 fit한다. classical 회귀는 task별 fitting을 사용하며, zero-shot TSFM과 학습 체계가 다르다. 이 차이를 숨기지 않고 비용과 함께 보고한다.
- Interaction은 모든 서로 다른 채널 쌍의 모든4×4 지연 조합이다. 전체 입력에서는720개 곱 항 +40개 raw 항 +4개 target history 항, 총764개다. 참 생성 쌍만 넣은 모델이 아니다. Lasso는 전체 사전을 받은 sparse regression으로, 결과를 보고 선택한 사후 arm이 아니다.
- 과거 supervised507행 중320행으로 fit,96행으로 alpha 선택,416행으로 refit,남은91행으로 residual quantile을 추정한다. 이후16origin동안 계수·잔차 분위수는 고정한다. model fit/선택/잔차 calibration 어느 쪽에도 미래 평가값이 없다. 알파 후보는 protocol에 고정했다.
- 참 쌍·부호·지연·잡음 scale을 아는 Gaussian quantile oracle은 예측 여지 진단용이며, 실용 기준선으로 세지 않는다. 이 oracle은 과거의 pair 찾기를 해결해 주는 privileged reference다.

## 지표와 진행 기준

기존 checkpoint의21개 quantile에서 twice-pinball loss를 평균한다. 단위 signal scale이 동일하여 별도 scale-normalization은 적용하지 않는다. WQL/CRPS 또는 classification accuracy로 바꿔 부르지 않는다. quantile을 정렬하거나 crossing을 사후 보정하지 않았다.

각 task에서 origin·quantile을 평균한 뒤 `G_joint=min(L_u,L_v)-L_uv`, `P_null=L_uvN8-L_uv`를 계산한다. 최소값은 task별로 취한다. 이는 개별 손실의 모집단 평균에 최소값을 취하는 추정량과 다르며, single-view 유리한 변동을 포함하는 보수적인 진단이다. CI는64개 product **task**를 짝지어10,000회 bootstrap하며 time point/view를 독립 표본으로 세지 않는다. Additive32개는 별도 보고하고 primary gate를 대체하지 않는다.

Frozen Chronos-2의 G와 P가 각각 평균.01초과·CI하한0초과여야 두 문제의 동시 존재 gate를 통과한다. 전체 후보 interaction이 history-to-oracle gap의80%이상을 회수하고 full-context Chronos보다 유의하게 좋다면 product toy만으로 추가 TSFM 학습을 진행하지 않는다. 이 경우 제안2×2는 **미실행**이며 학습법의 실패로 세지 않는다.

## 문헌과 주장 범위

조건부 관련성·상보성은 기존 [정보이론적 feature selection](https://www.jmlr.org/beta/papers/v13/brown12a.html)의 범위에 있다. [Chronos-2 초기 보고서](https://arxiv.org/html/2510.15821v1)는 group attention을 이미 사용한다. [2026 확장 원고 A.4](https://arxiv.org/html/2610.41956v1)는 random MLP의 비선형 순간 변환과 RNN의 지연 변환을 명시한다. 따라서 기존 prior가 선형뿐이라고 주장하지 않는다. [t0](https://arxiv.org/abs/2609.24559)도 다변량 covariate를 지원하지만 이번에는 실행하지 않았다. 하나의 checkpoint 결과를 전체 TSFM의 한계로 확대하지 않는다.

검증하는 범위는 알려진 product/additive family·10채널·lag0..3·history512의 frozen 비교다. 낮은 clean-product 활용의 고유 원인을 비선형 prior 부족, patching, 최적화 중 하나로 확정하지 않는다. 새로운 training loss, real-series robustness, 임의 synergy mechanism, PFN 메소드 novelty는 이번에 입증하지 않는다.
