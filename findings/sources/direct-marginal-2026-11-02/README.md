# 직접 주변화하는 conditional graph circuit 실험

GSPN의 유효한 sum/product 계산과 MaM의 marginal 합 규칙에서 단서를 얻어 구현한 **새 소형 conditional tree circuit**이다. 두 논문의 전체 재현이라고 부르지 않는다.

`C_theta(X,Y | G,T,prior_family)`는 관측 graph와 독립적인 tie-breaking으로 정한 tree에 조건부인 정규화된 모델이다. 각 원본 node의 binary label과 각 연속 feature 좌표는 한 번씩만 등장한다. Subtree의 변수 scope는 서로 겹치지 않는다. Root·mixture·transition은 정규화하고, missing Gaussian leaf는 1로 적분한다. Circuit 파라미터를 만드는 encoder는 G와 알려진 family descriptor만 본다. X_obs, 원래 mask, context labels는 leaf evidence로만 들어간다.

학습 전 다음을 검사한다.

- 소형 graph에서 모든 binary label 배치 열거와 evidence·marginal 일치
- 모든 값이 미관측일 때 전체 질량 1
- label 합 규칙과 한 연속 좌표를 관측한 evidence의 적분
- conditional ratio와 sum-product marginal 일치
- evidence가 달라져도 circuit 파라미터 고정
- signed transition으로 반대 class를 예측할 수 있음
- 유한한 학습 gradient, 알려진 family의 Grid6와 독립 scalar 계산 일치

이 유효성은 **현재 모델 내부의 정확한 주변화**다. 원래 cyclic SBM의 참 posterior가 정확하다는 뜻은 아니다. 고정된 12 nodes / 2 relations / binary class / 2 numeric features의 kill test이며, 가변 feature·semantic schema나 대형 cyclic circuit은 실행하지 않는다.

비교 모델은 circuit, 같은 structural encoder를 쓰는 direct graph predictor, 실제 공개 Mask GraphPFN이다. 공개 PFN에는 typed relation별 context support를 포함한 강한 입력을 제공한다. 모델당 같은 2048 task exposures와 256 optimizer updates를 사용한다. Parameter 수는 서로 다르므로 그대로 보고한다. 학습은 query CE만 사용하고 target graph에서는 가중치를 갱신하지 않는다.

이전 training-support Grid6를 유지하되 Gaussian feature의 알려지지 않은 orientation을 적분하도록 확장했다. Exact와 64-step BP 양쪽을 보고한다. 별도의 true-parameter exact reference는 privileged 진단이다. 실제 density/signal·hidden role·query truth는 학생 모델의 입력이 아니다.

Primary는 matched 48 base graphs의 whole-node uniform/clustered 50/90% 조건 NLL이다. Accuracy/Brier·Grid6 predictive KL과 OOD는 별도 보고한다. Feature 99.5% 결측은 12 nodes에서 실제 관측이 0개일 수 있으므로 그 빈도를 숨기지 않는다. MaM식 합 규칙을 새 consistency loss로 추가하지 않는다.

`protocol.json`과 `results/lock.json`이 실행 범위를 고정한다. 완료 후 `REPORT.md`, `results/paired.json`, `results/interventions.json`, `results/benchmark.json`을 확인한다. 유효성 gate, 예측 gate, classical 기준선 대비 품질·비용 gate는 나눠 판정한다.

실제 비용 측정의 entry point는 `runtime/benchmark_corrected.py`다. Teacher 생성용 reference engine은 exact와 BP 출력을 함께 반환하므로, BP만 배포할 때 필요 없는 exact query marginal 계산을 측정에서 제외했다. 두 구현의 예측 일치를 먼저 검사하고 측정 source hash를 `runtime/measurement_lock.json`에 저장한다. 학습 코드·가중치·데이터·주 gate는 바꾸지 않는다.

`runtime/feature_controls.py`는 primary 설계 고정 뒤 추가한 **기술적·해석적 대조**다. 알려진 Gaussian family만 쓰는 feature-only 모델과 modal training signal을 쓰는 고정 tree를 비교한다. Test tuning이나 main gate 변경에 쓰지 않고 adaptive encoder의 기여를 제한하는 데만 사용한다.
