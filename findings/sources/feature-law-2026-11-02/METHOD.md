# Feature-law Factor 비교

사용자가 정한 다음 실험: 새 loss나 posterior head 없이 관측 cell의 함수만 바꾼다. 작은 별도 structured predictor이며 공개 GraphPFN에 결합한 모듈은 아니다.

입력은 native target8/helper4 이분 그래프, 관계2, 피처2와 관측 mask, 총 context1/2/4개다. 관측 노드는3/6개이며 context와 완전히 분리한다. 정답을 보지 않고 query 하나를 예약해 반드시 자체 피처가 없는 query가 남도록 한다. Primary는 이 query 집합의 accuracy다.

각 column은 Gaussian 평균 차이, Student-t3 평균 차이, Gaussian 분산 차이 중 하나를 따른다. Law·실제 파라미터·orientation은 입력에 없다. 학습은 각 law의 두 diagonal 파라미터 tuple, heldout은 off-diagonal tuple이다. 모든 scalar와 law는 학습에서 보았으므로 unseen-law 일반화가 아니다. 정확한 수치는 protocol.json을 따른다.

관측 cell은 exp[s_i z_f rho(x_if,S_f)]로 변환한다. Original은 x*positiveNN, static은 x의 제약 없는 공유 함수, adaptive는 관측 column 통계8개를 더 사용한다. Static/adaptive는 선형 basis 경로와 MLP를 쓰며 trainable 수를2% 이내로 맞춘다. 모든 함수의 초기 score는1.55*x로 같고 최종 score는[-8,8]로 제한한다. 초기화를0으로 두면 orientation 주변화의 대칭점에서 gradient가 사라지므로 같은 비영 Gaussian 초기값을 쓴다. Variance evidence를 허용하는 것과 실제로 학습하는 것은 별도다.

Original CE seed11 checkpoint의 graph encoder와 graph factor를 모든 Factor arm/seed에 동일하게 고정한다. 기존 feature head는 재사용하지 않는다. Graph factor의 likelihood 상수를 유지하고24-step BP/Bethe로 같은 potential의 role weights와 conditional predictions를 계산한다. 학습 gradient는 cell→conditional과 cell→role update 두 경로에 흐른다.

강한 fixed control은 law3종×각4tuple×orientation2를 모른 상태로 주변화한다. `fixed_exact`는 같은 frozen graph potential에서4096개 node labels와 column별 후보를 정확히 합산한다. `fixed_bp`는12² law/tuple branches×4 roles와 binary orientation latent를 같은24-step BP로 계산한다. 동일 normalization prior와 full feature likelihood 상수를 보존한다. 정확 계산과 BP는 다른 비용/근사이므로 둘 다 보고한다. `native_exact`는 native6 graph parameters까지 합산한 별도 normalized reference다. Broad feature prior로 정의된 소형 reference이며 true OOD Bayes 또는 특권 true-parameter teacher라고 부르지 않는다.

Mask/PCFI-PFN은 공개 PFN backbone을 동결하고 기존 adapter 학습 방식을 사용한다. 이들의 graph adapter는 Factor와 같은 아키텍처가 아니며, 동일 graph checkpoint 고정 대조는 세 Factor와 fixed-law controls에 적용한다. PCFI는 실제 계산하며 관측값을 보존한다. 숫자 completion을 만드는 것은 이 대조군뿐이다.

새 task2048개,256 updates×batch8,query CE만,seed11/12/13,총15개 모델을 학습한다. 같은 seed에서는 task schedule과 노출 수가 같다. 마지막 checkpoint만 평가하고 최종 graph에서 gradient update하지 않는다. 별도 dev12 graphs의 수치·gradient 검사3 step은 폐기한다. Sources/config를 잠근 뒤 final192 graphs×6 episodes를 생성한다. 새 final bank에서 구조/학습량/loss를 바꾸지 않는다.

Primary는 heldout96 base graphs이며 query 평균→episode6개 평균→seed3개 평균 순서로 묶어 graph 단위5000회 paired bootstrap한다. Gaussian guard는 pure Gaussian48 graphs의 사전2%p NI다. 다른 subgroup은 탐색적·다중 비교 미보정이다. 과제 적응 성공에는 adaptive가 original과 static 모두의 primary accuracy를 CI 하한>0으로 이겨야 한다. 이어 Gaussian guard와 강한 fixed control 품질/비용 gate를 적용한다. 근거가 부족해도 추가 head/loss/temperature/학습량으로 구제하지 않는다.

실행: tasks.py prepare → checks.py(설계 lock) → tasks.py finalize → run.py --seed11/12/13 → runtime/analyze.py audit/cost/summary. 분석 entrypoint는 역사적 프로젝트에도 있는 run.py를 잘못 import하지 않도록 현재 run.py를 명시적으로 로드한다. 잠근 학습/분석 함수는 바꾸지 않는다. Bank/checkpoint/reference hashes, freeze, schedule과 재생 예측을 별도 audit한다. 보고서는 학습 성공, 표현력 효과, 과제 적응, 분류 성능과 확률 품질을 분리한다.
