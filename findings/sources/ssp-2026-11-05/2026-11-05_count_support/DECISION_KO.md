# Count support: 주 비교 통과, count-table의 최종 기본 채택은 유보

2026-11-05. 사용자 `gogo`에 따라 새 seed와24개 독립 target으로 count-table joint 추정 및 관측 정책을 구현·평가했다. 세 관점의 계획·코드 리뷰, 개발 전용 수치 검사, 코드·분석 freeze, 전체 원시 관측·쿼리·추천 감사와 독립 결과 리뷰를 수행했다. 목표는 root `RESEARCH_GOAL.md`의 size/composition 문제–해결 연결이다.

## 메타 판단

**사전 primary는 통과했다. 그러나 같은 probe 규칙을 쓴 SSPd3가 더 좋아 count-table+probe를 새 기본 방법으로 채택하지 않는다.** 복잡하거나 새로운 구성요소의 존재보다 문제별 실제 해결력을 기준으로 판단한다.

이번 실험은 중요한 두 사실을 드러낸다.

1. 기존 cubic 모델의 conditional posterior uncertainty도 잘못된 count 외삽을 놓칠 수 있다. 초기 narrow sigmoid48개 context의95% gap interval은1개만 true gap을 포함했고,45개에서는 실제로 나쁜 외삽 후보가 더 좋다고 interval 전체가 잘못 지지했다. 분산 계산 버그가 아닌지 독립 재구성으로 확인했다.
2. 정적인 외삽 실패가 실제 online 실패로 그대로 유지되지는 않는다. SSPd3는 새 관측으로 빠르게 회복했고, 소수 probe를 붙이면 이번 sigmoid에서는 count-table보다 유리했다. 반면 step처럼 cubic 밖 shape에서는 count-table이 composition을 훨씬 안정적으로 추정했다.

따라서 다음 병목은 **size shape misspecification에서 composition을 보호하면서도 count 사이의 유용한 일반화를 잃지 않는 것**이다. 임의의 size 함수를 안전하게 분리하면 데이터의 between-count 정보와 미관측 count 예측도 잃는다는 대가가 실제 선택 성능에서 드러났다.

## 구현과 실험

- 기존 SSPd3 원본은 변경하지 않고 재사용했다. 관측 count별 자유 효과와 Gaussian item prior를 joint 추정하는 모델을 새로 구현했다. Scale은 실제 N−관측 count수 차원의 REML error contrasts로 fit하며, overall label SD로 size 효과를 다시 섞지 않는다.
- Count-table은 미관측 count의 level을0으로 채우거나 외삽하지 않는다. Probe 슬롯은 관측 count에서 가장 먼 count를 택하고, 나머지는 관측 count 중에서 exploitation한다.
- SSPd3 mean, SSPd3 spread, SSPd3 probe, table probe, table spread의5 arms.
- n16, initial44, 추가16 unique sequential queries. 각 관측 후 refit 및 final refit. 총1,920 runs,32,640 selected fits, 새 query30,720개. 초기 공유 관측을 arm별로 세면 총115,200 observations.
-4CPU,464.86초. 후보748개/target-regime의 공통 finite catalog, 같은 potential-outcome noise, 같은 초기 관측. 물리 과제나 비싼 외부 objective 호출은 없다.
- 개발 fixture9000/9001의8 tests와80-run/1,360-fit 통합 감사 통과. 원시 관측·행동·추천·source hash 전체 감사 통과. Replayed label 최대 오차0.
- Primary selected fit 실패0/1,632; 전체 selected 실패5/32,640은 finite 결과를 유지했다. Nonfinite 재시작·행 삭제·대상 교체는 없었다.

## 사전 primary: narrow sigmoid, 두 archive regime 평균

추천은 현재까지 **관측된 noisy utility가 가장 큰 후보**다. 그 후보의 true latent utility를 평가해 catalog의 true 최선·최악 차이로 regret를 정규화한다. 추가0..16관측의17 checkpoint 평균이 주 지표다. True-best-observed로 추천을 대체하거나 regret를 누적 최솟값으로 다듬지 않았다.

| 항목 | 결과 |
|---|---:|
| SSPd3 spread mean incumbent regret | 0.03459062 |
| Count-table probe mean incumbent regret | 0.02651718 |
| 평균 paired 감소량 | +0.00807344 |
| Target paired bootstrap95% CI | [0.00530370,0.01072307] |
| 대상별 승리 | 20/24 |
| 최악 paired 감소량 | −0.00775887 |
| 구현의 fit+selection wall ratio | 0.07797982 |
| Primary selected optimizer 실패 | 0/1,632 |

평균·CI·승수·최악 손실·비용·optimizer·audit의 모든 사전 gate를 통과했다. 평균 regret의 상대 감소는 약23.34%이지만 **AUC 개선율이나 물리 BO의 개선율이 아니다**. 이 주 비교는 모델과 정책을 함께 바꾼 package 비교다.

## 더 강한 대조를 포함한 실제 선택

동일한 narrow sigmoid에서:

| Arm | Mean incumbent regret | Final incumbent regret | Final same-count composition regret |
|---|---:|---:|---:|
| SSPd3 mean | 0.022892 | 0.000600 | 0.027383 |
| SSPd3 spread | 0.034591 | 0.003761 | 0.030746 |
| **SSPd3 probe** | **0.012200** | **0.000340** | 0.030122 |
| Count-table probe | 0.026517 | 0.012078 | **0.001926** |
| Count-table spread | 0.031845 | 0.000365 | **0.001374** |

SSPd3 probe는 table probe보다23개 target에서 좋고1개에서 동률이었다(CSV에서 target23 paired 차이가 정확히0). Table의 composition은 더 정확하지만 실제 incumbent 선택 성능은 더 나빴다. Probe 규칙이 같아도 exploitation이 support를 바꾸므로 두 arm의 실제 count 경로는 동일하지 않다.

Count-table spread와 SSPd3 spread는16개 query의 count 경로가 동일하다. 이 matched-count 대조의 mean regret는0.034591→0.031845로 조금 개선되고 최종 regret도 좋아졌다. 따라서 table 모델의 신호는 있지만, 사전 package가 모든 대조를 이긴다는 해석은 틀리다. Secondary 비교를 primary의 새 성공 기준으로 바꾸지 않는다.

Table probe는 평균8개 count만 관측했고 SSPd3 probe는 약10.02개를 관측했다. 미관측 count를 예측하지 않는 모델이 유용한 count 일반화를 포기한 비용과, 서로 달라진 관측 경로가 함께 포함된다. 관측 개수 차이만으로 전체 성능 차이를 인과적으로 설명했다고 주장하지 않는다.

## 정적 실패와 online 회복

초기 narrow sigmoid SSPd3의 posterior 추천 regret는0.844752였다. 실제 관측16개 후에는:

- SSPd3 spread의 posterior 추천 regret0.013259, 실제 incumbent regret0.003761.
- SSPd3 probe의 posterior 추천 regret0.006457, 실제 incumbent regret0.000340.

따라서 이전 cycle의 K2 외삽 실패는 실제 online 절차에서 상당 부분 교정됐다. 처음 틀린 모델이 탐색 전체에서도 같은 손실을 계속 입는다고 해석하면 안 된다. 초기 posterior 추천과 당시 관측된 incumbent도 서로 다른 대상이다.

## 기존 uncertainty는 오류를 경고했는가?

H1은 초기 narrow sigmoid의 ssp3_mean만,24 target×두 regime=48 context를 보았다. 다른 SSP arm의 동일 초기 posterior를 반복 표본으로 세지 않았다.

- 모든48 context에서 예측상 최선 후보의 count는 미관측이었다.
- 후보와 observed-count 최선 후보 사이의 true utility gap이95% conditional interval 안에 들어간 경우1/48.
- 실제 gap은 음수인데 conditional interval 전체가 양수인 false certainty45/48.
- Gap SD 평균2.571804. Independent reviewer가 원시 fitted parameters와 별도19차원 precision solve로 SD를 재구성해 최대 절대 차이1.62e-13로 확인했다. 해당48개는 optimizer boundary에 걸린 사례도 아니었다.

이는 지정한 misspecified cubic stress 조건의 설명적 진단이다. Conditional Gaussian SD는 hyperparameter·함수족 오지정의 uncertainty를 포함하지 않는다. 일반적인 Bayesian interval의 보장이 틀렸다는 주장이나,48개 dependent context를 독립 표본으로 본95% coverage inference가 아니다. 숫자가 맞는 posterior variance도 틀린 size 함수족의 외삽을 충분히 보호하지 못할 수 있다는 근거다.

## Shape별 이득과 대가

계단형 size 효과에서는 count-table이 유리했다.

| 조건 | SSPd3 probe mean regret | Table probe mean regret | SSPd3 spread | Table spread |
|---|---:|---:|---:|---:|
| Narrow step | 0.012278 | **0.003921** | 0.010948 | **0.003236** |
| Wide step | 0.027990 | **0.003866** | 0.024710 | **0.005736** |

Wide step의 최종 same-count composition regret는 SSPd3 probe0.196294,table probe0.001788이었다. 특정 count shape를 강하게 가정하지 않는 composition 추정이 유용한 조건이다.

반대로 linear/cubic에서는 table이 between-count 정보를 버리는 대가가 있고, 이미 좋은 SSP mean에 probe를 강제로 넣으면 평가 기회비용이 생긴다. Wide linear의 mean incumbent regret는 SSPd3 mean0.001813,SSPd3 probe0.003140,table probe0.003147이었다. 불필요한 탐색을 넣어도 항상 좋아진다는 주장은 하지 않는다.

비용 수치는 이 구현의 actual fit+selection time이다. SSP wrapper의 posterior 재구성 비용은 포함하고, 별도 gap 진단 SVD 시간은 제외한다. 동일 벽시계 budget의 독립 우위나 일반적인13배 속도 보장으로 확대하지 않는다. 비용이 큰 물리 objective에서는 모델 fitting 절약의 상대적 중요성이 달라진다.

## 다음 메소드 판단

1. Count-table+probe를 새 기본값으로 승격하지 않는다. 이번 smooth-size stress의 후속 확인 기준선에는 **기존 SSPd3+소수 support probe**를 반드시 포함한다. 이 정책의 관측된 우위도 아직 외부 과제 전체에 대한 독립 확인은 아니다.
2. Count-table/within-count inference는 composition 보호 대조 및 step-like size의 후보로 유지한다. 임의 count shift의 강건성을 얻는 대가를 숨기지 않는다.
3. 다음에 수정할 대상은 composition 보호와 count 일반화의 연결이다. 가능한 가설은 composition 추정을 보호한 뒤 유연한 count prediction을 연결하는 방식과, 같은 count 모델을 완전히 joint fitting하는 방식의 비교다. 단순 모듈 결합만으로 새 기여를 주장하지 말고, feedback·상관·불확실성·관측 비용을 명시해야 한다.
4. 이 결과 bank는 앞으로 개발 근거다. 새 후보·policy fraction·kernel scale을 여기에 맞춰 고르고 같은 결과를 확인 성공으로 재사용하지 않는다. 후속은 별도 사전 계획과 새로운 bank를 쓴다.

## 독립 결과 검토와 산출물

- `RESULT_REVIEW_THEORY.md`: bootstrap·실제 incumbent 검증, 초기48개 gap SD 독립 재구성, source hash 확인.
- `RESULT_REVIEW_EMPIRICAL.md`: 원시 관측·쿼리·noisy incumbent·일치 count 경로·통계 단위와 gate 검증.
- `RESULT_REVIEW_NOVELTY.md`: primary 통과와 최강 대조 열세를 함께 인정, 문제–해결의 과학적 범위 검토.
- `SUMMARY.json`, `per_run.csv`, `CURVES.json`, `GAP_DIAGNOSTICS.json`, `runs/`에 원본 저장.
- `count_support.png`/`.pdf`는 고정 결과의 설명적 평균 곡선이다. 두 panel은 서로 다른 y축 scale이며, uncertainty band나 독립 확인 결과를 표시한 그림이 아니다.

[관측 곡선 PNG](count_support.png) · [관측 곡선 PDF](count_support.pdf)
