# 부호 전달 진단 — PCFI 복원과 GraphPFN 분류의 한계 분리

같은 PCFI→PFN checkpoint에서 부호 법칙을 반영한 복원의 예측적 가치는 확인했다. 그러나 강한 직접 예측·graph-only 대조를 모두 넘는 기준은 통과하지 못해, 새 학습 모델이 필요하다는 근거로 확대하지 않는다.

기존 Mask Plain/PCFI Plain의 최종 checkpoint를 3개 seed씩 그대로 재사용했다. 새 모델과 학습 update는 0개다. 두 arm은 이전 official graph/SCM bank에서 같은 schedule과 192-update 예산으로 학습됐다. 이번 graph-signal family는 그 학습 prior 밖이며, frozen 실패를 architecture의 표현력 불가능성으로 해석하지 않는다. 여기서 Plain은 공개 GraphPFN Mask Plain이고 과거 소형 typed-relation Plain이 아니다.

## 무엇을 분리했나

192-node connected bipartite graph에서 저주파 Gaussian field a를 만들었다. Feature law g=0은 이를 그대로, g=1은 bipartition b를 곱해 관측한다. Label law t=0은 a, t=1은 b*a를 median threshold한 binary class다. 여덟 column은 같은 field의 서로 다른 방향·scale·작은 noise 버전이다. Feature law와 label law를 독립적으로 바꿨다.

각 label 조건 내 graph·labels·context·query·observation mask는 feature-law 두 설정에서 같다. 실제 달성된 label homophily와 feature energy를 기록했으며 원하는 수치가 나올 때까지 graph를 재추출하지 않았다. 새 base graphs32개와 개발 graphs2개, 최종32조건/graph=1,024 episodes다. Edge observation missing은 없다.

Context는 클래스당1/4개, 관측 source는 context 밖에서 정확히0/1/8개다. 1개 source는99.48%,8개는95.83% whole-node missing이고 Bernoulli99.5%와 다르다. Full-feature reference만 context에도 피처가 있다. Primary는 자기 feature가 없는 query다. Hidden values와 query labels는 predictor에 들어가지 않는다.

## Primary

Label/feature 둘 다 sign-gauged(t=1,g=1), sources8, context4/class 조건이다. 세 고정 training seed를 base graph 안에서 평균한 뒤32개 graph를 paired bootstrap5,000회 했다. CI는 training-seed population에 대한 값이 아니다.

| 방법 | Missing-query accuracy % | NLL | Brier |
|---|---:|---:|---:|
| 기존 Mask Plain | 52.405028 | 0.709999 | 0.512094 |
| 기존 PCFI Plain | 50.190226 | 0.715966 | 0.520255 |
| 같은 PCFI Plain + known-law signed 복원 | 56.026584 | 0.707501 | 0.506394 |
| 고정 양의 LP | 46.683098 | 0.796353 | 0.572964 |
| 고정 signed LP | 56.260834 | 0.687817 | 0.492625 |
| Uniform | 50.265421 | 0.693147 | 0.500000 |
| 같은 PCFI checkpoint, 피처 제거·동일 query | 53.693289 | 0.750753 | 0.535923 |
| 같은 PCFI checkpoint, full·동일 query | 91.944505 | 0.323175 | 0.179793 |

Signed 복원 reference−대조 accuracy 차이(%p):

- known_signed-pcfi: +5.836358 [95% CI +2.291311, +9.440175]
- known_signed-mask: +3.621556 [95% CI +0.027501, +7.207857]
- known_signed-positive_lp: +9.343486 [95% CI +5.338770, +13.482302]
- known_signed-signed_lp: -0.234249 [95% CI -5.629258, +4.934528]
- 같은 PCFI checkpoint의0-source 대조, 동일 query 집합: +2.333295 [95% CI -1.034197, +5.630727]

원 PCFI checkpoint의 full−missing accuracy 차이, 동일 query 집합: +41.754278 [95% CI +38.860289, +44.667451]%p.

사전 reconstruction headroom=True, 강한 direct/LP/feature-use 기준=False. Primary signed−PCFI의 NLL 차이는 -0.008464 [95% CI -0.034323, +0.020560], Brier 차이는 -0.013862 [95% CI -0.035024, +0.008416]다.

원 PCFI−피처 제거의 동일-query accuracy 차이는 -3.503063 [95% CI -6.429188, -0.848897]%p다. Signed 복원−피처 제거 차이의 CI는0을 포함한다. 따라서 signed 교체가 원 PCFI의 분류를 개선한 것은 맞지만, 피처가 없는 입력보다 추가 예측 정보를 유의하게 회수했다고 주장할 수는 없다. 이 cross-budget 보조 CI는 다중비교 미보정이다.

**Known-law reference는 참 feature-law g를 안다.** Bipartition b는 adjacency만으로 구하지만, g가 주어졌다는 점은 privileged information이다. 실제 missing X를 공개하지 않고 g=1에서 b*x를 PCFI로 복원한 뒤 b를 다시 곱했다. 새 그래프에서 g를 알아낸 모델이 아니며 이 결과만으로 신경망 학습이 필요하다고 할 수 없다. 양의/signed LP도 두 후보를 모두 보고하며 query truth로 좋은 쪽을 고르는 predictor는 만들지 않았다.

## Label/feature law factorial

Sources8,context4/class의 보조 결과다. Label H는 edge 양끝의 클래스가 같은 비율이며 feature energy와 다른 관측량이다.

| Label law t | Feature law g | Label H | Mask % | PCFI % | Known signed % | 양의 LP % | Signed LP % | PCFI MSE | Signed MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.753531 | 54.664126 | 55.679568 | 55.679568 | 57.751757 | 51.194184 | 1.866379 | 1.866379 |
| 0 | 1 | 0.753531 | 54.933418 | 54.453823 | 52.257249 | 57.751757 | 51.194184 | 2.320895 | 1.866379 |
| 1 | 0 | 0.243647 | 51.680733 | 47.266016 | 47.266016 | 46.683098 | 56.260834 | 1.319515 | 1.319515 |
| 1 | 1 | 0.243647 | 52.405028 | 50.190226 | 56.026584 | 46.683098 | 56.260834 | 1.909480 | 1.319515 |

MSE는 실제 관측값에서 구한 RMS scale과 clip10을 적용한 hidden query feature에 대한 값이다. Full-X의 통계로 model 입력을 정규화하지 않았다. MSE 개선은 classification 이득과 별개다. 원 PCFI−Mask 차이에 대한 feature flip difference-in-differences는 +2.199915 [95% CI -1.199287, +5.699655]%p다. 이는 보조 interaction 분석이고 primary를 대체하지 않는다.

특히 label은 smooth하고 feature만 sign-gauged인 t0/g1에서는 signed 복원이 MSE를 낮췄지만 accuracy 점추정은 54.453823%에서 52.257249%로 바뀌었다. 보조 차이 -2.196574 [95% CI -5.777905, +1.506777]%p다. 복원 이득을 모든 task의 분류 이득으로 확대할 수 없다.

## Source 수와 context 수

Label/feature 모두 sign-gauged인 보조 분할이다. Source0은 피처 제거 대조이며, full은 관측 query도 포함하는 별도 reference다.

| k/class | Sources | Mask accuracy % | PCFI accuracy % | Known signed accuracy % | 양의 LP % | Signed LP % |
|---|---|---:|---:|---:|---:|---:|
| 1 | 0 | 51.595052 | 49.967448 | 49.967448 | 49.609375 | 51.953125 |
| 1 | 1 | 50.695144 | 50.697245 | 51.452243 | 49.719632 | 52.063382 |
| 1 | 8 | 50.860961 | 48.957768 | 52.540564 | 49.632781 | 52.299444 |
| 1 | full | 63.834635 | 64.713542 | 64.713542 | 49.609375 | 51.953125 |
| 4 | 0 | 53.417969 | 54.101562 | 54.101562 | 46.875000 | 55.957031 |
| 4 | 1 | 51.490045 | 47.375882 | 50.477781 | 46.654486 | 55.994834 |
| 4 | 8 | 52.405028 | 50.190226 | 56.026584 | 46.683098 | 56.260834 |
| 4 | full | 92.447917 | 91.829427 | 91.829427 | 46.875000 | 55.957031 |

개별 분할 CI와 전체2×2×2×4 factorial, seed별 결과는 statistics.json에 있다. 보조 CI는 multiplicity 미보정이다. Source 수 표는 각 source 조건의 자체 missing-query 집합이므로 직접적인 intervention 차이는 위 primary와 동일한 query 집합으로 다시 계산한 값을 사용한다.

## 보조 집계 감사 보정

원 statistics.json의 zero-source/full cross-budget 차이는 각각의 missing-query 집합(full은 전체 query)을 사용해 query 집합 변화가 섞였다. Primary8-source의 missing-query 집합으로 두 예측을 모두 제한해 results/common_query_interventions.json에 재계산했다. 원 statistics·예측·bank·checkpoint는 그대로 보존했다. Primary와 같은 cell 안의 모든 비교 및 strong-floor 판정은 바뀌지 않았다. Full 입력은 context feature도 관측하며 자체 RMS를 사용하므로 완전 입력 reference로만 해석한다.

## 수학적 반례와 실험의 범위

단일 column·연결된 path에서 관측값이+1이고 결측 시작값이0이면 PCFI의 양의 row-stochastic 전파는 음수 값을 생성하지 못한다. 한 column이라 channel correction도0이다. 개발 검사에서 일반 PCFI는 음수 값을 만들지 못했고 signed 변환은 필요한 반대 부호를 만들었다. 이는 복원 연산의 반례다. GraphPFN이 graph/context로 정답을 복구할 수 없다는 증명은 아니다.

Signed diffusion/BP는 우리 기존 연구에서 이미 구현했다. 이 실험은 새로운 attention·posterior·loss·메소드가 아니라, 현재 PCFI와 frozen 공개 PFN의 역할을 분리하는 known-family 진단이다. 고정 Gaussian/signed 추론의 기존 연구와 구분되는 학습 원리를 확보한 결과도 아니다.

## 비용과 검증

| 경로 | Median latency ms | 추가 peak work memory MiB |
|---|---:|---:|
| 기존 Mask Plain | 50.623700 | 13.604 |
| 기존 PCFI Plain | 54.287450 | 13.609 |
| 같은 PCFI Plain + known-law signed 복원 | 59.609350 | 13.609 |

같은 RTX3090에서 입력→CPU 확률, seed11/첫 graph primary cell,5warmups/30repeats다. PCFI 경로는 실제 PCFI를 다시 계산하고 signed reference는 parity 탐색도 포함한다. 로드/파일I/O를 제외하고 외부 GPU 부하는 통제하지 않았다. 이는 추가 작업 메모리이며 총 GPU 메모리가 아니다.

원 checkpoint replay18건을 lock 전에 확인했다. 저장 예측 9216건을 재채점했고 새 replay18건의 최대 확률 차이는 0였다. Source/config/bank/checkpoint/prediction SHA, 입력 factorial pairing, hidden feature poison, 관측값 clamp, zero-source 입력/출력 동일성과 평가 전후 모델 파라미터 해시를 확인했다. 새 모델/updates는0이다.

## 최종 판정

같은 PCFI→PFN checkpoint에서 부호 법칙을 반영한 복원의 예측적 가치는 확인했다. 그러나 강한 직접 예측·graph-only 대조를 모두 넘는 기준은 통과하지 못해, 새 학습 모델이 필요하다는 근거로 확대하지 않는다.

Frozen 결과의 고유 원인은 분리하지 않았다. 새로운 graph-signal law와 학습 prior의 차이, 입력/표현·최적화·decoder 한계 등이 섞일 수 있다. 모든 heterophily나 PCFI/PFN 통합의 무효로 확대하지 않는다. 현재 bank를 보고 온도·checkpoint·학습량·feature-law를 조정하지 않는다.
