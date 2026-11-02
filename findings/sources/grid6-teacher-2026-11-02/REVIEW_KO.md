# Grid6 teacher 판정

이번 비교는 공개 Mask GraphPFN에 대한 primary 가설을 지지하지 못했다. Exact teacher를 주는 것만으로 현재 입력·구조·예산의 GraphPFN이 강한 추론기를 학습해 압축한다는 방향은 여기서 확대하지 않는다. 다만 Factor PFN의 teacher-distribution fit 개선과, feature/label source를 연결하는 효과는 별도로 보존한다.

## 같은 bank에서 비교한 결과

| 모델 | 감독 | Accuracy % | NLL | Exact Grid6 predictive KL |
|---|---|---:|---:|---:|
| 공개 Mask PFN | Sampled-label CE | 55.3475 | 0.678973 | 0.104788 |
| 공개 Mask PFN | Exact predictive distill | 54.6255 | 0.679253 | 0.104134 |
| Factor PFN | Sampled-label CE | 69.8468 | 0.575272 | 0.067257 |
| Factor PFN | Exact predictive distill | 69.5877 | 0.568642 | 0.049731 |
| Fixed-mode factor BP | 학습 없음 | 67.5182 | 0.610127 | 0.107381 |
| Grid6-BP | 알려진 family 추론 | 68.3284 | 0.551872 | 0.000010 |

새96 graph 중 matched disjoint48 graphs의 결과다. 같은 기존 train2048 tasks에서 원래 초기화로 새6개 모델을 학습했고, 기존 CE9개 모델을 추가 학습 없이 이 bank에서 재평가했다. 세 fixed training seed와 context/관측수 조건을 graph 내 평균한 paired bootstrap5000회다. 이전 bank의 절대 수치와 빼서 distillation 효과라고 쓰지 않는다.

## 1. 공개 GraphPFN primary는 미통과

Mask distill−CE teacher KL은−0.000654,95%CI[−0.002034,+0.000583]였다. Accuracy−0.7220%p, NLL+0.000280의 CI도0을 포함한다. 따라서 이 예산에서 확실한 이득을 확인하지 못했다. CI가0을 포함한다고 distillation 효과가 항상0이라고 입증한 것은 아니다.

추론 median도 Mask59.735ms, Grid6-BP12.425ms로 압축의 latency 이점이 없었다. Mask teacher KL0.104134는 사전 .01 기준을 넘는다. Primary teacher-effect와 competitive compression gate 모두 False다.

다만 이것은 작은 known-family의 현재 frozen-backbone/adapters·입력 경로·256-update 구성에서 감독만 바꾼 결과다. Teacher 자체가 틀렸거나 모든 PFN이 불가능하다고 결론 내리지 않는다. 입력 경로, 표현력, 최적화, 학습량 중 고유 원인은 분리하지 않았다.

## 2. Factor 학생의 확률분포 근사 개선은 실제로 남는다

Factor distill−CE teacher KL은−0.017526,95%CI[−0.026031,−0.010538]였다. CE 대비 약26.1% 감소다. Fixed-mode factor BP 대비도−0.057650,CI[−0.089157,−0.031682]로 낮았다.

그러나 Factor의 accuracy 차이−0.2590%p,CI[−1.2112,+0.7551]와 NLL 차이−0.006630,CI[−0.023930,+0.007280]는0을 포함한다. 따라서 teacher fit의 개선을 분류 정확도·실제 label log score의 확실한 우위로 확대하지 않는다. ECE를 계산하지 않았으므로 calibration 개선이라고도 쓰지 않는다.

Grid6-BP 대비 Factor distill accuracy+1.2594%p,CI[−0.9537,+3.5799], NLL+0.016771,CI[−0.016408,+0.053202]였다. 우위·동등성 모두 입증한 것이 아니다. Grid6 KL0.049731은 여전히 남고, median8.764ms 대12.425ms의 약1.42배는 사전2배 기준에 미달한다. Factor는 사전 secondary이므로 Mask primary 실패를 대체하지 않는다.

특히 query 자신도 physical feature가 없는 사전 secondary에서는 Factor CE64.7164%/NLL0.614173, distill63.5556%/0.618377이었다. Accuracy 차이−1.1609%p,95%CI[−2.4190,−0.0482]로 하락했고 NLL 차이+0.004205,CI[−0.002231,+0.010960]는0을 포함한다. 같은 bank의 다중 보조 비교이며 multiplicity를 보정하지 않았지만, 전체 teacher KL 감소만 보고 이 중요한 분할을 생략하지 않는다.

## 3. Disjoint feature 활용의 재현은 보존한다

사전 matched disjoint context2/observed6 조건에서 Factor distill의 physical observations 제거는 accuracy−11.9213%p,CI[−19.0972,−4.8611]였다. Feature shuffle−20.7176%p와 graph permutation−29.0509%p의 CI도0을 제외했다.

기존 CE Factor도 이 새 bank에서 제거−12.0370%p,CI[−19.6759,−4.2824]였다. 따라서 이번 새 bank는 기존 factor 연산이 실제 관측정보를 활용한다는 근거를 보완하지만, 그 활용 자체를 distillation이 새로 만들었다고 해석할 수 없다. 제거에 대한 NLL/Brier 차이의 CI는0을 포함한다. 이 intervention은 전체 primary와 다른 보조 조건이다.

## 남기는 결론

이번에는 생성기가 달라진 것이 아니다. 기존 Grid6 생성 prior의 정확한 query predictive를, 한 번 샘플링한 class label 대신 감독으로 사용했다. 기대 CE와 exact predictive KL은 teacher entropy 상수를 제외하면 같은 목표이므로, 이 비교는 조밀한 감독이 제한된 예산에서 도움이 되는지를 시험한다.

현재 공개 Mask 학생에서는 그 변경만으로 부족했다. Factor 학생에서는 예측분포 근사에 도움이 됐다. 이를 새로운 메소드나 unseen-schema PFN의 완성으로 부르지 않는다. Exact-grid teacher의 열거는 지수 비용이고, 이번은12 nodes/2 Gaussian features/2 relations/50·75% whole-node missing뿐이다. 다른 feature law·real heterograph·99.5% missing·큰 graph·새 schema의 distillation은 미실행이다. 작은 수치를 살리기 위한 temperature/head/loss/학습량 추가는 이번 실험에 넣지 않았다.

재현·검증: `REPORT_KO.md`, `protocol.json`, `METHOD.md`, `results/paired.json`, `results/interventions.json`, `results/benchmark.json`, `results/final_integrity.json`. 새6개 모델1536 successful updates/12288 task exposures, 기존9개 재평가, source/bank/teacher hash와45개 예측 재실행을 확인했다. 이전125개 연구기록의 관찰과 판정은 수정하지 않는다.
