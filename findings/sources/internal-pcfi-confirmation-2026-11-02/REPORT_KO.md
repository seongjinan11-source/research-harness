# 내부 고정 PCFI — 독립 graph bank 재검증

새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다.

기존 개발 bank에서 선택한 후보를 확인하는 실험이다. 기존 checkpoint 6개를 그대로 재사용했고 새 모델·학습 update는 모두 0개다. 같은 학습 seed 11/12/13과 같은 official generator family이므로 학습 seed 집단·실제 graph·미지 schema 일반화 검증은 아니다.

## Primary

새 base graph 128개에서 uniform/clustered whole-node 90/99.5% missing 네 조건, 클래스당 최대 context 4개를 평가했다. 자기 피처가 없는 query의 episode 평균을 구한 뒤 네 조건과 세 고정 학습 seed를 graph 안에서 평균했다. 아래 CI는 graph 128개에 대한 paired bootstrap 5,000회이며, bank 생성 전에 단일 primary 기준과 full-feature 2%p guard를 고정했다.

| 방법 | Accuracy % | NLL | Brier |
|---|---:|---:|---:|
| external | 49.821540 | 0.856885 | 0.552034 |
| internal_fixed | 50.041098 | 0.856498 | 0.552432 |

내부 고정−외부 차이:

- accuracy %p: +0.219558, 95% CI [-0.331926, +0.796665]
- nll: -0.000388, 95% CI [-0.003777, +0.002808]
- brier: +0.000398, 95% CI [-0.001679, +0.002518]

Primary 확인=False, full-feature guard=True, FP/FISF 후속 진행 기준=False. Full-feature accuracy 차이는 -0.130208, 95% CI [-0.716146, +0.431519]%p다. CI가 0을 포함한 차이를 동등성이나 효과 없음으로 해석하지 않는다.

## 사전 보조 분할

| 분할 | External accuracy % | Fixed accuracy % | Fixed−external %p [95% CI] |
|---|---:|---:|---|
| uniform90% | 50.712277 | 50.891532 | +0.179255, 95% CI [-0.721513, +1.037430] |
| uniform99.5% | 49.128227 | 49.371109 | +0.242882, 95% CI [-0.950258, +1.449185] |
| clustered90% | 49.963865 | 50.561192 | +0.597327, 95% CI [-0.314759, +1.525249] |
| clustered99.5% | 49.481792 | 49.340558 | -0.141234, 95% CI [-1.435553, +1.131264] |
| cell99.5% | 49.338469 | 49.525492 | +0.187024, 95% CI [-1.029852, +1.404660] |
| uniform99.5%, context1/class | 43.955143 | 44.043430 | +0.088287, 95% CI [-1.265130, +1.457881] |
| full, 전체 query | 51.782227 | 51.652018 | -0.130208, 95% CI [-0.716146, +0.431519] |

Full을 제외한 표는 자기 피처 없는 query이며, 모든 query 결과는 statistics.json에 별도로 있다. 개별 분할은 다중비교 미보정 보조 결과로 primary를 대체하지 않는다.

| 고정 학습 seed | External accuracy % | Fixed accuracy % | 차이 %p |
|---|---:|---:|---:|
| 11 | 49.954117 | 50.577065 | +0.622948 |
| 12 | 49.818108 | 49.848964 | +0.030856 |
| 13 | 49.692395 | 49.697264 | +0.004868 |

독립된 새 학습 seed를 뽑은 결과가 아니므로 세 행을 독립 graph 표본처럼 bootstrap하지 않았다.

보조 prior 기준은 uniform: accuracy 42.722479%, NLL 0.880644; context_frequency: accuracy 46.135015%, NLL 0.858419다. Context 빈도만 사용하는 기준은 동률의 argmax 규칙에 민감할 수 있다.

## 원 결과와의 관계

원 bank의 fixed−external 차이는 +0.65743%p, 95% CI [−0.11984,+1.37358]였다. 네 primary 조건과 세 학습 seed에서 점추정이 모두 양수였지만 이 배치 비교는 원 context-router primary가 아닌 보조 관찰이었다. 그 관찰을 근거로 후보를 선택한 뒤 이번 새 bank의 기준을 잠갔다. 두 bank의 절대 accuracy를 빼서 개선폭으로 쓰거나, 합친 CI를 독립 재현으로 제시하지 않는다.

내부 고정은 PCFI 전파 계수가 고정된 방법이다. 전체 모델이 학습되지 않았다는 뜻이 아니다. 원 실험에서 공개 GraphPFN adapter/encoder/decoder를 각 seed당 256회 학습한 checkpoint를 사용했다. 두 방법 모두 실제 PCFI 100회·correlation 1회·PFN 12층 한 번이며, 내부는 25회씩 layer0/3/6/9에 배치한다. 원 코드·원 checkpoint·학습 bank를 수정하지 않았다.

## 검증과 비용

원 bank replay 18건을 design lock 전에 확인했다. 새 bank는 164~192 nodes, physical features 8~16이며 raw graph fingerprint가 원 train/dev/test와 겹치지 않는다. Primary 512개 episode 중 실제 관측 cell이 0개인 47개도 유지했다. Odd padding은 primary membership과 관측 수에서 제외했다.

저장 예측 5376건을 독립 재채점하고 source/config/bank/checkpoint/prediction SHA와 관측값 clamp, 100-step/1-correlation/12-layer 연산 수를 확인했다. 새 bank replay 18건의 최대 확률 차이는 0였다. 평가 전후 모델 파라미터 해시는 같았다.

| 방법 | 네 조건 median latency의 평균 ms | 추가 peak work memory MiB 범위 |
|---|---:|---:|
| external | 53.567025 | 19.360~19.360 |
| internal_fixed | 61.986337 | 22.561~22.561 |

같은 RTX3090에서 seed11/첫 graph 네 primary 조건, 5회 warmup·30회 반복의 CPU 입력→CPU 확률 시간이다. Confidence/거리 계산·PCFI·PFN을 포함하고 로드/파일 I/O를 제외했다. 총 GPU 메모리가 아닌 추가 작업 메모리다. 외부 GPU 부하는 통제하지 않았다.

## 판정 범위

새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다.

실패·불확실한 원인이 graph 변동, 제한된 학습 bank/seed, backbone/input interface, placement 자체 중 무엇인지 분리하지 않았으므로 원인 미상으로 남긴다. 동일 bank를 보고 학습량·온도·router·전파 횟수를 변경하는 구제 실험은 하지 않는다. 관측값을 전달하는 연산의 구현 자산과 원 개발 결과는 그대로 보존한다.

실행 원본: protocol.json, confirmation.py, METHOD_KO.md, bank/manifest.json, results/design.json, results/bank_lock.json, results/statistics.json, results/audit.json, results/cost.json. Raw graph와 checkpoint 및 개별 확률은 workspace에 유지하고 공유용 source snapshot에는 넣지 않는다.
