# 관측 좌표만 고정하는 Sheaf-PCFI: 개발 실험 결과

2026-11-03. 실제 관측된 첫 좌표만 clamp하고 두 번째 좌표는 관측 노드에서도 추론하는 변경을 검증했다. 기존 모델의 추론 변경과 같은 예산의 새 학습을 분리했다. 기존 결과와 checkpoint는 보존했다.

**현재 판정.** 현재 개발 bank에서 기존 Node와 PCFI를 함께 넘어선다는 주장은 확보하지 못했다. 계산의 표현력 차이는 확인됐지만, 그것을 실제 분류의 이득으로 연결했는지는 각 비교를 나누어 판단한다.
Partial-trained−Node -0.118%p, CI [-0.411,+0.160]; Partial-trained−PCFI -0.135%p, CI [-0.369,+0.083].

## Primary: 자기 피처가 없는 query

새 known-family base graph 32개, label homo/hetero × feature smooth/rough 각 8개다. 두 context 수·네 primary missing 조건·세 고정 학습 seed를 graph 안에서 평균했다. 실제 graph는 기존 개발 입력을 재사용했다.

| 방법 | Accuracy % | NLL | Brier |
|---|---|---|---|
| PCFI | 50.367 | 1.18267 | 0.58739 |
| 기존 Node 전체 clamp | 50.350 | 1.18438 | 0.58805 |
| Partial-frozen | 50.362 | 1.18370 | 0.58793 |
| Partial-trained | 50.232 | 1.18314 | 0.58809 |
| Partial 초기 network | 50.326 | 1.18255 | 0.58727 |
| Full-random | 48.268 | 1.24157 | 0.61703 |
| Partial-random | 47.927 | 1.23863 | 0.61710 |
| Smooth signed scalar | 49.946 | 1.19092 | 0.59282 |
| 기존 clipped scalar | 50.260 | 1.18483 | 0.58859 |

| 비교 | Accuracy 차이 %p | 95% CI | 판정 |
|---|---|---|---|
| Partial-trained − 기존 Node 전체 clamp | -0.118 | [-0.411, +0.160] | CI가 0 포함; 방향 미확인 |
| Partial-trained − PCFI | -0.135 | [-0.369, +0.083] | CI가 0 포함; 방향 미확인 |
| Partial-frozen − 기존 Node 전체 clamp | +0.011 | [-0.190, +0.209] | CI가 0 포함; 방향 미확인 |
| Partial-trained − Partial-frozen | -0.130 | [-0.413, +0.149] | CI가 0 포함; 방향 미확인 |
| Partial-trained − Partial 초기 network | -0.093 | [-0.314, +0.114] | CI가 0 포함; 방향 미확인 |
| Partial-trained − Partial-random | +2.306 | [+1.137, +3.648] | 양의 차이를 지지 |
| Partial-random − Full-random | -0.341 | [-0.714, +0.017] | CI가 0 포함; 방향 미확인 |
| Partial-trained − Smooth signed scalar | +0.286 | [-0.086, +0.637] | CI가 0 포함; 방향 미확인 |
| Smooth signed scalar − 기존 clipped scalar | -0.313 | [-0.611, +0.000] | CI가 0 포함; 방향 미확인 |
| Smooth signed scalar − PCFI | -0.421 | [-0.734, -0.097] | 음의 차이를 지지 |

Graph 단위 paired bootstrap 5,000회이며 다중비교 미보정이다. 세 고정 seed 및 저장된 확률에 조건부로 계산한 CI이며 GPU 수치 replay 불확실성을 포함하지 않는다.

## 해석 범위

**추론 때 경계 변경.** 평균 +0.011%p, CI가 0 포함; 방향 미확인. 같은 학습 frame/PFN에서 clamp만 바꾼 효과다.

**변경한 경계에 맞춘 학습.** 평균 -0.130%p, CI가 0 포함; 방향 미확인. 같은 원래 초기화부터 재학습한 효과이며 추가 update를 받은 fine-tuning이 아니다.

**기존 Node 대비.** 평균 -0.118%p, CI가 0 포함; 방향 미확인. 관측 경계 변경과 그 방식의 동일 예산 학습을 합친 비교다.

**PCFI 대비.** 평균 -0.135%p, CI가 0 포함; 방향 미확인. 현재 제안이 기존 강한 scalar confidence 전파를 넘는지에 대한 직접 비교다.

**Frame 학습.** 평균 +2.306%p, 양의 차이를 지지. 무작위 각도 대조와 학습된 입력 조건부 각도의 차이다. 무작위 범위를 조정하지 않았다.

두 번째 좌표가 실제로 0이 아니거나 소형 예제의 표현력이 늘었다는 것만으로 분류 메소드의 성공을 선언하지 않는다. 성능 차이의 원인이 식별 불가능성·증폭·학습 부족 중 무엇인지는 이 비교만으로 확정하지 않는다. 특히 개선이 없다면 clamp 변경의 현재 효용을 보류하는 것이며 모든 sheaf/PCFI/PFN 연구의 종료 판정이 아니다.

## Source 수와 결측 조건

| Source | 방법 | Accuracy % | NLL | Brier |
|---|---|---|---|---|
| zero | PCFI | 42.784 | 1.34245 | 0.65698 |
| zero | 기존 Node 전체 clamp | 42.784 | 1.34245 | 0.65698 |
| zero | Partial-frozen | 42.784 | 1.34245 | 0.65698 |
| zero | Partial-trained | 42.784 | 1.34245 | 0.65698 |
| zero | Partial-random | 42.784 | 1.34245 | 0.65698 |
| zero | Smooth signed scalar | 42.784 | 1.34245 | 0.65698 |
| nonzero | PCFI | 50.707 | 1.17356 | 0.58350 |
| nonzero | 기존 Node 전체 clamp | 50.675 | 1.17589 | 0.58437 |
| nonzero | Partial-frozen | 50.702 | 1.17521 | 0.58432 |
| nonzero | Partial-trained | 50.576 | 1.17420 | 0.58432 |
| nonzero | Partial-random | 47.905 | 1.23854 | 0.61802 |
| nonzero | Smooth signed scalar | 50.203 | 1.18313 | 0.58981 |
| one | PCFI | 47.752 | 1.21357 | 0.60866 |
| one | 기존 Node 전체 clamp | 47.573 | 1.21707 | 0.61035 |
| one | Partial-frozen | 47.581 | 1.21706 | 0.61035 |
| one | Partial-trained | 47.608 | 1.21450 | 0.60957 |
| one | Partial-random | 48.642 | 1.22521 | 0.61315 |
| one | Smooth signed scalar | 46.761 | 1.22836 | 0.61951 |
| two_or_more | PCFI | 52.197 | 1.15240 | 0.56992 |
| two_or_more | 기존 Node 전체 clamp | 52.202 | 1.15445 | 0.57061 |
| two_or_more | Partial-frozen | 52.233 | 1.15344 | 0.57054 |
| two_or_more | Partial-trained | 52.043 | 1.15306 | 0.57073 |
| two_or_more | Partial-random | 47.588 | 1.24450 | 0.61986 |
| two_or_more | Smooth signed scalar | 52.023 | 1.15961 | 0.57394 |

Source0와 nonzero는 서로 다른 과제 부분집합이다. 각 부분집합에서도 해당 조건의 seed/context/mask를 graph 안에서 먼저 평균하고, 포함된 base graph에 같은 가중치를 준다. 두 평균의 차이를 관측 피처의 인과 효과로 해석하지 않는다. Source0/full은 같은 PFN 입력으로 돌아가는 correctness 조건이다.

| Source 제어 조건 | PCFI % | Node % | Partial-trained % | Partial-random % |
|---|---|---|---|---|
| full | 98.432 | 98.432 | 98.432 | 98.432 |
| sources2 | 50.519 | 50.457 | 50.557 | 47.887 |
| sources8 | 52.222 | 52.643 | 52.541 | 47.852 |

**추가 기전 분석.** 학습 구조와 primary를 고정한 뒤, 같은 frame에서 연결 성분별 관측 source가 하나뿐이면 전체/부분 clamp가 같은 계산이 됨을 유도하고 소형 24개 예제로 확인했다. 공통 좌표의 상태가 그 source 방향의 span에 머물기 때문에 새 성분을 얻지 못한다. Source0/one/two+ 조건은 전체 primary를 바꾸지 않은 secondary 분석이다.
Primary 256 episode 중 source0 42개, source1 74개, source2+ 140개다. 새로 학습한 frame과 기존 frame은 다를 수 있으므로, 별도 학습한 Partial-trained와 Node까지 source1에서 동치라는 뜻은 아니다.

## 생성 조건과 학습 seed

| Label / feature | PCFI % | Node % | Partial-frozen % | Partial-trained % | Smooth signed % |
|---|---|---|---|---|---|
| homo_smooth | 58.522 | 58.765 | 58.453 | 58.235 | 58.070 |
| homo_rough | 58.931 | 59.123 | 59.245 | 58.852 | 58.396 |
| hetero_smooth | 50.356 | 49.930 | 50.075 | 50.382 | 50.111 |
| hetero_rough | 33.662 | 33.584 | 33.674 | 33.460 | 33.207 |

| Seed | Partial-trained − Node %p | Partial-trained − PCFI %p |
|---|---|---|
| 11 | -0.102 | +0.067 |
| 12 | -0.062 | -0.275 |
| 13 | -0.190 | -0.197 |

Label heterophily와 feature roughness는 별도 축이다. Label heterophily만으로 모든 numeric feature의 부호를 반전해야 한다고 가정하지 않는다. Context별 결과와 mask별 결과는 statistics.json에 함께 보존했다.

## 실제 graph: 개발 평가

| 방법 | Cora % | CiteSeer % | Texas % | Wisconsin % |
|---|---|---|---|---|
| PCFI | 50.229 | 30.916 | 27.844 | 21.557 |
| 기존 Node 전체 clamp | 49.835 | 31.223 | 28.178 | 22.048 |
| Partial-frozen | 49.800 | 31.514 | 26.424 | 22.388 |
| Partial-trained | 49.824 | 31.158 | 27.422 | 22.058 |
| Partial 초기 network | 50.181 | 30.996 | 27.858 | 21.916 |
| Full-random | 38.042 | 25.905 | 29.302 | 24.059 |
| Partial-random | 38.578 | 25.320 | 26.871 | 24.034 |
| Smooth signed scalar | 49.272 | 31.045 | 28.301 | 23.378 |
| 기존 clipped scalar | 49.424 | 30.622 | 28.218 | 22.007 |
| native_pcfi | 52.030 | 33.867 | 33.260 | 24.445 |
| negative_lp | 32.026 | 24.757 | 63.537 | 42.725 |
| mask_reference | 40.046 | 25.012 | 21.282 | 23.864 |

새 아홉 arm은 동일 real 입력에 이번 실행에서 평가했다. Native-PCFI/negative-LP/Mask는 기존 동일 입력 cache 기준이다. Native-PCFI는 원본 고차원에서 먼저 전파하는 더 강한 전체 pipeline이며 geometry-only 대조와 구분한다. 네 실제 graph는 독립 holdout이 아니다. Downstream은 기존 synthetic-trained 실험용 wrapper이며 공식 GraphPFN 전체 pipeline의 재현 성과가 아니다.

## 잠재 상태와 smooth scalar

| 입력 | Partial arm | 관측 노드 두 번째 좌표 평균 절댓값 | 마지막 step 변화 최댓값 | 출력 절댓값 최댓값 |
|---|---|---|---|---|
| development | Partial-frozen | 0.266995 | 0.018078 | 3.415431 |
| development | Partial-trained | 0.110501 | 0.008517 | 3.415431 |
| development | Partial-random | 0.197576 | 0.006455 | 3.415431 |
| real | Partial-frozen | 0.343472 | 0.100982 | 4.408170 |
| real | Partial-trained | 0.162693 | 0.115872 | 4.373160 |
| real | Partial-random | 0.258006 | 0.058482 | 7.703210 |

고정 100회 전파가 실제 실행한 메소드다. 마지막 변화가 작다는 사실만으로 잠재 상태의 유일성이나 정답성을 주장하지 않는다. Source가 거의 없을 때 자유로운 성분의 식별 문제가 남을 수 있지만, 성능 저하 원인으로 확인된 것은 아니다.

| 입력 | Smooth 음수 gain 비율 | 평균 gain | abs(g)>0.999 비율 | 평균 tanh 미분 |
|---|---|---|---|---|
| development | 0.000000 | 0.989851 | 0.000000 | 0.020190 |
| real | 0.000000 | 0.989821 | 0.000000 | 0.020251 |

Hard clipping은 제거했지만 smooth한 함수도 saturation될 수 있다. 계수의 실제 음수 사용과 미분을 보고하며, 이 한 parameterization을 전체 signed 함수족의 최강 대조라고 주장하지 않는다.

## 검증과 보존

새 frontend 6개, 6144 update. 기존과 초기 parameter·train task schedule·PFN hash가 일치한다. PFN/real update는 모두 0회다. 사전 검사 optimizer step은 폐기했다.
기존 파일 20639개, 입력 2072개 및 새 예측 13716개 hash를 확인했다.
전체 clamp의 기존 구현 차이 0, 부분 clamp의 독립 좌표계 출력 차이 5.55e-16, parameter gradient 차이 1.09e-14. Identity는 PCFI와 정확히 같았다.
소형 분석 예제에서 기존 출력 0.707107, 부분 clamp 출력 1.414214였다. 이는 다른 함수족이라는 검증이며 분류 성능 개선이 아니다.
Full/source0 확률 환원 2904건, 최대 차이 0.0000000.
같은 frame의 source1 Partial-frozen−Node 확률 비교 246건, 최대 차이 0.0192748. 수학적 연산 동치와 실제 GPU 확률 replay 차이를 구분한다.
선택된 replay에서 저장 확률 대비 최대 차이 0.0126960, 같은 session 반복 최대 차이 0.0078100, argmax 변경 최대 0.700%. 0.1 threshold는 큰 손상 검사일 뿐 엄격한 재현 보장이 아니다.

근거: [메소드](METHOD_KO.md), [고정 프로토콜](protocol.json), [전체 통계](results/statistics.json), [audit](results/audit.json).
