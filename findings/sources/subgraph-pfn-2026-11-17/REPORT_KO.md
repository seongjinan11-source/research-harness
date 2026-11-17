# Subgraph-PFN: 관측 라벨이 적은 클래스를 위한 context 압축

**구현 및 실험 완료.** 합성 그래프에서 같은 토큰 예산의 원형 노드 검색보다 그룹 요약이 정확했지만, 모든 소수 앵커를 남기는 배정 규칙의 별도 이득은 입증하지 못했다. 실제 그래프에서는 단순 PPR보다 균형 정확도가 낮았고, 전체 앵커보다 토큰은 적어도 더 빨라지지 않았다. 현재 후보의 확장 gate는 **FAIL**이다.

공개 NodePFN 가중치를 개조하거나 교사 모델을 distill하지 않았다. 3,705개 파라미터의 작은 class-shared attention head를 degree-corrected SBM/graph-conditioned nonlinear feature prior에 대해 합성 태스크만으로 학습했다. Real model weights는 고정이다. 클래스당 관측 라벨이 1–4개면 앵커를 그대로 유지하고, 그보다 많은 클래스의 라벨 앵커는 최대 두 그룹으로 묶는다. 각 그룹에는 전체 centroid와 query에서 도달하는 PPR 가중 centroid, 질량, 실제 그룹 크기와 클래스의 관측 수를 제공한다. 소수는 숨은 모집단 크기가 아니라 관측 라벨 수를 뜻한다. 이 계산은 전체 그래프에 sparse diffusion을 적용하며, ego-net을 물리적으로 추출하는 구현은 아니다.

## 합성 held-out 96 graphs

동일 prior에서 훈련한 4개 arm × 3 seeds × 320 updates. 학습과 겹치지 않는 그래프 96개(크기 holdout 48), 클래스마다 query 8개다. 아래 값은 seed 평균 후 graph 평균이다.

| 방법 | Tail recall (%) | 전체 bAcc (%) | NLL | Brier |
|---|---:|---:|---:|---:|
| Tail-keep + group moments | 74.51 | 74.68 | 0.6354 | 0.3338 |
| 동일 토큰 수 비례 배정 + group moments | 73.85 | 74.73 | 0.6367 | 0.3339 |
| 동일 클래스별 예산의 PPR+피처 raw top | 71.90 | 71.59 | 0.7153 | 0.3716 |
| Full anchors (scalar-only diffusion) | 75.12 | 74.66 | 0.6473 | 0.3399 |
| Class-mean PPR | 28.80 | 34.61 | 1.5778 | 0.7524 |
| Feature centroid | 59.07 | 69.70 | 1.1379 | 0.6050 |
| Balanced ridge | 61.63 | 71.90 | 1.0648 | 0.5632 |
| Uniform class probability | 24.31 | 25.78 | 1.3830 | 0.7422 |

| 고정한 비교 | 차이 (%p) | nominal 97.5% paired graph CI (%p) | 판정 |
|---|---:|---:|---|
| Tail recall: tail − proportional | +0.658 | [−0.340, +1.657] | FAIL |
| Tail recall: tail − raw top | +2.611 | [+1.183, +4.066] | PASS |
| 전체 bAcc: tail − full | +0.023 | [−0.616, +0.678] | −2%p guard PASS |

토큰 가드와 무학습 기준선 floor도 통과했다. `top`도 동일한 클래스별 쿼터를 쓰므로 관측 라벨 1–4개 클래스의 앵커는 모두 보존한다. 이 arm 대비 향상을 **tail singleton 보존 자체의 효과**로 읽을 수 없다. 대조가 비교하는 것은 같은 토큰으로 나머지 앵커의 정보를 집계하는 방법과 일부 원형 anchor를 선택하는 방법이다. 반면 quota 효과를 비교하는 `proportional`은 두 arm에서 그룹 수가 같은데 그 차이는 확실하지 않았다. 차이가 0임을 증명한 것도 아니다. Full 대비 guard는 전체 bAcc에 관한 것이며 tail recall 동등성까지 주장하지 않는다.

## 실제 그래프 개발 패널과 비용

고정 RP24 이후 5개 split을 공유했다. 공식 train mask 내에서 관측 2/16개, 공식 test에서 클래스마다 query 32개. Full PubMed 19,717노드를 포함한다. 재사용 그래프의 기술 점수이며 release PFN, native feature, arXiv 또는 독립 real holdout 평가를 대신하지 않는다.

| 그래프 | Tail recall: 제안 / PPR (%) | 전체 bAcc: 제안 / PPR (%) | CPU 준비+점수: 제안 / full (ms) |
|---|---:|---:|---:|
| Cora | 43.44 / 59.84 | 41.37 / 62.95 | 41.64 / 23.07 |
| CiteSeer | 31.25 / 31.04 | 24.90 / 42.60 | 46.76 / 20.86 |
| PubMed | 51.04 / 52.19 | 48.13 / 60.00 | 112.49 / 32.48 |

Timing은 각 그래프 고정 첫 draw, CPU 1 thread, warmup 후 3회 중앙값; 공통 RP 변환(그래프별 약 2.64/8.61/7.01ms), raw 파일 로드, 모델 로드, 결과 저장은 제외한다. 전체 범위와 산출물은 `results/artifact_manifest.json`에 있다. 큰 합성 held-out에서 query당 평균 **9.65 vs 59.19** 토큰(6.14× 차이)은 배정 규칙이 보장하는 decoder 크기 절감이다. 전체 실행의 속도나 peak RAM 절감이 아니다. Full은 singleton당 질량 한 열만 확산하지만 묶음의 local feature centroid는 `1+3F`개 열을 확산한다. 예를 들어 Cora 측정 draw에서 tail은 446열, full은 56열이다. 이 차이가 diffusion 비용에 실제로 드러났다. 기록된 seed 블록 바이트와 pair bytes는 전체 RAM 피크가 아니다.

이번 비교로 남은 것은 **같은 예산에서 선택보다 집계가 정보를 더 잘 보존할 수 있다**는 제한된 증거다. 반면 **관측 tail에 더 많은 토큰을 줘야 한다**, **실제 불균형 그래프에서도 더 잘 분류한다**, **전체 추론이 빨라진다**는 주장까지 뒷받침하지 못했다. Local moment는 유한 diffusion에서 정확히 계산되지만 비선형 분류나 실제 Bayesian posterior에 충분한 통계라는 보장은 없다. 기존 [LoGIC](https://arxiv.org/html/2609.05955v1)은 graph ICL의 context 예산과 PPR 검색을, [Ephris](https://arxiv.org/html/2609.37057v1)는 sparse graph ICL을 이미 다룬다. 여기서는 희소 관측 label과 같은 예산의 요약/검색 비교만 검증했다.

**남겨둘 다음 조건:** (i) 단순 클래스별 PPR에서 보존되는 구조적 증거가 실제 데이터까지 넘어가는 비교, (ii) `min(|group|,1+3F)` 기준으로 singleton→모멘트와 직접 group diffusion 중 좁은 쪽을 계산하는 동등 연산을 구현한 후 시간 재측정. 둘 다 이번 성과에 포함하지 않는다. 현재 합성·실제 bank를 기준 성능을 보고 튜닝하는 평가용으로 다시 잠그지 않는다.

재현 범위: 12 fits / 3,840 successful updates, prelock 폐기 dev 5 updates, 96+15 cases에 걸친 확률 배열 1,776개 재채점, target update 0. 원 계획과 세 리뷰, 수치 검사, input/answer identity 및 checkpoint/schedule 감사가 별도로 보존돼 있다.
