# 완전 비선형 구성 모델 실험: 실행 완료, projection 확대 중단

2026-11-05. 기존 SSPd3는 count 곡선만 비선형이고 composition은 선형이었다.
이번에는 composition도 비선형인 Hamming GP와 quadratic GP를 직접 구현했다.

## 실행과 검증

- 5 families x4 targets x7 arms=140 runs, 14,000 charged objective rows.
- 새 선형·비선형 count·독립 pair interaction·약한 명시적 size 추세·corrected
  physical tomography. 모든 모델이 같은 finite catalog와 초기 관측을 사용했다.
- 같은 fixed-count schedule, 같은 100평가 예산, archive/batch 중복 제외.
- Kernel PSD·hypergeometric projection·원래 kernel 복원·SSP 환원 등 8개 검사,
  5종 GP의 analytic likelihood gradient와 posterior 직접 계산 검사 통과.
- 모든 raw objective와 HV/AUC 독립 replay 14,000행 통과.
- 전체 온라인 실행이 끝난 후에만 51,476개 catalog reference 값을 별도로 평가했다.
  공통 초기 archive만 사용하는 120개 사후 fit/ranking 진단도 별도 기록했다.
- GPU 없음, 4 CPU workers. 온라인 wall time 10.18초. 원본 SSPd는 optimizer
  status 미공개이며 GP fit의 flag 실패는 1,000회 중3회였다. Primary 두 GP에는
  실패 flag가 없었다. 경계 fit이 많아 그 범위 한계를 유지한다.

## 사전에 정한 주 비교

gp_orth = a²(k-A)+b²R_count 대 gp_plus = a²k+b²R_count.
이는 tied count covariance A를 제거하는 비교다. gp_plus도 독립 count/residual
GP로 정확히 재표현되므로 독립성의 신규 획득이나 새로운 표현력 비교가 아니다.

| Family | gp_orth 대 gp_plus 평균 AUC 변화 |
|---|---:|
| 선형 | -0.03015% |
| 비선형 count + 선형 composition | -0.00132% |
| pair interaction + size 곡선 | +0.08541% |
| pair interaction + 약한 명시적 추세 | +0.12154% |
| corrected physical tomography | +0.03635% |

Equal-family 개선은 **+0.04237%**, 사전 기준1%에 미달했다. 3/5 family 양성,
family별 -2% 하한, 비용0.997배는 통과했지만 모든 조건을 충족해야 하므로
**primary 실패**다. Projection kernel의 확대·독립 확인을 진행하지 않는다.

주 비교의 실제 count는1,600개 paired proposal 모두 같았다. 선택된 catalog
index도1,068/1,600에서 같고20개 target 중6개는80개 action 전체가 같았다.
초기 ranking과 catalog composition regret에도 projection의 일관된 개선이 없다.

## 보존할 보조 결과

비선형 구성 모델의 작은 이득은 존재한다. 기존 quadratic GP는 SSPd3 대비
interaction_size +0.16955%, interaction_smallsize +0.20707%, tomography +0.12387%.
반면 선형 -0.26827%, count 곡선만 비선형인 family -0.15308%다. 이 값은
이번 공통 catalog 안의 개발 관찰로, 기존 full-set SSPd3+spread 확인의
+1.31%나 +3.66%와 직접 비교할 수 없다.

기본 GP에 count kernel을 더하면 초기 prediction과 일부 같은-count ranking이
개선된다. 하지만 그 count kernel에서 A를 제거하는 추가 조작의 실용적 이득은
현재 확인되지 않는다. 단순 baseline을 넘지 못한 작은 양성 관찰을 새 방법의
성공으로 바꾸지 않는다.

## 신규성 판정

정확한 Hamming count projection은 구현·수학적으로 타당하지만 group-invariant
kernel, zero-mean RKHS, orthogonal GP, invariant BO 선행이 있다. 기존 원고의
matrix-retention과 고차 size basis도 이미 존재한다. 이번 구현 자체나 새 이름만으로
방법론 신규성이 확보되지 않는다. NOVELTY_STATUS_KO.md와 PRIOR_ART_KO.md 참조.

현재 강한 근거는 기존 size-prior 메커니즘 분석과 고정 후보의 독립 확인이다.
이번 primary를 뒤집거나 이 bank에서 kernel/lengthscale/noise를 재튜닝하지 않는다.
이전 negative cycle, confirmation40, 원래 operator 결함과 budgeted ROI의
미완료 물리적 검증은 보존한다.

## 부가 설명 진단

POSTHOC_LEAK_PLAN.md는 실패 결과를 본 뒤 명시한 사후 설명 진단이다. 저장된
초기 GP hyperparameters와 원래 SD를 고정한 채, 알려진 synthetic size-only
평균이 같은-count 예측에 미치는 반응을 분리한다. 새 objective/fit/online run은
없으며, 이 결과로 primary 실패나 메소드 신규성 판정을 바꾸지 않는다.

세 결과 리뷰: RESULT_REVIEW_THEORY.md, RESULT_REVIEW_EMPIRICAL.md,
RESULT_REVIEW_NOVELTY.md. 분석 작성자의 역할과 별도 독립 집계를 구분해 기록했다.

사후 진단도 완료했다(POSTHOC_LEAK_RESPONSE.json). 16 synthetic target x5 GP의
저장된 초기 posterior80개를 재구성했고 추가 optimizer/objective/online 호출은0.
m_y=m_g+m_h 분해 오차 최대3.00e-14, saved 초기 예측 RMSE 재현 오차1.98e-16.
초기20개를 제외한 같은-count catalog에서, size-response RMS의 count/target/family
계층평균은 GP base .002866, GP plus .001043, GP split .000666, GP orth .001283,
quadratic .001239였다. Oracle composition에 그 response만 가한 controlled
loss는 각각 .000982/.000169/.000107/.000245/.000239였다.

따라서 비선형 GP에도 finite-archive size-response가 같은-count 구성 순위를
교란할 수 있음은 확인했다. 그러나 orth는 fitted plus보다 두 지표 모두 나빠
projection의 우월성을 지지하지 않는다. 각 모델의 fitted hyperparameters가
달라 'A 제거 자체가 원인'이라고도 주장하지 않는다. 이 oracle perturbation
손실은 실제 online AUC 손실의 인과 분해가 아니며 새 일반 정리가 아니다.
Root는 producer와 정규화·동일-count 손실식을 검토했고, 실험 reviewer는 입력
hash·80records·degenerate 처리·집계를 별도 감사했다. Scientific reviewer도
이 범위에서 진단에 동의했다. 원래 primary 실패와 novelty 판정은 유지한다.
