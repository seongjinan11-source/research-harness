# SPFN01 결과 검토: 추론·수학·구현

2026-11-17. **실행·저장 감사 PASS, 사전 composite gate FAIL, 과학 실행의 재실행 불필요.** 같은 토큰 예산에서 tail 압축은 top 선택보다 합성 tail balanced accuracy가 높았고 full 대비 전체 balanced accuracy guard도 통과했다. 그러나 tail quota가 proportional quota보다 우월하다는 근거는 부족했다. 구현 오류로 이 실패를 설명할 증거는 없었다. 아래의 검사·프로파일 한계는 주장 범위를 좁히며, 잠긴 점수를 고쳐야 할 근거는 아니다.

사전 수학 검토에 참여했지만 이번 model/run/data/checks 구현 및 실행에는 참여하지 않았다. 잠긴 코드·checks·audit·training·manifest·summary를 검토하고, production 모델이나 분석 모듈을 import하지 않은 별도 NumPy 식으로 저장 결과를 재채점했다. GPU, 모델 forward, 새 fit, 기존 파일 변경은 하지 않았다.

## 저장 산출물과 사전 결론

잠긴 7개 소스와 3개 raw Planetoid SHA, 12 checkpoint 파일 SHA를 대조했다. Training 320 episode IDs, seed별 320회 공통 permutation, 동일 seed의 공통 초기화 digest, 3,840개 loss/grad-norm 로그의 유한성을 확인했다. Checkpoint 내부 state/arm/seed/lock과 restore 검사는 저장된 production audit 범위다. 이번 독립 재채점에서 모델을 복원하거나 optimizer trajectory를 재실행하지는 않았다.

Final 합성 96개 및 real 15개 NPZ의 파일 SHA·query/context 분리·관측 class count·query class support·학습 입력 fingerprint와 final의 비중복을 확인했다. 각 NPZ의 16개 확률 배열, 총 1,776개를 confusion matrix와 별도 NLL/Brier 식으로 재계산했다. Summary와 최대 차이 `8.88e-16`이었다. 동일한 8개 strata의 base graph 단위, 그래프 안 고정 3 seeds 평균, bootstrap 5,000회로 세 CI를 절대 오차 `1e-14` 안에서 재현했다. 이 검사는 저장 확률의 통계적 정합성을 확인했으며 1,776회 모델 forward의 독립 복제를 뜻하지 않는다.

| 사전 비교 | 차이 %p | nominal 97.5% CI %p | 판정 |
|---|---:|---:|---|
| Tail−proportional, tail bAcc | +0.6583 | [−0.3400,+1.6565] | FAIL |
| Tail−top, tail bAcc | +2.6114 | [+1.1826,+4.0656] | PASS |
| Tail−full, 전체 bAcc | +0.0231 | [−0.6160,+0.6775] | guard PASS |

Holdout 토큰 비율 6.136×와 synthetic 규칙상 floor도 통과했다. Tail−proportional의 CI가 0을 포함하므로 양 효과가 없음을 확인했다는 뜻도, 제안 quota의 우월성을 확인했다는 뜻도 아니다. Top은 같은 class별 quota와 원 count, true singleton metadata, PPR·cosine rank fusion을 받았지만 선택하지 않은 anchors의 모멘트를 보존하지 않는다. 따라서 양의 contrast는 **이 decoder와 학습 절차에서 같은 예측 토큰 수의 요약 대 이 선택기** 비교 결과다. 이를 arbitrary selector에 대한 정리나 quota 단독 효과로 해석하지 않는다. 전체 bAcc guard도 tail recall 동등성, calibrated posterior 또는 real 동등성을 입증하지 않는다. Reused real 3개 데이터셋 모두 proportional의 tail bAcc 점추정치가 tail보다 높았고, PPR 전체 bAcc가 tail보다 높았다. 이들은 데이터셋별 5개 paired draws의 기술 비교다.

## 보존 수식과 수치 구현

- `transition`의 CSR 최대 대칭화, sparse identity 추가 후 nonzero를 1로 채우는 방식은 중복 edge/기존 loop를 simple adjacency로 접고 정점당 self-loop를 한 개 유지한다. Row-normalized P에서 isolates는 Pii=1이다. `diffuse`는 정확히 `D8=.15 Σ(k=0..7) .85^k P^k + .85^8 P^8`의 적용이다. Tiny check는 차수가 다른 무향 그래프에서 P의 비대칭과 row sum을 확인하고, FP64 polynomial과 직접 대조한다. 실제 FP32에서 `D8 B_g`와 `D8(B_g H)`에 해당하는 질량과 모멘트를 열별로 계산하고, singleton 질량 합 및 dense local centroid와 비교했다. **가중 질량·일차 모멘트의 선형 합산**이 부동소수점 허용 오차까지 보존된다는 뜻이다. 비선형 cosine, distance, softmax readout 또는 within-group 분산은 그 항등식의 보존 대상이 아니다.
- `moments`는 singleton에 feature-seed 열을 만들지 않는다. Singleton i는 도달할 때 `m H_i/m=H_i`, 미도달할 때 global fallback도 `H_i`다. Full/top은 같은 8-round singleton mass-only를 계산해 H를 gather한다. Tiny all-singleton tail/full pair pack의 exact parity와 zero-reach finite prediction을 확인했다. Pooled zero mass에는 안전한 분모와 미리 채운 global centroid를 쓰며, reached flag가 token mask를 만들지 않는다. Head의 `score+log(size)` class softmax에서 detached max는 수치 안정화 공통 이동일 뿐이고, 모든 class에 token이 하나 이상 있으므로 `den.clamp_min`이 정상 유한 입력에서 의도한 정규화를 바꾸지 않는다.
- 코드의 proportional 규칙은 PLAN_v2에 고정한 `n_c/allocated_slots`이며, 사전 리뷰에서 예시로 제시했던 `n_c/(allocated_slots+1)`과는 다른 허용 규칙이다. 모든 평가 row에서 quota 합·capacity, tail/top의 class별 quota 동일성, full의 실제 |C|, seed 열 개수를 대조했다. SVD는 정렬된 물리 node IDs의 H를 중심화하고 부호·근퇴화·projection tie를 정해 둔다. Class label rename이 quota/retrieval tie를 결정하지 않는다. 저장 checks의 context 순서·class rename·query 순서/chunk를 동시에 바꾼 오차는 최대 `2.98e-8`이었다. 물리 ID까지 다시 매기는 그래프 순열이나 모든 수치 동률에서 무조건 동일하다는 검사는 아니다.

## 누출과 gradient 검증의 정확한 범위

모델 `Input`은 `x,edge,context,yc,query`뿐이다. 그룹·quota·rank·diffusion과 모든 학습 head의 평가에서는 y_Q, 실제 class population, synthetic role을 받지 않는다. Scoring의 K는 관측 count 벡터 길이로부터 얻고, y_Q는 synthetic training CE, 저장/metric 계산에만 쓰인다. Real은 고정 raw hashes, 공식 train context/공식 test query, 고정 RP와 완전 관측 X·G를 공통으로 사용하며 target optimizer가 없다. Episode builder에서 class-balanced C/Q를 뽑기 위해 y를 사용하는 것은 계획된 표집 조건이다. 전체 graph feature 표준화와 전파는 이 transductive 완전 관측 문제의 허용 입력이며 inductive/no-query-feature 조건을 구현한 것이 아니다.

`checks.py`의 `poisoned=labelperm[y]`는 forward에 연결되지 않고 길이만 검사한다. 따라서 이 줄 자체는 hidden label을 실제 교체한 행동적 누출 검사가 아니다. 이번 결론은 허용 입력 계약과 실제 `prepare/scores` 호출 경로, 분리된 NPZ identities에 근거한다. 전역 label cache나 query truth에서 예측으로 가는 경로는 발견하지 못했다. 저장 checks는 balanced CE의 전체 episode class별 식과 head parameter gradient의 일치, 유한 gradient, arm별 discarded AdamW step, inference chunk/order parity를 지원한다. 훈련은 full episode 한 batch이므로 chunk마다 class balance를 다시 계산하는 문제는 없다. NumPy/SciPy preprocessing과 PCA/선택은 고정이며 이를 관통하는 end-to-end 미분을 검사하거나 주장하지 않는다. 간단한 재실행이 필요한 과학 오류가 아니라 향후 checks 문구에서 정확히 한정할 사항이다.

## 토큰·작업공간·실측 시간은 별개다

Holdout 48개 그래프에서 계산한 평균이다. `seed MiB`는 가장 큰 **seed 한 배열**의 크기, `pair MiB`는 실제 MLP에 넘기기 전 `Q×G×16` pair pack 크기다. 이 수치를 resident peak RSS나 전체 임시 배열 크기로 해석하면 안 된다.

| Arm | 예측 토큰/쿼리 | 전체 diffusion seed 열/episode | 최대 seed 블록 MiB | Pair pack MiB |
|---|---:|---:|---:|---:|
| Tail | 9.646 | 246.646 | 0.3827 | 0.0205 |
| Proportional | 9.646 | 668.146 | 0.6329 | 0.0205 |
| Top | 9.646 | 59.188 | 0.0089 | 0.0205 |
| Full | 59.188 | 59.188 | 0.0089 | 0.1264 |

여기서 압축 arm의 정확한 diffusion 열 수는 `G + (3F)×pooled_groups`, full/top은 |C|다. CSR의 연산량은 8회×`(2E+N)`×실제 열 수 규모다. 고정 group block은 N×열 작업공간을 줄이지만 `moments`는 Q×G×3F local/broadcast intermediate를 만들고, top은 검색 중 Q×|C| mass/rank 및 feature cosine도 계산한다. Scoring만 Q를 64개로 나누므로 전체 전처리 메모리를 그 크기로 제한하지 않는다. 전체 peak 메모리는 기록되지 않았다. `edges` 로그는 `p.nnz-N`, 즉 self-loop 제외 뒤의 양방향 CSR 항목 수다.

고정 real draw0의 preproc→scoring 단일 CPU 프로파일은 같은 thread limit와 1회 warm-up 뒤 3회 반복의 median [min,max]이다. 입력은 이미 로드·RP된 배열이다. Native read/convert/RP를 포함한 총 latency가 아니며, RP wall time은 별도 1회 기록으로 남았다.

| Graph | Tail ms | Full ms | 별도 공통 RP ms |
|---|---:|---:|---:|
| Cora | 41.64 [41.48,43.10] | 23.07 [23.02,23.20] | 2.64 |
| CiteSeer | 46.76 [46.18,51.34] | 20.86 [20.81,21.59] | 8.61 |
| PubMed | 112.49 [110.15,114.06] | 32.48 [31.33,32.55] | 7.01 |

토큰 감소와 별개로 세 CPU 프로파일 모두 tail이 최적화된 full보다 느리다. Tail diffusion median 34.07/39.73/97.42 ms가 각 전체 41.64/46.76/112.49 ms의 대부분이다. Top은 full과 동일한 diffusion 열을 사용하지만 Cora/CiteSeer profile의 순위 계산과 feature evidence 검색에 추가 비용이 있다. 6.136×는 설계로 정한 **예측 토큰 수 비율**이며 speedup, peak memory 감소 또는 N과 무관한 추론 시간 주장을 뒷받침하지 않는다.

Prelock의 Windows int64/scatter 및 test assertion 수리는 본 학습 전이었다. 현재 수식·코드·saved results 사이에 영향을 받은 재실행이 필요한 새 결함은 발견하지 못했다. Gate FAIL을 보존하고, 제한된 압축 대비 선택 효과 및 전체 full guard를 구분해 보고하는 것이 타당하다.

| 검토한 artifact | SHA256 |
|---|---|
| `results/lock.json` | `18e8aba6244630f5257fe2f92e44250276661d19e08d41ebb8c98efdf3796ea1` |
| `results/checks.json` | `86f5cd1477236a4ac1acbc373b8f29438b4635d02ca11eb0cc754c2583d3cc3b` |
| `results/audit.json` | `405ec4ef6f83d4352b31cc34a947f9773e060aac33d65e26daf9cfcaf78c1c4f` |
| `results/training.json` | `b95d420361d13a50e06d9f6af9cae738d9897bc862a464324040142faa407bdb` |
| `results/artifact_manifest.json` | `07afee7451a733b29e7a051926ba1676a7471bac51a85bfeb2ca8a5ecef5c9d2` |
| `results/summary.json` | `921ced6618dec806576dea7fe224ddee92a696aef080bb7a5165a7a2b0be9c55` |
