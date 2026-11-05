# FP05 final-bank input audit

2026-11-05. **입력·저장 식별 감사 PASS.** Root가 고정 design 아래 생성한 final96 raw base tasks와1056 processed NPZ를 전수 CPU 검사했다. Model prediction/result/metric 파일은 읽지 않았고, 새 forward·학습·GPU 작업이나 source/bank/lock/branch 수정은 하지 않았다. 이 문서는 성능 gate나 CI의 PASS를 판정하지 않는다.

검토자는 데이터 구현자다. 별도 NumPy 코드로 hashes·masks·정규화·paired membership과 분모를 재계산했으며 production data helper로 그것을 대신하지 않았다. Data 저자와 독립된 구현 재현이라고 과장하지 않는다. 작은 모델/trace correctness checks 및 실제 예측 통계 감사는 별도 담당/단계다.

## 고정 식별과 provenance

| Artifact | SHA256 |
|---|---|
| final manifest | `415e28c4f0becd89286964012aee763ba92068ad9e1e1386f9052662019dda79` |
| final generator configs | `af8040cb525339ba140dafa12324db7679f70098a9fa2cd155bbacb1e9337e6b` |
| rl_data.py | `2d72eff6ac0a8a1de243fc95511cfa20e13e37b7c33696921bac0a622770dc99` |
| design | `8d790c1a3c962c47c6fc01ae60433bfeaff5302e0607f9fb7483a850d12900fe` |
| full lock | `18b27b3d9a4eb3c8fc321824bf32407ae8f888d27c4e2de4a10b66eba325c575` |

Manifest의 protocol·plan·data·design SHA, 모든1056 episode SHA와96 raw-task SHA, generator-config SHA, 기록된 모든 dependencies를 현재 bytes와 대조했다. Design/full lock의 source/checkpoint/checks SHA 및 full lock의 bank-manifest SHA도 확인했다. **Final의 모든 preparation dependencies는 current와 일치**한다. Train/dev/real의 이전 rl_util bytes 미보존 한계는 `IMPLEMENTATION_DATA_AUDIT.md`/`PRELOCK_SOURCE_DELTA.json`에 별도로 보존되며, final이 일치한다고 그 과거 한계가 사라지는 것은 아니다.

Raw fingerprints는 final96 내부에서 모두 다르고 기존 train/dev264와도 중복되지 않았다. Raw task의 실제 N/F/K, graph, official n_train, context/query IDs 및 query/context labels를 processed rows와 대조했다. Context/query는 중복·교집합이 없고 각각 공식 train/evaluation segment 안에 있다. 원 official prior의 train/eval class-coverage 조건도 accepted raw tasks에서 모두 성립한다. 이는 새 model-score/class-balance acceptance를 추가했다는 뜻이 아니다.

## 실제 크기와 factorial 구성

요청된 shape2×context cap3×regime2×rep8로 **12 strata 각각8 graphs**다. k는 task class 수K가 아니라 context cap1/3/5다. Actual K는 아래처럼 official prior 결과이며 균등화하지 않았다.

| Requested shape | Graphs | Actual N | Actual F | K2 / K3 / K4 | Actual Q | Odd-F graphs |
|---|---:|---:|---:|---:|---:|---:|
| matched | 48 | 112…192 | 8…12 | 34 / 7 / 7 | 전부16 | 0 |
| heldout | 48 | 202…256 | 15…16 | 35 / 8 / 5 | 전부16 | 1 |

Heldout는 requested256/F16의 shape 구분이지 actual N이 모두256이라는 뜻은 아니다. 이번 actual N/F는 학습 범위(max N192/F12) 밖이지만 semantic schema holdout이나 실제 graph population 검증은 아니다. Padding 열은 bank가 추가하지 않았다.

Final의 outer AssertionError retry 합계는87, 성공 호출에서 기록한 task-type rejection 합계는23이다. 원 sampler 내부 class-coverage/feature sanity rejection 총수는 API가 모두 반환하지 않아 여전히 미상이다. 이 수들을 모두 structural-only rejection 또는 모든 rejected draw 수라고 부르지 않는다.

## 전수 mask/control·전처리 검사

각 base는7 standard settings와 node95/node995 각각 none/shuffle의4 control rows, 합계11개다. 동일 base의 모든 row가 실제 graph arrays·context/query IDs·target labels·forward seed를 공유한다. Row ID와 `(base,setting,intervention)`은1056개 모두 유일하다.

- Uniform node masks는 독립 stream6 벡터, cell mask는 stream8 행렬로 재생성해 완전 일치를 확인했다. Cluster는 stream7의 randomized BFS forest를 별도 구현으로 재구성해 rounded-count mask와 완전 일치를 확인했다. Node995⊆node95⊆node90, cluster995⊆cluster95도 각 base에서 성립한다.
- Disjoint는 원 mask 이후 context/query rows를 제거하며 모든 해당 rows에서 실제로 제거돼 있다. Natural full에서는 query를 포함한 모든 physical cells가 실제 관측된다.
- None는 실제 mask와 visible values가 전부0이다. 두 primary rate의 none rows는 values/observed/visible_values/pcfi_raw/forward_seed가 같다. 그러나 원래 standard query subset/source eligibility는 각 rate별로 유지하므로 두 독립 관측 표본으로 다루면 안 된다.
- Shuffle은 동일 observed rows 안에서 고정 stream30+setting-index의 전체 raw 행 순열을 재현했다. Mask/graph/context/query/target는 바뀌지 않으며 row별 표준 eligibility는 유지됐다. Metadata의 actual/standard source, observed cell, context overlap, column-support0/1, own-missing/query-observed flags와 hash도 직접 계산과 일치한다.
- 모든 row의 독립 float64 observed RMS·scale1·clip10 재계산이 visible_values와 정확히 같다. Hidden raw 위치를NaN으로 바꿔도 같다. PCFI raw는 유한하고, 관측 위치는 visible_values와 정확히 같으며 values는 clip(pcfi_raw,−10,10)과 같다. Source가 없는 column은 모두0이다(전체3454 episode-columns). Literal 관측0은 mask로 유지된다.

모든 input-array/graph/context/query-truth/standard-membership hashes를 재계산했다. 이 감사는 전체100-step PCFI kernel을1056번 별도 수식으로 재구현한 검증은 아니다. Pre-PCFI 입력·clamp·empty support·캐시/clip 및 고정 구현 출처를 확인했으며 numerical kernel correctness는 독립 checks의 별도 범위다.

## Primary의 확정된 입력 분모

Uniform node95와 node995에서 원래 physical query row가 완전히 missing인 `Q_gm`을 계산했다. **96 graphs 모두 두 rates가 유효해192 graph-rate cases**이며, empty subset에 의한 undefined case는0이다. Missing-query 수는 node95에서12…16, node995에서15…16이다. 모든 arms/training seeds 및 해당 none/shuffle의 scoring subset은 이 원래 standard membership으로 고정해야 한다.

Primary는 고정3 training seeds와 두 rates를 graph 안에서 먼저 평균한 뒤 graph 단위 paired comparison/bootstrap을 한다. 이 데이터에서 사용할12 strata는 다음 각각8 graphs다.

```text
matched_k1_natural, matched_k1_disjoint
matched_k3_natural, matched_k3_disjoint
matched_k5_natural, matched_k5_disjoint
heldout_k1_natural, heldout_k1_disjoint
heldout_k3_natural, heldout_k3_disjoint
heldout_k5_natural, heldout_k5_disjoint
```

Source0/1은 primary에서 그대로 유지된다. 특히 node995의 **76/96(79.17%)**가 source0/1이다. 이 조건을 보고 rates나 primary denominator를 바꾸지 않는다.

| Standard setting | Source0 | Source1 | Source2+ | Own-missing / observed-query targets |
|---|---:|---:|---:|---:|
| full | 0 | 0 | 96 | 768 / 768 |
| node90 | 0 | 0 | 96 | 1447 / 89 |
| node95 | 0 | 0 | 96 | 1489 / 47 |
| node995 | 38 | 38 | 20 | 1529 / 7 |
| cluster95 | 0 | 0 | 96 | 1492 / 44 |
| cluster995 | 5 | 91 | 0 | 1534 / 2 |
| cell95 | 0 | 0 | 96 | 1160 / 376 |

Cell95의 observed-query는 적어도 한 cell이 관측됐다는 뜻이며 full-observed와 다르다. Cluster/cell은 secondary이므로 이 지원 분포를 보고 primary로 승격하지 않는다.

## Utility의 확정된 조건부 분모

사전 조건인 disjoint, standard source>=2, 적어도1 own-missing query를 적용하면 **48 eligible graphs / 58 rate cases**다. Node95는48 cases, node995는10 cases다. Graph38개는 한 rate만, graph10개는 두 rates가 eligible이다. 따라서58 cases를 pooled 평균하는 것이 아니라, eligible rates와 fixed seeds를 graph 안에서 먼저 평균해48 graphs에 동일 weight를 줘야 한다.

| Utility stratum | Planned disjoint graphs | Eligible graphs | Eligible rate cases |
|---|---:|---:|---:|
| matched_k1_disjoint | 8 | 8 | 9 |
| matched_k3_disjoint | 8 | 8 | 9 |
| matched_k5_disjoint | 8 | 8 | 8 |
| heldout_k1_disjoint | 8 | 8 | 11 |
| heldout_k3_disjoint | 8 | 8 | 12 |
| heldout_k5_disjoint | 8 | 8 | 9 |
| Total | 48 | 48 | 58 |

유효6 strata가 모두 남았다. Utility는 primary96/192와 다른 conditional estimand이며48×3×rates를 독립 graph 수로 늘리면 안 된다. None의 actual source0을 보고 eligibility를 다시 계산하지 말고 반드시 해당 standard_source_nodes를 써야 한다.

## Shuffle의 실제 입력 변화와 full-natural guard 지원

192 shuffle rows 중87개는 순열 후 실제 visible input이 standard와 같다(node95 2개, node995 85개). 이들은 cached represented values까지 같음을 확인했다. Source0/1 외에도 적은 source에서 identity permutation/같은 행이 생길 수 있다. **Utility58 cases 안에서도8 cases(node95 2,node995 6)의 shuffle input이 같다.** 지원이 빈 사례가 아니라 사전에 허용한 permutation의 실제 결과이며, 바뀌도록 다시 뽑거나 제외하지 않는다. 이 사실은 shuffle contrast의 정보량/검정력 한계이지 예측 효과를 확인한 결과가 아니다.

Full-natural guard는 natural48 graphs,6 strata 각각8 graphs이며 **768 query 전부가 실제 own-row fully observed**임을 확인했다. 반대 disjoint48 full rows의 query768개는 모두 own-missing이다. Full-natural guard에 full_nonquery를 섞거나 두 조건을 동일한 완전 관측 성능으로 부르면 안 된다. 이 입력 지원이 확인됐다는 사실만으로 미래 noninferiority CI가 통과한다고 말하지 않는다.

## 감사 종료 범위

Final의 입력·target identity·masks/controls·cached representation과 고정된 primary/utility/full-guard 지원은 이번 입력 감사로 확정됐다. 모델 점수는 사용하지 않았다. 새 source나 bank를 만들지 않았고 원 manifest/data SHA가 끝까지 유지되는 것을 확인했다. 성능 방향·CI·학습 정책의 routing/value 기여·real 개발 결과는 저장 prediction이 생성된 뒤 별도로 검토해야 한다.
