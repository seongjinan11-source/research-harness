# FP05 prior art — full-paper verification

검토일: 2026-11-05. Abstract 확인을 넘어 두 arXiv 본문의 method·실험·결론을 읽었다. Policy-GNN은 reproducibility appendix A/B도 확인했다. 아래는 논문 근거 기록이며 FP05 결과 판정이나 새 방법 계획이 아니다. 실행 중인 locked source·학습·GPU·bank·branch·YAML은 변경하지 않았다.

## Policy-GNN

**서지:** Kwei-Herng Lai, Daochen Zha, Kaixiong Zhou, Xia Hu. *Policy-GNN: Aggregation Optimization for Graph Neural Networks*. KDD ’20, ACM, 2020, pp.461–471. DOI **10.1145/3394486.3403088**. 읽은 버전은 **arXiv:2006.15097v1, 2020-06-26**, 11쪽이다. [출판 DOI](https://doi.org/10.1145/3394486.3403088), [arXiv metadata](https://arxiv.org/abs/2006.15097), [전문](https://arxiv.org/html/2006.15097), [PDF](https://arxiv.org/pdf/2006.15097). Page range는 [DBLP 서지](https://dblp.uni-trier.de/rec/conf/kdd/LaiZZH20.html)로 보완 확인했다.

**본문에서 확인한 계산/전제:** Node attribute가 state, aggregation hop 수가 action인 DQN이다. 다음 node는 선택한 k-hop에서 sampling한다. GCN은 supervised training하고, policy reward는 최근 baseline 대비 **validation accuracy**다. Shared layers/action buffers를 사용한다. 실험은 complete graph/features의 Cora·CiteSeer·PubMed이며, validation 성능으로 checkpoint도 고른다(§§3.1–3.3,4.1; Appendix A.3). 따라서 “validation labels를 전혀 쓰지 않는 policy”로 읽으면 틀리다. [본문 근거](https://arxiv.org/html/2006.15097).

**저자가 직접 명시한 한계 — 1개:** 유효 policy에 수백 training iterations가 필요해 큰 graph에서 학습 효율이 문제가 될 수 있다고 결론에서 인정한다(§6, PDF p.9). [저자 한계](https://arxiv.org/pdf/2006.15097#page=9).

**분석자의 범위 판단 — 최대 2개:**

1. Complete-feature transductive 실험은 extreme missing/source0·1 또는 synthetic-only PFN 전이를 입증하지 않는다(§4.1에서의 추론).
2. RL로 propagation 횟수를 조절하는 원리는 선행이다. 다만 node-attribute DQN과 FP05의 detached hidden-state 3-bit controller는 같은 알고리즘·재현이 아니다(§3에서의 비교).

## RMS

**서지:** Wentao Ning, Reynold Cheng, Jiajun Shen, Nur Al Hasan Haldar, Ben Kao, Xiao Yan, Nan Huo, Wai Kit Lam, Tian Li, Bo Tang. *Automatic Meta-Path Discovery for Effective Graph-Based Recommendation*. CIKM ’22, ACM, 2022, pp.1563–1572. DOI **10.1145/3511808.3557244**. 읽은 버전은 **arXiv:2112.12845v5, 2022-09-07**, 10쪽이다. 저자 순서는 이 버전과 공저자 기관 repository 기준이다. [출판 DOI](https://doi.org/10.1145/3511808.3557244), [arXiv metadata](https://arxiv.org/abs/2112.12845), [전문](https://arxiv.org/html/2112.12845v5), [PDF](https://arxiv.org/pdf/2112.12845), [UWA 공식 서지](https://research-repository.uwa.edu.au/en/publications/automatic-meta-path-discovery-for-effective-graph-based-recommend/).

**본문에서 확인한 계산/전제:** 알려진 HIN relation vocabulary의 count vector를 합산·L2-normalize해 state를 만들고, DQN이 relation 또는 STOP을 고른다. Recommender 형식에 맞춰 symmetric extension하고 기존 paths를 보존한다. Reward는 짧게 학습한 recommender의 validation NDCG@10 증가다. HRec은 meta-path subgraphs와 두 attention 단계, BPR을 사용한다(§§3–4). [본문 근거](https://arxiv.org/html/2112.12845v5).

**저자가 직접 명시한 한계 — 최대 2개:**

1. 짧은 학습 성능과 수렴 성능의 상관이 불완전해 accuracy penalty가 있고 최적 path 선택을 보장하지 못한다(§3.3; §5.2).
2. Meta-path 선택의 gain은 downstream recommender에 의존하며 MCRec처럼 path에 덜 민감하면 개선도 작다고 인정한다(§5.2). [저자 논의](https://arxiv.org/html/2112.12845v5).

**분석자의 범위 판단 — 최대 2개:**

1. Count-state는 relation 순서가 다른 path sets를 구분하지 못할 수 있다. Markov-sufficient encoding 보장이 있다는 뜻은 아니다(§3.2 식에서의 추론).
2. Yelp/Douban/TCL recommendation 평가는 missing-feature node classification이나 RL-backbone PFN 전이의 근거가 아니다(§5.1에서의 판단).

## 최종 보고서에 적용할 novelty 경계

**RL 기반 aggregation 횟수 조절과 typed meta-path 탐색 자체는 기존 원리다.** FP05는 pretrained PFN 내부 연산 선택의 제한된 응용 비교로 구분해야 하며, 두 논문이 feature utility·heterogeneous schema transfer·PFN backbone의 policy-gradient 학습을 대신 검증해 주지는 않는다. 이 문서는 FP05 성능 결과를 읽거나 성공을 주장하지 않는다.

접근 기록: DOI endpoint 직접 열기는 도구에서 실패했으므로 method/한계 근거는 공개 arXiv 전문을 사용했다. 두 논문이 기재한 code URL([Policy-GNN](https://github.com/datamllab/Policy-GNN), [RMS-HRec](https://github.com/Stevenn9981/RMS-HRec))의 직접 열기도 이 도구에서 404를 반환했다. 이는 현재 코드 내용을 검증·재현했다는 뜻이 아니며, 논문 본문의 검증 상태와 분리한다. 공식 서지/원문 URL만 knowledge record의 근거로 남긴다.

Harness 추가 확인: canonical 중복 검색에서 기존 일치가 없어 신규 IDs를 `PolicyGNN`, `RMS`로 정했다. [KDD 공식 proceedings](https://www.kdd.org/kdd2020/proceedings/index.html)에서 Policy-GNN의 게재·원문 제목·저자 순서를 추가 확인해 inproceedings로 기록했다. Page range는 앞서 DBLP로 확인했지만 citation의 verified_from을 공식 원문/학회로 제한하며 pages를 생략했다. RMS는 ACM DL 403/학회 사이트 미접근으로 CLAUDE의 엄격한 확인 규칙을 충족하지 못해 citation만 보수적으로 misc/arXiv를 유지했다. 이는 전문 검토·공저자 기관의 CIKM 출판 서지 확인과 별개다. 기존 paper/findings는 수정하지 않았고 revisit/validate/build/git은 root 담당으로 남겼다.
