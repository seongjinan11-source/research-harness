# FP04: End-to-end observed-cell PFN

감사 PASS; 사전 복합 gate FAIL. 새9 fits/3 seeds, query CE만 사용, 실제 그래프 업데이트0회.

## 합성 primary: 자기 피처가 없는 query, 95/99.5% whole-node missing

| Arm | Accuracy % | BAcc % | NLL | Brier |
|---|---:|---:|---:|---:|
| observed | 48.253 | 46.999 | 1.02142 | 0.61350 |
| mask | 47.830 | 46.542 | 1.02359 | 0.61510 |
| pcfi | 47.830 | 46.549 | 1.02359 | 0.61511 |

| 비교 | Accuracy 차이 %p | 사전97.5% CI %p | Graph 수 | Gate |
|---|---:|---:|---:|---|
| observed_minus_mask | +0.423 | [-0.293, +1.134] | 96 | FAIL |
| observed_minus_pcfi | +0.423 | [-0.271, +1.118] | 96 | FAIL |
| observed_minus_none | +0.043 | [-0.629, +0.629] | 48 | FAIL |
| observed_minus_shuffle | -0.326 | [-0.825, +0.087] | 48 | FAIL |

Primary는96개 base graph에서 고정3 train seeds와2 rates를 먼저 평균했다. Utility는 disjoint48개 예정 graph 중 source>=2인 48 graphs/68 cases의 별도 조건부 비교다.

## 실제 그래프 개발 평가

각 셀은3 training seeds×3 고정 context/mask 반복 평균이다. Cora/CiteSeer 전체, PubMed 기존2000-node induced graph. 모두 이전에 검토한 데이터이며 독립 새 graph 일반화 CI가 아니다.

| Graph | Mode | Observed acc % | Mask acc % | PCFI acc % |
|---|---|---:|---:|
| Cora | 95 | 52.876 | 47.819 | 47.924 |
| Cora | 99 | 52.818 | 48.217 | 48.275 |
| Cora | 995 | 52.924 | 48.486 | 48.544 |
| Cora | none | 52.923 | 48.999 | 49.023 |
| Cora | shuffle95 | 52.922 | 47.819 | 47.913 |
| CiteSeer | 95 | 29.977 | 25.840 | 25.828 |
| CiteSeer | 99 | 29.590 | 26.109 | 26.144 |
| CiteSeer | 995 | 29.649 | 26.179 | 26.296 |
| CiteSeer | none | 30.119 | 26.566 | 26.601 |
| CiteSeer | shuffle95 | 30.001 | 25.828 | 25.828 |
| PubMed | 95 | 64.002 | 55.331 | 55.097 |
| PubMed | 99 | 64.023 | 55.426 | 55.258 |
| PubMed | 995 | 64.125 | 55.536 | 55.529 |
| PubMed | none | 64.264 | 56.181 | 56.217 |
| PubMed | shuffle95 | 64.023 | 55.331 | 55.221 |

## 외부 강한 기준선: 정확히 같은 context/query/mask의3반복 평균

| Graph | Mode | Observed % | Public PFN % | PCFI+Public PFN % | FISF+Public PFN % | Restart LP % | Harmonic LP % |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | 95 | 52.876 | 17.023 | 54.796 | 54.347 | 59.056 | 38.246 |
| Cora | 99 | 52.818 | 28.692 | 52.219 | 55.195 | 59.056 | 38.246 |
| Cora | 995 | 52.924 | 21.294 | 51.678 | 56.902 | 59.056 | 38.246 |
| Cora | none | 52.923 | 37.751 | 37.751 | 59.995 | 59.056 | 38.246 |
| Cora | shuffle95 | 52.922 | 20.062 | 54.008 | 57.189 | 59.056 | 38.246 |
| CiteSeer | 95 | 29.977 | 14.614 | 34.078 | 34.189 | 38.410 | 34.228 |
| CiteSeer | 99 | 29.590 | 26.683 | 32.636 | 34.602 | 38.410 | 34.228 |
| CiteSeer | 995 | 29.649 | 16.092 | 39.074 | 29.498 | 38.410 | 34.228 |
| CiteSeer | none | 30.119 | 25.493 | 25.493 | 31.922 | 38.410 | 34.228 |
| CiteSeer | shuffle95 | 30.001 | 12.863 | 35.454 | 35.304 | 38.410 | 34.228 |
| PubMed | 95 | 64.002 | 40.629 | 55.888 | 56.091 | 73.690 | 51.248 |
| PubMed | 99 | 64.023 | 40.629 | 62.321 | 55.262 | 73.690 | 51.248 |
| PubMed | 995 | 64.125 | 40.629 | 64.284 | 60.264 | 73.690 | 51.248 |
| PubMed | none | 64.264 | 40.629 | 40.629 | 56.091 | 73.690 | 51.248 |
| PubMed | shuffle95 | 64.023 | 40.629 | 54.220 | 53.984 | 73.690 | 51.248 |

외부 모델과 새 모델은 물리적 관측값·graph/context/query/truth가 같지만 전처리·구조·학습 예산은 다르다. 새 모델은 native 전체 column/RMS, public 모델은 원래 preprocessing/PCA/ensemble을 사용한다. 같은 예산의 세 arm 대조와 외부 pretrained 기준선의 비교를 혼합해 인과 해석하지 않는다.

관측값 이용은 none/shuffle 비교로 별도 판정하며, 단순히 피처 없이 유지되는 성능을 missing-feature inference 성공으로 부르지 않는다. Source0/1과 실패한 family/shape/real 데이터도 제거하지 않는다.
