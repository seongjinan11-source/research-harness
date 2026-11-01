# FISF -> frozen GraphPFN: pilot results (2026-10-30)

All 110 registered predictions and 26 official-reference predictions completed. No model training or hyperparameter tuning.

The substantial gain is from graph propagation. FISF adds a modest benefit under 99.5% whole-node missingness, but does not consistently improve on FP/PCFI across conditions.

## Setup

- Cora LCC: 2,485 nodes, 1,433 raw features, 122 context / 459 validation / 915 test nodes.
- CiteSeer LCC: 2,120 nodes, 3,703 raw features, 80 context / 328 validation / 663 test nodes.
- Public splits restricted to the largest connected component; masks include both context and queries.
- Three missingness seeds (101/102/103), structural and uniform masks, r=.9/.995. Exactly paired across methods.
- Frozen released GraphPFN 1.3; official 10-member ensemble and 8 random features; quantile-normal input and PCA64; Windows fp16.
- Raw features are masked BEFORE any imputation, transform, or PCA. Imputers never receive labels or true missing values.
- Fixed author defaults: 100 iterations; PCFI alpha=.9/beta=1; FISF alpha=.9/beta=.9/gamma=.7, injection seed=0.
- Most-frequent and propagation arms fill BEFORE official transforms. The separate official arm transforms BEFORE filling; see OFFICIAL_BASELINE_ADDENDUM.md.

## Test accuracy (%)

Mean +/- sample SD over three masks of one fixed graph/split. These are not independent-graph confidence intervals. All conditions are shown.

| Dataset | Mask | Missing | Official order | Fill first | FP | PCFI | FISF |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | structural | 90% | 63.75 +/- 0.99 | 64.74 +/- 0.55 | 73.08 +/- 1.21 | 72.42 +/- 3.33 | 71.77 +/- 0.82 |
| Cora | structural | 99.5% | 41.42 +/- 1.33 | 42.33 +/- 2.15 | 73.15 +/- 2.33 | 73.08 +/- 1.78 | 73.62 +/- 1.20 |
| Cora | uniform | 90% | 61.46 +/- 2.36 | 61.42 +/- 0.87 | 74.50 +/- 0.23 | 76.21 +/- 1.21 | 74.21 +/- 0.83 |
| Cora | uniform | 99.5% | 63.50 +/- 0.76 | 62.77 +/- 0.69 | 73.84 +/- 0.17 | 74.94 +/- 1.60 | 73.48 +/- 0.64 |
| CiteSeer | structural | 90% | 50.53 +/- 5.64 | 48.87 +/- 5.23 | 62.95 +/- 0.53 | 63.95 +/- 1.97 | 62.29 +/- 1.38 |
| CiteSeer | structural | 99.5% | 24.43 +/- 10.97 | 22.93 +/- 9.93 | 62.09 +/- 1.82 | 62.80 +/- 0.44 | 63.90 +/- 0.88 |
| CiteSeer | uniform | 90% | 49.77 +/- 0.66 | 49.77 +/- 1.85 | 65.01 +/- 0.54 | 68.88 +/- 1.46 | 65.11 +/- 1.03 |
| CiteSeer | uniform | 99.5% | 45.50 +/- 3.93 | 46.96 +/- 1.13 | 64.61 +/- 0.35 | 64.91 +/- 0.44 | 63.10 +/- 0.23 |

Full-feature references (same predictor/preprocessing): Cora 72.90%, CiteSeer 73.30%.

## FISF minus PCFI, all metrics (percentage points)

| Dataset | Mask | Missing | Accuracy | Balanced accuracy | Macro-F1 | Accuracy wins / masks |
|---|---|---:|---:|---:|---:|---:|
| Cora | structural | 90% | -0.66 | -0.11 | -0.44 | 1/3 |
| Cora | structural | 99.5% | +0.55 | +2.40 | +1.68 | 2/3 |
| Cora | uniform | 90% | -2.00 | +0.38 | -0.63 | 0/3 |
| Cora | uniform | 99.5% | -1.46 | +0.94 | -0.26 | 0/3 |
| CiteSeer | structural | 90% | -1.66 | -1.74 | -1.90 | 0/3 |
| CiteSeer | structural | 99.5% | +1.11 | +2.43 | +2.08 | 3/3 |
| CiteSeer | uniform | 90% | -3.77 | -2.85 | -3.15 | 0/3 |
| CiteSeer | uniform | 99.5% | -1.81 | -1.07 | -1.27 | 0/3 |

At 99.5% whole-node missingness, FISF improves accuracy over PCFI by about 0.55 pp (Cora) and 1.11 pp (CiteSeer); balanced accuracy improves by about 2.40 and 2.43 pp. At 90% missingness, PCFI is better in all four dataset/mask conditions. Uniform 99.5% missingness also favors PCFI in accuracy.

## Imputation-graph shuffle control

Only the imputation graph is shuffled; frozen GraphPFN still receives the true graph. Fixed r=.995, three masks.

| Dataset | Mask | FISF true | FISF shuffled | Difference (pp) |
|---|---|---:|---:|---:|
| Cora | structural | 73.62 | 56.50 | +17.12 |
| Cora | uniform | 73.48 | 58.32 | +15.15 |
| CiteSeer | structural | 63.90 | 48.27 | +15.64 |
| CiteSeer | uniform | 63.10 | 47.41 | +15.69 |

The true imputation topology matters in these conditions. This does not establish a FISF-specific advantage over ordinary propagation.

## Runtime and implementation checks

- CPU optimized FISF/PCFI agrees with the unmodified author code on connected toy graphs; max absolute difference 2.39e-7.
- FISF retains the public code behavior: it selects anchors from all nodes, and can overwrite an observed cell. Every method receives copies of the original mask/values. Overwrite counts and feature variance diagnostics are in diagnostics.csv.
- Additional official baseline prompted by a preprocessing-order audit, after some original pilot results were visible. No parameters or original comparisons changed.
- Independent recomputation of accuracy, balanced accuracy and macro-F1 from saved probabilities passed; all 136 predictions finite, grid complete, masks match, and source/checkpoint hashes verified.
- Fresh-model full-feature references agree within 2e-5.

| Dataset | FP impute seconds | PCFI impute seconds | FISF impute seconds |
|---|---:|---:|---:|
| Cora | 1.03 | 2.71 | 6.13 |
| CiteSeer | 2.13 | 5.78 | 12.94 |

## Interpretation limits and decision

This validates propagation as a useful preprocessing baseline for missing-feature GraphPFN in this recipe. FISF has a narrow positive signal at extreme whole-node missingness; the current evidence does not justify replacing PCFI generally or training a new PFN on the strength of FISF alone.

Only two citation graphs, one public split per graph, three missingness draws and one fixed injection seed were tested. PCA64, feature transforms and Windows fp16 may affect relative performance. Defaults were not tuned. No claim of state-of-the-art performance, statistical significance, unseen-mechanism generalization or native heterogeneous-graph support is made.

Artifacts: summary.csv, paired_differences.csv, contrasts.csv, diagnostics.csv, metrics.jsonl, audit.json; masks and full probability outputs are retained. Original plan: PROTOCOL.md; added reference: OFFICIAL_BASELINE_ADDENDUM.md.
