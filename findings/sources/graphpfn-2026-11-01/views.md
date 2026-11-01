# Zero-shot missing-feature GraphPFN: input extension results

All 144 candidate/control predictions and four parent-parity checks completed. No GraphPFN weights were updated. All inputs use the same context labels as the parent PCFI/FISF baselines.

## Global comparison

Accuracy, balanced accuracy and macro-F1 (%), equal weight across the eight dataset/mask/rate conditions and three mask draws. This is development evidence on two fixed graph/public-split pairs.

| Method | Accuracy | Balanced acc | Macro-F1 | Delta accuracy vs PCFI |
|---|---:|---:|---:|---:|
| PCFI + GraphPFN (parent) | 69.65 | 68.18 | 66.61 | +0.00 |
| FISF + GraphPFN (parent) | 68.43 | 68.22 | 66.12 | -1.21 |
| PCFI + support information | 69.64 | 68.11 | 66.53 | -0.00 |
| FISF + support information | 68.80 | 68.32 | 66.28 | -0.85 |
| Short/medium/PCFI views | 69.61 | 68.21 | 66.47 | -0.04 |
| PCFI/FISF views | 69.73 | 69.12 | 67.13 | +0.09 |
| PCFI values, matched width | 69.59 | 68.14 | 66.58 | -0.06 |
| FISF values, matched width | 68.64 | 68.46 | 66.29 | -1.01 |

## All conditions: test accuracy (%)

| Dataset | Mask | Missing | PCFI baseline | FISF baseline | PCFI + info | FISF + info | Multiscale | PCFI/FISF views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | structural | 90% | 72.42 | 71.77 | 72.53 | 71.80 | 72.60 | 72.90 |
| Cora | structural | 99.5% | 73.08 | 73.62 | 72.53 | 74.06 | 73.08 | 73.95 |
| Cora | uniform | 90% | 76.21 | 74.21 | 75.63 | 74.54 | 73.99 | 75.48 |
| Cora | uniform | 99.5% | 74.94 | 73.48 | 73.01 | 72.90 | 73.70 | 73.95 |
| CiteSeer | structural | 90% | 63.95 | 62.29 | 63.65 | 62.29 | 66.21 | 65.01 |
| CiteSeer | structural | 99.5% | 62.80 | 63.90 | 63.90 | 64.15 | 64.00 | 64.35 |
| CiteSeer | uniform | 90% | 68.88 | 65.11 | 68.33 | 64.96 | 67.72 | 66.67 |
| CiteSeer | uniform | 99.5% | 64.91 | 63.10 | 67.57 | 65.71 | 65.56 | 65.56 |

## Metadata versus equal-width value controls

| Candidate | Mean accuracy difference (pp) | Per-condition differences (Cora structural .9/.995, uniform .9/.995; then CiteSeer) |
|---|---:|---|
| PCFI + support information | +0.05 | +0.07, +0.26, -0.44, -1.42, -0.40, +0.40, +0.15, +1.81 |
| FISF + support information | +0.17 | +0.15, +0.11, +0.55, -0.84, +0.25, +0.45, -0.90, +1.56 |

## Fixed decision

Fresh-graph confirmation criterion, fixed before these results: >=1 pp mean accuracy gain over BOTH parent baselines, and metadata must also beat its corresponding equal-width control. No method switching by graph or condition.

Eligible candidates: none.

## Construction and checks

- PCFI/FISF value blocks preserve the original quantile-transform -> PCA64 preprocessing. Masks and original-source pseudo-confidence are separately projected (uniform) or appended as scalar summaries (structural). FISF additionally receives the per-node synthetic-anchor fraction.
- The multiscale input contains confidence-weighted propagation at 5/20 steps (16 components each) and PCFI100 (32 components). PCFI/FISF views use 32 components each. Both remain within 64 input columns.
- Metadata candidates have additional columns; value-only controls match each actual column count exactly. All predictors use the same ten-member frozen GraphPFN and eight random features.
- These support summaries are low-dimensional inputs, not a native cell-wise provenance encoder or calibrated predictive uncertainty.
- Projection-cache equivalence tests passed exactly. Four full predictor parity checks match the saved parent predictions within 2e-5.
- Independent recomputation of all classification metrics, saved-feature finiteness, complete grid, paired-mask hashes, source fingerprints and unchanged checkpoint fingerprints passed.
- No target training, validation-based per-graph choice, label-driven imputation or hidden-feature access. PCA and graph propagation use observed/transductive features only.
- Cora/CiteSeer and the same public splits/masks were already inspected in the parent experiment. These results are development evidence; three mask draws do not provide independent-graph uncertainty.
- A negative frozen-input result does not establish that synthetic pretraining on these inputs cannot improve performance.

Artifacts: results/dev/summary.csv, overall.csv, paired.csv, audit.json, decision.json, metrics.jsonl, predictions/ and features/.


## Secondary fixed probability average

Added during the primary run, before computing any fusion result; parent and some extension development results were already visible. One globally fixed .5 PCFI + .5 FISF probability average, with no fitted weights or condition-wise selection. See ENSEMBLE_ADDENDUM.md.

Mean accuracy: 70.16%; mean difference versus PCFI +0.52 pp, versus FISF +1.73 pp.

Deployment requires both 10-member predictors (20 forwards total), so this is not a compute-matched comparison to the single-predictor extensions. It is development evidence and another useful baseline, not an independent final validation.

## Secondary PubMed confirmation

Whole-node missingness was narrowed as a secondary hypothesis after development inspection. The PCFI/FISF input-view method and new masks were fixed before PubMed evaluation; no PubMed-specific training or selection. This does not change the original all-condition primary decision. See FRESH_GRAPH_PROTOCOL.md and PUBMED_REPORT.md.

Mean PubMed accuracy: pcfi 70.05%, fisf 71.50%, pcfi_fisf_views 73.00%, prediction_average 71.97%.

Input-view difference versus PCFI +2.95 pp; versus FISF +1.50 pp; versus fixed prediction averaging +1.03 pp.