# Secondary PubMed check: whole-node missing features

The PCFI/FISF input-view method was selected on Cora/CiteSeer whole-node missingness and fixed before this check. This is a secondary hypothesis, not the original all-condition primary endpoint. No PubMed-specific tuning or training.

| Missing nodes | PCFI | FISF | PCFI/FISF input views | Fixed prediction average |
|---|---:|---:|---:|---:|
| 90% | 70.60 +/- 3.20 | 72.23 +/- 0.35 | 72.83 +/- 2.02 | 72.37 +/- 1.26 |
| 99.5% | 69.50 +/- 1.75 | 70.77 +/- 3.71 | 73.17 +/- 2.28 | 71.57 +/- 2.31 |

Mean accuracy over both rates: pcfi 70.05%, fisf 71.50%, pcfi_fisf_views 73.00%, prediction_average 71.97%.

Input-view difference: +2.95 pp vs PCFI, +1.50 pp vs FISF, +1.03 pp vs prediction averaging.

The two baselines and input-view candidate each use one 10-member frozen GraphPFN with 64 input columns. Prediction averaging requires both baseline predictors (20 forwards). All six masks and public context labels are identical across methods. Original features are masked before imputation/PCA.

Accuracy SD is across three masks of the same graph/split, not a graph-level CI. Balanced accuracy and macro-F1 are retained in summary.csv. The single full-feature reference is retained in metrics.jsonl.

All 25 probability outputs passed finite/mask/metric/source/checkpoint audits. See results/pubmed/audit.json and FRESH_GRAPH_PROTOCOL.md.
