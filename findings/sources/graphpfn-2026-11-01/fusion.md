# Fixed fusion and matched-node error overlap

Posthoc development check motivated by observed cross-dataset strengths. No new inference, training, temperatures or dataset-specific weights. PCFI/FISF retain raw-coordinate imputation followed by the prior SVD/path/PFN pipeline. LP is the uniform metapath label-propagation ensemble. All use identical context labels and public test nodes.

## Accuracy means (%) at 99.5% missing target rows

| Dataset | Labels/class | PCFI-PFN | FISF-PFN | Uniform LP | PCFI/LP 50:50 | FISF/LP 50:50 | PCFI/FISF/LP equal thirds |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACM | 5 | 58.52 | 64.45 | 80.79 | 81.04 | 81.11 | 80.24 |
| ACM | 20 | 75.45 | 74.36 | 85.55 | 83.65 | 82.20 | 80.74 |
| IMDB | 5 | 32.54 | 32.66 | 32.04 | 33.79 | 33.76 | 33.04 |
| IMDB | 20 | 42.61 | 44.55 | 35.41 | 40.80 | 42.54 | 42.89 |

## Correctness overlap: fraction of the same public test nodes (%)

| Dataset | Labels/class | PFN input | Both correct | PFN only correct | LP only correct | Both wrong |
|---|---:|---|---:|---:|---:|---:|
| ACM | 5 | pcfi | 50.06 | 8.46 | 30.73 | 10.75 |
| ACM | 5 | fisf | 56.43 | 8.03 | 24.37 | 11.18 |
| ACM | 20 | pcfi | 70.39 | 5.05 | 15.16 | 9.39 |
| ACM | 20 | fisf | 69.48 | 4.88 | 16.07 | 9.57 |
| IMDB | 5 | pcfi | 18.42 | 14.12 | 13.62 | 53.84 |
| IMDB | 5 | fisf | 18.55 | 14.11 | 13.48 | 53.85 |
| IMDB | 20 | pcfi | 20.37 | 22.24 | 15.05 | 42.34 |
| IMDB | 20 | fisf | 20.80 | 23.75 | 14.62 | 40.84 |

Cross-dataset strengths alone do not prove within-task complementarity. The overlap table distinguishes that issue. An oracle could exploit either model being correct, but observing those correctness events requires test labels; it is not an achievable predictor. The fusion table reports only predefined equal weights. Means/SDs, balanced accuracy, macro-F1 and NLL are in fusion_results/summary.csv. All 36 outputs pass alignment, equal-weight reconstruction and metric readback checks.
