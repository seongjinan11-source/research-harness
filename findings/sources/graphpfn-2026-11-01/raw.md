# Supplementary raw-coordinate PCFI/FISF at 99.5% missingness

Added after partial ACM primary-grid results were visible, before any raw-coordinate predictions. Same fixed algorithms, masks, context draws, typed paths and GraphPFN. Raw imputation precedes SVD32; the primary grid projects observed inputs before imputation. This compares complete pipelines and is not a new method or an independently held-out confirmation.

| Dataset | Labels/class | Method | Raw accuracy % | Raw minus reduced pp | Raw minus full pp |
|---|---:|---|---:|---:|---:|---:|
| ACM | 5 | raw_fisf_paths | 64.45 | +2.22 | -6.13 |
| ACM | 20 | raw_fisf_paths | 74.36 | -1.08 | -11.76 |
| ACM | 5 | raw_pcfi_paths | 58.52 | -3.55 | -12.06 |
| ACM | 20 | raw_pcfi_paths | 75.45 | +0.84 | -10.68 |
| IMDB | 5 | raw_fisf_paths | 32.66 | -0.39 | -0.72 |
| IMDB | 20 | raw_fisf_paths | 44.55 | +2.44 | -1.66 |
| IMDB | 5 | raw_pcfi_paths | 32.54 | -0.33 | -0.84 |
| IMDB | 20 | raw_pcfi_paths | 42.61 | -0.37 | -3.59 |

Full metric means and context SDs: raw_results/summary.csv. FISF retains its original all-node random-anchor behavior and may overwrite observed cells; counts and maximum changes are retained in each prepared JSON. These are raw-coordinate imputation algorithms followed by this experiment's SVD, typed-path feature construction and frozen PFN, not the original papers' downstream classifiers.

All 24 supplementary predictions and inputs pass independent metric/context/mask/source audits. No target gradients or test-driven path/algorithm selection.
