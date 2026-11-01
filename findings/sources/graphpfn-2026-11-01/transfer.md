# Locked baseline transfer across ACM, IMDB and Freebase

This is a development-stage transfer check on three previously explored datasets. The DBLP context-LOO path selector and all algorithm settings were fixed before these new predictions. New context/mask draws do not make these untouched dataset holdouts. No new heterogeneous PFN training, per-target gradient updates or formal unseen-schema validation.

## Frozen GraphPFN accuracy (%)

| Dataset | Labels/class | Full paths | Structure only | Missing rows | Zero + paths | PCFI + paths | FISF + paths | Featureless context-LOO LP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACM | 5 | 70.58 | 61.48 | 90% | 56.58 | 56.92 | 58.08 | 69.42 |
| ACM | 5 | 70.58 | 61.48 | 99.5% | 59.78 | 62.07 | 62.23 | 69.42 |
| ACM | 20 | 86.12 | 74.53 | 90% | 74.01 | 78.29 | 77.94 | 76.98 |
| ACM | 20 | 86.12 | 74.53 | 99.5% | 72.49 | 74.61 | 75.44 | 76.98 |
| IMDB | 5 | 33.38 | 33.78 | 90% | 32.63 | 33.28 | 33.41 | 27.53 |
| IMDB | 5 | 33.38 | 33.78 | 99.5% | 33.18 | 32.86 | 33.06 | 27.53 |
| IMDB | 20 | 46.20 | 41.55 | 90% | 39.71 | 41.09 | 41.13 | 34.57 |
| IMDB | 20 | 46.20 | 41.55 | 99.5% | 40.41 | 42.98 | 42.10 | 34.57 |
| FREEBASE | 5 | n/a | 54.37 | n/a | n/a | n/a | n/a | 41.80 |
| FREEBASE | 20 | n/a | 56.13 | n/a | n/a | n/a | n/a | 50.80 |

PCFI/FISF are applied to observed-only, post-mask SVD coordinates (up to 32), then fed through the same typed-path feature constructor and unchanged published GraphPFN. These are reduced-coordinate baselines, not raw-coordinate reproductions of the published imputation pipelines. Existing verified CPU propagation code and original FISF all-node anchors are preserved.

## Label-only methods and context-based selection

| Dataset | Labels/class | Method | Accuracy % | Balanced accuracy % | Macro-F1 % | NLL |
|---|---:|---|---:|---:|---:|---:|
| ACM | 5 | lp_P-P | 72.46 | 72.03 | 71.98 | 0.5904 |
| ACM | 20 | lp_P-P | 76.98 | 77.27 | 76.76 | 0.5146 |
| ACM | 5 | lp_PAP | 61.42 | 58.48 | 59.28 | 0.8102 |
| ACM | 20 | lp_PAP | 67.06 | 65.50 | 65.84 | 0.7483 |
| ACM | 5 | lp_PSP | 56.14 | 62.94 | 60.40 | 3.1771 |
| ACM | 20 | lp_PSP | 66.16 | 66.72 | 67.25 | 2.5458 |
| ACM | 5 | lp_context_loo | 69.42 | 71.61 | 70.89 | 0.8277 |
| ACM | 20 | lp_context_loo | 76.98 | 77.27 | 76.76 | 0.5146 |
| ACM | 5 | lp_uniform | 80.79 | 79.55 | 79.90 | 0.6026 |
| ACM | 20 | lp_uniform | 85.55 | 85.16 | 85.29 | 0.5307 |
| ACM | 5 | lp_union | 66.52 | 68.88 | 67.33 | 0.6341 |
| ACM | 20 | lp_union | 73.67 | 72.71 | 72.69 | 0.6081 |
| FREEBASE | 5 | lp_MAM | 48.50 | 53.41 | 47.55 | 1.0805 |
| FREEBASE | 20 | lp_MAM | 50.80 | 54.51 | 50.44 | 0.9403 |
| FREEBASE | 5 | lp_MDM | 38.13 | 33.76 | 19.19 | 1.1150 |
| FREEBASE | 20 | lp_MDM | 39.10 | 35.31 | 22.25 | 1.2806 |
| FREEBASE | 5 | lp_MWM | 41.00 | 37.95 | 27.86 | 1.8381 |
| FREEBASE | 20 | lp_MWM | 42.30 | 39.94 | 31.93 | 1.7728 |
| FREEBASE | 5 | lp_context_loo | 41.80 | 40.93 | 28.80 | 1.1711 |
| FREEBASE | 20 | lp_context_loo | 50.80 | 54.51 | 50.44 | 0.9403 |
| FREEBASE | 5 | lp_uniform | 48.33 | 51.31 | 46.56 | 1.0073 |
| FREEBASE | 20 | lp_uniform | 50.90 | 52.92 | 49.94 | 0.9855 |
| FREEBASE | 5 | lp_union | 48.97 | 53.67 | 48.06 | 1.0737 |
| FREEBASE | 20 | lp_union | 50.50 | 54.50 | 50.19 | 0.9376 |
| IMDB | 5 | lp_MAM | 31.90 | 32.36 | 26.17 | 1.3100 |
| IMDB | 20 | lp_MAM | 34.57 | 36.89 | 33.41 | 1.1464 |
| IMDB | 5 | lp_MDM | 27.53 | 33.47 | 14.82 | 1.4390 |
| IMDB | 20 | lp_MDM | 28.43 | 34.24 | 16.87 | 2.1144 |
| IMDB | 5 | lp_context_loo | 27.53 | 33.47 | 14.82 | 1.4390 |
| IMDB | 20 | lp_context_loo | 34.57 | 36.89 | 33.41 | 1.1464 |
| IMDB | 5 | lp_uniform | 32.04 | 32.52 | 26.44 | 1.1398 |
| IMDB | 20 | lp_uniform | 35.41 | 37.70 | 34.32 | 1.0942 |
| IMDB | 5 | lp_union | 33.61 | 34.10 | 27.97 | 1.2287 |
| IMDB | 20 | lp_union | 36.55 | 38.95 | 35.34 | 1.1146 |

The selector chooses the minimum mean context leave-one-out NLL. Each held-out seed contribution is removed from all diffusion paths. Independent reruns excluding the first seed verify the subtraction. No query/test labels enter inference or path selection. All per-path results are retained. The test-best path below is only a diagnostic reference; it is never used for a prediction.

| Dataset | Labels/class | Seed | Chosen path | Selected accuracy % | Test-best path accuracy % (diagnostic) | Regret pp |
|---|---:|---:|---|---:|---:|---:|
| ACM | 5 | 11 | PSP | 58.99 | 68.10 | 9.10 |
| ACM | 5 | 12 | P-P | 76.48 | 76.48 | 0.00 |
| ACM | 5 | 13 | P-P | 72.79 | 72.79 | 0.00 |
| ACM | 20 | 11 | P-P | 78.50 | 78.50 | 0.00 |
| ACM | 20 | 12 | P-P | 77.42 | 77.42 | 0.00 |
| ACM | 20 | 13 | P-P | 75.02 | 75.02 | 0.00 |
| IMDB | 5 | 11 | MDM | 27.46 | 32.95 | 5.49 |
| IMDB | 5 | 12 | MDM | 27.63 | 30.62 | 2.99 |
| IMDB | 5 | 13 | MDM | 27.49 | 32.12 | 4.63 |
| IMDB | 20 | 11 | MAM | 34.42 | 34.42 | 0.00 |
| IMDB | 20 | 12 | MAM | 35.34 | 35.34 | 0.00 |
| IMDB | 20 | 13 | MAM | 33.96 | 33.96 | 0.00 |
| FREEBASE | 5 | 11 | MDM | 38.00 | 47.70 | 9.70 |
| FREEBASE | 5 | 12 | MAM | 49.30 | 49.30 | 0.00 |
| FREEBASE | 5 | 13 | MDM | 38.10 | 48.50 | 10.40 |
| FREEBASE | 20 | 11 | MAM | 50.80 | 50.80 | 0.00 |
| FREEBASE | 20 | 12 | MAM | 50.80 | 50.80 | 0.00 |
| FREEBASE | 20 | 13 | MAM | 50.80 | 50.80 | 0.00 |

## Observed native source rows

| Dataset | Missing rate | Seed | Observed rows | Projected coordinates |
|---|---:|---:|---:|---:|
| ACM | 0.9 | 11 | 399 | 32 |
| ACM | 0.995 | 11 | 19 | 19 |
| ACM | 0.9 | 12 | 380 | 32 |
| ACM | 0.995 | 12 | 20 | 20 |
| ACM | 0.9 | 13 | 388 | 32 |
| ACM | 0.995 | 13 | 15 | 15 |
| IMDB | 0.9 | 11 | 420 | 32 |
| IMDB | 0.995 | 11 | 19 | 19 |
| IMDB | 0.9 | 12 | 409 | 32 |
| IMDB | 0.995 | 12 | 22 | 22 |
| IMDB | 0.9 | 13 | 421 | 32 |
| IMDB | 0.995 | 13 | 16 | 16 |

## Paired missing-minus-full accuracy changes (pp)

| Dataset | Labels/class | Missing rate | Method | Mean delta pp |
|---|---:|---:|---|---:|
| ACM | 5 | 0.9 | zero_paths | -14.00 |
| ACM | 5 | 0.9 | pcfi_paths | -13.66 |
| ACM | 5 | 0.9 | fisf_paths | -12.50 |
| ACM | 5 | 0.995 | zero_paths | -10.80 |
| ACM | 5 | 0.995 | pcfi_paths | -8.51 |
| ACM | 5 | 0.995 | fisf_paths | -8.35 |
| ACM | 20 | 0.9 | zero_paths | -12.12 |
| ACM | 20 | 0.9 | pcfi_paths | -7.84 |
| ACM | 20 | 0.9 | fisf_paths | -8.18 |
| ACM | 20 | 0.995 | zero_paths | -13.64 |
| ACM | 20 | 0.995 | pcfi_paths | -11.51 |
| ACM | 20 | 0.995 | fisf_paths | -10.69 |
| IMDB | 5 | 0.9 | zero_paths | -0.75 |
| IMDB | 5 | 0.9 | pcfi_paths | -0.11 |
| IMDB | 5 | 0.9 | fisf_paths | +0.03 |
| IMDB | 5 | 0.995 | zero_paths | -0.20 |
| IMDB | 5 | 0.995 | pcfi_paths | -0.52 |
| IMDB | 5 | 0.995 | fisf_paths | -0.33 |
| IMDB | 20 | 0.9 | zero_paths | -6.50 |
| IMDB | 20 | 0.9 | pcfi_paths | -5.12 |
| IMDB | 20 | 0.9 | fisf_paths | -5.07 |
| IMDB | 20 | 0.995 | zero_paths | -5.80 |
| IMDB | 20 | 0.995 | pcfi_paths | -3.22 |
| IMDB | 20 | 0.995 | fisf_paths | -4.10 |

## Scope and provenance

ACM and IMDB load only their native target feature block. Existing helper features derived from complete target data are never loaded. Hidden raw rows are removed before basis fitting; hidden-value poisoning leaves observed-only projected inputs exactly unchanged. The graph and original vocabulary remain observed, so numerical attribute masking does not remove all semantic information encoded in edges. Freebase is naturally featureless and only tests structural/path selection transfer.

All 204 saved probability outputs, actual PFN inputs, source/data/checkpoint fingerprints, contexts, masks and metrics pass audit.py. Full metric means/SDs (including balanced accuracy, macro-F1 and NLL) are in results/summary.csv. Three seeded runs share each graph and public test set. Freebase at 20 labels/class exhausts its public train pool: all three runs have the SAME context indices, so LP repeats exactly and PFN variation reflects inference RNG only. Unique context counts are explicit in summary.csv. Different masks also change the fitted observed-only basis, so paired loss is a complete-pipeline loss, not a pure imputation-error estimate.

Missing-vs-full loss alone is not evidence of recoverable information or a new model contribution. A poor selector relative to test-best paths can motivate a context-selection problem, but the test-best path uses test labels and is not an achievable baseline. Any new prior/training protocol must be fixed and evaluated on independent graph families/schemas.
