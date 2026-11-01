# Expanded context-only LP candidate-pool diagnostic

Posthoc development check registered after the ACM k=5 results were visible. Add uniform and union LP to the original individual-path candidate pool, keeping all propagation and context scoring settings fixed. No trained blend, query labels or test-driven per-cell selection.

| Dataset | Labels/class | Original path-only LOO % | Uniform LP % | Expanded LOO % |
|---|---:|---:|---:|---:|
| ACM | 5 | 69.42 | 80.79 | 69.42 |
| ACM | 20 | 76.98 | 85.55 | 75.78 |
| FREEBASE | 5 | 41.80 | 48.33 | 41.93 |
| FREEBASE | 20 | 50.80 | 50.90 | 50.50 |
| IMDB | 5 | 27.53 | 32.04 | 27.53 |
| IMDB | 20 | 34.57 | 35.41 | 35.72 |

| Dataset | Labels/class | Seed | Context-selected candidate | Test accuracy % |
|---|---:|---:|---|---:|
| ACM | 5 | 11 | PSP | 58.99 |
| ACM | 5 | 12 | P-P | 76.48 |
| ACM | 5 | 13 | P-P | 72.79 |
| ACM | 20 | 11 | P-P | 78.50 |
| ACM | 20 | 12 | P-P | 77.42 |
| ACM | 20 | 13 | union | 71.42 |
| IMDB | 5 | 11 | MDM | 27.46 |
| IMDB | 5 | 12 | MDM | 27.63 |
| IMDB | 5 | 13 | MDM | 27.49 |
| IMDB | 20 | 11 | union | 36.46 |
| IMDB | 20 | 12 | uniform | 36.11 |
| IMDB | 20 | 13 | uniform | 34.59 |
| FREEBASE | 5 | 11 | MDM | 38.00 |
| FREEBASE | 5 | 12 | union | 49.70 |
| FREEBASE | 5 | 13 | MDM | 38.10 |
| FREEBASE | 20 | 11 | union | 50.50 |
| FREEBASE | 20 | 12 | union | 50.50 |
| FREEBASE | 20 | 13 | union | 50.50 |

All 18 outputs pass source, mask/context, held-out NLL, selected-component and metric audits. This diagnostic can expose a candidate-pool omission; success would not establish a new method or independent generalization.
