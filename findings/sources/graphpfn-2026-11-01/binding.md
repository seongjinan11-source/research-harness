# Development check: common-context product

2026-11-01. Training updates: **0**. Existing development graphs/contexts only; no independent holdout claim.

Accuracy (%), full public test, mean of cached context seeds 11/12/13:

| Dataset | Labels/class | Original uniform LP | Product after class pooling | Product before class pooling |
|---|---:|---:|---:|---:|
| ACM | 5 | 80.791 | 80.822 | 72.548 |
| ACM | 20 | 85.555 | 84.757 | 76.390 |
| IMDB | 5 | 32.039 | 32.039 | 27.477 |
| IMDB | 20 | 35.413 | 35.423 | 28.311 |
| FREEBASE | 5 | 48.333 | 48.300 | 41.033 |
| FREEBASE | 20 | 50.900 | 50.900 | 43.100 |

The fixed common-context product is not a useful next method on these graphs. At k=20, IMDB has zero common seed support in approximately 96.1% of test rows, averaged across the three contexts. Multiplication therefore discards most available evidence. With more than two views the reported zero-support quantity averages over view pairs, not the fraction with no support in any pair.

An exact constructed histogram collision demonstrates that class pooling can lose context identity. This is an expressivity example, **not evidence that this loss explains the real-data errors**. The real-data comparison does not support universally enforcing a common context witness.

All 18 reproduced uniform-LP probability arrays agree with their cached references exactly (maximum absolute error 0). Context/test disjointness, balanced context counts, finite normalized probabilities, class permutation, context-seed permutation, and view permutation were checked. Full task-level accuracy/NLL/Brier and input/source hashes are in `results/`.

This check uses no features; it establishes no missing-feature robustness. No existing experiment/report/checkpoint was edited. No new model was trained, no threshold was selected, and no test-based view selection was performed.

Decision: **do not promote this fixed product to a method**. The more general problem of learning a task-dependent relational computation is separate and remains an untested research proposal.
