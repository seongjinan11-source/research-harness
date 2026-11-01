# Context accuracy versus NLL selection diagnostic

Posthoc development analysis, motivated after the earlier test results were visible. Uses exactly the saved held-out context probabilities and the original expanded candidate pool. Accuracy ties are resolved by context NLL and fixed candidate order. Test labels report output accuracy only. No new propagation, PFN inference or learned parameters.

| Dataset | Labels/class | Context-accuracy selected test accuracy % |
|---|---:|---:|
| ACM | 5 | 75.79 |
| ACM | 20 | 85.55 |
| FREEBASE | 5 | 45.27 |
| FREEBASE | 20 | 50.50 |
| IMDB | 5 | 27.53 |
| IMDB | 20 | 33.55 |

| Dataset | Labels/class | Seed | Chosen candidate |
|---|---:|---:|---|
| ACM | 5 | 11 | P-P |
| ACM | 5 | 12 | P-P |
| ACM | 5 | 13 | uniform |
| ACM | 20 | 11 | uniform |
| ACM | 20 | 12 | uniform |
| ACM | 20 | 13 | uniform |
| IMDB | 5 | 11 | MDM |
| IMDB | 5 | 12 | MDM |
| IMDB | 5 | 13 | MDM |
| IMDB | 20 | 11 | union |
| IMDB | 20 | 12 | union |
| IMDB | 20 | 13 | MDM |
| FREEBASE | 5 | 11 | MDM |
| FREEBASE | 5 | 12 | union |
| FREEBASE | 5 | 13 | union |
| FREEBASE | 20 | 11 | union |
| FREEBASE | 20 | 12 | union |
| FREEBASE | 20 | 13 | union |

All 18 saved outputs pass metric and context-only selection readback. This is an accuracy-oriented baseline diagnostic on already explored data, not an independent generalization result.
