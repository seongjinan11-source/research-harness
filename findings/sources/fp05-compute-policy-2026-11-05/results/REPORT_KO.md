# FP05: missing-graph computation policy

Audit PASS; prespecified gate FAIL. Controller policy gradient + adapter CE; no new RL algorithm or heterogeneous claim.

| Arm | Missing-query accuracy % | Base graphs |
|---|---:|---:|
| fixed | 49.290 | 96 |
| uniform | 49.475 | 96 |
| direct | 48.776 | 96 |
| group_pg | 48.793 | 96 |

| Comparison | Difference pp | 97.5% CI pp | Verdict |
|---|---:|---|---|
| group_pg_minus_fixed | -0.497 | [-1.034, +0.012] | FAIL |
| group_pg_minus_direct | +0.017 | [-0.241, +0.275] | FAIL |
| feature_utility_none | -1.063 | [-5.567, +4.069] | FAIL |
| feature_utility_shuffle | -0.260 | [-2.127, +1.584] | FAIL |
| full_natural_guard | +0.130 | [-0.825, +1.042] | PASS |

Rates and fixed seeds are averaged inside base graph before stratified bootstrap. Utility is a separate eligible-disjoint estimand. See summary.json for all secondary mechanism tests and real results.

Real graphs are reused development data; representation is fixed visible-row RP16→RMS→PCFI. Uniform inference uses8 traces; greedy policies1 trace. Historical native public baselines have different preprocessing/budget.

No tuning after these locked results. Full posterior/Transformer RL training and new imputer claims are outside this pilot.
