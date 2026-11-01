# Direct marginal GraphPFN — valid conditional circuit kill test

- Circuit validity: True; mechanism gate: True; competitive gate: False.
- Newly implemented normalized graph-conditioned tree mixture; not a GSPN or MaM reproduction. Exactness holds inside the learned conditional tree model, not the original cyclic SBM.
- Three fixed training seeds; 2048 task exposures and256 updates per model. Public Mask GraphPFN uses accumulated graphs and strong typed-context metadata. Architecture parameter counts differ and are reported.
- Known-family exact Grid6 marginalizes parameters, roles, all hidden node labels and feature orientations. Grid6-BP uses the same coherent weights,64 damped iterations and observed-feature unary fields.
- Primary: matched48 fresh base graphs, whole-node uniform/clustered50/90%, contexts1/2/4. No test selection.

| Method | Accuracy (%) | NLL | Brier | Grid6 predictive KL |
|---|---:|---:|---:|---:|
| circuit | 61.3839 | 0.605128 | 0.430740 | 0.064716 |
| direct_graph | 51.1512 | 0.693134 | 0.499986 | 0.154796 |
| mask_pfn | 53.4846 | 0.687080 | 0.493828 | 0.147684 |
| grid6_exact | 69.3680 | 0.541211 | 0.375627 | 0.000000 |
| grid6_bp | 69.5602 | 0.540755 | 0.375136 | 0.000099 |
| oracle_true_parameters | 70.1389 | 0.525782 | 0.364317 | 0.007823 |
| lp | 51.4137 | 1.170692 | 0.655766 | 0.644926 |
| class_prior | 49.6445 | 0.769901 | 0.566413 | 0.227662 |
| uniform | 52.1577 | 0.693147 | 0.500000 | 0.154244 |

| NLL contrast | Difference | Graph95% CI |
|---|---:|---|
| circuit − direct_graph | -0.088005 | [-0.106528, -0.069288] |
| circuit − mask_pfn | -0.081952 | [-0.101972, -0.062402] |
| circuit − grid6_bp | 0.064373 | [0.024247, 0.105302] |
| circuit − grid6_exact | 0.063917 | [0.024041, 0.104503] |

A valid marginalization operator is not evidence that the learned probability family fits the data. Gaussian features and the tree approximation are restrictive.
Only the declared tiny schema was run. Full sparse observation tokens, continuous variable-schema MaM, and a globally valid cyclic graph circuit remain untested.
See protocol.json and results/summary.json, paired.json, interventions.json, benchmark.json, audit.json for all secondary conditions and raw measurement scopes.
