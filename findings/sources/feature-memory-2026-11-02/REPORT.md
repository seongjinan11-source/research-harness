# Feature-memory GraphPFN — matched pilot

- Mechanism gate: False; strong-method gate: False.
- Fresh 64 graph seeds; three fixed training seeds. Same declared families; real-graph/schema transfer untested.
- Primary is accuracy. NLL/Brier remain separate outcomes. CIs resample base graphs, not individual queries or training seeds.
- Observed memory is native two-coordinate feature-group memory with original two-bit mask. It is not a fully sparse cell-token architecture.

| Model | Primary accuracy (%) | NLL | Brier |
|---|---:|---:|---:|
| context | 52.3336 | 0.687776 | 0.494768 |
| all_nodes | 51.9552 | 0.689573 | 0.496561 |
| observed | 52.2888 | 0.692156 | 0.498759 |
| pcfi | 53.9246 | 0.683446 | 0.490617 |
| lp | 53.3203 | 0.777961 | 0.519445 |
| uniform | 49.9512 | 0.693147 | 0.500000 |
| class_prior | 51.0254 | 0.689640 | 0.496611 |

| Accuracy contrast | Difference (%p) | Paired graph 95% CI (%p) |
|---|---:|---|
| observed − context | -0.0448 | [-1.5015, 1.4608] |
| observed − all_nodes | 0.3337 | [-0.9481, 1.6135] |
| observed − pcfi | -1.6357 | [-3.4202, 0.0633] |
| observed − lp | -1.0315 | [-4.4214, 2.3562] |

The context-only key/value restriction is verified in the source; this run does not establish it as the cause of earlier failed experiments.
All-node feature attention is a direct prior-work control. Observed-only must improve it to establish an observation-specific effect.
Details: protocol.json, results/paired.json, results/interventions.json, results/costs.json, results/audit.json. No automatic epoch/loss rescue.
