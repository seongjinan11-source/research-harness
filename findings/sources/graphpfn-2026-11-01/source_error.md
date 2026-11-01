# Shared-imputation-error GraphPFN: short synthetic pilot

One training seed, 64 updates x 2 episodes per arm. Same initialization, tasks, original masks, inputs and common trainable parameters. Final checkpoints only. All final comparisons use results/paired_eval with graph-wise matched stochastic feature-position embeddings (evaluation_addendum.json); preliminary root evaluation files are superseded. Independent synthetic evaluation graphs; no real-graph results or publication/novelty claim.

| Arm | Accuracy % | Balanced acc % | Macro-F1 % | NLL | Brier | Working error score |
|---|---:|---:|---:|---:|---:|---:|
| initial | 55.95 | 45.69 | 43.15 | 1.0347 | 0.5901 | 0.9339 |
| vanilla | 57.01 | 43.89 | 36.94 | 0.8162 | 0.5139 | 0.9339 |
| matched | 57.06 | 43.94 | 36.99 | 0.8159 | 0.5138 | 0.8821 |
| log | 57.04 | 43.90 | 36.94 | 0.8162 | 0.5139 | 0.8822 |
| diag | 57.04 | 43.93 | 37.00 | 0.8159 | 0.5137 | 0.8820 |
| covariance | 57.04 | 43.91 | 37.00 | 0.8161 | 0.5138 | 0.8826 |
| covariance_shuffled | 57.01 | 43.89 | 36.96 | 0.8161 | 0.5138 | 1.0221 |

Context-majority accuracy: 54.35%. Full-feature initial-model accuracy: 61.77%.

| Condition | Initial | Vanilla | Matched | Log | Diagonal | Shared error |
|---|---:|---:|---:|---:|---:|---:|
| uniform_nodes, 90% | 57.05 | 58.23 | 58.24 | 58.30 | 58.22 | 58.26 |
| uniform_nodes, 99.5% | 54.71 | 56.38 | 56.41 | 56.26 | 56.36 | 56.34 |
| clustered_nodes, 90% | 58.13 | 57.96 | 57.96 | 58.09 | 58.04 | 58.06 |
| clustered_nodes, 99.5% | 53.92 | 55.46 | 55.64 | 55.50 | 55.52 | 55.50 |

| Shared-error method minus | Accuracy difference pp | Descriptive 95% graph bootstrap |
|---|---:|---:|
| initial | +1.089 | [-0.640, +2.805] |
| vanilla | +0.032 | [-0.092, +0.157] |
| matched | -0.021 | [-0.168, +0.100] |
| log | +0.004 | [-0.075, +0.115] |
| diag | +0.004 | [-0.080, +0.082] |
| covariance_shuffled | +0.027 | [-0.022, +0.083] |

## Interpretation

No incremental classification gain is established for the shared-error method in this pilot. Its difference versus every matched dependency control is very small, and every displayed paired bootstrap interval includes zero. This is not proof that the method cannot work with another prior or a larger budget.

The working error score increases by 0.1395 when source correspondence is destroyed, but classification changes little. The error model is sensitive to source structure; this does not demonstrate useful task-level error correction.

Compared with the initial checkpoint, shared-error accuracy increases but balanced accuracy changes by -1.78 pp and macro-F1 by -6.15 pp. Generic continued training must not be presented as a method-specific improvement. No larger training run or real-graph validation was started on the basis of these results.

Zero-observation evaluation cases: 8/64; retained, never resampled. All 448 saved probability outputs have independently recomputed metrics and reconstructed masks.

The bootstrap varies the 16 held-out synthetic graphs, not training seeds or real domains. This small prior is explicitly restricted in size/degree/class balance and does not reproduce full GraphPFN pretraining. Five mirror steps approximate the attention optimum. The PSD error matrix is a working residual second-moment model, not certified posterior uncertainty. Gaussian scoring treats feature columns as replicate residual vectors.

Runtime preflight: initialize Windows DGL/CUDA bootstrap before CUDA allocation; fp16 GradScaler initial scale 128. Before pilot training, a gradient-only probe moved corrections from layers 10/11 to 8/10 because layer-11 feature updates have no later path to the label readout. Invalid restricted-prior SBM draws are deterministically retried and logged; no performance rejection.

See README.md for the objective/solver, protocol.json for fixed choices, checks.json for algebra/gradient/parity tests, results/metadata.json for fingerprints, and results/paired.json for per-graph differences.
