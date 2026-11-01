# diag_context results (DEV 60 tasks, r=0.5, NaN input, pilot checkpoints s0; acc %, paired task bootstrap 95% CI, B=10000)

Conditions: correct; permctx (context labels permuted among context nodes); reversed (complement S_j, sanity: in-distribution); mismatch (context uses S_j, query uses complement).

| strength | quantity | init | B4 | Ours | Ours-B4 |
|---|---|---|---|---|---|
| q.3 b.5-2 (train-like) | correct | 72.66 | 74.35 | 75.40 | +1.05 [0.69, 1.47] |
| | permctx | 62.26 | 62.34 | 62.32 | -0.02 [-0.06, 0.01] |
| | reversed | 72.82 | 74.29 | 75.37 | +1.08 [0.73, 1.47] |
| | mismatch-correct | -5.19 | -9.64 | -12.44 | Dmismatch -2.81 [-4.04, -1.75] |
| q1 b1-4 (strong) | correct | 87.58 | 90.99 | 93.23 | +2.24 [1.56, 3.01] |
| | permctx | 62.28 | 62.30 | 62.29 | -0.02 [-0.04, -0.00] |
| | reversed | 87.78 | 91.51 | 93.68 | +2.16 [1.62, 2.73] |
| | mismatch-correct | -41.93 | -48.28 | -52.72 | Dmismatch -4.44 [-5.73, -3.22] |
| MCAR control | correct | 70.77 | 71.04 | 71.08 | +0.04 [-0.08, 0.19] |
| | permctx | 62.29 | 62.34 | 62.30 | -0.04 [-0.10, -0.00] |

Dcontext (Ours-B4 correct minus permctx): +1.07 [0.70, 1.49] (q.3), +2.26 [1.58, 3.02] (q1), +0.08 [-0.04, 0.23] (MCAR).

Notes
- permctx collapses every model to ~62.3 (majority-class level): it removes all label information, not only the mask-label link. It shows the Ours gain needs context labels, but it is not mask-specific.
- reversed equals correct within noise for all models (expected; S_j is drawn symmetrically). Sanity pass.
- mismatch is the sharp test: all models, including init, transfer the context mask->label relation to queries; Ours relies on it more than B4 (Dmismatch < 0, CI excludes 0). init already shows the behaviour (-41.9 pp at strong strength), so the in-context capability pre-exists in GraphPFN 1.3; mixed-prior training amplifies it.
- Single training seed; DEV split only; hold-out not used.
