# Joint imputation inside public GraphPFN

This experiment separates one-time task-trained imputation from interleaved
numeric-state refinement inside the released GraphPFN. It does not reuse the
earlier quadratic feature-label prototype or change subgraph selection.

Read `protocol.json` for the fixed design. Trainable modules are the original
GraphPFN graph adapters, feature encoder and label decoder, plus a small shared
refiner or generic token control. LimiX transformer blocks remain frozen.

The refiner updates numeric feature values, clamps observed cells, and injects
the difference of the common feature encoder into feature tokens. Context label
tokens may condition the update. Query labels are loss targets only. Masks are
kept explicitly; imputed values are not silently declared observed. The new
common input interface intentionally omits native recentering and PCA, and is
not an official benchmark reproduction.

All arms run the same 12 PFN layers. Frontend and interleaved share exactly the
same refiner architecture and four steps; only the position of those steps
changes. The plain controls use a small generic token residual adapter. Their
parameter difference is reported. PCFI Plain receives the same continued
training, rather than serving only as a frozen weak reference.

Run with the existing Windows GraphPFN Python environment:

```
python joint_imputation_pfn/experiment.py prepare
python joint_imputation_pfn/experiment.py preflight
python joint_imputation_pfn/experiment.py run
python joint_imputation_pfn/experiment.py analyze
python joint_imputation_pfn/runtime/correct_feedback_diagnostic.py
python joint_imputation_pfn/runtime/finalize.py
```

No feature reconstruction loss, posterior head, target-graph optimization,
test-tuned checkpoint, or automatic longer training is included. Results are a
small same-generator-family pilot, not evidence of unseen-schema generalization.

For an interrupted process with a completed `training.json` and checkpoint,
`runtime/resume.py` restores the saved trainable parameters and runs evaluation
without extra updates, then continues the remaining original runs. The resume
runner is separately fingerprinted; it does not change the locked experiment
implementation. `results/completed.json` appears only after all runs and audits.

The bank precomputes PCFI for the comparison arm. Saved forward timings exclude
that bank construction and PCFI cost; they are not end-to-end latency claims.
An odd number of original features is padded with one observed constant-zero
channel to preserve the native two-feature groups. Raw observation counts in
the bank manifest exclude that padding channel.

The first run exposed FP16 nonfinite gradients before the original clipping
step allowed GradScaler to react. `runtime/numerical_repair.json` records the
execution-only fix and original source snapshot: retry the identical episode
at a lower scale, without updating parameters, until gradients are finite.
Every completed arm still performs exactly 192 successful optimizer updates.
No model, prior, loss, learning-rate schedule, or checkpoint selection changed.
Completed pre-fix controls are retained; retry counts and time are reported.

The no-feedback diagnostic had a separate FP16 cancellation issue in the order
of embedding subtraction. `correct_feedback_diagnostic.py` repairs only that
intervention in memory and pairs it with a standard forward in the same process.
It preserves original main predictions and records cross-process replay drift.
Use `results/paired_corrected.json` for the updated diagnostic; primary main
contrasts are unchanged. The correction script also runs finalization.

Prior art motivating the distinction: GRAPE (NeurIPS 2020), GCNmf (FGCS 2021),
PCFI (ICLR 2023), and GraphPFN. Joint imputation and downstream learning by itself
is not a new contribution. This pilot tests its explicit in-context placement.
