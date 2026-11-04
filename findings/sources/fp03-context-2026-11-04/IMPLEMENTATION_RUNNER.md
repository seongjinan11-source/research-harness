# FP03 runner and independent saved-probability reconstruction

`run.py` implements the frozen root-owned protocol and PLAN_v1/v2. It owns CPU
preparation, gated single-worker inference, immutable arrays and descriptive
reporting. `operators.py` is the sole production implementation of fold generation,
context scoring and contrast BP. The report independently reconstructs split,
scores, weights and controls with NumPy. No optimizer is instantiated.

## CLI and root-owned execution gate

Use the existing graph environment with sklearn-before-torch imports and no
bytecode writes:

```powershell
python -B -X utf8 factor_context_cycle/run.py selfcheck
python -B -X utf8 factor_context_cycle/run.py prepare
# Root runs full independent numerical checks, writes the source lock once.
python -B -X utf8 factor_context_cycle/run.py run
python -B -X utf8 factor_context_cycle/run.py report
```

The implementer runs only the tiny CPU selfcheck. Root performs preparation,
source lock and actual real CUDA evaluation. `selfcheck` reads protocol and tiny
NumPy arrays only, writes no files, and does not import historical code or run a
model. `prepare` is CPU-only; it imports unchanged FP02 run.py by explicit path
and a unique module name, verifies its source lock, and reads its prepared ten
episode/70 reference manifest. No historical arrays are rewritten or copied.

The new `bank/dev_manifest.json` references the exact old rows/paths/hashes and
stores compact calibration split metadata. Node-feature matrices are not copied.
The manifest validates common graph/context between none/95 and expected totals:
139 full context labels,126 calibration targets,19 balanced folds,285 calibration
branches,150 final branches,50 derived controls,70 references,120 aggregate scores.
`prepare --resume` fully rebuilds/compares this state, ignoring only original
preparation time. Original context order is retained; selected slots follow the
canonical operator's minimum-stable-ID class ordering and fold-major flattening.

Root calls `source_state()` and writes its exact result create-only to
`results/lock.json`:

```python
{
  'status': 'LOCKED', 'cycle': 'FP03',
  'files': {absolute_path: sha256, ...},
  'manifest_sha256': '...', 'checks_sha256': '...',
  'inherited_lock_sha256': '...',
  'optimizer_updates': 0, 'target_gradient_updates': 0,
}
```

`source_state` itself never writes a lock. It requires current checks.status=PASS,
nonempty tested_sources with unchanged hashes, no real bank/predictions used by
checks, verified preparation, and unchanged inherited FP02 lock/full files map.
The new map includes run/operators/checks/protocol, both plans, response,
PRIOR_ART_SCOPE, ANALYSIS_SPEC, three current reviews and runner/inference/checks
implementation documents. IMPLEMENTATION_API is also included if present. It
includes the checked source map, inherited lock/manifest, and every inherited
locked file, including bank/reference/checkpoint and earlier API lineage.

`run` refuses without an exact matching root source lock and uses cuda/float32,
one worker, candidate/fold chunks at most16. A create-only active marker prevents
overlapping writers. A stale marker requires root to verify no worker remains
and preserve interruption evidence before removing it. Git branch/commit is read
once per invocation for descriptive metadata, outside scientific identity.

## Production fold and operator API

The runner does not implement a second production fold/scoring algorithm:

```python
spec = O.balanced_folds(context, classes=K)
weights, info = O.score_weights(heldp, heldlabels)
p, info = O.contrast_bp(src, dst, N, contexts, K, observed, LL, log_weights,
                       a, b, iterations=24, damping=.5, details=False)
```

`context[N]` is −1 except for allowed context labels; all row IDs are stable IDs
for selection only. C+ contains represented classes, m is the minimum positive
count, and each fold hides one selected node/class. All unselected extras are
unclamped in EVERY calibration fold. CS uses14 classes×3=42 targets and leaves13
extra context labels unused for calibration; final inference uses all55 labels.
There is no class-count cue from leaving only one class with fewer clamps.

Split metadata saves original/selected/unused IDs/labels, represented classes,
class counts and each fold's held/clamped IDs/labels. Dense fold_contexts and
stable_ids[N] are not duplicated into JSON. The manifest saves its split hash.
The report's separate `_numpy_split` verifies these memberships, counts and order.
No context-derived labels, held IDs, query IDs, truths or split metadata are
passed to BP beyond the legitimate clamped context tensor. Query IDs are used
only to extract final probabilities after BP.

Candidate order is a-major/b-minor, fixed-edge index10, graph-off index4. Candidate
and fold axes are independent lanes in contexts[B,N], a[B], b[B]; runner chunks
these jobs at16. Graph-only calibration supplies observed[N,0], LL[N,0,1] and
log_weights[1]=0, with every feature absent. All BP calls start fresh; no
label-dependent message warm cache is used across folds.

Calibration is performed once per graph and reused bit-identically for none/95.
One branch predicts all held nodes in its balanced fold. The actual branch count
is19×15=285, while saved held probability vectors are126×15=1890; neither is an
independent sample count. m0 uses empty targets and prior weights. m1 has no
calibration anchors/features; symmetric inference must give uniform class
predictions and prior weights, checked numerically.

The frozen feature model is inherited FP01 C_global_s11 with the same3 fitted
coefficients/GH41. FP02's unchanged `feature_table` builds its exact literal
Normal table. Model parameters are requires_grad=False and the state digest is
checked before/after. Final contexts use all original labels. Each episode runs
15 candidates once and derives adaptive/prior/MAP/fixed-edge/graph-off without
extra BP. Weights are float64; model inference and stored final probabilities
are float32. Graph-off own-feature-missing query probabilities must be uniform.
`operator_info.sources` counts observed cells, not physical source nodes; episode
source-node counts remain in the immutable manifest.

## Immutable probability schema and resume

Five calibration NPZ/JSON pairs are stored under `results/calibration/{graph}`:

- `p[15,S,K]`: raw held predictions in saved fold-major selected-node order.
- `held_nodes[S]`, `held_labels[S]`, full `context`/`yc`, `unused_nodes`.
- `scores[15]`, `weights[15]` float64, `a[15]`, `b[15]`.

Sidecars include split/identity/hash, MAP index, compact score diagnostics,
branch/chunk counts, timing and memory. Normalized_held tensors and repetitive
per-cell values are not printed to JSON. Production score is represented-class
macro mean log probability after float64 clip[1e-8,1] and row normalization,
unit mean-score scale, uniform candidate prior. Raw stored p is unchanged.

Ten final NPZ/JSON pairs are under `results/predictions/{case_id}`:

- `candidate_p[15,Q,K]`, plus five named `[Q,K]` control arrays: adaptive, prior,
  map, fixed_edge, graph_off.
- `query`, `y`, `context`, `yc`, and bit-identical shared calibration scores/weights.

Sidecars record exact source/checkpoint/bank/input/target/calibration/split hashes,
15 branch forwards, cold table preparation, inference+D2H, CPU aggregation,
row-read time, absolute/incremental CUDA peak, small operator diagnostics and zero
optimizer/gradient counts. Long degree/theta vectors are summarized rather than
duplicated. No dense N×N×K messages are allocated by the runner or saved.

Creation uses x/xb semantics. Complete pairs require `--resume`, exact identity,
NPZ/sidecar hash and all known arrays. A partial pair is an error even with resume;
root must preserve and resolve it explicitly. Evaluation.json is complete only
after all5 calibration and10 final pairs. Completed index contents must match
exactly on resume. Per-invocation attempt logs retain failure traceback, elapsed
wall, new branch counts and verified episodes; completed/skipped predictions are
not silently regenerated. A report set is also create-only; `report --resume`
verifies a COMPLETE artifact manifest/all hashes, while partial report sets error.

## Independent analysis and ten fixed gate checks

`report` does not call a model or production score/mixture helpers. NumPy code
independently reconstructs split membership, clips/normalizes raw held arrays,
macro log scores, float64 softmax, MAP and all5 final controls. It verifies scores
and weights within1e-12 and float32 controls within1e-7, with raw common weights
bit-identical across none/95. It recomputes120 aggregate scores and all150
candidate diagnostics from saved p, with confusion-matrix BAcc/macro-F1 and fixed
NLL/Brier convention. Class support/recall and predicted/true hard class masses
are retained; CS's unseeded class9/query4 are never removed.

All ten gate comparisons use none and equal weighting of5 source graphs:

1. Physics accuracy−old-full≥.20.
2. CS accuracy−old-full≥.05.
3. CiteSeer accuracy−old-full≥−.01.
4. CiteSeer NLL−old-full≤.05.
5. Mean BAcc−old-edge≥0.
6. Mean NLL−old-edge≤.05.
7. Mean accuracy−prior mixture≥.01.
8. Mean accuracy−fixed-edge≥0.
9. Mean NLL−prior mixture≤.01.
10. Mean NLL−fixed-edge≤.01.

All are conjunctive, including BOTH NLL control guards. Only1e-12 tolerance
protects floating subtraction boundaries. MAP, a single candidate, or95 results
cannot replace adaptive/none primary. 95−none is descriptive per graph and as the
equal5-graph mean; there is no independent-real CI or feature-alignment claim.

Summary includes held scores/weights/entropy/ESS/prior KL, weighted a/b moments,
uniform-pair mean TV and weighted TV-to-mixture, plus all per-candidate metrics.
ANALYSIS_SPEC's query-truth best single-candidate accuracy/NLL is separately
marked family-headroom diagnostic only. It feeds no operator, score, primary,
hyperparameter choice or rerun. Min NLL/max accuracy may select different oracle
candidates and are not one deployed predictor.

`summary.json`, `audit.json`, `budget.json`, `REPORT_KO.md` and completion marker
`artifact_manifest.json` are stored under results. Audit PASS is distinct from
performance gate PASS. Artifact manifest contains absolute paths/SHA256/bytes for
new arrays/metadata, references, reports, attempts and full locked lineage; its
own hash is excluded to avoid circularity. Large NPZs can remain shared/untracked
while compact summaries/manifests are Git artifacts managed by root.

Budget separates preparation, calibration setup/inference/score, final table
setup/inference+D2H/CPU aggregation, model load, lock validation, audit and full
attempt wall. The first invocation is cold end-to-end; saved-pair resume is not a
new warm timing trial. Failed/abrupt attempts and historical baseline/training
costs are not assumed zero. Actual branch count on a failure inside an unfinished
call may be incomplete, while wall/traceback are retained. No extra repeats are
performed to select a favorable latency. This is inference-stage context
calibration on reused real development data, not new neural pretraining or a
calibrated Bayesian mechanism posterior.
