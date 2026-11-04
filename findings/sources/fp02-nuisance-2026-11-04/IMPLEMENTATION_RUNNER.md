# FP02 Stage 0 runner and saved-probability audit

`run.py` owns only preparation, single-worker execution, immutable probability
storage, and descriptive saved-probability analysis. `operators.py` owns the
scientific inference operation; `checks.py` independently checks correctness.
PLAN_v2 and protocol.json govern the fixed ten development episodes. No optimizer
is created and the runner never creates a source lock.

## Commands and execution gate

Use the existing graph environment with `-B -X utf8`. Import order is sklearn
before torch. Root alone performs the actual real-graph GPU run.

```powershell
python -B -X utf8 factor_nuisance_cycle/run.py selfcheck
python -B -X utf8 factor_nuisance_cycle/run.py prepare
# Root runs independent numerical checks, then creates results/lock.json once.
python -B -X utf8 factor_nuisance_cycle/run.py run
python -B -X utf8 factor_nuisance_cycle/run.py report
```

`selfcheck` reads protocol.json, checks parsing/scoring/array identity and the
positive/negative gate paths with tiny CPU arrays. It writes no files, imports no
historical runner, performs no model forward, and allocates no GPU tensors.

`prepare` imports the unchanged FP01 experiment.py by explicit absolute path and
unique module name. It uses `R.G.real_rows()` and `R.G.read_real(row)` to select
exactly rep0/seed11 Cora, CiteSeer, PubMed, CoauthorCS, CoauthorPhysics in none/95
modes. Historical bank arrays remain in place; a new manifest references their
absolute paths and immutable hashes. `bank/dev_manifest.json` contains ten rows,
input/target-array hashes, strict own-feature-missing query validation and seven
references per row. Preparation validates all reference probability arrays and
the inherited FP01 source/checkpoint state on CPU. It does not call a predictor.

Root runs checks and then calls `run.source_state()` to obtain this exact lock:

```python
{
    "status": "LOCKED", "cycle": "FP02-Stage0",
    "files": {absolute_path: sha256, ...},
    "manifest_sha256": "...", "checks_sha256": "...",
    "inherited_lock_sha256": "...",
    "optimizer_updates": 0, "target_gradient_updates": 0,
}
```

Root writes that object to `results/lock.json` with create-only semantics.
`source_state()` itself never writes. It requires checks.status=PASS, unchanged
checks.tested_sources, no real predictions used by checks, and a fully verified
prepared manifest. Its file map includes run/operators/checks/protocol/PLAN_v2/
RESPONSE/this document/three current-cycle review copies; manifest/checks;
inherited FP01 lock/checks/C_global_s11 checkpoint/training/evaluation index;
recursive inherited locked sources/checkpoints; each selected bank NPZ and bank
manifest; and all 70 reference NPZ/sidecars. If a root-owned review is not yet
present, source_state fails rather than locking an incomplete map.

`run` requires an existing lock exactly equal to current source_state. Device is
fixed to cuda, float32, one worker. A create-only `.active_run.json` marker prevents
overlapping writers. A marker remaining after process termination requires root
to verify no writer remains and preserve operational evidence before removal.
Git branch/commit are read only once per invocation, outside prediction identity.
No per-prediction Git processes are started.

## Inference API and frozen feature factor

The unchanged FP01 `R.load_model('C_global', 11, device='cuda')` supplies the exact
three fitted feature coefficients and registered normalized GH41 masses. All
parameters have requires_grad=False; the digest is checked before/after. For each
episode the runner calls unchanged `model.hyperparameters(visible,{})` and builds
the same literal Normal `[N,F,41]` table. It passes only visible inputs to:

```python
p, info = nuisance_bp(
    src, dst, N, context, K, observed, log_likelihood, log_weights,
    variant=arm, iterations=24, damping=.5, block=6, details=False,
)
```

`context` is `[N]` long, −1 for unknown nodes and known context labels otherwise;
src/dst are symmetric directed edges, observed is `[N,F]` bool, table and weights
are float32. Query IDs and truth never enter nuisance_bp. Root checks edge
representation and operator mathematics independently. The three arms are
poisson_fixed, sbm_update, dc_update, with nuisance_updates=0/3/3 respectively.

New outputs are 30 NPZ/JSON pairs under
`results/predictions/{arm}/{graph}_r0_{mode}`. NPZ contains `p` (query×K), `y`,
`query`, `context`, `yc`. It also preserves all O(N+K²) tensor statistics from
info, including fresh/update-history statistics, using keys such as
`operator__theta_mean` and nested `operator__update_history__0__...`. Message banks
are not requested; arrays above 100,000 elements are rejected. JSON records
input/query/checkpoint/source identity, prediction hash, shared preparation costs,
inference seconds, CUDA peak/incremental peak, zero optimizer counts and compact
statistics. Vectors above 256 elements are summarized with shape/min/max/mean/hash
in JSON, with full operator values retained in NPZ. `max_rate` is the maximum
conditional pair intensity across classes, not its q-weighted mean; no clipping.

Same C_global/Normal table is used for all graph variants. Each mode estimates its
own nuisance parameters, so observed−none is the total joint inference effect.
There are no newly computed shuffle cases and no alignment success claim.

## Historical references and strict resume

For each row, old_full/old_edge are selected from FP01
`results/C_global_s11/evaluation_real.json` and corresponding sidecars. Their full
role weights must concentrate on homophilic role 2 within 1e-6. Otherwise
preparation raises an error for root; the runner never chooses another branch.
Input hash, row, target-array hash and feature checkpoint hash must match.

Five external references per row are unchanged edge-benchmark NPZ/JSON arrays:
lp_harmonic, lp_restart, public_graphpfn, pcfi_public_graphpfn,
fisf_public_graphpfn. Sidecar row/bank hash and all NPZ query/y/context/yc arrays
must match exactly. All reference hashes are frozen in the manifest and lock.
There are 70 reused reference records, counted once rather than per new variant.

No existing output is overwritten. `prepare --resume` rebuilds and compares the
full immutable manifest, ignoring only its original preparation duration.
`run --resume` verifies complete NPZ/sidecar pairs against the current identity,
full file hash and exact target/context arrays before skipping them. A partial
pair is an error even with resume. Root must preserve/resolve it explicitly.
Completed evaluation.json is compared exactly. Attempt logs are create-only and
retain failure traceback, wall cost and verified prediction count. A completed
report can be verified/skipped with `report --resume`; incomplete report sets are
errors, never silently repaired or overwritten.

## Rescoring, gate, reports and artifacts

`report` requires the same live lock and complete 30-record evaluation index. It
independently reopens all 100 probability arrays, validates NPZ/sidecar/index
hashes and query/context identities, then computes metrics in float64. For the
historical scoring convention probabilities are clipped to [1e-8,1] and row
normalized **only for metrics**. Stored probabilities are unchanged. Accuracy,
balanced accuracy, macro-F1, NLL (nats), Brier, class recall/support and hard
predicted/true class mass are saved. Missing true-class recall is null and class
support is explicit; sklearn's established BAcc/macro-F1 convention is used.

The gate uses none and source-graph-equal means, never pooled query weighting:

- Physics accuracy − old-full ≥ .20; CS ≥ .05.
- CiteSeer accuracy − old-full ≥ −.01 and NLL − old-full ≤ .05.
- Five-graph mean BAcc ≥ both old-full and old-edge.
- Five-graph mean NLL ≤ old-edge + .05.

Only 1e-12 floating subtraction tolerance protects exact boundaries. dc_update
is primary; sbm_update gets a separately displayed gate and cannot rescue DC.
95−none is reported per graph and as the equal-five-graph mean with no independent
real CI, no ensemble and no graph-specific winner selection.

Outputs are `results/summary.json`, `audit.json`, `budget.json`, `REPORT_KO.md`,
then `artifact_manifest.json` as the completion marker. Audit PASS means identity
and numerical integrity; it does not mean the development performance gate passed.
All original negative comparisons and external baselines remain in the summary.
Budget separates shared table/read cost (once per row), arm inference, cold model
load/attempt wall, peak memory and failed attempts; inherited training/reference
cost is explicitly not zero. Unlogged termination cost is not inferred to be zero.
The artifact manifest records absolute path, SHA256 and bytes for compact reports,
all new probabilities/metadata, reference arrays, source lock and locked lineage.
Its own hash is excluded to avoid a circular digest. Root may track only compact
summaries in Git while keeping prediction files shared/untracked by these hashes.

This is classical operator development on reused real data. It does not establish
new method novelty, independent transfer, feature alignment or a retrained PFN.
Stage 1 remains separately reviewed/locked and uses actual equal-budget fits and
new final graphs. Scientific settings may not change in response to these ten
episode outcomes.
