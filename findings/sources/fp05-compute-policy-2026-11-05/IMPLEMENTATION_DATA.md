# FP05 data API and provenance

2026-11-05. Ownership: `rl_data.py` and this note only. Preparation/training/source-lock execution is root owned. Import creates no files, samples no task, loads no real arrays/checkpoints and starts no GPU model. Development checks use only small hand-built CPU arrays. The original GraphPFN/prior and FP04 files are read only.

## Public interface

`prepare(split)` supports `train`, `dev`, `final`, `real`, writing `bank/{split}/manifest.json` exclusively at successful completion. An existing or partially prepared split directory is refused. Files use `open('x'/'xb')`, and a new split directory uses exclusive `mkdir()`. Shared parent directory creation does not replace any content. Interrupted artifacts are preserved for root review, not silently restarted/overwritten. Final generation additionally requires root-created `results/design.json` and records its SHA; it does not create a design lock itself.

`rows(split)` reads the manifest and checks the protocol SHA. `read(row)` checks the NPZ SHA and returns `(inp,y)` with exactly these input keys:

```text
values          float32 [N,F], PCFI after represented-value clip10
observed        bool [N,F], physical availability, independent of numerical zero
src,dst         int64 [E], literal official graph records / inherited real graph
context_ids     int64 [C]
context_labels  int64 [C], same order as context_ids
query_ids       int64 [Q]
classes         Python int, actual task K
forward_seed    Python int
```

`y:int64[Q]` is separate and never appears in `inp`. `raw X_full/y_full`, generator statistics, intervention membership, missing-query flags, source counts and lineage are not in this model whitelist. Actual F is preserved, even if odd after the official prior drops constant columns; GraphPFNLight retains its original internal preprocessing/padding. Data never adds an observed dummy feature that would change source0 or own-missing membership.

`read_extra(row)` and alias `extra(row)` return:

- `own_missing` and `standard_own_missing`: identical bool[Q] from the corresponding **STANDARD physical mask**, even on none/shuffle rows.
- `query_observed`: complement of standard_own_missing; at least one actual physical feature is observed in that original mask. Full-natural means all features are observed. None interventions do not change this paired scoring membership.
- `actual_own_missing`: bool[Q] from the intervened mask, separately retained to avoid confusing the two notions.
- `standard_source_nodes`: Python int from the original mask; `source_nodes`: Python int from the current actual mask.
- `physical_features`: Python int; `visible_values`: float32[N,F], current observed-only normalized values, exactly0 elsewhere; `pcfi_raw`: actual unbounded PCFI cache before clip10.

Rows have `id=row_id`, `base=base_id`, `shape`, `k` (context cap, not K), `regime`, `stratum`, `repetition`, `setting`, `intervention`, `source_nodes`, `standard_source_nodes`, `path` relative to FP05 ROOT, SHA and identity hashes. `classes` denotes actual K. Synthetic intervention rows keep `setting='node95'/'node995'` and `intervention='none'/'shuffle'`; there is no independent synthetic setting named none. Real rows have `setting=mode` in95/99/995/none/shuffle95, `intervention='standard'`, graph/repetition/mode and `k='legacy'`. Their inherited shuffle is already applied and is not shuffled a second time.

`simple_reference(inp,'prior')` returns the Laplace1 context-frequency distribution on query IDs. `simple_reference(inp,'lp')` uses binary symmetric adjacency+self-loops, row-normalized restart.9,40 iterations with true context clamped each time, then `.99*normalized_propagation+.01*prior`. Both use supplied K and context only, with no query-target/feature access beyond array shape. Output is NumPy[Q,K].

`historical_references()` reads the frozen FP04 external index with its manifest hash checked. These references have physical sample identity but different processed feature representations and are not new RP16 causal controls. Root runner separately evaluates frozen projected-NaN/projected-PCFI000 references.

## Counts and deterministic factorial design

| Split | Base graphs | Episodes/base | Rows |
|---|---:|---:|---:|
| train | 256 | 7 standard | 1792 |
| dev | 8 | 7 standard | 56 |
| final | 96 | 7 standard + 4 paired interventions | 1056 |
| real | 3 graphs×3 repetitions | 5 inherited modes | 45 |

Seven standards are full,node90,node95,node995,cluster95,cluster995,cell95. Final node95 and node995 each have their own none/shuffle rows. The none processed arrays/forward seed are identical within base, while standard membership/source metadata may differ. This deliberate duplicate does not create another independent graph or query sample.

Final enumeration is requested shape(matched,heldout) × k(1,3,5) × regime(natural,disjoint) × rep0…7:12 strata with8 graphs each. Matched N/F are drawn from protocol128/160/192 and8/10/12 with their own RNG; heldout requests256/16. Actual N/F can differ because the original graph sampler/constant-column handling is retained, so requested/actual N/F/K are stored separately. No draw is retried to force exact observed class balance or feature count. `stratum='{shape}_k{k}_{regime}'` denotes the requested design cell.

Train/dev alternate natural/disjoint by base index; k cycles in pairs1,3,5. Train256 gives128 bases per regime; dev8 gives4 per regime. Full-natural16 reserved update scheduling is root runner owned. Data records every query membership/count, and root runner counts actual observed/missing-query CE exposures. A50:50 regime split is not a50:50 observed-query split.

Seeds are protocol51050000/51060000/51070000+base index. Independent `SeedSequence([requested_seed,stream])` streams are0 requested shape,5 context/query,6 whole-node uniform,7 cluster BFS,8 cell,30+standard-setting-index shuffle,90 evaluation forward seed. One forward seed is shared by all masks/interventions of the base; training overrides it with the common update epsilon, root owned.

## Official task generation

`generator_config(N,F)` deep-copies the pinned official prior and changes only: requested N scalar; degree uniform4…12; min first-level child16; requested feature count F scalar; task class min2/max4. Official quantile and other SCM/task distributions are unchanged. In particular, FP05 does not silently copy an earlier pilot's extra quantile .3….7 restriction. Configuration dictionaries are saved in create-only `generator_configs.json` with a manifest hash.

Official `mgpfn.prior_adapter.sample_official_task` is invoked on CPU with allowed binclass/multiclass types. That existing wrapper's task-type rejection count is stored separately. Up to50 outer retries catch only official AssertionError for invalid structural/sanity draws; actual seed is requested_seed+attempt*100000000. Each caught error/attempt/seed is recorded for a successful base. Other failures stop preparation and preserve the partial folder rather than rejecting according to a model result. Full synthetic class remapping defines supplied contiguous K; absent context classes do not trigger resampling. Actual class/context counts are private audit metadata.

Context is sampled classwise up to k from the official `[0,n_train)` segment and then permuted. Query up to16 is sampled uniformly from the official `[n_train,N)` segment, before masks, with actual count retained. Missing classes simply contribute no anchors. There is no query-label quota, forced balance or feature-dependent query selection. Literal source/destination records are preserved, and self/duplicate edge counts are audited rather than silently changing graph multiplicities.

Private raw tasks in `bank/{split}/raw/{base}.npz` hold X_full/y_full and graph/split identities for replay. They never enter read()'s input. Row hashes, raw-task fingerprint/SHA and generator key link the processed episodes to these artifacts.

## Availability, interventions and preprocessing

Uniform node masks use one uniform vector per base with thresholds.90/.95/.995, so node995⊆node95⊆node90 exactly. Cell95 uses a separate uniform matrix. Cluster uses one randomized BFS forest ordering per base: a uniformly shuffled root list starts new components, and neighbors are shuffled before enqueueing; take rounded `N*(1−rate)` nodes. Cluster995⊆cluster95. The BFS uses only graph edges and private mask RNG, never labels/feature values.

For the disjoint regime, clear context and query rows **after** the original mask is drawn, including full. Thus actual cluster source count can fall below the initial rounded count, including0/1. Natural leaves the mask unchanged. Full-natural is physically fully observed; full-disjoint is nonquery/context-visible and must be described as full_nonquery. Raw rates and actual observed/source counts are both retained. No mask retry forces support.

None changes M to allfalse and current raw values to0; original paired membership remains. Node shuffle permutes complete available rows among the same rows, leaving M/context/query unchanged. The helper also supports columnwise shuffling of observed values if a cell intervention is separately specified later; no cell-intervention episode is currently generated. Scaling and actual PCFI are freshly computed after every intervention.

Observed-only float64 RMS normalization, scale1 for empty/allzero columns, clip10/float32 and hidden-zero are reused read-only from FP04. Hidden NaNs/finite poison cannot enter these statistics. Actual PCFI is likewise FP04's unique-file import of the literal `fisf_graphpfn/imputation.py`,100 iterations/alpha.9, no fabricated anchors, explicit final observation clamp, isolated Python/NumPy/Torch RNG states. No-source input uses the mathematically equivalent allzero shortcut. Its raw result is privately cached, then values is represented-clipped to[-10,10]. The mask identifies literal observed zeros correctly.

Metadata records actual and standard source counts, observed cells, source0/1-compatible support, column support0/1, query counts, source/context overlap, original query-membership hash and graph/context/query-truth/input hashes. These are audit/estimator inputs, not a policy whitelist extension. PCFI wall/CPU seconds, source implementation SHA, raw maxabs and represented clipping fraction are stored per row; total preparation wall/CPU/cache cost is in the manifest.

## Real reconstruction and projection

The FP04 source is imported by its explicit unique filesystem module name, with no legacy generic `data` import. Its frozen real manifest/NPZ and legacy pilot/homophily manifests are read only. Existing FP04 native recovery helpers load hashed immutable PyG sources, reconstruct observed raw rows and historical shuffled value IDs, verify graph/context/query/truth/observed-mask identity and historical RP replay. The actual FP04 NPZ target/context/mask arrays are compared too. No clipped FP04 RMS array is mistaken for native raw values.

Only visible native values are multiplied by projection. `project_visible(raw,observed,R)` rejects partial-cell native masks, sanitizes hidden values before multiplication, and returns projected values plus the identical row availability broadcast across16 columns. A cancellation to numerical0 remains observed. Query/context masks remain inherited own-missing. All Cora2708/CiteSeer3327 full nodes and PubMed inherited2000 nodes are preserved.

For graph index0/1/2 in the protocol list, R seed is51089000+index and entries are independently±1/sqrt16, shape rawF×16. The same matrix is reused across all reps/modes/arms/seeds for that graph and stored as a separate immutable projection NPZ with source/array hashes. Pipeline is **raw visible rows→RP16→observed RMS→literal PCFI100→clip10**. Neither hidden values nor labels fit R. Native source/mask hash, raw source/projection SHA, FP04 bank SHA and all reconstruction identities are retained. Original raw sources are rehashed at the end.

The real manifest references the225 historical LP/public/PCFI/FISF records without copying or changing them. This establishes physical sample identity only; RP16/new RMS and historical native/PCA/public ensemble/model budgets differ. No native-dimensional representation claim or new real-generalization CI is implied.

## Dependency/source lock and bounded selfcheck

Each manifest exposes `dependencies`: this data/rl_util/protocol/PLAN_v2, FP04 data, literal PCFI, mgpfn upstream/prior_adapter; synthetic manifests also hash the original pretrain TOML, Windows bootstrap and all pinned prior Python sources. Root source lock must include these imported dependencies and actual public checkpoint hashes. A reused FP04 function is not treated as an invisible dependency. Runtime generation records scientific code/config and imported original bytes, independently of Git metadata.

`selfcheck()` samples **zero official tasks**, loads **zero real inputs**, writes **zero bank files**, executes **zero optimizer updates**, and runs **no GPU**. Small CPU hand-built arrays verify observed normalization poison/zero-column invariance, nested masks, disjoint clearing, paired query/context/truth and original missing-membership invariance for standard/none/shuffle, source0 PCFI, RP hidden-poison/row-mask identity, and finite prior/LP probability output. Actual tiny CPU PCFI is invoked. The broader independent model/gradient/trace suite remains checks-agent owned.

Root CLI invocations:

```text
python pfn_rl_cycle/rl_data.py selfcheck
python pfn_rl_cycle/rl_data.py prepare --split train
python pfn_rl_cycle/rl_data.py prepare --split dev
python pfn_rl_cycle/rl_data.py prepare --split real
# root must first create results/design.json
python pfn_rl_cycle/rl_data.py prepare --split final
```

Implementation stopped after the CPU selfcheck/documentation. No bank generation, checkpoint loading or GPU job was run by the data agent.
