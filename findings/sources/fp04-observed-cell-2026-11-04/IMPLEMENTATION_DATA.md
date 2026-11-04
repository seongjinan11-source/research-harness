# FP04 data implementation contract

`data.py` prepares immutable NumPy episodes explicitly. Import does not load raw real data, generate banks, execute PCFI, or initialize CUDA. The root owns every actual prepare invocation, source lock, and experiment. Existing sources, banks and predictions are read only.

## Public API and artifacts

- `cfg()` reads the frozen `protocol.json`.
- `sha(path)` streams SHA256; `dump(path,obj)` creates UTF-8 JSON exclusively and refuses replacement.
- `prepare(split='train'|'dev'|'final'|'real')` refuses an existing **or partial** split directory. It writes `bank/{split}/manifest.json` only after all episodes and checks succeed. A failed partial split remains for investigation; it is never silently resumed or replaced.
- `rows(split)` returns `manifest['episodes']` after checking the protocol hash.
- `read(row)` validates the NPZ SHA and returns `(inp,query,y)`. The exact input whitelist is NumPy `x:float32[N,F]`, `observed:bool[N,F]`, `pcfi:float32[N,F]`, `src/dst:int64[E]`, `context:int64[N]` (`-1` unknown), and Python `classes:int`. Query IDs/truth are separate. Private stored `context_ids/labels_context` preserve historical reference ordering and never enter this whitelist.
- `normalize_visible(raw,observed)` returns only normalized visible values. `actual_pcfi(x,observed,src,dst,return_info=False)` returns the dense imputation, or `(imputation,info)` if requested. `shuffle_visible(raw,observed,seed,whole_node=True)` exposes the value-only intervention.
- `external_records()` validates the real reference-index SHA and returns the identity-checked external records, including absolute NPZ paths for independent rescoring.

Commands (the root invokes preparation):

```text
python observed_cell_cycle/data.py prepare --split train
python observed_cell_cycle/data.py prepare --split dev
python observed_cell_cycle/data.py prepare --split final
python observed_cell_cycle/data.py prepare --split real
python observed_cell_cycle/data.py selfcheck --pcfi
```

Each row includes `base_id,split,stratum,family,shape,repetition,setting,variant,classes,sources,disjoint,path,sha256`, plus source/query/context/graph fingerprints and support diagnostics. A final stratum is `family_shape_K{k}` (12 cells); final `shape` is `matched` or `heldout`. `variant` is `standard` or `shuffle`. Real rows have `base_id='{graph}_r{rep}'`, `graph`, `mode`, `setting=mode`, and `variant='standard'`; the inherited real shuffle is encoded by `mode='shuffle95'` and is not applied twice.

Expected counts: train 384 bases × 7 settings = **2688** episodes; dev 12 × 7 = **84**; final 96 × (7 standard + 2 node-mask shuffles) = **864**; real 3 graphs × 3 repetitions × 5 modes = **45**. Each base has exactly one `none` row. Root training draws 3072 presentations/seed uniformly from the 2688 episodes; this is not one pass or 3072 distinct episodes. All three arms use the same root-owned schedule and cached arrays.

## Synthetic generation and RNG

All randomness uses `default_rng(SeedSequence([base_seed,stream]))`: 0 shape, 1 centroid-family labels, 2 graph/degree/eta, 3 feature law/values, 4 nonlinear label rule, 5 context/query, 6 whole-node mask, 7 cell mask, 8 shuffle. There is no global reseeding, rejection, class balancing, quantile relabeling, retry for zero sources, or conditioning on outcome quality. Supplied K is retained even for absent classes.

Train uses `start_seed+i`, alternating centroid/field family, repetition `i//2`, and independently draws N/F/K from the declared lists using stream0. Thus each family has equal overlap/disjoint design counts. Final order is shape, K, family, repetition0…7; its 2×3×2×8 factorial gives 96 bases. Dev uses one base per factorial cell with repetition `cell_index//2`; each family has three overlap-allowed and three disjoint bases. Stream5 first draws the context cap from 1/2/4, independently permutes each available class's IDs and takes up to cap, then permutes context IDs. It samples 16 query IDs uniformly without replacement outside context **before any availability mask**. Query selection does not use feature values or query-label quotas.

Centroid family: y iid uniform K; d uniformly 4/8/12; eta uniform .15… .8. Each unordered edge uses the declared capped Bernoulli SBM probability. The task centroid SD is U(.6,1.5), noise SD is U(.35,.9). Column laws are uniform over Gaussian mean, `centroid + noise*Student_t(3)/sqrt(3)`, zero-mean Gaussian with SD `noise*exp(.4*centroid)`, and Bernoulli with logit `2*centroid+b_f`, `b_f~U(-3,-1)`. Independent signs ±1 and scales `Lognormal(0,.4)` apply to continuous columns only. The Student-t scaling gives unit variance before the task noise multiplier. Centroid/law/orientation values are audit metadata, not model inputs.

Field family: ER edges with d uniformly 4/8/12; row-normalized `P=A+I`; five `Z=(1-rho)Z0+rho*PZ` steps with rho U(.4,.85). Column transforms are uniform identity/cubic-divided-by3/tanh, followed by iid Gaussian noise SD .1. A task-shared MLP maps `[X,PX]` to K logits: hidden width16, tanh, W1 iid normal divided by sqrt(2F), W2 divided by sqrt(16), both biases iid N(0,.2²); iid Gumbel scale .2 then argmax defines labels. The graph, feature field and rule streams are independent. The rule and hidden feature values never enter the input whitelist.

## Masks, controls and denominators

`disjoint=bool(repetition%2)`. Eligible nodes exclude all 16 queries, and additionally exclude context for disjoint bases. `full` observes all eligible cells, not the query cells. `none` observes no cell. Whole-node settings use one base-specific node-uniform vector and thresholds .9/.95/.995; cell settings use a separate base-specific matrix and thresholds .95/.995. Therefore mask nesting is exact without enforcing realized source counts. Source0/1 are retained. In overlap-allowed cases context observation is possible but not forced.

Node shuffle permutes the whole available row values among the same available rows. The cell helper permutes each column's values among its observed cells. Masks, graph, query and context remain fixed. Scaling and PCFI are recomputed from intervened values. Two final shuffles use setting `node95`/`node995` and variant `shuffle`; their standard/none partners share base/query/context fingerprints.

`sources` counts nodes with any observed cell. Rows retain `observed_cells`, `source_bin` (0/1/2+), column support0/1/2+ counts, actual context-source overlap and source-class counts (audit only). The primary all-96 estimand includes source0/1. The separate utility estimand is disjoint cases with source>=2, with actual graph/case denominators calculated by root analysis; an empty conditional set is undefined. Preparation does not filter this bank for the primary or utility gate.

## Normalization and literal PCFI

Only observed cells enter float64 RMS statistics. Empty/allzero columns use scale1. Divide by the column RMS, clip to [-10,10], cast to float32 and explicitly zero hidden cells. Observed numerical zero remains a genuine included observation. Hidden NaN/finite poison does not affect normalization.

The unique-file import is `fisf_graphpfn/imputation.py`. PCFI runs on these normalized visible values with NaNs at hidden cells, the native F unchanged, symmetric graph/self-loops, 100 iterations and alpha .9. Whole-node masks use the source's structural distance option (identical source set in every column); cell masks use per-column distances. Every observed value is explicitly clamped after the source routine. Empty masks return the mathematically equivalent allzero result, with `empty_zero_shortcut=True`; nonempty masks invoke the actual implementation. This is not a proxy row-mean fill.

Python random, legacy NumPy and Torch CPU RNG state are saved/restored in `finally`. Already-initialized CUDA RNG states are also preserved; this CPU helper never initializes CUDA to obtain them. RNG isolation is not a claim about an external process. Timing contains native PCFI wall/CPU seconds, iteration/config/source SHA, and the F² correlation-matrix size; total prepare wall/CPU and summed cache time are in each manifest. Native PCFI still incurs its O(NF²) correlation operation separately from the model's feature-chunk computations.

The tiny CPU selfcheck verifies normalization hidden-poison behavior, zero columns, deterministic two-family construction, query/context masking, exact nested masks, actual tiny PCFI finiteness/clamp, and exact Python/NumPy/Torch CPU RNG restoration. It creates no bank, loads no real graph and runs no GPU job. Independent checks own broader numerical/model assertions.

## Native real recovery and frozen external references

The real preparation reads only the historical manifest/NPZ and immutable PyG raw sources. Cora/CiteSeer preserve full graphs; PubMed preserves the inherited 2000-node induced graph and original-node mapping. Source/raw/bank hashes are checked. Cora/CiteSeer replay the original context/source RNG using the stored forward seed, reconstruct shuffle row permutation, and verify query/context/source IDs. PubMed checks native values against raw `value_original_ids`, source IDs against `original_ids`, finite-mask identity and the induced graph edges. All three check native labels against stored context/query truth. Reprojecting reconstructed observed native values through the historical RP16 and RMS reproduces the old visible input to maximum error <=1e-5. That RP replay is an identity audit only: no projection enters any new arm.

Only the observed native rows reach normalization and PCFI. Original full hidden values, raw labels and query labels are used for reconstruction/support auditing only and excluded from read()'s input. All modes within a graph/repetition must have identical graph/context/query/truth fingerprints. Source files are rehashed after preparation.

Five frozen edge-benchmark reference arms are examined for every real episode: `lp_harmonic`, `lp_restart`, `public_graphpfn`, `pcfi_public_graphpfn`, `fisf_public_graphpfn`. With all files available this is **225** records. Each NPZ and sidecar hash, bank association, graph/repetition/mode/arm, class shape and exact ordered query/truth/context labels must pass. Saved probabilities are rescored into accuracy, balanced accuracy, macro-F1 (all K), NLL (clip1e-8 then row normalization) and Brier. Valid entries go to `bank/real/references.json`; absent entries are recorded separately as missing, never silently pooled. A changed/mismatched existing reference fails preparation.

`identity_pass` means matching graph, context, query/truth and physically recovered native observations. It does **not** mean equal processed features or model/training budgets. Public references use the legacy native raw/public scale-fill/PCA pipeline, whereas new arms use observed-only normalized full-native channels and PCFI on those coordinates. LP has no feature input. Each record and the manifest state this distinction. These repeatedly inspected real graphs remain development evidence with no new independent-real CI claim.

Root may preserve a byte-exact synthetic-builder snapshot while the real/helper portion is being completed. Such snapshot provenance belongs to BUILD_EXECUTION/source lock; the synthetic manifest retains the SHA of the actual builder used. No prepared split is overwritten to replace that execution record.
