# FP03 operators implementation and API

Implements the reviewed `PLAN_v1.md`/`PLAN_v2.md`/`protocol.json` in the new isolated `operators.py`. No I/O, pretrained model loading, bank loading, branch action, neural fit, training/evaluation job, or historical-source edit occurs in this module or was performed by its implementation agent. Imports sklearn before torch as required on Windows. FP01/FP02 feature equations were read and reimplemented directly; no ambiguous historical module import is used.

## Production API

```python
p, info = contrast_bp(
    src, dst, nodes, context, classes, observed, log_likelihood, log_weights,
    a, b, iterations=24, damping=.5, details=False)
```

- `src/dst[D]`: symmetric directed binary edges, no self-loop/duplicates. Each undirected edge appears in both orientations. Arbitrary input order is accepted and reverse indices are found by sorted keys, without an adjacency matrix.
- `nodes`: integer N≥1; `classes`: integer K≥2.
- `context[N]` or `[B,N]`: −1 or a context label. Batch length1 broadcasts. Held-node IDs, selection labels, full-context counts, and evaluator metadata are **not accepted** by this function.
- `observed[N,F]`, literal Normal `log_likelihood[N,F,R]`, and finite centroid `log_weights[R]` or `[F,R]` are shared label-independent inputs. Only observed likelihood entries are read; hidden NaNs are ignored. Grid prior masses are normalized once, without adding a second Gaussian prior. `F=0,R=1` is the graph-only calibration interface.
- `a/b[B]`, or scalars/length1 tensors, broadcast with the context batch. Every non-singleton batch length must agree. Device/dtype follow the floating likelihood tensor. The reviewed execution uses FP32; local numerical checks also use FP64.
- Output is always `p[B,N,K]`, even for B=1. Context outputs are exact one-hot. The caller chunks independent candidate/fold lanes at16; there is no public candidate-chunk argument. Internally, directed edge work is chunked at4096 edges. No dense N×N×K message array exists.

`candidate_table(device='cpu', dtype=torch.float32)` returns `(a[15], b[15])` in a-major/b-minor order. Values exactly follow the protocol. Fixed-edge index is10; graph-off index is4.

## Graph and feature state

`theta[N]=degree/sqrt(sum degree)`, with all zeros for an empty graph. For old beliefs q, compute

\[
v_i(c)=\sum_j\theta_jq_j(c)-\theta_iq_i(c)
-\sum_{j\in N(i)}\theta_jq_j(c),\qquad H_i(c)=b\theta_iv_i(c).
\]

The **signed H is added** to the node field. It is not subtracted as a Poisson penalty. Observed edges use the complete Potts score `a*1[label_i=label_j]`; no b term is applied to them. Neighbor sums are sparse scatter. Negative cancellation roundoff in volume is replaced by0, and the pre-correction minimum is recorded. Strengths/intensities/fields are not clipped.

Directional messages have shape `[B,D,K]`, with target label as the last axis. For i→j use `cavity=field_i−message_j→i`. The finite protocol grid uses the equivalent Potts identity

\[
\log m_{i\to j}(c)=\log\{1+(e^a-1)\operatorname{softmax}(cavity)_c\}.
\]

Context source labels are handled explicitly by `a*onehot(label_i)` before zero-mean log gauge and log damping. The safe bounded path is used when all a lie in[−10,10], including the reviewed grid [−log2,log16]. An extreme-a fallback uses `logaddexp(a+cavity_c, logsum_except(cavity))` and the same gauge. The two formulas define the same Potts factor; the fallback is a numerical implementation choice, not another candidate family.

The first implementation used prefix/suffix logsumexp for every graph update. A no-anchor FP32 smoke found class-index roundoff up to8.94×10⁻⁸ on an equal-class input. Before locking or any real prediction, the bounded Potts identity above replaced that path, preserving equal class values componentwise and improving the graph-off/m1 symmetry boundary. The centroid `logsum_except` remains unchanged and avoids subtracting a probability from1.

For S observed cells, centroid messages are `to_node[B,S,K]`, `to_centroid[B,S,K,R]`, with beliefs `[B,K,F,R]`. Each label/centroid cavity strictly excludes its own opposite message. Node→centroid messages use zero-mean log gauge over R. Literal Normal factors, including constants, enter exactly once per observed cell for its active class.

All graph/feature proposals read OLD messages and the MF field from OLD node beliefs, then are log-damped and replaced together. New node beliefs read NEW messages with that just-used MF field. There is no hidden extra final MF step and no cached/warm-start messages across calls, folds, candidates, or feature interventions. All calls start messages at0 and unknown labels uniform. Zero-source output retains a zero derivative connection through the empty selected likelihood, without reading hidden values.

`info` always contains a/b, batch/iteration/damping metadata, degree/theta, raw volume minima and approximation description. Its `sources` key is **S observed cells**, not the number of distinct observed rows; the caller's bank metadata remains the source-node count. With `details=True`, additionally return sparse graph/centroid messages, final node field/belief, centroid belief, normalized prior, observed cell indices, endpoints/reverse indices and context. `field_node_belief` is the actual previous q used in the final MF field; reconstructing that field with final q would instead perform an extra step.

The public check helper `nonneighbor_field(src,dst,q[B,N,K],theta[N],b[B])` returns `(signed_field, volume, raw_min[B])`. It expects symmetric directed edges.

## Balanced folds: split-only CPU helper

```python
folds = balanced_folds(context, stable_ids=None, classes=None)
```

Returns CPU tensors and integer counts. `context[N]` is the full allowed context vector, not query truth. `stable_ids[N]` must be unique; if omitted, original row IDs `arange(N)` are used. Under node permutation those IDs must travel with nodes, rather than being regenerated. They never enter BP as features. `classes` is optional for full K-length `class_counts` and range validation; pass it in production so missing classes and the C-empty case are explicit.

Let C+ be represented classes and m the minimum positive context count. Select the m smallest-ID nodes per class. Class slots are ordered by each class's minimum stable ID rather than numeric class name, preserving the held-node table under class relabeling. Fold t hides selected row t in all represented classes; it clamps only other selected nodes. All extras are hidden in every calibration fold. The caller uses the original full context for final prediction.

Dictionary keys:

- `fold_contexts[m,N]`, the only label tensor from this helper passed to BP.
- `held_nodes[m,C+]`, `held_labels[m,C+]`, `held_flat[S]` and `labels[S]`, flattening in fold-major/class-slot order, S=m|C+|.
- `selected_nodes/selected_labels`: flattened aliases of the same selected nodes/labels.
- `unused_nodes[E]`: extra original-context nodes, sorted by stable ID.
- `class_counts[K]`, `represented_classes[C+]`, `stable_ids[N]`, `selected_stable_ids[m,C+]`.
- integer `m`, `points`, `original_context_count`.

C-empty yields zero folds/points and empty represented classes. An unseen class remains a valid BP label but receives no fake calibration score. If m=1 every calibration context vector is all−1; graph-only inference is class symmetric, so weights must remain prior uniform. In the specified CS case, m=3,C+=14 gives42 held nodes/3folds/13 extras; every calibration fold has28 clamped labels, while final uses all55. The module does not assert a real bank identity or open that bank.

## Scoring and diagnostics

```python
weights, score_info = score_weights(heldp, heldlabels)
```

`heldp[G,S,K]` contains raw held-node probabilities in the helper's flattened order; `heldlabels[S]` is read **only after** forward prediction. Scores are computed in float64: clip every cell to[1e−8,1], row-normalize, take own-label log probability, average within each represented class, then average these class means. Candidate prior is uniform and score scale is fixed at1. Weights are `softmax(scores−log(G))`. No posterior calibration, independent-fold likelihood, stacking optimization, or effective statistical sample-size claim is made.

For S=0 scores are0 and weights uniform. Invalid/nonfinite/negative probability arrays or nonempty rows of zero total are rejected, rather than silently dropping candidates. Score info includes `scores`, per-class scores/classes/counts, `normalized_held`, floor, clipped/zero cell counts per candidate, entropy, ESS, prior KL, and `map_idx`. MAP is `scores.argmax()` with the first exact score tie, as required; it is not `weights.argmax()`, whose finite-precision softmax could introduce extra ties. This final metadata correction does not change primary weights or predictions.

`mixture_diagnostics(weights,a,b,candidate_probabilities=None)` returns float64 entropy/ESS/prior KL, weighted contrast means/variances and a/b covariance. Optional probabilities `[G,N,K]` add two explicitly different statistics: `mean_pairwise_total_variation` averages uniformly over candidate pairs g<h and nodes; `weighted_mean_tv_to_mixture` averages each candidate's TV from the weighted mixture using w. These are predictive dispersion, not uncertainty credible intervals. The runner may independently reconstruct these quantities in NumPy for audit.

## Local verification and freeze

Only generated tiny CPU inputs were used by this implementation agent:

- N=6,K=3,F=2,R=5,3 independent lanes with negative/zero/positive b: batched-versus-scalar agreement in FP64, normalized finite probabilities, and finite gradients with respect to likelihood, a and b.
- Zero-source versus one source per column prediction max difference1.11×10⁻¹⁶ in the initial tiny boundary smoke.
- Fold helper with counts[4,3,1,0]: m1/3points, extras masked; source-free inference yielded uniform candidate weights and ESS15. Empty context also produced finite prior weights.
- After the graph identity correction, a nonempty N=8,K=15 star with no label anchors gave **bitwise uniform1/K across all15 FP32 candidates**. Batch/scalar parity and finite gradients were rechecked after this numerical change.

The separately assigned reviewer reported preliminary independent NumPy MF/production agreement≤2.22×10⁻¹⁶, tree agreement≤2.22×10⁻¹⁶ and last-field direct-sum agreement≤1.33×10⁻¹⁵, with fold/poison/permutation boundaries passing. That preliminary message is not a final suite/PASS artifact; the independent checks report owns the final verification. The final operators source is now held unchanged pending those checks. No GPU smoke, real-bank evaluation, neural training, or branch changes were performed by this implementation agent.
