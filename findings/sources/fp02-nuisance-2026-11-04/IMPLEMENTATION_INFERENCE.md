# FP02 inference implementation

Implements `PLAN_v2.md` in `operators.py`. The implementation adds no trainable parameters, optimizer, checkpoint/bank access, GPU job, or historical source modification. Its numerical model is a Poisson working score with a finite centroid grid, independent Gamma nuisance distributions, observed-edge BP, and nonedge node-product mean field. It does not claim exact Bernoulli, exact degree-conditioned, continuous-centroid, or all-pair Bayesian inference.

## Public contract

```python
prob, info = nuisance_bp(
    src, dst, nodes, context, classes, observed, log_likelihood, log_weights,
    variant='dc_update', iterations=24, damping=.5, block=6, details=False)
```

- `nodes` is integer N, `classes` is integer K≥2. `src/dst[D]` are binary-adjacency directed edges: both orientations exactly once, no self-loops. The input order is retained; arbitrary directed-edge reordering is allowed. Validation rejects duplicate, asymmetric, or invalid edges.
- `context[N]` contains −1 or a known class. `observed[N,F]` selects feature factors. `log_likelihood[N,F,R]` is the literal class-shared Normal log likelihood; only selected entries are read. Unobserved values may be poisoned without changing inference. `log_weights[R]` or `[F,R]` provides finite grid-prior log masses and is normalized once. A second Normal prior is never added.
- Device/dtype follow floating-point `log_likelihood`. Index/context tensors are moved to its device. No CPU model import, dense N×N messages, role mixture, or neural feature head occurs here. The fixed C_global caller constructs the same GH41 table for every arm.
- The result `prob[N,K]` sums to one, with exact one-hot context clamps. A zero-source case retains a zero derivative connection to selected (empty) likelihood entries for future training callers; it performs the actual graph inference rather than a cached graph-only shortcut.

`damping` is the **new-proposal fraction**. Both message log values and nuisance distribution parameters use the same fraction. The reviewed execution fixes it to .5. Generic shorter/longer `iterations` and `block` are available to numerical checks, without a special tree solver.

## State and updates

For a directed edge i→j, `graph_messages[D,K]` is indexed by the target label. `reverse[D]` indexes j→i. Its finite gauge is zero mean over target classes. Each proposal is the full generic K×K sum-product update, not the earlier Potts shortcut:

\[
u_{i\to j}(b)\leftarrow\log\sum_a\exp\{h_i(a)-u_{j\to i}(a)
+L_{ab}-t\bar\theta_i\bar\theta_j R_{ab}\}.
\]

The cavity is normalized and context-clamped before summation. Here `L=digamma(shape)-log(rate)` and `R=shape/rate` come from the same symmetric Gamma B state. The density offset is observed mean degree/(N−1), or zero for N=1/empty adjacency. Class-independent A log(t theta_i theta_j) is omitted only from class messages; observed degree is retained in the theta Gamma shape.

The nonneighbor volume at a step is

\[
v_i(b)=\sum_j\bar\theta_j q_j(b)-\bar\theta_i q_i(b)
-\sum_{j\in N(i)}\bar\theta_jq_j(b).
\]

The class field subtracts `t*theta_i*(v_i @ R.T)` once and adds incoming observed-edge messages, population Elogpi, and centroid-to-node messages. Observed edge factors already contain their complete negative Poisson term. The nonneighbor volume is computed by sparse scatter and N×K storage. Cancellation-roundoff below zero is replaced by zero, with its raw minimum exposed; no intensity or posterior parameter is clipped.

All graph/centroid proposals in an inner step use the previous messages and node beliefs. After their simultaneous damped replacement, node probabilities are recomputed using that step's saved nonneighbor field. There is no extra final field update. The final node belief and final observed-edge belief are computed from the **same** saved field/cavities and current Gamma state. Finite-iteration edge marginal consistency is not asserted.

For an observed cell e=(i,f), feature messages have shapes `to_node[S,K]`, `to_centroid[S,K,R]`. Their updates are the existing class-gated finite-grid factorization, reimplemented for a single role: centroid cavity excludes its own vector message; the label cavity excludes its own active-class scalar message. Node→centroid messages use a zero-mean log gauge over grid states. The centroid prior enters once. These proposals run jointly with graph BP at every inner iteration.

## Nuisance statistics and schedule

At completed steps 6,12,18, and only for update variants, compute qedge from the saved cavity state. `gamma_statistics` constructs all sufficient statistics from this one qnode/qedge state, the **old** theta mean, and the **old** B mean. All B/theta/pi proposals are then formed simultaneously:

\[
\begin{aligned}
a^*_{ab}&=5+M_{ab}, &r^*_{ab}&=5/B^0_{ab}+tV_{ab},\\
a^*_{\theta i}&=1+d_i, &r^*_{\theta i}&=1+t I_i,\\
\alpha^*_a&=1+\sum_{i:\,context_i=-1}q_i(a).
\end{aligned}
\]

Each new distribution parameter is `(1−damping)*old+damping*proposal`. A previous posterior is never the base of a fresh evidence addition. The old directional messages survive block boundaries. The final six-step block uses the third updated state, and there is no unused fourth update. Gamma priors and fixed density set a scale convention for the otherwise confounded theta/B product; no theta renormalization/rescaling is performed.

`poisson_fixed` makes zero nuisance updates, keeps theta=1, affinity at its Gamma prior, and uniform population. `sbm_update` updates B and population three times, keeps theta=1, and records theta proposals only for audit. `dc_update` applies all three proposals. Population counts exclude context, but graph exposure and degree statistics include every node. Population unary is Elogpi, while reported population is alpha/sum(alpha).

`gamma_statistics(src, dst, node_belief, edge_belief, theta_mean, affinity_mean, context)` is a public check helper. Here each undirected edge is supplied **once**, in either orientation, and its `edge_belief[U,K,K]` label axes match the supplied endpoints. Its outputs are:

- `edge_count[K,K]`: unordered observed-edge M.
- `pair_exposure[K,K]`: theta-product-weighted unordered V, split into `nonedge_exposure` and `edge_exposure`.
- `population_count[K]`: unweighted unknown-node counts.
- `theta_interaction[N]`: I_i=sum_{j≠i} theta_j E_qij[B], without density t.
- `class_volume[K]`: all-node theta-weighted class volume; `nonneighbor_volume_raw_min` audits subtraction roundoff.

Counts/exposures use a symmetric full K×K storage convention: diagonal H_aa=q(a,a); off-diagonal H_ab=q(a,b)+q(b,a) is stored in both (a,b) and (b,a). Sum the **upper triangle** to recover the total, never the full matrix. Nonedge exposure uses weighted N×K products with self/neighbor subtraction. Observed-edge exposure uses the edge pair belief instead of node product. Theta interaction uses nonneighbor node products plus each observed-edge pair expectation. No dense N×N×K tensor is used.

`nonneighbor_field(src,dst,node_belief,theta_mean,affinity_mean,density_offset)` is another public check helper, taking symmetric **directed** edges. It returns positive penalty[N,K], nonneighbor volume[N,K], raw minimum. These edge conventions intentionally differ and are explicit in both docstrings.

## Audit outputs

With either details setting, `info` contains variant/schedule, `nuisance_updates`, theta shape/rate/mean[N], affinity prior/current shape/rate/mean/meanlog[K,K], population alpha/mean/unary[K], degree[N], density offset, final fresh statistics, and a three-entry `update_history` for update arms. Each entry contains `after_steps`, `before`, `statistics`, `proposal`, and `after`; all recorded proposals refer to that entry's old snapshot. Final fresh statistics are diagnostic and cause no fourth update. These outputs are O(N+K²) per snapshot.

`max_rate` is `t * product(top2(theta_mean)) * max(affinity_mean)`: the largest class-conditional pair intensity over distinct nodes and unconstrained label assignments, hence an upper bound on a posterior-averaged pair rate. It is **not** the maximum posterior-averaged rate. `rate_diagnostics` also reports the observed-edge q-weighted maximum/mean rate, fraction above 1, `1-exp(-max_rate)`, theta maximum, and affinity maximum. All diagnostics are computed without clipping; these are binary-Poisson saturation diagnostics, not calibrated probabilities of graph fit.

With `details=True`, additionally return final node/undirected-edge beliefs and endpoint order, directed src/dst/reverse, sparse graph and centroid messages, centroid beliefs, final field, nonneighbor penalty/volume, observed-cell indices, normalized log prior, `field_node_belief` (the actual previous q used to produce the final penalty), and `field_theta_mean`. Thus checks need not incorrectly reconstruct the last field using the final q.

## Verification performed by this implementation agent

Tiny CPU float64 in-memory smoke: N=5,K=3,F=2,R=5, six directed edges, three observed rows. All three variants had finite normalized probabilities, exact context clamps, finite feature gradients, correct nuisance count 0/3/3, and unordered edge-count sum 3. For both theta=1 controls, exposure upper-triangle sum was 10=N(N−1)/2. DC means remained positive, including a degree-zero node (mean approximately .4885). Empty-edge inference had density/max-rate zero and theta means all 1. These are local smoke checks, not a replacement for the separately assigned independent correctness suite. No training or development prediction was run by this agent.

A second tiny float64 boundary smoke (R=7) checked every variant: zero-source versus one source per column differed by at most 2.23×10⁻¹⁶, zero-source likelihood gradients were exactly zero, unobserved likelihood NaN poison preserved predictions bitwise, and exposure upper-triangle sums equaled `((sum theta)**2−sum(theta**2))/2` with zero numerical error. The independent suite separately checks the numerical equations; its results belong to its own report. The implementation source is now held unchanged pending that suite.
