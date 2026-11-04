# FP04 revision: fixed computation and estimands

This revision supersedes conflicting details of PLAN_v1 before implementation or scored data generation. Three independent review perspectives were obtained; their documents and RESPONSE_v1 preserve objections. No model/data results were inspected to make these choices.

## Exact shared operator

Let A_if=M_if for observed, A_if=1 for mask/pcfi. Dense-mask is allowed to learn to ignore hidden entries but is NOT hard-gated by M. Included counts n_i=sum_f A_if and n_f=sum_i A_if use positive counts, never signed weight sums; divide by max(count,1) and zero empty aggregates. Observed x=0 is an included cell. Dense and sparse are evaluation strategies for this same formula.

Observed-only RMS scaling divides raw observed values by sqrt(mean_observed(x²)), with scale1 for empty/allzero columns. Clip normalized visible values to[-10,10]. For all interventions recompute scaling, counts, moments and PCFI. None is an allzero M, not values0 with M1. Mask zero-fills in normalized coordinates. PCFI uses the existing literal100-step/alpha.9 implementation on those normalized visible values and clamps true observations. Precompute outside training with isolated NumPy/Python/Torch RNG state. Its extra preprocessing cost is retained.

Prelock numeric clarification: all represented value inputs, including PCFI completions, are clamped to[-10,10] immediately before the common value MLP. Raw actual PCFI caches remain intact. Report out-of-range fraction/max magnitude; this bounded representation loses extreme completion magnitude and is not an imputation-algorithm change. It was fixed before training or performance inspection.

The value MLP phi has input [v/10,abs(v)/10,v²/100,M,observed mean,observed second moment,log1p(observed count)/log1p(N)] with moment inputs bounded through tanh, and output4 via tanh. It does not receive class, node, column IDs or hidden values. Four separate shared projections P_r,V_r are permitted. In class channel c, the initial node state is a shared ternary positive-context/other-context/unknown seed embedding plus observed-only node/count/degree/static class-support statistics. There is NO dynamic global context or global node feature pooling. Static context counts are allowed and value-independent.

For each of four shared rounds, first use a shared residual graph MLP of node and row-normalized neighbor states. Then:

u_cf = (sum_i sum_r A_if phi_ifr P_r(h_ci))/max(n_f,1)

z_cf = LayerNorm(column_MLP(u_cf, observed_column_stats))

v_ci = (sum_f sum_r A_if phi_ifr V_r(z_cf))/max(n_i,1)

h_ci = LayerNorm(h_ci + .25 read_MLP(h_ci,v_ci)).

The shared scalar head produces z_ic, then one softmax across all supplied K classes. No per-chunk class normalization. No dependence between class channels except final normalization. Feature chunks use complete-axis counts. Dense reduction/matrix product and sparse/scatter must agree in forward and gradient. Full-M1 observed/mask/PCFI inputs agree exactly apart from tolerance; PCFI full rows are clamped. Include a two-component bridge derivative test: context→graph→sourceA→shared feature-column→sourceB→graph→query; cross-component derivative must vanish if column exchange is disabled. This verifies an implemented route, not predictive utility or novelty.

## Fixed synthetic generator

All tasks use one shared rule per graph and retain supplied K even if a class is absent. No outcome rejection or quantile relabeling.

A (centroid/community): y iid uniformK; d chosen4/8/12; eta U(.15,.8); unordered Bernoulli edge probability d/(N−1) times 1+(K−1)eta for same class,1−eta otherwise, clipped.98. Task class-feature centroids iid N(0,s²), s U(.6,1.5). Column law sampled uniformly from Gaussian mean, scaled Student-t3 mean, zero-mean class-dependent variance, Bernoulli logistic. Noise scale U(.35,.9); variance sigma=exp(.4*centroid) times noise scale; Bernoulli logit=2*centroid+b, b U(−3,−1). Continuous columns receive independent random sign and lognormal scale. These law/parameter/orientation IDs are not inputs.

B (field/nonlinear): ER graph with d4/8/12 independent of task labels. Draw iid Gaussian Z0 and iterate Z←(1−rho)Z0+rho PZ five times, rho U(.4,.85), P row-normalized A+I. Per-column transformation is identity, cubic/3 or tanh, chosen uniformly; add Gaussian noise sd.1. A single task-random MLP16 on [X,PX], with scaled iid normal weights and bias sd.2, produces K logits; add iid Gumbel noise scale.2 then argmax for y. No rule ID enters input. This family does not force semantic class=graph community.

Train384 base graphs: two families alternate; N sampled128/256/512,F4/8/16,K2/3/4. Dev12 is separate. Final96 is factorial: shapes(256,8)/(384,12) ×K2/3/4 ×families A/B ×8 independent graph repetitions. Shape384/12 is heldout shape, not semantic schema. Final generator seeds are separate from all train/dev ranges and set in protocol before training.

Each base graph has fixed context up to1/2/4 available labels/class and16 query nodes sampled outside context before masking. Odd/even repetition determines context/source overlap allowed vs disjoint, balanced within every final factorial cell. Queries never have features. Whole-node masks use nested independent node uniforms among eligible nodes; rates90/95/99.5. Cell masks use nested cell uniforms at95/99.5. In disjoint stratum context is additionally ineligible. Include all-nonquery-visible and none. Retain source0/1; do not retry masks. Train uses all seven settings, uniform scheduled sampling from the full bank, exactly four draws/update. The same precomputed schedule is used in all arms for a given train seed.

## Locked evaluation

Primary: all96 final base graphs, whole-node95 and99.5, mean fixed three training seeds and two rates within each base graph, with source0/1 included. Both observed−mask and observed−pcfi accuracy paired97.5% CI lower must exceed0. No threshold based on inspected effect size. Source0/1 and shape/family strata reported separately.

Feature utility: disjoint base graphs, retain eligible rate cases with at least2 observed source nodes; average eligible cases and fixed seeds within each base graph. Require observed−none and observed−shuffle accuracy paired97.5% CI lower>0. Report eligible graph/case counts and identify this as a different conditional estimand. If no eligible graph remains, the utility gate is undefined, not PASS. Source1 need not be invariant under this learned mixed prior. Shuffle permutes entire available rows for node masks and values within each observed column for cell masks; mask/query/context unchanged.

Evaluate standard/none/shuffle for primary masks in every arm, recomputing all preprocessing. Secondary90/full/cell masks standard only. An additional small bridge correctness test is not substituted for feature utility. CE is the only training loss; last checkpoint only; no target updates. Training9 fits ×768updates ×4 tasks =27648 exposures, fixed random initialization per seed across3 arms. Equal parameters/updates/exposures are NOT an equal-FLOPs/walltime claim.

Real development: all3 seeds ×3 datasets ×3 fixed repetitions ×(95,99,99.5,none,shuffle95), each of3 arms. Cora/CiteSeer full, PubMed inherited2000-node induced graph. Recover native observed rows from immutable source inputs; preserve original mask/query/context ordering. Actual native PCFI caches must record source/config/hash/time; native columns are never reduced. Keep legacy public GraphPFN/PCFI/FISF and LP references only after exact identity validation; disclose preprocessing/model/budget differences. No new final-real generalization CI. All real results run regardless of synthetic outcome, without a rescue architecture. Gate failure stops this candidate under the locked budget, not all observed-cell PFNs.

## Compute and checks

Use one GPU worker, FP32 initially and no TF32. Inference class chunk1 and feature chunk256 are semantic-preserving resource settings, not tunable predictive parameters. Full native resource smoke before lock must report latency and peak memory without reading test scores. PCFI O(NF²) cost is separate. Model/checks/data files may be repaired before lock, with discarded dev smoke steps logged. Thereafter source edits require an explicit repair record; preserve prior artifacts. Schedule, final graph fingerprints, all nine checkpoints, predictions, scalar metrics, independent rescore and three result reviews are required.
