# Cycle FP01: projection-matched training and class-specific feature centroids

Status: proposal for three independent reviewers, not an implemented or validated method.

## Question

Can correcting the train/real RP16 interface and replacing one preferred-class column state with a class-specific centroid vector recover real observed-feature utility in frozen Factor inference? Preserve the original evidence that nonedge deletion helps Coauthors but hurts CiteSeer. Do not choose a graph arm per dataset after scoring.

Known limitation: original synthetic columns have one preferred class and identical other-class means; real inputs use RP16 of native features. This family is not closed under such projection. This is a mathematical mismatch, not established unique cause of poor feature utility. Earlier binary feature-law pilot did not test multiclass projection closure.

## Comparisons and training budget

- A: old preferred-class feature operator, unprojected synthetic train values.
- B: same preferred-class operator and parameterization, projection-matched synthetic values.
- C: projection-matched synthetic values, class-specific latent Gaussian centroid vector per column.
- C-fixed: identical centroid inference with globally fixed hyperparameters, no adaptive encoder; strong statistical comparison.

All use each seed's original frozen graph encoder/potentials checkpoint. Reset trainable feature modules, no graph parameter updates. Seeds11/12/13, same512 base graph/label/noise/mask/context tasks and task schedule,256updates×2tasks, query CE only. A/B feature initial state identical; C has different representation/parameter counts to report. Original trained Factor remains a separate historical reference, not mislabeled A. Train on the same original full graph-factor definition; evaluate full and edge-only in separate panels. This keeps graph training fixed and avoids doubling compute. No claim of equal FLOPs between architectures.

Construct raw16-column synthetic values from the same original preferred-class means/noise, retain their class centroids and labels only as evaluation/supervision-side generator variables. A consumes raw values; B/C consume raw @ R, with R an independent task-seeded16×16 Rademacher matrix divided by sqrt16. Apply the exact observed-only RMS/clipping and whole-node masks afterward. Projection changes covariance/SNR as well as mean shape; B−A estimates this whole train-interface change, not a pure single-cause correction. Evaluation uses paired raw and projected views plus new scalar/shape settings. Consider a richer raw native dimension only in a distinct later cycle.

## Candidate C inference definition

For each class c and feature f, latent centroid mu_cf has a shared Gaussian prior. Observed cell likelihood is Normal(x_if;mu_{s_i,f},sigma_f^2). Missing cells introduce no likelihood. Node class context is clamped; own-feature-missing query information reaches it through graph paths, not a nonexistent query-feature edge.

First implementation proposal: fixed Gaussian quadrature (11 points) for each centroid, avoiding stochastic variational labels/point pseudo-label fitting. The latent centroid grid is centered/scaled by observed-column statistics; a small shared feature encoder predicts location/scale/noise. These input-dependent hyperparameters define a conditional structured score, not a calibrated generative likelihood. C-fixed uses the same statistics/inference with fixed coefficients.

Factorize each cell by class: psi_icf(s_i,mu_cf)=Normal(x_if;mu_cf,sigma_f^2) if s_i=c, and1 otherwise. Their product over c recovers the desired cell factor. Sum-product graph/centroid cavity messages,24 synchronous iterations, log-space damping0.5; no query labels or true centroids in forward. Need a fully specified finite-iterate Bethe score including centroid priors and cell factors before role mixing. If that score is impractical or unreliable, reviewers should propose a simpler honest shared role protocol for A/B/C, rather than retaining a wrong partition formula.

## Mandatory correctness before training

Independent exact enumeration on tiny heterogeneous-state factor trees (node class3, centroid grid3/5), including context clamps, Normal normalization constants and role scores. Singleton-observation behavior, class/node/column permutations, hidden-value poison invariance, observed-row order invariance, finite feature gradients, frozen graph digests, no query truth in inference. Check quadrature versus high-resolution1D integration where possible. No equality claim between finite-loopy beliefs and exact posterior. Keep all source revisions before final lock.

## Evaluation and success

New synthetic development bank for repairs, then lock code and generate new synthetic final seeds with paired raw/projected tasks; own-feature-missing queries primary. Report all class counts and missing rates, not only successful subgroups. Final synthetic bank must not be used to retune.

Real reuse all5 homophilic graphs, same existing75episodes (95/99/99.5%, none, shuffle95). Cora/CiteSeer full graphs; PubMed/Coauthors previous2000node induced graphs. All real graphs are development evidence, never new independent holdouts. Both full and edge-only retained. No real gradient updates. Existing public PFN/PCFI/FISF probabilities and two fixed LP baselines hash/identity-checked and reused.

Primary: real per-graph/rate missing-query accuracy; alongside balanced accuracy/NLL/Brier, all three seeds and paired repeated masks. Feature-full minus no-observed and feature-alignment shuffle are mandatory; structure-only gains do not count as feature-inference success. C must beat B and C-fixed in meaningful feature utility/quality to support adaptive centroid inference; a gain only against A does not establish new inference contribution. Synthetic final statistical unit base graph; existing real repeats descriptive, no node bootstrap. Costs include inference and quadrature, report capacity and inherited graph-checkpoint cost.

## Iteration boundary

Three reviewers can amend this plan before data generation. Correctness/compute-only repairs use dev data and preserve originals. After final results, review them and begin a new explicit hypothesis/cycle if needed; do not rescue this cycle with post-hoc loss/head/hyperparameter sweeps or per-graph full/edge selection.
