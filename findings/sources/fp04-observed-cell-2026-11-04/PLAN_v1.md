# FP04: End-to-end observed-cell GraphPFN

Date: 2026-11-04. User authorized an actual experiment following the missing-native PFN discussion. This is a fresh fully trainable model, not a frozen attention access edit or a Gaussian Factor repair. No novelty claim is implied by this pilot.

## Question

Can a jointly trained observation-set encoder and relational predictor use distant observed feature cells to classify feature-missing queries, better than the same trainable predictor with zero+mask or actual PCFI input?

Inputs are G, observed (node,column,value) cells, mask and context labels only. Query labels and hidden numeric values are evaluator/training targets, never model inputs. Model weights are fixed on all real graphs. No heterogeneous or heterophily success claim is targeted.

## Model and matched arms

Three arms share the same initialization, trainable parameter shapes and logical task schedule:

- observed: cell messages exist only at truly observed entries.
- mask: all cell locations are present, observed x or zero, plus the actual observed flag.
- pcfi: all locations have literal PCFI values and the actual observed flag. PCFI is recomputed after every intervention and clamped to observed values.

All encoders, graph blocks, feature-column messages and query head are learned from random initialization. Use width32, four shared recurrent reasoning rounds, a learned four-dimensional value basis, and no dropout. A shared value MLP reads normalized value, absolute value, square, observed bit and observed-only column count/moments. Whole columns with zero support remain explicitly marked. No global column/relation IDs are learned.

For every candidate class c, a shared node state receives a ternary seed indicating context-positive/context-other/unknown. It uses neighbor aggregation and context pooling. Column memory is formed by value-weighted sums of node states; node memory then reads column states through the same observed cell links. Both directions use trained projections and signed nonlinear value functions. This permits context→graph→source→column→other source→graph→query. Class scores share one function and are normalized over the supplied class vocabulary. Class chunks preserve exact forward semantics; no intermediate probability ensemble. Label/node/column permutations must commute with outputs.

Implementation may use sparse observed-cell reductions and feature/class chunking. Dense arms may use matrix multiplication of value-basis weights. Numerical checks must compare both computations on a small complete example. There is no BP, likelihood-ratio head, exact posterior claim, output gate or imputation loss.

## Fresh synthetic bank

Generate independent train/dev/final base-graph seeds in separate ranges. Train384 base graphs, dev12, final96; retain every draw without outcome-based rejection. Two equally weighted task families: (A) homophilic multiclass SBM with task-random dense class-feature centroids and per-column Gaussian/Student-t/nonlinear observation laws; (B) label-independent random graph with graph-smoothed latent features and random nonlinear per-node/neighbor label rules. Keep the complete class vocabulary even when a realized class is absent. Exact definitions are recorded in code before final creation; no generator role/parameter IDs enter inputs.

Train n64/96/128, f4/8/16 and K2/3/4. Heldout panel n160 and f6/12 changes shapes but is not unseen semantic schema. Context up to1/2/4 per available class; query16 sampled outside context before masking and deliberately feature-free. Source/context overlap varies, with a separate disjoint stratum. Whole-node90/95/99.5% and cell95/99.5% masks; include no-source draws and do not reject them. Training also includes a full/nonquery-visible setting. Real extreme missing is reported separately from synthetic masks.

Three train seeds11/12/13. Each arm768 AdamW updates, four task exposures/update, learning rate3e-4, warmup32 and cosine final0.1, clip1, weight decay.01. Same last checkpoint, no test-based selection or auxiliary loss. Total9 fits/6912 successful updates/27648 task exposures. Save realized seed schedules, losses, gradient checks, checkpoint hashes, wall times and peak memory.

Final primary is feature-free-query accuracy, average fixed3 train seeds and prespecified extreme whole-node settings/families within each base graph. Primary comparisons observed−mask and observed−PCFI require both paired97.5% CI lower>0. Additionally observed−none and observed−shuffled must have paired97.5% CI lower>0 in the declared source>=2 disjoint utility subset. Report unsupported source0/1 separately without removing them from primary. The no-observed/shuffled evaluations use identical queries/context/graphs; shuffle keeps mask and permutes only available source rows (or observed values within a column for cell masks). Interpret all four conditions conjunctively; do not swap primary to NLL if accuracy fails. NLL/Brier and per-family/mask/shape results are secondary. Bootstrap5000 resamples of base graphs, not nodes or training seeds.

## Real development transfer

Use Cora, CiteSeer and PubMed, three fixed context/mask repetitions,95/99/99.5% whole-node missing, plus none/shuffle95. Reuse established allowed context and query sets where possible to retain external comparisons; Cora/CiteSeer full and PubMed existing2000-node induced graphs. Native feature coordinates are kept; normalization uses only observed cells. Newly trained model weights are frozen. Do not learn a projection from hidden values. If real native PCFI is resource-intensive, cache it once with timing/hash; do not substitute zero fill. All three arms are evaluated with all three train seeds. Existing public GraphPFN/PCFI/FISF scores are comparable only after exact context/query/mask identity checks; otherwise recompute or mark unmatched, never silently pool.

Real graphs have already influenced research decisions, so these are development transfer results. Repetitions are not independent new graph datasets. Report all graphs/rates and fixed seed/repetition spread; no node-bootstrap generalization claim. A public PFN reference and restart LP remain strong external controls. Real transfer is run even if synthetic gate fails, but no additional training or architecture rescue follows.

## Preflight, lock and audit

Three independent reviews precede implementation. Require finite gradients through value/column/graph/head, hidden-value poison invariance, no query labels in input, node/class/column permutation, empty-source/context, singleton support, CPU/GPU agreement and chunk/dense reductions. Tiny development smoke training is discarded and counted. Lock source/config/train/dev/final bank provenance before scored execution; any actual fault is logged as repair, not hidden restart. Final predictions must be independently rescored, confidence intervals reproduced, and original locks/historical files preserved. Three independent result reviews precede the decision.

## Prior-art and limits

TabImpute/PartialVAE provide arbitrary observed-set conditioning; StructuralMissingness PFN provides direct incomplete-context label prediction; MA-PFN learns value/mask task priors; Ephris already uses feature compression and graph message passing. None establishes this pilot's novelty or success. Prior crossing, frozen feature-memory routing, completion and circuit variants were already tried and are not called new here. This pilot tests full learned computation and native observed-cell input. Small synthetic budgets and a finite generator cannot establish universal robustness or MNAR identifiability.
