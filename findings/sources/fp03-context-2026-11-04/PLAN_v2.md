# FP03 reviewed amendments: balanced context masking

This version supersedes the calibration, numerical scoring and computation counts of PLAN_v1 before implementation or any new inference. The conditional score,15-candidate grid,24 iterations/damping.5,frozen feature model,controls,10 real-development episodes and all primary gates remain exactly as v1. Both plan files enter the source lock.

## 1. Remove the single-label count cue

Ordinary one-label LOO on class-balanced context changes class counts4,4,4 to3,4,4. A global absence/imbalance effect can then identify the hidden class without learning useful connectivity. The primary uses balanced leave-one-per-class folds instead.

Let C+ be the classes actually represented in the allowed context; m=min_{c in C+} n_c. For each c select the m context nodes with smallest stable node IDs, ordered by that ID. Fold t=0,...,m−1 hides selected node t in every represented class simultaneously and clamps the other m−1 selected nodes in each class. All other context nodes are also hidden in that calibration computation, including extra nodes from classes with more than m. Each selected node is scored once. The full-context final predictions use ALL original context labels, not the trimmed calibration subset. These are context-only fold labels, not query truth.

Stable IDs are existing graph row IDs, used only for reproducible fold construction, never as numeric model features. Node permutation tests carry the IDs through the same permutation. Class relabeling must leave the selected node sets/folds unchanged. Selection by ID is deterministic rather than a claimed random sample and that limitation is disclosed. No label-dependent messages or full-context warm start may be shared across folds.

Actual manifest counts: Cora28,CiteSeer24,PubMed12,CS55,Physics20. CS has class counts[4,4,4,3,4,4,4,4,4,0,4,4,4,4,4]. Its calibration uses42 nodes,3folds,14representedclasses;13extra context labels are withheld from calibration only. The other graphs have m4. Total126 scored held-context nodes,19 balanced folds,285 graph-only candidate/fold branch forwards, plus150 full-context candidate episode forwards. These counts replace139individual folds/2085forwards. Missing context classes get no fake score; prediction still has allK classes, including CS class9.

Boundary: C empty→zero scores/uniform candidate weights; m1→each calibration fold has no label anchors and no features, hence uniform class predictions for every candidate from symmetric initialization. The implementation must verify this and preserve prior weights rather than inventing relation identification. A class represented zero times is not validated by this risk estimate. Balanced trimming/full-context shift remains a limitation.

## 2. Score convention and algorithm state

Save raw held-context probabilities P[g,selected_node,class]. For scoring only, convert to float64, clip each cell to[1e−8,1], then row-normalize (same inherited scoring convention). Let s_g be the mean over represented classes of the mean log probability of their own held labels. It is a log score (larger is better), not positive loss. Save s_g and weights=softmax(log(1/15)+s_g). Every weight must be reconstructible from saved arrays. This choice can reduce the effect of underflow but is not a calibrated Bayesian posterior or effective-sample-size estimate. Do not tune the floor or score scale after outcomes.

For each BP step compute nonneighbor mean field from OLD node beliefs, all graph/centroid message proposals from OLD messages and that same field, log-damp messages, then read node beliefs with the just-used nonneighbor field plus NEW messages. The final output uses this last field/message state with no hidden extra MF step. The b term is added once; observed edges never receive it. All candidates/folds use the same zero initialization and uniform class unary, except strict context clamps. Metadata distinguishes graph0/off/feature0 conditions.

## 3. Additional mandatory checks and diagnostics

Compare batch size1 versus16 (same probabilities/held scores/weights within FP32 tolerance), direct brute-force nonneighbor sums, self/neighbor exclusion, negative/positive b signs, b0 independent tree oracle, finite differences, class/node/column permutation and hidden feature poison. Verify the actual fold generator: identical remaining counts across represented classes; its held label absent from context tensor and every label-dependent cache; extra CS labels all hidden; represented-class-only scoring; class labels permutable and stable node IDs consistently mapped. Test m1/no-anchors gives uniform candidate predictions and prior weights. An own-fold held-label poison test is performed on branch prediction before the label is legitimately used by the score.

Report connected-known support3/0/1/2/2, calibration subset/fold IDs,count table,held predictions,weights/entropy/ESS,weighted contrast moments,prior-to-weight KL, mean predictive disagreement among candidates, and per-candidate full-context metrics (evaluation only). None and95must share bit-identical calibration weights/manifest. No path is counted as an independent label observation. Entropy/ESS describes a finite predictive mixture only.

## 4. Reviewed scope

The work is an inference-stage context calibration of an inherited learned feature factor. It is not a new15-expert network, new gradient training, exact Bayes update, stacking implementation or universal heterophily method. Real data is reused development. A score/calibration improvement alone does not pass the primary accuracy/BAcc/NLL gates. No new final synthetic bank or neural fit is executed unless this fixed development gate passes and the separate protocol is reviewed.

All primary mean/gain comparisons use none and equal weighting of the five source graphs. The v1 NLL guard means BOTH adaptive<=prior-mixture+.01 AND adaptive<=fixed-edge+.01. Calibration weights are reused bit-identically for none/95. The primary is the adaptive mixture; MAP or individual candidate success cannot replace it. Derived controls count50 saved-score records, plus70 reused references=120 aggregate score records; the150 candidate conditionals and285 calibration branches are separately counted computation, not independent observations.
