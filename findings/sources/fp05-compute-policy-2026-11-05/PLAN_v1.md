# FP05: reinforcement learning for missing-graph inference computation

2026-11-05. User authorized the experiment. Objective remains missing-feature node classification with a frozen trained model on real target graphs. First pilot is homogeneous/homophilic-oriented; heterogeneous schema transfer is NOT tested. This plan precedes substantive implementation and scored data. Three reviews and revisions are required before execution.

## Concrete operation and limits

Start from the actual released GraphPFN1.3 graph adapters and LimiX16M checkpoint (pinned upstream3b9b115), never a new 19k scratch predictor. Keep all original12 Transformer blocks. At layers2,5,8 (zero-based), after the normal wrapper forward choose a binary action: 0=continue; 1=apply that layer's same pretrained residual graph conv AND residual graph MLP once more. Three decisions produce8 traces. All-zero trace must match unmodified GraphPFNLight exactly with paired RNG. Number of extra conv+MLP applications is cost0..3. This changes computation before output, not a choice among LP/PFN final probabilities. It is not literal internal-PCFI routing, node-frame sheaf or the previous subgraph selector.

All inputs use literal PCFI100/alpha.9 on observed-only RMS-normalized values, with observation clamping, ±10 represented-value clipping fixed for every arm. PCFI is a shared strong front end, not a new imputer. No hidden value or query label is available to policy or predictor. Also retain frozen native-NaN GraphPFN, PCFI+GraphPFN and LP as references. A successful adaptive model here would be an RL controller+supervised-adapted PFN; do NOT claim backbone parameters receive policy gradients if they only receive CE.

Policy pi_phi(a_t|state_t) is a shared MLP32, independently seeded but exactly same initialization in the two learned-policy arms. Inputs: t/3, observed node/cell fraction, context count/N, logN, logF, mean degree/N, and permutation-invariant summaries of current label-token states (mean norm, variance, context-vs-query normalized mean contrast and within-context same-class vs different-class cosine; use explicit validity flags for insufficient support). Do not feed raw class IDs. Hidden states read by the policy are detached in ALL arms; graph-adapter CE gradients remain live. Include static statistics and enough state to distinguish action prefixes. Query label used only externally in training loss/reward, never evaluation routing. Additional observations cannot be requested.

All GraphPFN graph adapters are trainable; LimiX backbone/feature embedding/y-decoder frozen. No new mask embedding/refiner. Same starting public checkpoint in all arms. Policy-only and backbone-only effects are reported honestly. Evaluate policy deterministically by sequential argmax, not test-oracle best trace.

## Arms and objectives

Four trained arms,3 seeds11/12/13, same bank/schedule/128 updates and8 trajectory presentations per update:

1. fixed: a=000, original GraphPFN continued CE. Eight paired forward presentations per update (distinct pre-fixed feature RNG seeds, same seeds used by all arms).
2. uniform: uniformly draw8 traces per update, CE-adapt adapters, no learned policy; inference fixed uniform probability mixture over8 traces (charge8 forwards), plus separately report train-only best-global trace using train calibration predictions (charge its selection).
3. direct: enumerate all8 traces, minimize exact expected (CE + lambda*cost) under their policy probabilities. Adapters receive the policy-weighted CE, detached policy weights; policy learns by direct differentiation through the trace probabilities, detached hidden read. Entropy regularization as below. Inference sequential greedy.
4. grpo: sample8 traces with replacement from current policy; adapters receive mean sampled CE. Policy receives group-relative advantages from R=-CE-lambda*cost, clipped GRPO/PPO-style one on-policy update. Same hidden-detach convention. No claim this is a new RL algorithm. Inference sequential greedy.

Lambda=.01 NLL per extra adapter application; fixed before data. Policy entropy coefficient .01 for direct and GRPO; AdamW policy lr1e-3, adapter lr3e-5, weight_decay.01, warmup16/cosine floor.1, clip1. Query CE only for backbone. GRPO rewards detached; A=(R-meanR)/(population_sd+1e-6); ratio=exp(logpi_new-logpi_old.detach()), clip.2; single optimizer update means initial ratio1, no fake multi-epoch reuse. For proper old-logp freeze full trajectory conditional probabilities at sampling. New training stochasticity may cause branch differences: force model.eval() (grad enabled), no dropout and paired per-episode feature-position RNG across traces. Report advantage-zero fraction and policy entropy/cost. Cost is actual extra adapters, not total GPU latency proxy.

The direct expected-gradient arm is deliberately strong: for8 traces exact enumeration may be better than RL. Difference in trajectory distribution/variance is part of the comparison. Same updates and trajectory count do not prove equal wall time; log actual forwards/backwards, unique traces, latency and memory. Use one GPU worker and FP16 with finite-gradient checks/recorded scale retries. No target-graph optimizer updates.

## Tasks and masked observations

Use official GraphPFN prior and retain class-dependent features/graph generation. Fixed restricted size N128..192, F8..12, K2..4, degree4..12, graph sampler minimum child16. Any invalid structural draw retry is deterministic/logged; never reject by model performance. Sample256 train base graphs,8 dev,96 final after scientific source lock. Train base distinct from all earlier seed ranges (train51050000,dev51060000,final51070000). Context k1/3/5 per realized class sampled from official training segment; query16 from remaining nodes, include observed-query and feature-free-query episodes in equal proportions. Whole-node rates0/.90/.95/.995 and clustered-node .95/.995, cell.95. Train schedule samples graph/setting with independent fixed RNG; none/shuffle additional eval controls. Do not retry masks to force sources. Include source0/1 and report counts. Half tasks use context/source-disjoint restriction and always-missing query, half have natural mask/query availability. Full means actual full observations for natural half, all eligible nonquery otherwise.

Final96 graphs: same-size48 vs heldout N256/F16 shape48; contexts rotate k1/3/5; source-disjoint balanced. Primary is node95/995 whole-node, query nodes whose own row is missing. Synthetic95/995 rates averaged and fixed3 train seeds averaged WITHIN base graph before paired bootstrap5000. Do not call shape holdout semantic schema or real-generalization.

## Primary and gates

1. GRPO vs fixed accuracy paired97.5% CI lower>0 AND GRPO vs direct lower>0; all96 graphs, same own-missing query subset, undefined cases counted/reported not resampled. NLL/Brier/cost separately.
2. Feature utility: disjoint final graphs with>=2 source nodes and>=1 own-missing query, GRPO observed−none AND observed−shuffle accuracy paired97.5% CI lower>0. Recompute all input scaling and PCFI per intervention. All matching query/context/graph preserved.
3. Full natural-query accuracy loss vs fixed must not exceed2%p. Confidence absence not rewarded. Compare group rewards within identical task/mask/context only; never group different missingness difficulties.

Primary failure stops this candidate under fixed budget. Preserve any SFT or adaptive-computation improvement separately. No new RL algorithm novelty claim even if successful. Candidate is not a family of tuning sweeps.

## Checks, real panel, budget

Before source lock: original000 forward/gradient parity on tiny dev; naive trace enumeration vs recursive tree probabilities sum1; direct expected policy gradient vs finite-difference and score-function exact expectation; GRPO gradient sign and no detach loss; class/node/feature-order sensitivity with paired random feature embeddings; all hidden label/value poison; missing0 source and singleton support; no query label appears in serialized state; checkpoint save/load. Check actual AdamW finite updates and policy effect on branch predictions. Dev smoke discarded and logged. No data-dependent chosen lambda/actions/steps after final lock.

Real reused development panel: Cora/CiteSeer/PubMed inherited2000-node,3 context/mask reps, rates95/99/99.5 plus none/shuffle95, all3 training seeds. Native physical observations read as previous FP04 bank; use a pre-fixed random projection to16 feature dimensions fitted without labels/hidden values to keep GraphPFN feasible on3090; disclose reduction and preserve masks/physical input identity. External historical public PCFI/FISF/LP results are references with different preprocessing/budget, not same-capacity causal controls. Primary synthetic run all arms regardless of intermediate scores; real panel after fits without tuning. Charge preprocessing separately and retain original manifest hashes.

Budget ceiling:12 fits×128 updates×8 traces=12,288 trajectory presentations, plus mandatory checks/dev smoke, final96×(7standard+4intervention)×12 fits and real panel. First resource smoke reports projected walltime/memory; may reduce planned update count BEFORE scientific lock if hardware makes it necessary, in reviewed revision. Heavy artifacts immutable/untracked; code/reviews/compact results local branch research/fp05-missing-compute-policy. Source lock includes all imported local files/public checkpoint hashes. Audit every saved output and obtain same three result perspectives.

## Prior art / why not assume novelty

Policy-GNN (arxiv2006.15097) already learns aggregation counts; RMS (arxiv2112.12845) already learns heterogeneous meta-paths. GraphPFN supports fine-tuning; our prior observation policy failed against fixed evidence paths and a context-frequency baseline. This pilot tests whether internal adaptive computation on an actual strong pretrained PFN adds anything over CE and exact expected-gradient training. Graph-GRPO (arxiv2603.10395/PMLR306 Zhu26b) is graph GENERATION via flow models, not node-classification GraphPFN. Do not use the shared acronym as novelty evidence.
