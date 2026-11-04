# FP03: Context-calibrated edge/nonedge contrast

2026-11-04. User authorized implementation of context-adaptive relational strength after FP02 failed. This is a small inference experiment using the existing learned C_global feature potential, not new neural pretraining, a posterior-identification claim or a completed novel PFN. No historical bank/checkpoint/source is overwritten.

## Hypothesis and changed operation

The old neural graph potential does not read context labels directly; FP02 instead updated affinity/population from its own wrong predictions. FP03 holds degree/opportunity fixed and uses held-context predictive quality to weight a prespecified family of conditional graph scores. It never selects full/edge per dataset or uses query labels to set weights. Homophily and nonedge reliability are separate; label-context support is sparse and not IID.

For symmetric directed edges, d_i is degree and v=sum_i d_i. Define theta_i=d_i/sqrt(v) when v>0, otherwise zero. The undirected pair score is

    J_ij * 1[y_i=y_j],
    J_ij = a for observed edges;
    J_ij = b*theta_i*theta_j for nonedges (excluding self).

This is a finite conditional Potts working score on the given graph, not a normalized Bernoulli graph likelihood. Edge a and nonedge b are not tied to a single homophily estimate. Pair opportunity may exceed1 and is not called an edge probability. Fix 15 candidates: a in [-log2,0,log2,log4,log16], b in [-2,0,2], Cartesian product with uniform prior. a=log4,b=0 is the common fixed edge control. (0,0) is an uninformative graph control.

Inference: sparse observed-edge cavity BP plus product mean-field nonneighbor field,24 synchronous steps/log damping.5. Nonedge field is b*theta_i*(sum_j theta_j p_j - self - neighbors), added once; observed edges use full Potts a and do not get this term. Class prior is uniform; no nuisance class-count/affinity update. This contrasts with FP02 but does not guarantee removal of all class bias, loopy errors or overconfidence. Frozen C_global_s11 Normal/GH41 observed-cell factors use the same strict centroid cavities as FP01/FP02. No numeric completion is made.

## Context calibration

For each context i hide its label from every input/initialization and run each candidate with all other context clamped, on graph only (all features off). Keep predictions at i only. No query truth is passed. Let loss_g be the mean of per-class means of held-context log probability; thus class-balanced design is not treated as representative population. Define weights w_g=softmax(log(1/15)+loss_g). The unit weight on mean log score is fixed before outcomes, not a temperature sweep. It is a conservative generalized predictive weighting, NOT exact Bayes posterior. Context folds share nodes; the number of paths is not an IID sample size. With zero context, use prior weights. All held-context outputs, score vectors and candidate weights/ESS/entropy are saved.

Calibration is graph-only so weights are common across none/95 feature interventions. The final full-context run jointly updates node labels and centroid messages for each candidate, then averages candidate probabilities with those weights. This is an inference mixture in an explicitly defined conditional family, not a learned joint feature-conditioned hypothesis posterior. Do not attribute a common none/95 weight to feature-driven mechanism adaptation.

Controls (same candidate predictions): uniform prior mixture; MAP candidate by the same held-context score (ties lowest candidate index); fixed edge(a=log4,b=0); graph-off(a=b=0); adaptive mean-score mixture. MAP is diagnostic, not a data-specific replacement for the primary mixture. Add old full/edge C_global, LP harmonic/restart, public PFN and PCFI/FISF-PFN on identical real queries. Compare the adaptive mixture with prior mixture and fixed edge to isolate weighting versus model family. All candidates and controls are retained regardless of results.

## Fixed computation budget and phases

No new neural fits/optimizer updates. One CUDA worker, float32. Batch candidate/fold axes in memory-bounded chunks (mathematically independent; fixed chunk size16 initially) with one common24-step operation. Save final candidate probabilities and held-context probabilities so every mixture can be independently reconstructed without rerunning BP. Do not allocate dense N*N*K messages. All synthetic/real calibration folds exclude their own seed before computation. Topology/features/centroid priors may be shared; label-dependent messages may not be cached across folds.

Numerical/dev checks precede source lock: tree oracle for b=0, tiny dense nonedge operator vs independent dense implementation and exact enumeration (approximation errors reported), edge/nonedge signs, reverse-cavity and self/neighbor exclusion, singleton/empty graph, class/node/column permutation, candidate/fold chunk parity, context label poison in its own fold, hidden features, own-feature-missing queries, probability normalization, and inherited file hashes. Feature0/1 exchangeability and finite gradients are retained. Numerical correction is allowed before lock with provenance; no scientific tuning after fixed evaluations.

First actual locked run: existing five real graph rep0/seed11 none/95 ten episodes. These are reused development, no independent real CI. Feature C_global_s11 and70 old references are the FP02 manifest/hashes; no graph-specific selection. Calibration has sum_C=139 held nodes across five graphs,15 candidates=2085 graph-only branch forwards; final150 candidate episode forwards. Folds may be batched. Report cold/warm full end-to-end CPU->GPU->CPU time, calibration cost versus final conditionals, peak memory and failure costs. All calculations are inference, not target neural training.

Development gate for the adaptive mixture: none Physics accuracy>=old_full+20pp, CS>=old_full+5pp; CiteSeer accuracy>=old_full-1pp and NLL<=old_full+.05; five-graph mean BAcc>=old_edge and mean NLL<=old_edge+.05. Additionally mean accuracy>=prior mixture+1pp and>=fixed-edge; mean NLL<=either+.01. These criteria are feasibility only, not paper success. All five graphs same rule, no setting changes after results.

If that gate passes, a separately specified fresh synthetic factorial bank is run with the locked operator: homophily, degree heterogeneity and semantic-class coarsening crossed, contexts1/4 per class,95/99.5 node missing, disjoint context/source/query. It tests whether observed context supports the right contrasts, including ambiguity, and whether feature utility survives. Do not generate/tune the final synthetic bank until its precise generator and statistics are reviewed/locked. If real development fails, do not spend new neural training or rescue by a/ b/ score-temperature grid changes. Close the scope and retain any supported effects.

## Scope and prior-art boundary

Conditional Potts scores, composite/predictive likelihood, Bayesian/Gibbs model averaging, held-out calibration and graph BP are existing techniques. FP03 only tests whether context-derived contrast adaptation helps the existing learned feature inference. Strong LP/classical baselines must stay. No unseen-schema/heterophily-universal/novel learning-principle claim follows. Real graphs remain the current homophily project scope; negative-a candidates test conservative role uncertainty rather than establish arbitrary heterophily generalization.
