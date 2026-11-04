# FP02 reviewed Stage0 specification

This supersedes PLAN_v1 before implementation. Three synthesis reviews are METHOD_REVIEW/INFERENCE_REVIEW/EXPERIMENTS_REVIEW in candidate_synthesis_20261104. Root adopts their strongest common recommendation: joint affinity/population/degree-propensity updating, not a free nonedge gate. This is an inference repair using established DC-SBM principles; no new-method or PFN-training success is claimed by Stage0.

## Model and schedule locked for the development execution

Binary undirected adjacency, no self-loops. Density offset t=mean_observed_degree/(N−1) is fixed from adjacency; an empty graph has t=0. This is a graph-conditioned Poisson working score, not an exact Bernoulli or degree-conditional likelihood. Score per unordered pair is A_ij L_ab − t theta_i theta_j R_ab, with L=ElogB and R=EB from the same Gamma distribution. Class-independent A*log(t theta_i theta_j) drops from class messages but observed degrees enter the theta update. Context labels are clamped; target truth is never an input.

All class-pair Gamma priors have shape5 and rate5/B0_ab; B0 diagonal1+(K−1)*.6, offdiagonal.4. Prior theta_i~Gamma(1,1). Population Dirichlet prior is 1 per class. Class-pair B is symmetric and otherwise unrestricted. No target neural optimizer, gradient fitting, label-aware parameter initialization or outcome-selected multistart is allowed. Label/node/column permutation equivariance is required.

Graph cavity BP and class-gated feature/centroid messages run synchronously for24steps,damping.5. Four6-step blocks: block0 uses priors; before blocks1/2/3 update nuisance distributions from the preceding state (3updates total), then freeze those nuisance parameters for that6-step block. Within every BP step update the nonneighbor field from the previous node beliefs. End prediction uses the last block's parameters and final messages; no unused fourth nuisance update is made. Old directional messages are retained across blocks.

The graph external field includes **only nonneighbors**, using current degree-propensity-weighted class volumes after subtracting self and neighbors. Each observed edge factor contains its full positive and negative Poisson score. Pair beliefs/counts are computed from one saved field/cavity state under the same L,R; no edge negative-term double count. Nonedge pairs use node-product mean-field; observed edges use edge pair beliefs. Report this approximation and tiny dense-vs-sparse difference rather than claiming exact all-pair inference.

At each nuisance update, compute fresh sufficient statistics and set posterior proposals from **prior + current statistics**, never previous posterior + same evidence again. Gamma B shape=5+expected unordered edge count; rate=prior_rate+t*expected theta-product pair exposure. Diagonal/offdiagonal unordered membership conventions must be independently checked. Theta posterior proposal has shape=1+degree_i and rate=1+t*sum_{j!=i}Etheta_j E_{qij}[B_zizj]. Population proposal Dirichlet=1+sum_{i not in C}node_belief_i; context is excluded from population counts but included in graph exposure/degree statistics. Use .5 damping of Gamma shape/rate and Dirichlet parameters at each nuisance update. All count/volume quantities are distinct and saved for audit.

Population unary is Elogpi=digamma(alpha_c)−digamma(sum alpha), while reported population is posterior mean alpha/sum alpha. For pair statistics, use unordered class-pair indicator H_ab: diagonal q_ij(a,a), offdiagonal q_ij(a,b)+q_ij(b,a). Edge count M_ab=sum_edges A_ij H_ab, each undirected edge once. Exposure V_ab=sum_nonedge theta_i theta_j H_ab under node-product plus sum_edges theta_i theta_j H_ab under edge belief; Gamma rate adds t*V_ab once. Theta rate uses all other nodes under the same nonedge/edge convention, never population count as volume.

## Prespecified Stage0 arms, inputs and controls

Ten development cases: same rep0/seed11 Cora,CiteSeer,PubMed,CoauthorCS,CoauthorPhysics episodes in modes none/95. Same frozen FP01 C_global_s11 three coefficients and same centroid grid41 are used in every method. Feature source/context/query are exactly the old bank. Model-input whitelist excludes all query labels, true class sizes and diagnostic homophily. For new arms only homophilic prior is used, as the current project scope is homophily. Old full/edge conditional references must use the same homophilic branch; verify their existing all-homophilic weights or compute it explicitly. No relation-selection claim.

New score controls:

1. `poisson_fixed`: theta=1, Gamma B stays at prior (use ElogB, not logEB), population uniform. Tests changed link/potential strength without adaptation.
2. `sbm_update`: theta=1, update B and population. Tests class-volume/rate inference without degree correction.
3. `dc_update`: update theta,B,population using one common schedule. Tests joint nuisance correction.

Also show unchanged old-full/old-edge C_global references, both LP, public PFN and PCFI/FISF-PFN on the same episodes. Each fixed algorithm is shown on all five graphs; no per-graph winner selection. All none/95, accuracy, balanced accuracy, macroF1,NLL,Brier, population estimates and parameter/rate/saturation diagnostics are retained. Report observed−none and existing shuffle references separately. Transductive nuisance-distribution updates are permitted inference; describe neural optimizer updates as zero rather than pretending no statistical parameters changed.

Stage0 has no newly computed shuffle case, so it cannot establish alignment or a new feature-inference success. Its fixed development go-condition for further study is: on none, Physics improves old-full by at least20pp and CS by5pp; CiteSeer loses at most1pp accuracy and .05nats NLL versus old-full; mean five-graph balanced accuracy is no lower than both old-full and old-edge, and mean NLL no worse than old-edge+.05. Report comparisons with poisson_fixed/sbm_update and both LP regardless. Passing this gate alone is not a main-method result. If sbm_update suffices and dc_update adds no value, preserve the simpler class/rate repair; do not force a degree-specific claim or select a different algorithm per graph. Any follow-on choice is development and needs separately locked Stage1 training/final evaluation.

## Correctness before these predictions

Independent tiny dense Poisson score/enumeration and dense BP implementation; sparse nonneighbor approximation error is diagnostic. Edge-only/tree oracle, full graph/nonneighbor self-neighbor exclusion, exact edge pair count/exposure, Gamma moments and no repeated evidence accumulation, class/population vs volume, context clamps, zero/one feature source, hidden-value poison, class/node/source-row/feature-column permutation, finite gradients for later training, empty-edge graph, and unchanged historical artifacts. Inspect high-rate hub pairs: Poisson binary saturation is a limitation, not silently clipped away.

Do not change scientific settings after seeing these ten outputs. Actual implementation faults may be repaired with original outputs preserved and a written reason. Stage0 is a concrete model repair and development feasibility result, not independent final validation.

Stage1 remains a separately locked follow-on: actual synthetic-only equal-budget training under original versus degree/skew priors (3seeds each,256updates). Strong classical updates/LP are retained. Trainable parameters, comparison arms and final seeds must be reviewed before that training; Stage0 does not authorize hidden extra sweeps or a new affinity/posterior neural head.
