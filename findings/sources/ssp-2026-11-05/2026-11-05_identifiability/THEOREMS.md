# What count flexibility costs: fixed-archive identification and decisions

2026-11-05. Proof draft for independent review. Classical foundations are stated explicitly; no priority or adaptive-policy guarantee is asserted. This document concerns linear composition and arbitrary count effects, not nonlinear composition.

## Model and quantities

Fix a nonadaptive archive of binary rows X in R^(N×n). Let Z be the one-hot matrix for its m observed count levels. Every column occurs, so D=Z'Z is invertible. Choose B in R^(n×(n−1)) with B'B=I and B'1=0. Write U=XB and

    y = U theta + Z a + epsilon,
    epsilon ~ N(0, sigma² I), sigma>0,
    Q = I − Z D^−1 Z', M = U' Q U.

The count values a are unrestricted. All results concern likelihood information, rather than identification created by a prior. For a candidate binary set x at a fixed count K, only h_theta(x)=x'B theta affects its ordering. For a finite feasible slice F_K, define its contrast space C_K=span{B'(x−x'):x,x' in F_K}. A slice with one candidate has no nontrivial contrasts. The full nontrivial slice has 1<=K<=n−1.

## Proposition 1: exact contrast identification

A contrast c'theta is globally identifiable from the archive distribution if and only if c belongs to row(QU)=range(M). Therefore all decisions' composition contrasts on F_K are globally identifiable exactly when C_K is contained in range(M). On the full nontrivial slice this is equivalent to rank(M)=n−1. In particular, N−m>=n−1 is necessary, but not sufficient. This is global identification over the parameter class, not a claim that every individual mean fiber remains ambiguous on a fixed-norm sphere.

Proof. Two parameters have the same observation law iff U delta_theta+Z delta_a=0. Such a delta_a exists iff QU delta_theta=0. Hence all equal-law parameters agree on c'theta iff c is perpendicular to ker(QU), which is row(QU). The same global identification conclusion holds when ||theta||=s>0 is fixed: for a null vector affecting c, the antipodal points ±s v give an equal-law counterexample after choosing the two a values appropriately. The full K-slice contrasts span 1-perp: each e_i−e_j is the difference of two K-sets sharing K−1 other items. B is an isometry on that space, so C_K=R^(n−1). Finally rank(QU)<=rank(Q)=N−m. Sufficiency cannot follow just from counts: repeating one identical set at each observed count makes QU=0 even for arbitrarily large N. QED.

This is standard linear-model estimability after removing fixed effects. Its research use is to state the archive requirement in count/composition terms.

A unique-query counterexample also exists: with n=8, take six distinct sets at each count2,...,6, always including items1 and2 together or excluding both. There are at least16 such sets at every one of these counts, so these30 observations can all be distinct. Here N−m=25>=7 but the direction e1−e2 is never observed. Thus the necessary sample budget is insufficient even when repeated identical queries are forbidden. This example is a proof illustration; the numerical validation below uses the simpler repeated-row excitation counterexample.

## Proposition 2: an exact pairwise decision lower bound

Fix s>0 and a unit vector v in R^(n−1), with positive width of h(x)=x'Bv on F_K. Let

    theta_± = ±s v,
    a_± = −D^−1 Z' U theta_±,
    mu = s Q U v.

These two allowed environments have observation distributions P_±=N(±mu,sigma² I). For each, define normalized simple regret by dividing its fixed-slice objective loss by its own (equal, positive) objective range. Then

    inf_rules max{ E_+ r_+(x_hat), E_- r_-(x_hat) }
      = Phi(−||mu||/sigma)
      = Phi(−s sqrt(v'Mv)/sigma).

The infimum allows randomized observation-based recommendations in F_K and knowledge of the two-point family. It is not an achievable upper bound over the full unknown-parameter class.

Proof. Put h_max=max_F h, h_min=min_F h, w=h_max−h_min>0. For every x,

    r_+(x)=(h_max−h(x))/w,
    r_-(x)=(h(x)−h_min)/w,
    r_+(x)+r_-(x)=1.

Thus the equal-prior Bayes loss of any rule is at least (1/2) integral min(p_+,p_-) dy: at every y the smallest possible p_+ r_++p_- r_- is min(p_+,p_-), attained by an endpoint of h. If mu!=0, the likelihood-ratio rule uses the sign of mu'y, recommending an h-maximizer for nonnegative sign and an h-minimizer otherwise. Under P_+, mu'y has mean ||mu||² and variance sigma²||mu||²; under P_- its mean is negative with the same variance. Both error probabilities equal Phi(−||mu||/sigma), and endpoint regret is exactly0 or1. This attains the Bayes lower bound with equal environment risks and hence is minimax for the pair. If mu=0, the two laws coincide; any rule's two expected losses sum to1, so their maximum is at least1/2. A fair coin between the endpoints attains1/2. Finally ||mu||²=s² v'Mv. QED.

Corollaries. A null direction that affects the feasible slice imposes a worst-environment expected normalized regret of at least1/2, even as sigma tends to zero through positive values. On a full nontrivial slice every nonzero v has positive width; taking a minimum-eigenvalue direction gives the lower bound Phi(−s sqrt(lambda_min(M))/sigma) for the full parameter class {||theta||=s,a unrestricted}. This is a lower bound only. For a restricted slice a null vector may have zero width and provide no such obstruction.

Fixing the signal norm is essential for a nontrivial normalized-risk comparison. If arbitrarily small nonzero signals are allowed, taking s down to zero makes this lower bound approach1/2 even with full-rank M. No assertion is made for smooth, monotone, bounded or otherwise restricted a unless the constructed a_± satisfy those restrictions. The pair is chosen after X is fixed; it is not a pair established before an arbitrary adaptive sampling trajectory.

## Proposition 3: information bought by support versus replication

Let u_i=B'x_i and bar_u_k be its sample mean among observations with count k. Then

    M = sum_k sum_{i:K_i=k} (u_i−bar_u_k)(u_i−bar_u_k)'.

A first observation at a new count contributes zero to this M. A new observation u at a count with t>=1 existing observations changes M by

    Delta M = t/(t+1) (u−bar_u_k)(u−bar_u_k)'.

Proof. Q subtracts the mean of each observed count group, giving the first equality. For an existing group, substitute the updated mean (t bar_u_k+u)/(t+1) in its centered sum; cross terms sum to zero and the remaining contribution is t/(t+1) times the displayed outer product. A new singleton group has zero centered scatter. QED.

This is the classical online scatter identity. Its decision interpretation follows from Proposition2: in a fixed direction the pairwise testing difficulty depends on v'Mv, which a new count's first sample cannot improve when each count level has an unrestricted nuisance value. It can still be valuable for learning count levels, choosing a count, or for inference under a justified coupling prior. A new count does not reduce the previous M.

## Proposition 4: where count-prior assumptions enter

If count effects are restricted to a known observed linear design W=Z Psi, replace Q with Q_W=I−WW^+ and M with U'Q_W U. For nested column spaces W1⊆W2,

    U'Q_W2 U <= U'Q_W1 U   (Loewner order).

For a proper Gaussian count prior with covariance alpha G on observed levels, G positive definite, marginal covariance is V_alpha=sigma²I+alpha ZGZ'. Its inverse satisfies

    limit_{alpha→infinity} V_alpha^−1 = sigma^−2 Q.

Proof. Orthogonal projection onto a larger nested space subtracts a positive semidefinite matrix, giving the inequality. Let E=Z D^−1/2, so E'E=I, and C=D^1/2 G D^1/2 be positive definite. Decompose the observation space into col(E) and its complement:

    V_alpha^−1 = sigma^−2 Q
                  + E (sigma² I_m+alpha C)^−1 E'.

The second summand tends to zero, proving the limit. QED.

These are classical projection/mixed-model facts. A finite proper GP can use between-count differences because its prior couples count levels; that benefit is not uniformly available under an arbitrary deterministic count function. The limit is conditional on fixed covariance parameters and says nothing about empirical-Bayes optimization, nonlinear composition, calibrated credible intervals, or sequential regret.

## Consequence for SSP design, not yet a validated policy

The theorem distinguishes two purchases within a limited query budget: count coverage and within-count composition excitation. A flexible joint model can exploit both, but flexibility alone does not guarantee enough data to separate them. An information-aware query candidate should track the within-count information M and the contrasts that can change the recommended set, alongside the joint count posterior. A new count can have high count-decision value while adding no arbitrary-nuisance composition information. Any claimed advantage over ordinary joint GP plus a standard experimental-design acquisition needs a separate frozen experiment. There is no theorem here that the existing fixed-frequency spread probe is optimal or safe.

## Direct foundations and claim boundary

Linear estimability, removal of nuisance fixed effects, sufficient-statistic scatter updates, Gaussian binary testing, and optimal design are existing mathematics. Fiez et al.2019 establishes transductive linear-bandit experimental-design lower bounds; this document does not claim to originate decision-directed design. Krishnamurthy et al.2018 treats action-independent nuisance terms; count effects here are shared within count but change across count actions. The count-specific fixed-archive connection can support the paper's diagnosis if positioned alongside these sources. Independent novelty review determines the appropriate weight. Numerical identities cannot establish priority.

The direct augmented-linear mapping is measurement feature (B'x,e_K), target feature (B'z,0_m), and parameter (theta,a). M is the Schur complement of the nuisance block in the augmented Gram matrix. Thus the generic observation-versus-recommendation framework already covers this fixed-archive model. Rosa and Harman's nuisance-resistant treatment-contrast design supplies further direct design context. The proposed contribution is the count/composition diagnosis and its connection to subset decisions, not invention of residualized information.

- Fiez et al., Sequential Experimental Design for Transductive Linear Bandits, NeurIPS2019: https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html
- Krishnamurthy, Wu, Syrgkanis, Semiparametric Contextual Bandits, ICML2018: https://proceedings.mlr.press/v80/krishnamurthy18a.html
- Rosa and Harman, Optimal approximate designs for estimating treatment contrasts resistant to nuisance effects: https://arxiv.org/abs/1504.06079
