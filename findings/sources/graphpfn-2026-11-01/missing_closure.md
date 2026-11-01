# Missingness-aware GraphPFN line: closed (29 Oct 2026, user decision "지금 중단하고 결측 라인 종료")

## State at closure
- Matched-imputation run stopped at B4imp_s0 update 255/500 (results/imp/B4imp_s0; trainable.pt = update 200). Ours-Imp and seed 1 not trained; no imp evaluation ran.
  The final bank results/bank/final (seeds 700000-700119) was generated but never evaluated. Protocol: docs/imp_protocol.md (+ r=0.9 addendum).
- Prior effect on the official imputation input path: NOT CONFIRMED (not assessed; stopped before training finished).

## What was established (DEV unless noted; single training seed s0)
1. NaN reaches LimiX MaskEmbEncoder exactly; the official pipeline imputes all NaN (most_frequent), which acts as a min-indicator (mask AUROC 0.97 from imputed values).
2. NaN-input training: Ours-B4 on label_dep +0.75/+1.43/+0.78 (r=.2/.5/.8); ~0 elsewhere; hold-out (self_mask, degree_dep) within +-0.1.
3. Released GraphPFN with official imputation beats NaN-trained Ours on label_dep (-0.54/-0.63/-0.12); Ours@impute - init@impute only +0.14..+0.24.
4. Headroom is small: no-missing acc 72.9 vs majority ~62; MCAR r=.8 costs ~3.5 pp.
5. In-context test (docs/diag_context_results.md): all models, including init, transfer the context mask->label relation to queries; mixed-prior training amplifies it
   (mismatch penalty Ours vs B4 -2.81 / -4.44 pp). Unseen-MNAR generalization: not supported.

## Lessons carried forward
- Compare against the official preprocessing path, not only the raw-NaN path.
- Check headroom (signal-vs-majority gap) of synthetic tasks before training.
- Context-query mismatch is a necessary diagnostic for any in-context-adaptation claim.
