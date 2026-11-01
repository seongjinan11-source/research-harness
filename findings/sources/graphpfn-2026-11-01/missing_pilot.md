# Pilot results — Missingness-aware GraphPFN (Gate 3 + follow-up diagnostics)

Date 2026-10-29. All numbers are accuracy in %, from raw CSVs in `results/pilot/` and `results/diag_labeldep/`. Differences are paired over identical tasks and masks; brackets are task-level bootstrap 95% CIs (10,000 draws). Dev = 60 tasks (training mechanisms), hold-out = 120 tasks. One training seed per arm (500 updates x 4 tasks, lr 3e-5). **Hold-out results have now been looked at several times and are marked exploratory; a new, fixed final validation set is required before any claim.** Earlier result files were not modified.

## 1. Five input treatments, GraphPFN 1.3 without extra training

| split | mechanism | r | most-frequent (official) | minimum | mean | zero | NaN (LimiX path) | min − mf |
|---|---|---|---|---|---|---|---|---|
| dev | label_dep | 0.2 | 73.86 | 74.01 | 73.77 | 73.86 | 72.28 | +0.15 [-0.00, +0.33] |
| dev | label_dep | 0.5 | 75.50 | 75.78 | 74.66 | 74.73 | 72.28 | +0.28 [+0.06, +0.52] |
| dev | label_dep | 0.8 | 72.58 | 72.75 | 72.37 | 72.26 | 69.97 | +0.16 [+0.02, +0.34] |
| dev | mcar | 0.2 | 71.68 | 71.60 | 72.00 | 71.97 | 72.07 | -0.08 [-0.22, +0.06] |
| dev | mcar | 0.5 | 70.59 | 70.66 | 70.87 | 70.87 | 70.82 | +0.08 [-0.18, +0.41] |
| dev | mcar | 0.8 | 69.39 | 69.43 | 69.75 | 69.72 | 69.12 | +0.04 [-0.06, +0.15] |
| dev | neighbor_dep | 0.2 | 71.77 | 71.86 | 72.01 | 72.00 | 72.19 | +0.09 [-0.08, +0.29] |
| dev | neighbor_dep | 0.5 | 70.76 | 70.88 | 70.93 | 70.95 | 70.86 | +0.12 [-0.05, +0.36] |
| dev | neighbor_dep | 0.8 | 69.44 | 69.44 | 69.63 | 69.61 | 69.24 | +0.00 [-0.06, +0.08] |
| dev | node_mcar | 0.2 | 71.54 | 71.52 | 71.35 | 71.26 | 71.05 | -0.01 [-0.12, +0.09] |
| dev | node_mcar | 0.5 | 69.92 | 69.78 | 69.53 | 69.43 | 69.22 | -0.13 [-0.28, -0.01] |
| dev | node_mcar | 0.8 | 68.46 | 68.46 | 68.45 | 68.40 | 68.08 | -0.00 [-0.13, +0.16] |
| holdout | degree_dep | 0.2 | 69.45 | 69.42 | 69.13 | 69.07 | 69.04 | -0.03 [-0.14, +0.07] |
| holdout | degree_dep | 0.5 | 68.29 | 68.20 | 67.83 | 67.77 | 67.71 | -0.09 [-0.22, +0.02] |
| holdout | degree_dep | 0.8 | 67.17 | 67.23 | 66.80 | 66.81 | 66.87 | +0.06 [-0.03, +0.16] |
| holdout | self_mask | 0.2 | 69.74 | 69.85 | 69.57 | 69.57 | 69.64 | +0.10 [-0.00, +0.21] |
| holdout | self_mask | 0.5 | 69.09 | 69.15 | 69.06 | 69.05 | 68.66 | +0.06 [-0.05, +0.17] |
| holdout | self_mask | 0.8 | 68.27 | 68.37 | 68.20 | 68.25 | 67.94 | +0.10 [+0.00, +0.22] |

Mask recoverability from imputed values alone (CPU, dev, `results/pilot/impute_diag_summary.csv`): AUROC most-frequent 0.97, minimum 0.98, mean 1.00, zero 1.00. Value collisions (observed cells equal to the fill value) 0.14/0.06/0.02 for most-frequent and 0.11/0.04/0.01 for minimum at r = .2/.5/.8, 0 for mean/zero. Most-frequent equals the column minimum in 73–81% of features. Pipeline order here: standardise on observed context entries → impute → LimiX NanEncoder/NormalizationEncoder (context statistics). The official path is quantile transform → most-frequent impute → (PCA) → drop constant columns → LimiX.

## 2. Init / B4 / Ours × NaN input / official imputation input

B4 and Ours were trained with NaN input; evaluating them with imputed input is a distribution-shift test, not the performance of models trained on imputed input.

| split | mechanism | r | Init-NaN | B4-NaN | Ours-NaN | Init-Imp | B4-Imp | Ours-Imp | Ours-Imp − B4-Imp | Ours-Imp − Init-Imp |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | label_dep | 0.2 | 72.28 | 72.57 | 73.32 | 73.86 | 73.90 | 74.02 | +0.12 [+0.03, +0.21] | +0.16 [+0.07, +0.25] |
| dev | label_dep | 0.5 | 72.28 | 73.43 | 74.86 | 75.50 | 75.46 | 75.64 | +0.18 [+0.07, +0.30] | +0.14 [+0.00, +0.28] |
| dev | label_dep | 0.8 | 69.97 | 71.68 | 72.46 | 72.58 | 72.63 | 72.82 | +0.19 [+0.08, +0.31] | +0.24 [+0.01, +0.48] |
| dev | mcar | 0.2 | 72.07 | 72.18 | 72.14 | 71.68 | 71.77 | 71.76 | -0.00 [-0.10, +0.10] | +0.08 [-0.03, +0.21] |
| dev | mcar | 0.5 | 70.82 | 71.09 | 71.26 | 70.59 | 70.76 | 70.78 | +0.01 [-0.08, +0.13] | +0.19 [+0.03, +0.40] |
| dev | mcar | 0.8 | 69.12 | 69.82 | 70.02 | 69.39 | 69.54 | 69.55 | +0.02 [-0.06, +0.10] | +0.16 [+0.01, +0.35] |
| dev | neighbor_dep | 0.2 | 72.19 | 72.22 | 72.20 | 71.77 | 71.88 | 71.89 | +0.01 [-0.07, +0.09] | +0.12 [+0.04, +0.21] |
| dev | neighbor_dep | 0.5 | 70.86 | 71.13 | 71.21 | 70.76 | 70.77 | 70.82 | +0.05 [-0.01, +0.13] | +0.06 [-0.05, +0.16] |
| dev | neighbor_dep | 0.8 | 69.24 | 69.77 | 69.92 | 69.44 | 69.60 | 69.63 | +0.03 [-0.03, +0.09] | +0.19 [+0.05, +0.34] |
| dev | node_mcar | 0.2 | 71.05 | 71.59 | 71.54 | 71.54 | 71.60 | 71.60 | -0.00 [-0.08, +0.08] | +0.07 [-0.02, +0.15] |
| dev | node_mcar | 0.5 | 69.22 | 70.15 | 70.09 | 69.92 | 69.95 | 69.97 | +0.02 [-0.05, +0.12] | +0.06 [-0.06, +0.19] |
| dev | node_mcar | 0.8 | 68.08 | 69.03 | 69.00 | 68.46 | 68.57 | 68.55 | -0.02 [-0.10, +0.06] | +0.09 [-0.12, +0.32] |
| dev | none | 0.0 | 72.85 | 72.88 | 72.89 | 72.85 | 72.88 | 72.89 | +0.01 [-0.06, +0.07] | +0.04 [-0.04, +0.12] |
| holdout | degree_dep | 0.2 | 69.04 | 69.43 | 69.39 | 69.45 | 69.54 | 69.60 | +0.06 [-0.02, +0.14] | +0.15 [+0.01, +0.28] |
| holdout | degree_dep | 0.5 | 67.71 | 68.48 | 68.39 | 68.29 | 68.28 | 68.26 | -0.03 [-0.11, +0.06] | -0.03 [-0.23, +0.17] |
| holdout | degree_dep | 0.8 | 66.87 | 67.49 | 67.46 | 67.17 | 67.22 | 67.17 | -0.05 [-0.12, +0.00] | -0.00 [-0.18, +0.17] |
| holdout | none | 0.0 | 70.41 | 70.44 | 70.49 | 70.41 | 70.44 | 70.49 | +0.05 [-0.01, +0.12] | +0.09 [-0.03, +0.21] |
| holdout | self_mask | 0.2 | 69.64 | 69.69 | 69.76 | 69.74 | 69.69 | 69.74 | +0.05 [-0.03, +0.13] | -0.00 [-0.10, +0.10] |
| holdout | self_mask | 0.5 | 68.66 | 69.07 | 69.13 | 69.09 | 69.07 | 69.14 | +0.07 [-0.01, +0.15] | +0.05 [-0.08, +0.16] |
| holdout | self_mask | 0.8 | 67.94 | 68.43 | 68.40 | 68.27 | 68.27 | 68.29 | +0.02 [-0.03, +0.07] | +0.01 [-0.12, +0.13] |

## 3. Experiment A — mask-only and label-independent mask (dev, label_dep)

| input | r | Init | B4 | Ours | Ours − B4 |
|---|---|---|---|---|---|
| full | 0.2 | 72.28 | 72.57 | 73.32 | +0.75 [+0.34, +1.23] |
| full | 0.5 | 72.28 | 73.43 | 74.86 | +1.43 [+0.82, +2.11] |
| full | 0.8 | 69.97 | 71.68 | 72.46 | +0.78 [+0.47, +1.15] |
| maskbin | 0.2 | 71.47 | 71.51 | 71.50 | -0.01 [-0.11, +0.08] |
| maskbin | 0.5 | 72.63 | 72.48 | 72.55 | +0.07 [-0.05, +0.18] |
| maskbin | 0.8 | 71.98 | 71.94 | 71.91 | -0.04 [-0.11, +0.04] |
| masknan | 0.5 | 72.85 | 73.42 | 73.46 | +0.04 [-0.21, +0.30] |
| ratematch | 0.2 | 72.22 | 72.27 | 72.18 | -0.08 [-0.17, +0.00] |
| ratematch | 0.5 | 70.77 | 71.04 | 71.20 | +0.16 [-0.01, +0.40] |
| ratematch | 0.8 | 69.15 | 69.86 | 70.02 | +0.16 [+0.01, +0.34] |

full = observed X with NaN at M; maskbin = M as 0/1 features only; masknan = NaN at M and 0 elsewhere; ratematch = same per-column missing rate but label-independent mask.

## 4. Experiment B — label_dep strength and context size (dev, r = 0.5)

| condition | Init | B4 | Ours | Ours − B4 | B4 − Init |
|---|---|---|---|---|---|
| strength: q.3_b.5-2 | 72.63 | 73.89 | 74.86 | +0.97 [+0.54, +1.50] | +1.26 [+0.82, +1.75] |
| strength: q.3_b2-4 | 80.57 | 83.28 | 84.52 | +1.24 [+0.81, +1.72] | +2.71 [+2.05, +3.46] |
| strength: q.6_b.5-2 | 74.06 | 75.86 | 77.13 | +1.28 [+0.77, +1.89] | +1.80 [+1.14, +2.54] |
| strength: q1_b1-4 | 87.87 | 91.43 | 93.50 | +2.08 [+1.47, +2.74] | +3.55 [+2.57, +4.64] |
| context_label_dep: f0.1 | 61.67 | 62.18 | 62.37 | +0.19 [+0.04, +0.35] | +0.51 [+0.13, +0.92] |
| context_label_dep: f0.25 | 66.36 | 67.20 | 67.79 | +0.59 [-0.05, +1.50] | +0.84 [+0.03, +1.75] |
| context_label_dep: f0.5 | 70.44 | 71.85 | 72.46 | +0.61 [+0.27, +0.98] | +1.41 [+0.84, +2.08] |
| context_label_dep: f1.0 | 72.28 | 73.43 | 74.86 | +1.43 [+0.82, +2.11] | +1.16 [+0.72, +1.64] |
| context_mcar: f0.1 | 61.84 | 61.91 | 61.94 | +0.03 [-0.06, +0.12] | +0.06 [-0.23, +0.35] |
| context_mcar: f0.25 | 66.16 | 66.33 | 66.25 | -0.08 [-0.22, +0.04] | +0.16 [-0.48, +0.69] |
| context_mcar: f0.5 | 69.49 | 69.84 | 69.79 | -0.04 [-0.28, +0.14] | +0.34 [+0.14, +0.59] |
| context_mcar: f1.0 | 70.82 | 71.09 | 71.26 | +0.17 [+0.06, +0.30] | +0.27 [+0.07, +0.54] |

strength: q = max share of label-dependent features, b = dependence strength; training used q_max 0.3, b ~ U(0.5, 2). context: fraction f of context labels kept (f = 1.0 equals the main evaluation).

## 5. Hypotheses: supported vs. not supported

Supported by the data
1. Ours' advantage over B4 on label_dep comes from label information in the mask. With a label-independent mask of the same per-column rate, Ours − B4 falls to −0.08 to +0.16.
2. The advantage needs observed values and mask together. With the mask alone (maskbin, masknan), Ours − B4 is within ±0.1 (CIs include 0).
3. The advantage grows with context size (+0.19 at 10% of context labels, +0.61 at 50%, +1.43 at 100%) while the MCAR control stays near 0. This is consistent with inferring the task-specific label–mask relation in context, though it is not proof.
4. The advantage extends beyond the training strength range (+2.08 [+1.47, +2.74] at q = 1, b ~ U(1, 4), versus +0.97 train-like).
5. Constant imputation keeps the mask almost fully recoverable from the numbers (AUROC 0.97–1.00). Minimum ≈ most-frequent, and minimum is slightly better on label_dep (+0.15 to +0.28). Mean and zero fill are worse on label_dep (−0.8 at r = .5) but better on MCAR-type masks (+0.2 to +0.36). So recoverability alone does not explain the official path's strength; the position of the marker matters to the model.

Not supported / refuted
1. Generalisation to unseen mechanisms (self_mask, degree_dep): Ours − B4 is within ±0.1 with NaN input and with imputed input. Ours-Imp − Init-Imp is within ±0.1 except degree_dep r = 0.2 (+0.15 [+0.01, +0.28]), a single cell out of six. The novelty review notes degree_dep is MAR given the graph, so a null result there is expected.
2. "Ours beats the official GraphPFN path": with NaN input, Ours is below official imputation on label_dep (−0.54, −0.63, −0.12 at r = .2/.5/.8).
3. Under imputed input the prior effect is small. Ours-Imp − B4-Imp on label_dep is +0.12 to +0.19 (CI > 0), and Ours-Imp − Init-Imp is +0.14 to +0.24 (lower CI at or just above 0). This is a shift test for NaN-trained models.
4. neighbor_dep gains are small in every setting (≤ 0.2).

## 6. Recommendation

The decision rule in the instructions puts this in Case A: Ours-Imp > B4-Imp and > Init-Imp on label_dep. But the margins are only 0.1–0.2 points, from one training seed and a shift setting. Following that rule:
1. Before retraining, check that the Case A margin repeats. Use a fresh dev-like bank with new seeds, and add a second training seed for B4/Ours if approved.
2. If it repeats, run a small matched experiment with imputed (minimum or most-frequent) input during both training and evaluation. Arms: MCAR prior, current mixed prior, and the unified sparse-coefficient prior. Use the same budget and init@impute as the reference.
3. Fix a new final validation set before that run. Use new task seeds, and new evaluation-only mechanisms such as neighbour-label-dependent missingness and naturally missing graphs. Current hold-out numbers stay exploratory.
4. Framing per the novelty review: claim in-context detection of informative missingness. Mask reliance should track |b_y|, with no loss under MCAR. Baselines must include the official GraphPFN fill, GNNmim/PEMix and an SMIM-style test-then-gate heuristic.

No new checkpoint will be trained until this is approved.
