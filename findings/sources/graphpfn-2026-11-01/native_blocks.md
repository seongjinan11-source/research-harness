# DBLP: whole native feature-block missingness baseline pilot

Frozen GraphPFN 1.3, 10 members, 8 random features, PCA64 cap, identical public splits and context draws. No target-graph parameter updates and no new PFN pretraining. This is a single-real-graph baseline headroom check, not unseen-schema validation or a new-method result.

A = author native keyword features (334 coordinates), P = paper title words (4231), T = term GloVe features (50). These coordinates are never averaged across feature types. Only observed source blocks contribute to a condition.

## Mean test accuracy (%) across three context draws

| Context labels/class | Missing blocks | Target-only paths | Cross-type paths | Adapted HetGFD paths |
|---:|---|---:|---:|---:|
| 5 | full | 87.32 | 89.96 | 89.92 |
| 5 | drop_A | 81.38 | 90.18 | 90.19 |
| 5 | drop_P | 87.32 | 90.00 | 89.97 |
| 5 | drop_T | 87.32 | 89.40 | 89.41 |
| 5 | drop_A_P | 81.38 | 82.29 | 82.58 |
| 5 | drop_A_T | 81.38 | 89.87 | 89.87 |
| 20 | full | 93.15 | 93.55 | 93.55 |
| 20 | drop_A | 91.72 | 93.65 | 93.66 |
| 20 | drop_P | 93.15 | 93.33 | 93.32 |
| 20 | drop_T | 93.15 | 93.66 | 93.66 |
| 20 | drop_A_P | 91.72 | 93.66 | 93.61 |
| 20 | drop_A_T | 91.72 | 93.22 | 93.22 |

## Paired change from each method's own full-feature condition (pp)

| Labels/class | Missing blocks | Target-only paths | Cross-type paths | Adapted HetGFD paths |
|---:|---|---:|---:|---:|
| 5 | drop_A | -5.94 | +0.21 | +0.27 |
| 5 | drop_P | +0.00 | +0.04 | +0.05 |
| 5 | drop_T | +0.00 | -0.56 | -0.51 |
| 5 | drop_A_P | -5.94 | -7.67 | -7.34 |
| 5 | drop_A_T | -5.94 | -0.09 | -0.05 |
| 20 | drop_A | -1.43 | +0.10 | +0.11 |
| 20 | drop_P | +0.00 | -0.23 | -0.24 |
| 20 | drop_T | +0.00 | +0.11 | +0.11 |
| 20 | drop_A_P | -1.43 | +0.11 | +0.06 |
| 20 | drop_A_T | -1.43 | -0.33 | -0.33 |

## Source counts and structural reference

- A: 4057 native observed rows when retained; direct features reach 4018/4057 target rows, adapted diffusion 4018/4057.
- P: 14328 native observed rows when retained; direct features reach 4057/4057 target rows, adapted diffusion 4057/4057.
- T: 7723 native observed rows when retained; direct features reach 4057/4057 target rows, adapted diffusion 4057/4057.
- Structure-only, 5 labels/class: 81.38% accuracy.
- Structure-only, 20 labels/class: 91.72% accuracy.

## Secondary label-only development diagnostics

These were added after the registered GraphPFN experiment started and its k=5 results were visible. Per-view reporting followed the aggregate LP results; the context-only selector was designed after the per-view test scores were visible. They are posthoc development checks, not independent confirmation or a new method. All use fixed alpha=0.9, 100 symmetric-normalized propagation steps, the same graph and contexts, and zero attribute inputs.

| Labels/class | Union LP | Uniform view LP | APA LP | APTPA LP | APVPA LP | Context-LOO selected LP |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 64.49 | 69.66 | 30.46 | 42.45 | 91.12 | 91.12 |
| 20 | 78.55 | 83.98 | 37.90 | 46.74 | 92.28 | 92.28 |

The selector removes each context label contribution globally (including all feedback paths), scores its held-out node with context leave-one-out NLL, and chooses the lowest-NLL view. Test labels are used only for reporting, not path selection. This selection rule is nevertheless a development-stage choice informed by the earlier test results.

| Labels/class | Context seed | Chosen view | LOO NLL APA | LOO NLL APTPA | LOO NLL APVPA |
|---:|---:|---|---:|---:|---:|
| 5 | 0 | APVPA | 4.7946 | 1.4361 | 0.7048 |
| 5 | 1 | APVPA | 1.5088 | 1.3670 | 0.7510 |
| 5 | 2 | APVPA | 1.3381 | 1.4711 | 0.6221 |
| 20 | 0 | APVPA | 1.1368 | 1.3225 | 0.6434 |
| 20 | 1 | APVPA | 1.4769 | 1.2984 | 0.5678 |
| 20 | 2 | APVPA | 1.4222 | 1.3301 | 0.6218 |

On this DBLP pilot, context-only selection recovers the strong APVPA reference in all six contexts. The term-only feature arm falling below its own full-feature result therefore does not by itself establish headroom over strong attribute-free prediction. Stronger native heterogeneous pretraining is not justified by that gap alone. Full-feature GraphPFN can still outperform label-only inference at the larger context budget; additional datasets and untouched schemas are needed to separate useful attributes, path choice and missingness robustness.

All 36 secondary probability outputs pass independent metric, context, source-hash, uniform-mean and selected-component checks. See secondary_lp/summary.csv and secondary_lp/audit.json.

## Interpretation limits and data provenance

Performance loss relative to a full-feature reference is not proof that the missing information is recoverable. Cross-type attributes and graph structure may provide redundant signal, and a missing-input arm can even outperform its own full reference. No per-condition winner is selected or deployed; all registered methods and negative results are retained.

HetGFD is adapted here: native source types are processed independently and their virtual author features are concatenated. The local original code has a fixed paper source and returns only its attributed rows. The adapted computation matches the original raw-coordinate DBLP propagation on toy graphs after exposing all virtual output rows; max projected error < 1e-6. Raw-space ranking and the preliminary state initializing final diffusion are preserved. Fixed-source linear projection commutes with the final propagation. This is not a reproduction of published HGT+HetGFD scores.

IMDB director/actor attributes are precomputed movie-feature averages, verified against MAGNN preprocessing and local arrays. ACM subject features equal target-paper aggregation in cosine; author blocks are also highly aligned. Keeping pre-mask aggregates after hiding target raw features would retain derived information. Those datasets were not silently treated as independently attributed heterographs. DBLP author keywords, paper title words and pretrained term vectors are retained as distinct native inputs; existing graph edges and the original vocabulary are part of the fixed benchmark.

All 114 saved probability outputs and actual feature matrices passed independent NumPy metric, context, type-access and checkpoint/data/source audits. Hidden-author target-only predictions equal the structure-only reference exactly. Query-label poison max error is zero. Three context draws share one graph/test set and do not establish independent-graph confidence.

See protocol.json, checks.json, prepared/metadata.json, results/summary.csv, results/audit.json, and results/predictions/.
