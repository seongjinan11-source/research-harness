# Gate 0 — environment and generator checks (no training)  [29-30 Oct 2026]

Location: mgpfn/heteropfn (user: "그냥 여기다해"). Reuses mgpfn/src/mgpfn/upstream.py (GraphPFN 1.3 loader, commit 3b9b115).

## 1. GraphPFN 1.3 graph interface (read from source)
- GraphPFNLight takes ONE untyped dgl graph; every layer adds graph attention (edge_softmax over in-edges, u->v) after the LimiX layer. No edge types or weights.
- Official ICL path (inference/icl.py): prepare_graph = remove self loops, to_simple, to_bidirected (direction is discarded);
  prepare_features_tensor = quantile-normal, most-frequent impute, drop constant (on context); GraphPFN wrapper = 10-member ensemble
  (feature and class permutations), 8 random features, bf16 autocast.
- Our predictor (heteropfn/eval/gpfn.py) reproduces this path. Deviations: fp16 instead of bf16 (the Windows DGL build has no bf16 kernels);
  torch.manual_seed(seed) before the ensemble is built (members are drawn at construction; without it results depended on RNG history).
- Pitfall found and fixed: GraphPFNLight orders context rows by ascending node index, so y_context must be in that order. The predictor now sorts.
- Importing sklearn before the DGL bootstrap crashes the process (0xC0000005); import it after Predictor().

## 2. Typed-graph prior (heteropfn/priors/typed_graphs.py)
Spec ranges as in the plan (100-500 nodes, 2-4 node types, 2-6 relations, 2-5 classes, one target type, shared feature space d 8-24).
Latents z (label) and z' (nuisance) on every node; features = random MLP of both + noise. Roles: inf / het / dis (homophilous on z', structured
but label-irrelevant) / noise; edge counts drawn independently of role. Scenarios A (target-target inf), C (target-target het), D (only the typed
2-hop target-B-target path informative), N (none). Relation IDs randomly permuted per task; roles kept only in meta.
Addition to the plan: the "dis" role, so that label-free structure (communities, triangles) cannot identify the useful relation.

## 3. 100-task inspection (seeds 100000-100099; results/gate0/relation_stats.csv, task_stats.csv)
- Scenario counts: A 42, D 27, N 16, C 15.
- Label homophily: inf 0.73 (target-target) / 0.73 (target-other), het 0.05, dis 0.36 on z but 0.74 on z', noise 0.35-0.36.
- Feature-only 5-fold logreg (all target labels) 0.78 mean vs majority 0.44 (median gap 0.34).
- Typed 2-hop vote accuracy: scenario D 0.62 vs collapsed 0.36; scenario A collapsed vote 0.84.
- Shortcut check within the same schema (AUROC of label-free statistics; results/gate0/shortcut_within_schema.json):
  degree/edge-count 0.37-0.61 for all role pairs; clustering inf vs dis 0.50, inf vs noise 0.60; het has lower clustering (AUROC 0.23-0.34),
  i.e. heterophilic relations are recognizable without labels (remaining risk, recorded).

## 4. Model-side checks with released GraphPFN (results/gate0/model_checks.json, 4 tasks, k=5)
- Determinism across different preceding RNG state: max |dp| = 0 (all 4 tasks).
- Query-label poison (all non-context labels randomized): max |dp| = 0.
- Relation-ID permutation: collapsed and meta-path edge sets identical; B1/B4 identical; B2/B3 equal up to 2e-16; B3 weights permute exactly.
- Cost: 7.6-17.9 s per task at k=5 (R=2..6).

## 5. Observation to carry into Gate 1
Feature-only GraphPFN (B0, empty graph) is weaker than a logistic regression fitted on the same context on these synthetic tasks
(12 tasks, k=5: 0.45 vs 0.61; 0.48 with 0 random features). On separable Gaussian data GraphPFN is close to logreg (0.99 vs 1.00 at high separation).
Gate 1 will report a context-only logreg reference row next to B0.

## 6. Real heterographs (user-added, gts_pfn/data/preprocessed; data/data/preprocessed is a byte-identical copy)
ACM (P 4019 / A 7167 / S 60; relations P-P, P-A, P-S), IMDB (M 4278 / D 2081 / A 5257), DBLP (A 4057 / P 14328 / T 7723 / V 20),
Freebase (M 3492 / A 33401 / D 2502 / W 4459; featureless). Label homophily of the standard meta-paths (results/gate0_real/view_homophily.csv):
ACM P-P .885, PAP .809, PSP .639 | IMDB MDM .614, MAM .444 | DBLP APA .799 (coverage .64), APVPA .670, APTPA .324 | Freebase MDM .830 (coverage .53), MAM .692, MWM .643.
Relation utility differs within every dataset, so relation choice can matter on real data. Dense meta-paths (PSP 2.2M, APTPA 3.5M, APVPA 2.5M edges)
need a label-free degree cap before GraphPFN (to be fixed before any real run). Freebase/DBLP feature handling: see heteropfn/data/real.py notes.

## Gate 0 verdict
Generator, views and baselines run; leakage (poison) and relation-permutation sanity tests pass. Proceed to Gate 1 (synthetic, 200 dev tasks).
