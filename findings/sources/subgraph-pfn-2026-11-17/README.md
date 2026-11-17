# Subgraph-PFN pilot (SPFN01)

This directory implements a graph ICL prototype for skewed **observed label counts**. It preserves all anchors in a class with 1–4 labels and uses at most two summary tokens for any more observed class. The shared neural head operates on a small query-to-class-token attention problem. Each token carries a local finite-PPR weighted centroid plus an ordinary centroid and support information. Preparing tokens visits the complete graph with a sparse operator; there is no globally constant-time claim and no materialized query ego-subgraph. See [the specification](PLAN_v2.md).

`model.Input(x, edge, context, yc, query)` takes complete node features, local edge indices `[2,E]`, context node indices and observed class IDs, and query indices. Every class must have at least one context example and class coordinates are zero-based contiguous. `model.prepare(inp, 'tail')` creates query/group statistics. `run.scores(head, pack, num_classes)` returns per-query class probabilities with a fixed pretrained head. Hidden labels are outside this API. The main model is trained only on synthetic episodes. The repository preserves exact assumptions, not claims about calibration, feature missingness or novel real schemas.

The local audit and run commands from the repository root were:

```powershell
$researchPy = 'C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe'
& $researchPy -B -X utf8 subgraph_pfn_pilot/checks.py
& $researchPy -B -X utf8 subgraph_pfn_pilot/run.py
```

Both are create-only against `results/`; do not rerun into an existing artifact directory. The real panel reads the trusted local Planetoid files identified by SHA-256 in the lock; the synthetic prior uses fixed, disjoint RNG sequences. Saved checkpoints and prediction arrays stay local and untracked; the compact artifact manifest and audit document their identities. Scientific sources are locked; a new variant requires its own experiment and evaluation lock.
