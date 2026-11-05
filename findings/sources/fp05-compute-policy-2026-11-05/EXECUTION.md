# FP05 execution receipt

This file records work performed by the active orchestrator. It is not a scheduled service or an independent daemon.

## Training completed — 2026-11-05, approximately 06:20 UTC

Unified execution session39869 completed with exit0. Its command used the existing research Python, `-B -X utf8 -u`, imported `rl_run`, and ran `train(seed,arm)` for seeds11/12/13 and fixed/uniform/direct/group_pg sequentially. One GPU worker only. `training.log` preserves stdout/stderr.

All12 `training.json` receipts exist: each reports128 successful updates, unchanged frozen public parameters, and zero target updates. Total scientific updates1,536; sum of recorded fit seconds2,811.497653398663. These are workload times, not a robust speed benchmark. The correctness runs discarded2 optimizer steps and the development preflight discarded8 groups; those are separate.

## Evaluation started — 2026-11-05, approximately 06:20 UTC

Unified execution session64086 runs the following logical sequence in one Python process, with stdout/stderr in `evaluation.log`:

```python
for seed in [11,12,13]:
    for arm in ['fixed','uniform','direct','group_pg']:
        for split in ['final','real']:
            evaluate(seed, arm, split)
for split in ['final','real']:
    references(split)
```

This includes all fixed checkpoints, uniform mixtures/global controls, direct and PG replay000 controls, and the PG source-defined global secondary. No final metric is used to select a checkpoint or adjust the method. At this receipt's creation evaluation is genuinely running; completion and audit are to be appended below. The complete untracked outputs are hashed by the prediction and supplementary provenance manifests after successful audit.

## Provenance boundary

`results/design.json` precedes final-bank sampling. `results/lock.json` precedes all scientific fits. The earlier utility-file preparation receipt limitation is preserved in `PRELOCK_SOURCE_DELTA.json` and `BUILD_EXECUTION.md`. Scientific source, original banks, original public checkpoints and already saved outputs are not overwritten.

## Evaluation completed — 2026-11-05, 07:29 UTC

Session64086 completed with exit0. All24 fit/split evaluation receipts and both `final_references.json`/`real_references.json` exist. No evaluation was interrupted or resumed and no scientific fit was added. The main saved-array analysis then started in session23408; independent CPU audits follow its completion. The files keep the actual workload timing and forward counts.

## Analysis and reviews completed — 2026-11-05

Main analysis session23408 completed exit0: auditPASS, composite gateFAIL. Independent deployment session22485 and statistics session10923 both completed exit0/PASS; each used only CPU. The separate saved-artifact routing diagnostic also passed without importing/forwarding the model. All6,606 direct/PG main/replay pairs are bitwise equal. No inference repair, scientific rerun or additional fit was performed.

The same three perspectives completed `RESULT_METHOD.md`, `RESULT_INFERENCE.md`, and `RESULT_EXPERIMENTS.md`. They disclose their implementation/audit participation and do not claim external blind review. Root addressed the distinctions and limitations in `RESPONSE_RESULTS.md` and `DECISION_KO.md`. Current operator/budget/greedy-deployment candidate is No-Go. No GPU job for this cycle remains running. Final knowledge archival/build and local versioning are the remaining bookkeeping work.
