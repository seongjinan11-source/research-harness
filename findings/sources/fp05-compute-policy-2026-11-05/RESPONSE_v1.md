# FP05 revision response

2026-11-05. Material review objections are incorporated in PLAN_v2; no scored final data or model fit was used to revise design.

1. All arms and all8 traces use ONE common feature-position epsilon per update; separate action RNG. Different epsilon per trace invalidates exact tree normalization. Fixed duplicate slots do not create independent examples or compute parity.
2. One-on-policy-update GRPO clipping is inactive. Rename group_pg, remove clip/entropy, state exact group-normalized REINFORCE loss. This is a standard RL baseline, not new GRPO.
3. Keep CE adapter updates separate from policy gradient; frozen LimiX receives neither. Report trained-checkpoint000 replay and feature-utility difference-in-differences rather than attributing all PG−fixed to routing/value inference.
4. Natural/disjoint halves are mask regimes, not50% observed queries. Actual training exposures and full_natural support recorded. Full guard uses natural fully observed-query accuracy CI lower>−2pp. Fixed handling of primary empty-query cases prevents outcome-based resampling.
5. Define RP→RMS→PCFI order on visible whole rows and compare projected frozen references. Historical native models remain distinct. Existing internal-PCFI already fine-tuned strong public GraphPFN; this is not the first pretrained experiment.
6. Count128 taskgroups/fit and12288 logical trace slots separately from physical unique episodes, actual calls, walltime and memory. Uniform global trace selection uses dev only and its cost is charged.
7. Preserve strong direct expected-gradient comparator. A small8-trace action space may make RL unnecessary; a failure is an informative bounded result, not a reason for a new temperature/rank sweep.
8. Final experimental review accepted the predefined available-rate primary and97.5% full guard. Global trace selection is explicitly dev calibration (8graphs×7settings×8traces), not train-only; keep its cost. Direct sums p-weighted losses without another /8; sampled means use /8. Clip adapter/policy parameter sets separately. Reserve16 full_natural training groups to guarantee observed-query support and record actual counts.
