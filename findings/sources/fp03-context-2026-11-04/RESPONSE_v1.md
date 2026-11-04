# FP03 implementation-before-results review response

Root accepts the following material objections before any new branch prediction:

- Independent reviewers found CS has one absent context class and another has only3labels. Macro calibration averages represented classes only; trimming/masking is recorded and missing-class validation is not claimed.
- Single-label LOO on balanced context reveals the held class through remaining counts. PLAN_v2 replaces it with balanced leave-one-per-class folds on m selected nodes/class, hiding extra context labels during calibration. Full final inference still uses all context labels. Counts change from2085 to285 calibration branch forwards; no compute result motivated the change.
- Finite strong fields can underflow FP32 softmax. Raw probabilities are retained and the fixed float64 clip/renormalize/logscore convention is explicit and independently reconstructible. The score is log score, not positive NLL.
- Final MF readout and all-cavity old/new states are specified; no self/neighbor double count or hidden extra field update is allowed. No full-context message warm cache is shared with a fold.
- Uniform prior/mean-score weights are generalized predictive averaging. They do not prove calibrated parameter uncertainty, identify true homophily or give a new PFN training contribution. Candidate support/fold uncertainty and final predictive gain are distinct diagnostics.

Historical artifacts and theFP02 failed operator remain immutable. This scientific revision startsFP03 in its ownbranch. Three reviews and fixedGate are recorded beforeimplementation; correctness tests/source lock precede actualreal predictions.
