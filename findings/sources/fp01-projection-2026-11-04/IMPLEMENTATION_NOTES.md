# Pre-lock execution notes

The reviewed PLAN_v2 snapshot remains unchanged. A pre-lock compute optimization drops only role lanes whose frozen mixture coefficient is exactly0 after casting to the node-probability dtype, the cast already used by the original mixture. There is no threshold/topk or renormalization of omitted positive coefficients. Full role weights and active ids remain in graph metadata. Single/multilane numerical parity must be checked. Conditional BP remains differentiable. This does not choose roles from feature/test outputs.

A/B compute only the original conditional BP messages and beliefs, omitting the unused Bethe score because all role weights are already fixed by the common graph core. Check against the original categorical_bp implementation on small inputs before lock.
