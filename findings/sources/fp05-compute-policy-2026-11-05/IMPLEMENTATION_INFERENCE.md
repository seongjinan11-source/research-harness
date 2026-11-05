# FP05 inference/controller implementation

2026-11-05. Owned files: `rl_model.py` and this document. No upstream, plan, protocol, bank, historical result or branch was changed by this implementation. AST syntax parsing passed. No Model construction, source-data generation, training or GPU run was executed by this agent; root runs the independent suite before source lock.

## Public API

```python
model = Model(seed, device='cuda')
out = model(inp, trace=None, mode='greedy', action_rng=None)
reference = model.original(inp)
```

`inp` must have exactly the whitelist `values,observed,src,dst,context_ids,context_labels,query_ids,classes,forward_seed`. Values are the already represented PCFI inputs. Missing physical entries may have legitimate nonzero imputations and are NOT zero-gated by the model. The raw hidden truth belongs outside this interface. `original()` also permits NaN entries for the declared frozen projected-NaN reference.

Output has:

- `logits[Q,K]`: normalized log probabilities in supplied physical class-ID order. They work with runner cross-entropy or softmax and remain live for adapter gradients.
- `logp`: live scalar sum of the three conditional policy log probabilities.
- `states[3,14]`: detached FP32 policy summaries.
- `action_logits[3]`: live FP32 Bernoulli logits, one per stage.
- `trace`: three actual binary actions; `cost`: their sum; `action_calls=3`; `decoder_class_order`: the deterministic context-first class-coordinate map.

Mode is `greedy` or `sample`. A supplied tuple overrides actions in either mode while still computing current conditional logp. Sampling requires an explicit `np.random.Generator`. Greedy is action1 iff logit>0; equal initial binary probabilities tie to0. Forced traces should use `model(inp, trace=(0,1,0))`, not a third mode name.

Additional interfaces are `adapter_parameters()`, `policy_parameters()`, `frozen_digest()`, `trainable_state()`, `load_trainable_state(state)`, and read-only `hook_calls_total`. `trainable_state()` stores CPU clones of adapter/policy tensors, public checkpoint SHA256s and frozen-parameter digest. Loading verifies names/shapes/public identity before copying. Full model pickling is not the checkpoint contract.

## Actual pretrained path

`rl_util.setup()` returns pinned `mgpfn.upstream`. `U.load_light(always_checkpoint=True)` creates the actual released12-wrapper predictor and loads both local public checkpoints. Public construction uses a forked constant seed0; training seed changes only the new policy initialization. All parameters in each released wrapper's `conv` and `mlp`, including their residual LayerNorms, are trainable. All remaining base parameters are frozen. Exact adapter names and checkpoint keys are exposed by `adapter_names` and `loaded_checkpoint_names`; checkpoint hashes and upstream commit are stored as attributes.

Only forward hooks at layers2/5/8 are added in memory. Their closure holds a weak reference to the owner. The original base→residual-conv→residual-MLP wrapper runs first. Action1 then executes the SAME layer's `conv` followed by the SAME `mlp` over the full hidden tensor and its actual internally permuted graph. The Transformer block is not repeated. Attention return entries are retained. Original Light still performs its normal random-feature augmentation, context-first permutation, preprocessing, all12 blocks and decoding. `original(inp)` executes it with controller hooks bypassed.

Autograd remains enabled in eval mode. Frozen Transformer parameters do not imply `no_grad()` around their operations. Policy reads use detached hidden states, so policy loss has no hidden→adapter path. CE through returned query scores remains live through the graph adapters and intervening frozen Transformer operations. No loss, optimizer, entropy term, advantage computation or training schedule is embedded in this file.

For each complete forward, `paired_rng(forward_seed,device)` forks/restores CPU and selected CUDA RNG state. It seeds the standard random node features and random orthogonal feature-position draw identically across traces. It does not reset the separate NumPy action generator. Hook decisions cannot be drawn again on backward: `_active` exists only during the original forward, each stage is accepted exactly once and `hook_calls_total` allows the independent checks to assert no decision count increases during checkpoint recomputation. Always-checkpoint changes storage/recomputation, not the requested operator. Actual parity and callback behavior must pass root's checks before lock.

## Fourteen-state definition

The standalone helper is `policy_state(label_states[N,D],context_labels[C],stage,observed[N,F],indegree[N])`, also available as `Model.summary`. Nodes are already context-first; the first C correspond to context labels sorted by original node ID. Stage is0/1/2. The label token is the final token `x[0,:,-1,:]`, without flattening feature tokens into it.

State order exactly follows V2: stage/3; observed cell fraction; observed node fraction; C/N; log(N)/10; log(F)/10; mean indegree/N; mean label-token L2 norm/20; mean population variance across nodes over hidden components/10; context-mean/unlabelled-mean cosine; same-class context pair cosine; different-class pair cosine; same-valid; different-valid. Unlabelled means all noncontext nodes. Cosine pairs are unordered i<j and require both norms>1e−12. Unsupported/zero-norm cases produce0, with same/different validity flags as applicable. All non-stage entries clamp to[−10,10]. Computation is explicitly FP32 even under runner autocast; nonfinite hidden/stat/logit fails visibly.

The policy is14→32→1 with Tanh hidden activation and zero-initialized final weight/bias. Thus every conditional initial probability is1/2. Hidden first-layer initialization differs by policy seed. No raw class ID, query target, generator parameter or hidden raw numeric value enters the state.

## Class-coordinate clarification approved before execution

The actual vendored `MulticlassTargetEncoder` maps the sorted context-unique class labels to consecutive ranks. Without output correction, context labels {0,2} would make native head1 mean actual class2. The common postprocess in BOTH `original()` and controlled forward is:

1. Take the native first K head slots and renormalize them.
2. Set decoder slot order to sorted context-represented IDs followed by sorted unrepresented supplied IDs.
3. Preserve the total probability mass in unrepresented head slots and distribute that mass uniformly across unrepresented classes. This prevents arbitrary absent-class head order from naming different unanchored classes.
4. Inverse-permute to the supplied class-ID coordinates and select queries in their original requested order.

When every class is represented, no absent-class averaging occurs, so the000 reference is literal upstream first-K renormalization. The mapping uses only context labels and supplied K, never query targets. It is a coordinate/postprocessing contract shared by every arm and matched frozen reference; it does not claim that unused pretrained head slots have learned absent-class semantics. The policy summaries use only context equality groups. Whole pretrained-model class or feature-order equivariance remains a diagnostic, not a new guarantee.

The released LimiX requires0<C<N; unsupported empty context raises a clear input error rather than fabricating labels. Source0 with nonempty context is supported. Context/query overlap is rejected because original Light returns dummy context predictions. K is supplied externally, and its maximum must fit the public decoder. Node/context IDs and labels are validated and aligned to Light's stable internal permutation.

## Precision and remaining verification

Adapter/new-policy parameters remain FP32 for normal GradScaler use; root runner controls FP16 autocast. Policy statistics, logits and logp are FP32. The returned query log probabilities are also FP32. The independent suite must check original000 output/adapter-gradient parity, all8 prefix consistency and probability sum1, actor/adapter gradient boundaries, coordinate correction including absent classes, RNG restoration/isolation, save/load, finite optimizer step and no new hook decisions during backward. Tiny independent tests and native resource smoke are root-owned. Passing these does not establish predictive utility, policy-only causality or novelty.
