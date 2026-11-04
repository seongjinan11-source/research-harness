# FP01 implementation contract

Root owns common.py, analysis.py and integration. Inference agent owns operators.py/checks.py. Experiment agent owns bank.py/protocol.json. Method agent owns experiment.py. No edits outside those assigned files without coordination. No training/final-bank creation until root's lock step.

## common.py (root)

Exports ROOT,WORK,PYTHON,E (read-only edge-only benchmark module),OLD_BP=E.BP,np,torch; setup();sha(path);dump(path,obj);digest(model); cfg() from protocol.json.

`GraphCore(seed, device='cuda', dtype=torch.float32)` loads inherited Factor checkpoint seed and freezes it. `.encode(inp, graph_arm='full', chunk=128)` returns graph dict tensors on device/dtype:

- `j`:1×H×N×N Potts coupling, graph only; H1–3 after dropping only EXACT-zero mixture coefficients in float-j dtype, no threshold/topk;
- `valid`:1×N×N bool; `context`:1×N int64,-1unknown;
- `weights`:H frozen graph-only role weights; full_role_weights3 andactive_role_ids also retained;
- `p0`:N×K graph-only mixed prediction;
- `z`:N×24 frozen structural representation used by feature head;
- `classes`:int; `graph_arm`:str; graph constants/logz may be extra fields.

All graph encoder/j/q are computed with no_grad and detached. Conditional feature BP must be differentiable through its feature input even when j is frozen. `GraphCore.check()` checks original model digest unchanged. `GraphCore` is not a child of FeatureModel; checkpoints save only FeatureModel parameters.

`tensor_input(inp, device, dtype)` returns dict `x`N×F, `observed`N×Fbool,`classes`int. Query ids/truth never passed to FeatureModel. `read_real(row)` / `real_rows()` wrap the exact prior75episodes. `graph_identity(inp)` hashes graph/context only; same across masks/views. `source_lock(create=False)` hashes final sources, protocol, train/dev manifests, inherited sources/checkpoints and PLAN_v2. Does not include evaluation outputs or final bank to allow final creation after source lock. Final manifest has its own hash stored at completion.

## operators.py (inference agent)

`FeatureModel(arm, seed=11, iterations=24, damping=.5, grid_points=41)` where arm in A/B/C_global/C. `.forward(inp_tensor, graph, details=False)` ->N×K mixed probabilities; with details returns `(p, details_dict)` (branches useful). Models contain only new trainable feature parameters; no graph params. A/B identical feature-head initialization for same seed. A/B use old28→24→1 preferred-class gain head and same bias initialization, frozen z. C/C_global as PLAN_v2 moment initialization (read updated v2); C30→22→3, C_global3raw shared coefficients. Needed grid/model constants registered buffers. Source0 return p0 plus zero-connected trainable parameters if needed for backward; singleton algorithm must be checked before any optimized shortcut.

Expose a low-level centroid conditional BP function usable by checks with arbitrary H (including1), small N/K/F, double CPU, fixed hyperparameters or direct log-likelihood table and GH masses. It must permit exact discrete reference comparison independently of encoder. Document signature in operators.py docstring and message it to root. `.hyperparameters(inp, graph)` for C returns location,tau,sigma (and/or grid/logweights) for reference checks.

checks.py: imports common/operators after files exist. CLI writes results/checks.json with status PASS, numerical errors, loop approximation diagnostics, quadrature diagnostics, gradients, permutation/singleton/noobs/freeze and resource. No final predictions or optimizer training; optional tiny dev optimizer smoke must be counted and discarded. Do not lock sources.

## bank.py (experiment agent)

`prepare()` creates immutable train512 and dev12 base-task/view banks and manifest(s), no PCFI/native imputation. `finalize()` only after results/lock.json and results/checks.json PASS creates final48 graphs and specified orthogonal diagnostic, then final_manifest.json.

`rows(split)` list records with base_id,view(raw/projected/orthogonal),mode,split, path/hash, graph/meta,class/source counts. `read(row)` -> `(inp, query, truth)` numpy; inp has x,observed,src,dst,context_ids,labels_context,classes,forward_seed. No PCFI/native features needed. Train paired rows raw/projected one episode/base using original rate cycle and same old-generator RNG graph/noise/context/source. Same original512 task seeds from old cfg. Independent projection RNG. Final all6 nested rates +none and shuffle95 controls for projected (raw full6rates sufficient); orthogonal first of eachN×K at75/95, as plan. Common own-feature-missing queries across rates; no hidden truth in inp. New final/dev seed constants fixed in protocol before generation. Record covariance/rank/moments generator-side only. Keep bank size modest.

Train raw A/B paired rows can have same base_id but distinct view; schedule ids0..511 as oldseed+94000 permutation. Need helper `train_rows(view)` sorted by base_id or use rows('train') filtered by view.

## experiment.py (method agent)

Actions train/evaluate, flags --seed11 etc and optional --arm; no silent partial training restart. Calls common.setup/source_lock and bank.rows/read, operators.FeatureModel. Training graph core full definition per seed. Cache graph encode once per base task shared across arms (CPU cache or GPU if bounded). A uses raw view; B/C_global/C projected, same original seed-specific schedule/256×2. AdamW schedule from PLAN_v2. CE can supervise original noncontext queries including observed source (disclose); final own-feature-missing queries only. Store initial/final digests, graph before/after, task order/view ids, updates/exposures, logs/cost, source lock and checkpoint hashes. Zero-source gradient steps valid (do not crash/backprop no-grad loss); no hidden retry changing math.

Eval frozen models on synthetic final rows and all real rows, both full/edge. All arms same role graph cache. Save p,y,query,context,yc and metadata row/bank/model hashes, seed,arm,graph_arm,stage,time. Real graph core cache by graph+rep+graph_arm, then mode/view-specific feature tensors. Keep GPU memory bounded and clear graph cache betweengraphs; zero source can reuse p0. Do not run final until root explicitly calls. No published PFN training or target updates. Prediction paths deterministic and no overwriting saved arrays; resume only if source/checkpoint/bank hashes/identities match. Return all outcomes, no best-epoch/arm selection. Root builds final summary/audit.
