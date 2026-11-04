"""FP03 immutable balanced-context calibration and saved-probability audit.

Root alone creates the source lock and executes the real CUDA experiment.
Production uses operators.balanced_folds/score_weights; report independently
reconstructs those operations with NumPy. Historical evidence is read-only.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows import order.
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
RESULTS = ROOT / "results"
MANIFEST = ROOT / "bank/dev_manifest.json"
LOCK = RESULTS / "lock.json"
CHECKS = RESULTS / "checks.json"
GRAPHS = ("Cora", "CiteSeer", "PubMed", "CoauthorCS", "CoauthorPhysics")
ARMS = ("adaptive", "prior", "map", "fixed_edge", "graph_off")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier")
_MODULES = {}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def arrays_digest(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode("utf8"))
        if isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            assert not a.dtype.hasobject
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        else:
            h.update(canonical(value).encode("utf8"))
    return h.hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def module(name, path):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, Path(path))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        _MODULES[name] = m
    return _MODULES[name]


def old():
    return module("fp03_readonly_fp02_runner", WORK / "factor_nuisance_cycle/run.py")


def operator():
    return module("fp03_production_operator", ROOT / "operators.py")


def config():
    c = read_json(ROOT / "protocol.json")
    assert c["cycle"] == "FP03" and c["version"] == 2
    assert c["graphs"] == list(GRAPHS) and c["modes"] == ["none", "95"]
    assert c["arms"] == list(ARMS) and c["candidate_order"] == "a-major,b-minor"
    assert c["a"] == [-math.log(2), 0., math.log(2), math.log(4), math.log(16)]
    assert c["b"] == [-2., 0., 2.] and c["fixed_edge_index"] == 10 and c["graph_off_index"] == 4
    assert c["seed"] == 11 and c["repetition"] == 0
    assert c["operator"]["iterations"] == 24 and c["operator"]["damping"] == .5
    assert c["operator"]["chunk"] == 16 and c["operator"]["dtype"] == "float32"
    assert c["calibration"]["clip"] == 1e-8 and c["calibration"]["weight_scale"] == 1.
    assert c["calibration"]["normalize_after_clip"] and c["calibration"]["none95_common_weights"]
    assert c["expected"] == dict(episodes=10, graphs=5, original_contexts=139,
                                  calibration_points=126, calibration_folds=19,
                                  calibration_branch_forwards=285, final_branch_forwards=150,
                                  derived_prediction_records=50, reference_records=70,
                                  aggregate_scored_records=120)
    assert c["execution"]["workers"] == 1 and c["execution"]["optimizer_updates"] == 0
    assert c["execution"]["target_gradient_updates"] == c["execution"]["new_neural_fits"] == 0
    assert not c["scope"]["confidence_intervals"]
    return c


def context_vector(inp):
    context = np.full(len(inp["x"]), -1, np.int64)
    context[np.asarray(inp["context_ids"], np.int64)] = np.asarray(inp["labels_context"], np.int64)
    return context


def compact_split(inp):
    """Production split comes only from the checked canonical operator helper."""
    context = context_vector(inp)
    spec = operator().balanced_folds(torch.as_tensor(context), classes=int(inp["classes"]))
    contexts = numpy(spec["fold_contexts"], np.int64)
    held = numpy(spec["held_nodes"], np.int64)
    labels = numpy(spec["held_labels"], np.int64)
    represented = numpy(spec["represented_classes"], np.int64)
    m = int(spec["m"])
    assert contexts.shape == (m, len(context))
    assert held.shape == labels.shape == (m, len(represented))
    folds = []
    for t in range(m):
        ids = np.flatnonzero(contexts[t] >= 0)
        folds.append(dict(fold=t, held_ids=held[t].tolist(), held_labels=labels[t].tolist(),
                          held_positions=list(range(t * len(represented), (t + 1) * len(represented))),
                          context_ids=ids.tolist(), context_labels=contexts[t, ids].tolist()))
    assert np.array_equal(numpy(spec["held_flat"], np.int64), held.reshape(-1))
    result = dict(m=m, points=int(spec["points"]), represented_classes=represented.tolist(),
                  class_counts=numpy(spec["class_counts"], np.int64).tolist(),
                  original_context_ids=np.asarray(inp["context_ids"], np.int64).tolist(),
                  original_context_labels=np.asarray(inp["labels_context"], np.int64).tolist(),
                  selected_ids=held.reshape(-1).tolist(), selected_labels=labels.reshape(-1).tolist(),
                  unused_ids=numpy(spec["unused_nodes"], np.int64).tolist(), folds=folds,
                  stable_ids="existing graph row indices; not numeric features",
                  class_slot_order="ascending minimum stable ID within represented class")
    assert result["points"] == len(result["selected_ids"])
    return result


def build_manifest():
    c = config()
    P = old()
    P.require_lock()
    inherited = P.verified_manifest()
    rows, splits = inherited["rows"], {}
    for graph in GRAPHS:
        pair = [x for x in rows if x["row"]["graph"] == graph]
        assert len(pair) == 2 and {x["row"]["mode"] for x in pair} == {"none", "95"}
        inputs = [P.episode(x["row"])[0] for x in pair]
        assert context_vector(inputs[0]).tolist() == context_vector(inputs[1]).tolist()
        assert pair[0]["graph_identity"] == pair[1]["graph_identity"]
        split = compact_split(inputs[0])
        splits[graph] = dict(split=split, split_sha256=fingerprint(split),
                             graph_identity=pair[0]["graph_identity"], nodes=pair[0]["nodes"],
                             classes=pair[0]["classes"],
                             calibration_branches=15 * split["m"])
    counts = dict(original_contexts=sum(len(s["split"]["original_context_ids"]) for s in splits.values()),
                  calibration_points=sum(s["split"]["points"] for s in splits.values()),
                  calibration_folds=sum(s["split"]["m"] for s in splits.values()),
                  calibration_branch_forwards=sum(s["calibration_branches"] for s in splits.values()),
                  final_branch_forwards=15 * len(rows), episodes=len(rows), graphs=len(splits),
                  derived_prediction_records=5 * len(rows),
                  reference_records=sum(len(r["references"]) for r in rows), aggregate_scored_records=120)
    assert counts == c["expected"]
    return dict(status="PREPARED", cycle="FP03", protocol_sha256=sha(ROOT / "protocol.json"),
                inherited_manifest=str(P.MANIFEST), inherited_manifest_sha256=sha(P.MANIFEST),
                inherited_lock_sha256=sha(P.LOCK), rows=rows, calibration=splits, counts=counts,
                bank_arrays_rewritten=False, final_bank=False)


def prepare(resume=False):
    began = time.perf_counter()
    result = build_manifest()
    if MANIFEST.exists():
        if not resume:
            raise FileExistsError("Prepared manifest exists; --resume verifies exact contents")
        saved = read_json(MANIFEST)
        assert {k: v for k, v in saved.items() if k != "preparation_seconds"} == result
        return saved
    result["preparation_seconds"] = time.perf_counter() - began
    write_json_new(MANIFEST, result)
    return result


def verified_manifest():
    saved = read_json(MANIFEST)
    assert {k: v for k, v in saved.items() if k != "preparation_seconds"} == build_manifest()
    return saved


def source_state():
    """Return exact root-owned lock schema; no source lock is created here."""
    verified_manifest()
    checks = read_json(CHECKS)
    assert checks["status"] == "PASS" and checks.get("optimizer_updates", 0) == 0
    assert not checks.get("real_bank_used", False) and not checks.get("real_predictions_used", False)
    assert checks["tested_sources"]
    for name, expected in checks["tested_sources"].items():
        assert sha(name) == expected, f"Checked source changed: {name}"
    paths = [ROOT / name for name in ("run.py", "operators.py", "checks.py", "protocol.json",
             "PLAN_v1.md", "PLAN_v2.md", "RESPONSE_v1.md", "PRIOR_ART_SCOPE.md", "ANALYSIS_SPEC.md",
             "REVIEW_METHOD_v1.md", "REVIEW_INFERENCE_v1.md", "REVIEW_EXPERIMENTS_v1.md",
             "IMPLEMENTATION_RUNNER.md", "IMPLEMENTATION_INFERENCE.md", "IMPLEMENTATION_CHECKS.md")]
    api = ROOT / "IMPLEMENTATION_API.md"
    if api.exists():
        paths.append(api)
    paths += [MANIFEST, CHECKS] + [Path(x) for x in checks["tested_sources"]]
    inherited = old().require_lock()
    paths += [old().LOCK, old().MANIFEST]
    for name, expected in inherited["files"].items():
        assert sha(name) == expected
        paths.append(Path(name))
    files = {str(p): sha(p) for p in sorted({p.resolve() for p in paths}, key=str)}
    return dict(status="LOCKED", cycle="FP03", files=files, manifest_sha256=sha(MANIFEST),
                checks_sha256=sha(CHECKS), inherited_lock_sha256=sha(old().LOCK),
                optimizer_updates=0, target_gradient_updates=0)


def require_lock():
    if not LOCK.exists():
        raise RuntimeError("Root must create results/lock.json after PASS checks")
    saved = read_json(LOCK)
    assert saved == source_state(), "Current sources/checks/banks differ from root lock"
    return saved


def git_snapshot():
    result = {}
    for key, args in (("branch", ["branch", "--show-current"]), ("commit", ["rev-parse", "HEAD"])):
        try:
            p = subprocess.run(["git", "-c", f"safe.directory={WORK.as_posix()}", "-C", str(WORK), *args],
                               capture_output=True, text=True, check=False)
            result[key] = p.stdout.strip() if p.returncode == 0 else None
        except OSError:
            result[key] = None
    return result


def validate_p(p, shape=None):
    p = np.asarray(p)
    if shape is not None:
        assert p.shape == shape, (p.shape, shape)
    assert np.isfinite(p).all() and np.min(p, initial=0) >= -1e-7 and np.max(p, initial=0) <= 1 + 1e-7
    if p.size:
        assert np.max(np.abs(p.sum(-1) - 1)) < 3e-5


def pair_paths(kind, name):
    path = RESULTS / kind / f"{name}.npz"
    return path, path.with_suffix(".json")


def artifact_identity(stage, name, extra):
    return dict(cycle="FP03", stage=stage, name=name, source_lock_sha256=sha(LOCK),
                manifest_sha256=sha(MANIFEST), protocol_sha256=sha(ROOT / "protocol.json"),
                feature_checkpoint_sha256=sha(WORK / "factor_projection_cycle/results/C_global_s11/checkpoint.pt"),
                optimizer_updates=0, target_gradient_updates=0, **extra)


def saved_pair(path, side, identity, required_arrays, resume):
    path, side = Path(path), Path(side)
    if not path.exists() and not side.exists():
        return None
    if not path.is_file() or not side.is_file():
        raise RuntimeError(f"Partial NPZ/sidecar pair; never overwrite: {path}")
    if not resume:
        raise FileExistsError(f"Complete pair exists; --resume required: {path}")
    meta = read_json(side)
    assert meta["status"] == "COMPLETE" and meta["identity"] == identity
    assert sha(path) == meta["sha256"], path
    with np.load(path, allow_pickle=False) as z:
        for key, expected in required_arrays.items():
            assert key in z.files and np.array_equal(z[key], expected), (path, key)
        data = {k: z[k].copy() for k in z.files}
    return meta, data


def save_pair(path, side, identity, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists() and not side.exists()
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    meta = dict(status="COMPLETE", identity=identity, path=str(path.resolve()), sha256=sha(path), **metadata)
    write_json_new(side, meta)
    return meta, arrays


def derive_predictions(candidate_p, weights, map_index=None):
    """Production probability aggregation only; report has a separate version."""
    p = np.asarray(candidate_p)
    w = np.asarray(weights, np.float64)
    assert p.ndim == 3 and p.shape[0] == 15 and w.shape == (15,)
    assert np.isfinite(w).all() and (w >= 0).all() and abs(w.sum() - 1) < 1e-12
    index = int(w.argmax()) if map_index is None else int(map_index)
    result = dict(adaptive=np.einsum("g,gqk->qk", w, p.astype(np.float64)),
                  prior=p.astype(np.float64).mean(0), map=p[index], fixed_edge=p[10], graph_off=p[4])
    return {arm: np.asarray(result[arm], np.float32) for arm in ARMS}


def _sync():
    torch.cuda.synchronize()


def _batch_forward(src, dst, n, contexts, k, obs, likelihood, logw, aa, bb):
    """Candidate/fold chunking is solely runner-owned and fixed at sixteen."""
    assert len(contexts) == len(aa) == len(bb) <= 16
    return operator().contrast_bp(src, dst, n, contexts, k, obs, likelihood, logw,
                                   aa, bb, iterations=24, damping=.5, details=False)


def _small_info(info):
    result = {}
    for key, value in info.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            a = numpy(value)
            if a.size > 256:
                assert np.isfinite(a).all()
                result[key] = dict(shape=list(a.shape), min=float(a.min()), max=float(a.max()), mean=float(a.mean()))
            else:
                result[key] = plain(a)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = plain(value)
    return result


def calibrate(graph, spec, case, inp, resume, git):
    setup_began = time.perf_counter()
    split = spec["split"]
    required = dict(held_nodes=np.asarray(split["selected_ids"], np.int64),
                    held_labels=np.asarray(split["selected_labels"], np.int64),
                    context=np.asarray(inp["context_ids"], np.int64),
                    yc=np.asarray(inp["labels_context"], np.int64),
                    unused_nodes=np.asarray(split["unused_ids"], np.int64))
    identity = artifact_identity("calibration", graph, dict(graph_identity=spec["graph_identity"],
                split_sha256=spec["split_sha256"], dtype="float32", classes=case["classes"]))
    path, side = pair_paths("calibration", graph)
    previous = saved_pair(path, side, identity, required, resume)
    if previous is not None:
        validate_p(previous[1]["p"], (15, split["points"], case["classes"]))
        return previous
    n, k = case["nodes"], case["classes"]
    src = torch.as_tensor(inp["src"], device="cuda", dtype=torch.long)
    dst = torch.as_tensor(inp["dst"], device="cuda", dtype=torch.long)
    obs = torch.zeros(n, 0, device="cuda", dtype=torch.bool)
    likelihood = torch.empty(n, 0, 1, device="cuda", dtype=torch.float32)
    logw = torch.zeros(1, device="cuda", dtype=torch.float32)
    aa, bb = operator().candidate_table(device="cuda", dtype=torch.float32)
    jobs = [(fold, g) for fold in split["folds"] for g in range(15)]
    held_p = np.empty((15, split["points"], k), np.float32)
    _sync(); torch.cuda.reset_peak_memory_stats(); baseline = torch.cuda.memory_allocated()
    began = time.perf_counter()
    setup_seconds = began - setup_began
    chunks = 0
    for start in range(0, len(jobs), 16):
        current = jobs[start:start + 16]
        cv = np.full((len(current), n), -1, np.int64)
        gi = []
        for j, (fold, g) in enumerate(current):
            cv[j, fold["context_ids"]] = fold["context_labels"]
            gi.append(g)
        contexts = torch.as_tensor(cv, device="cuda", dtype=torch.long)
        idx = torch.as_tensor(gi, device="cuda", dtype=torch.long)
        with torch.no_grad():
            p, info = _batch_forward(src, dst, n, contexts, k, obs, likelihood, logw, aa[idx], bb[idx])
            for j, (fold, g) in enumerate(current):
                nodes = torch.as_tensor(fold["held_ids"], device="cuda", dtype=torch.long)
                held_p[g, fold["held_positions"]] = p[j].index_select(0, nodes).detach().cpu().numpy()
        chunks += 1
        del contexts, idx, p, info
    _sync()
    seconds = time.perf_counter() - began
    peak = torch.cuda.max_memory_allocated()
    validate_p(held_p, (15, split["points"], k))
    score_began = time.perf_counter()
    weights, info = operator().score_weights(torch.as_tensor(held_p), required["held_labels"])
    weights, scores = numpy(weights, np.float64), numpy(info["scores"], np.float64)
    assert weights.shape == scores.shape == (15,)
    if split["m"] <= 1:
        if split["points"]:
            assert np.max(abs(held_p - 1 / k)) < 2e-6
        assert np.max(abs(weights - 1 / 15)) < 1e-12
    a, b = operator().candidate_table(device="cpu", dtype=torch.float64)
    arrays = dict(p=held_p, scores=scores, weights=weights, a=numpy(a), b=numpy(b), **required)
    return save_pair(path, side, identity, arrays, dict(graph=graph, split=split,
            map_index=int(scores.argmax()), score_summary={key: plain(value) for key, value in info.items()
                if key not in ("normalized_held", "per_class_scores", "scores")},
            branch_forwards=len(jobs), chunks=chunks, inference_seconds=seconds,
            setup_seconds=setup_seconds, scoring_seconds=time.perf_counter()-score_began,
            peak_mib=peak / 2**20, incremental_peak_mib=(peak-baseline) / 2**20,
            graph_only=True, git=git, neural_optimizer_updates=0))


def run(resume=False, device="cuda"):
    validate_began = time.perf_counter()
    if str(device) != "cuda":
        raise ValueError("FP03 real execution is one float32 CUDA worker")
    lock = require_lock()
    validation_seconds = time.perf_counter() - validate_began
    manifest = read_json(MANIFEST)
    RESULTS.mkdir(parents=True, exist_ok=True)
    token, marker = str(time.time_ns()), RESULTS / ".active_run.json"
    write_json_new(marker, dict(token=token, source_lock_sha256=sha(LOCK)))
    began, git, records, calibration_records = time.perf_counter(), git_snapshot(), [], []
    new_calibration_branches = new_final_branches = 0
    attempt_path = RESULTS / "attempts" / f"{token}.json"
    try:
        torch.set_num_threads(4)
        assert torch.cuda.is_available()
        P, O = old(), operator()
        R = P.inherited_runner()
        model_start = time.perf_counter()
        model, fitted = R.load_model("C_global", 11, device="cuda")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval(); _sync()
        model_seconds = time.perf_counter() - model_start
        initial_digest = R.G.digest(model)
        assert initial_digest == fitted["final_sha256"]
        for graph in GRAPHS:
            cases = [c for c in manifest["rows"] if c["row"]["graph"] == graph]
            none = next(c for c in cases if c["row"]["mode"] == "none")
            read_start = time.perf_counter()
            inp, _ = P.episode(none["row"])
            shared_read_seconds = time.perf_counter() - read_start
            cp, cs = pair_paths("calibration", graph)
            existed = cp.exists() and cs.exists()
            cal_meta, cal = calibrate(graph, manifest["calibration"][graph], none, inp, resume, git)
            if not existed:
                new_calibration_branches += cal_meta["branch_forwards"]
            calibration_records.append(dict(graph=graph, path=str(cp.resolve()), metadata=str(cs.resolve()),
                                            sha256=sha(cp), metadata_sha256=sha(cs)))
            for case in cases:
                row_start = time.perf_counter()
                inp, targets = P.episode(case["row"])
                assert arrays_digest(inp) == case["input_sha256"]
                assert arrays_digest(targets) == case["target_arrays_sha256"]
                row_read_seconds = time.perf_counter() - row_start
                identity = artifact_identity("final", case["case_id"], dict(input_sha256=case["input_sha256"],
                    target_arrays_sha256=case["target_arrays_sha256"], bank_sha256=case["bank_sha256"],
                    calibration_sha256=sha(cp), split_sha256=manifest["calibration"][graph]["split_sha256"],
                    graph_identity=case["graph_identity"], classes=case["classes"], dtype="float32"))
                required = dict(**targets, weights=cal["weights"], scores=cal["scores"])
                path, side = pair_paths("predictions", case["case_id"])
                previous = saved_pair(path, side, identity, required, resume)
                if previous is None:
                    prep_start = time.perf_counter()
                    n, k = case["nodes"], case["classes"]
                    visible = R.G.tensor_input(inp, device="cuda", dtype=torch.float32)
                    src = torch.as_tensor(inp["src"], device="cuda", dtype=torch.long)
                    dst = torch.as_tensor(inp["dst"], device="cuda", dtype=torch.long)
                    cv = torch.as_tensor(context_vector(inp), device="cuda").expand(15, -1)
                    with torch.no_grad():
                        obs, ll, lw, _ = P.feature_table(model, visible)
                    aa, bb = O.candidate_table(device="cuda", dtype=torch.float32)
                    _sync(); prep_seconds = time.perf_counter() - prep_start
                    baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                    pred_start = time.perf_counter()
                    with torch.no_grad():
                        all_p, info = _batch_forward(src, dst, n, cv, k, obs, ll, lw, aa, bb)
                        q = torch.as_tensor(targets["query"], device="cuda", dtype=torch.long)
                        candidate_p = all_p.index_select(1, q).detach().cpu().numpy()
                    _sync(); seconds = time.perf_counter() - pred_start
                    peak = torch.cuda.max_memory_allocated()
                    validate_p(candidate_p, (15, len(targets["query"]), k))
                    aggregation_began = time.perf_counter()
                    controls = derive_predictions(candidate_p, cal["weights"], int(cal["scores"].argmax()))
                    aggregation_seconds = time.perf_counter() - aggregation_began
                    assert np.max(abs(candidate_p[4] - 1/k)) < 2e-6, "Graph-off own-missing query must be uniform"
                    arrays = dict(candidate_p=candidate_p, **required, **controls)
                    meta, _ = save_pair(path, side, identity, arrays, dict(case_id=case["case_id"], graph=graph,
                        mode=case["row"]["mode"], branch_forwards=15, chunks=1,
                        inference_seconds=seconds, preparation_seconds=prep_seconds,
                        aggregation_seconds=aggregation_seconds,
                        row_read_seconds=row_read_seconds, common_calibration_input_read_seconds=shared_read_seconds,
                        peak_mib=peak / 2**20, incremental_peak_mib=(peak-baseline) / 2**20,
                        map_index=int(cal["scores"].argmax()), operator_info=_small_info(info),
                        git=git, neural_optimizer_updates=0, target_gradient_updates=0))
                    new_final_branches += 15
                    del visible, src, dst, cv, obs, ll, lw, aa, bb, all_p, info, q
                else:
                    meta, data = previous
                    validate_p(data["candidate_p"], (15, len(targets["query"]), case["classes"]))
                    for arm in ARMS:
                        validate_p(data[arm], (len(targets["query"]), case["classes"]))
                records.append(dict(case_id=case["case_id"], path=str(path.resolve()), metadata=str(side.resolve()),
                                    sha256=sha(path), metadata_sha256=sha(side)))
                print("FP03_FINAL", len(records), "/10", case["case_id"], round(meta["inference_seconds"], 3), flush=True)
                gc.collect(); torch.cuda.empty_cache()
        assert len(records) == 10 and len(calibration_records) == 5
        assert R.G.digest(model) == initial_digest
        assert require_lock() == lock
        index = dict(status="COMPLETE", cycle="FP03", source_lock_sha256=sha(LOCK),
                     manifest_sha256=sha(MANIFEST), calibration=calibration_records, records=records,
                     expected=config()["expected"], optimizer_updates=0, target_gradient_updates=0)
        dest = RESULTS / "evaluation.json"
        if dest.exists():
            assert resume and read_json(dest) == index
        else:
            write_json_new(dest, index)
        write_json_new(attempt_path, dict(status="COMPLETE", token=token, git=git,
            wall_seconds=time.perf_counter()-began, validation_seconds=validation_seconds,
            model_load_seconds=model_seconds, new_calibration_branches=new_calibration_branches,
            new_final_branches=new_final_branches, verified_final_episodes=len(records),
            source_lock_sha256=sha(LOCK), optimizer_updates=0))
        return index
    except Exception as exc:
        write_json_new(attempt_path, dict(status="FAILED", token=token, git=git,
            wall_seconds=time.perf_counter()-began, validation_seconds=validation_seconds,
            new_calibration_branches=new_calibration_branches, new_final_branches=new_final_branches,
            verified_final_episodes=len(records), error=repr(exc), traceback=traceback.format_exc(), optimizer_updates=0))
        raise
    finally:
        if marker.exists() and read_json(marker).get("token") == token:
            marker.unlink()


def _numpy_split(context, k):
    """Independent audit construction: never calls the production split helper."""
    context = np.asarray(context, np.int64)
    counts = np.bincount(context[context >= 0], minlength=k)
    classes = [c for c in range(k) if counts[c] > 0]
    classes.sort(key=lambda c: int(np.flatnonzero(context == c).min()))
    m = min((int(counts[c]) for c in classes), default=0)
    chosen = {c: np.flatnonzero(context == c)[:m] for c in classes}
    held = np.asarray([[chosen[c][t] for c in classes] for t in range(m)], np.int64).reshape(m, len(classes))
    selected = held.reshape(-1)
    unused = np.setdiff1d(np.flatnonzero(context >= 0), selected)
    folds = []
    for t in range(m):
        clamped = np.setdiff1d(selected, held[t])
        folds.append(dict(fold=t, held_ids=held[t].tolist(), held_labels=context[held[t]].tolist(),
                          held_positions=list(range(t*len(classes), (t+1)*len(classes))),
                          context_ids=clamped.tolist(), context_labels=context[clamped].tolist()))
    return dict(m=m, represented_classes=classes, class_counts=counts.tolist(),
                selected_ids=selected.tolist(), selected_labels=context[selected].tolist(),
                unused_ids=unused.tolist(), folds=folds)


def _numpy_calibration(raw, labels):
    """Independent NumPy reconstruction, including score clipping convention."""
    p = np.asarray(raw, np.float64)
    labels = np.asarray(labels, np.int64)
    g, length, k = p.shape
    assert g == 15 and labels.shape == (length,)
    normalized = np.clip(p, 1e-8, 1)
    normalized /= normalized.sum(-1, keepdims=True)
    scores = np.zeros(g, np.float64)
    if length:
        own = np.log(normalized[:, np.arange(length), labels])
        per_class = [own[:, labels == c].mean(axis=1) for c in np.unique(labels)]
        scores = np.mean(np.stack(per_class, axis=1), axis=1)
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    entropy = -sum(float(w) * math.log(float(w)) for w in weights if w > 0)
    return scores, weights, dict(entropy=entropy, ess=float(1 / np.dot(weights, weights)),
                                 prior_kl=float(math.log(g)-entropy), map_index=int(np.argmax(scores)))


def _numpy_controls(p, weights, map_index):
    """Separate summation implementation, not derive_predictions()."""
    p = np.asarray(p)
    weighted = np.zeros(p.shape[1:], np.float64)
    prior = np.zeros_like(weighted)
    for g in range(15):
        weighted += float(weights[g]) * p[g].astype(np.float64)
        prior += p[g].astype(np.float64) / 15
    return dict(adaptive=weighted.astype(np.float32), prior=prior.astype(np.float32),
                map=p[int(map_index)].copy(), fixed_edge=p[10].copy(), graph_off=p[4].copy())


def _numpy_metrics(raw, labels, details=True):
    p = np.clip(np.asarray(raw, np.float64), 1e-8, 1)
    p /= p.sum(1, keepdims=True)
    labels = np.asarray(labels, np.int64)
    prediction = p.argmax(1)
    k = p.shape[1]
    cm = np.bincount(labels*k+prediction, minlength=k*k).reshape(k, k)
    true_n, pred_n, correct = cm.sum(1), cm.sum(0), np.diag(cm)
    union, present = (true_n+pred_n)>0, true_n>0
    result = dict(accuracy=float(correct.sum()/len(labels)),
                  balanced_accuracy=float((correct[present]/true_n[present]).mean()),
                  macro_f1=float((2*correct[union]/(true_n[union]+pred_n[union])).mean()),
                  nll=float(-np.log(p[np.arange(len(labels)), labels]).mean()),
                  brier=float(((p-np.eye(k)[labels])**2).sum(1).mean()))
    if details:
        result.update(class_support=true_n.tolist(),
                      class_recall=[float(correct[c]/true_n[c]) if true_n[c] else None for c in range(k)],
                      predicted_class_mass=(pred_n/len(labels)).tolist(),
                      true_class_mass=(true_n/len(labels)).tolist())
    return result


def _mean(records):
    assert records
    return {m: float(np.mean([r[m] for r in records])) for m in METRICS}


def _gate(table):
    cfg = config()["gate"]
    groups = {}
    for arm in ("adaptive", "prior", "fixed_edge", "old_full", "old_edge"):
        groups[arm] = {graph: table[(graph, "none", arm)] for graph in GRAPHS}
    a, f = groups["adaptive"], groups["old_full"]
    mean = {arm: _mean(list(rows.values())) for arm, rows in groups.items()}
    am, pm, em, xm = (mean[arm] for arm in ("adaptive", "prior", "old_edge", "fixed_edge"))
    values = dict(physics_gain_old_full=a["CoauthorPhysics"]["accuracy"]-f["CoauthorPhysics"]["accuracy"],
                  cs_gain_old_full=a["CoauthorCS"]["accuracy"]-f["CoauthorCS"]["accuracy"],
                  citeseer_accuracy_floor_old_full=a["CiteSeer"]["accuracy"]-f["CiteSeer"]["accuracy"],
                  citeseer_nll_ceiling_old_full=a["CiteSeer"]["nll"]-f["CiteSeer"]["nll"],
                  mean_balanced_accuracy_floor_old_edge=am["balanced_accuracy"]-em["balanced_accuracy"],
                  mean_nll_ceiling_old_edge=am["nll"]-em["nll"],
                  mean_accuracy_gain_prior=am["accuracy"]-pm["accuracy"],
                  mean_accuracy_floor_fixed_edge=am["accuracy"]-xm["accuracy"],
                  mean_nll_ceiling_prior=am["nll"]-pm["nll"],
                  mean_nll_ceiling_fixed_edge=am["nll"]-xm["nll"])
    checks = {}
    for name, value in values.items():
        ceiling = "ceiling" in name
        passed = value <= cfg[name]+1e-12 if ceiling else value >= cfg[name]-1e-12
        checks[name] = dict(value=float(value), threshold=cfg[name],
                            comparison="<=" if ceiling else ">=", passed=bool(passed))
    assert len(checks) == 10
    return dict(passed=all(x["passed"] for x in checks.values()), checks=checks,
                primary="adaptive", mode="none", equal_source_graph_weight=True,
                means=mean, both_nll_controls_required=True, independent_final_validation=False)


def _disagreement(p, w, aa, bb):
    """NumPy descriptive moments and TV, without a production diagnostics call."""
    p, w = np.asarray(p, np.float64), np.asarray(w, np.float64)
    aa, bb = np.asarray(aa, np.float64), np.asarray(bb, np.float64)
    ma, mb = float(w@aa), float(w@bb)
    mixture = np.sum(w[:, None, None]*p, axis=0)
    tv = [float((.5*np.abs(p[i]-p[j]).sum(-1)).mean()) for i in range(15) for j in range(i+1,15)]
    return dict(a_mean=ma, b_mean=mb, a_variance=float(w@(aa-ma)**2),
                b_variance=float(w@(bb-mb)**2), ab_covariance=float(w@((aa-ma)*(bb-mb))),
                mean_pairwise_total_variation=float(np.mean(tv)),
                weighted_mean_tv_to_mixture=float((w[:, None]*.5*np.abs(p-mixture[None]).sum(-1)).sum(0).mean()))


def report(resume=False):
    began = time.perf_counter()
    require_lock()
    manifest, index = read_json(MANIFEST), read_json(RESULTS / "evaluation.json")
    assert index["status"] == "COMPLETE" and index["source_lock_sha256"] == sha(LOCK)
    assert index["manifest_sha256"] == sha(MANIFEST)
    outputs = [RESULTS / name for name in ("summary.json", "audit.json", "budget.json", "REPORT_KO.md", "artifact_manifest.json")]
    if any(p.exists() for p in outputs):
        if not resume or not all(p.is_file() for p in outputs):
            raise FileExistsError("Complete/partial report exists; never overwrite. --resume requires a complete set")
        complete = read_json(outputs[-1])
        assert complete["status"] == "COMPLETE" and complete["source_lock_sha256"] == sha(LOCK)
        for f in complete["files"]:
            assert sha(f["path"]) == f["sha256"], f["path"]
        return complete
    assert len(index["records"]) == 10 and len(index["calibration"]) == 5
    P = old()
    cal_records = {r["graph"]: r for r in index["calibration"]}
    final_records = {r["case_id"]: r for r in index["records"]}
    assert len(cal_records) == 5 and len(final_records) == 10
    files, calibrations, cal_data, cal_metadata = {}, {}, {}, {}
    def track(path, role):
        path = Path(path).resolve()
        files[str(path)] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, role=role)
    max_score_error = max_weight_error = max_control_error = 0.
    for graph in GRAPHS:
        case = next(c for c in manifest["rows"] if c["row"]["graph"] == graph)
        inp, _ = P.episode(case["row"])
        split = manifest["calibration"][graph]["split"]
        independently = _numpy_split(context_vector(inp), case["classes"])
        for key, expected in independently.items():
            assert split[key] == expected, (graph, key, "split differs from independent NumPy")
        assert fingerprint(split) == manifest["calibration"][graph]["split_sha256"]
        rec = cal_records[graph]
        path, side = pair_paths("calibration", graph)
        assert Path(rec["path"]).resolve() == path.resolve() and Path(rec["metadata"]).resolve() == side.resolve()
        assert sha(path) == rec["sha256"] and sha(side) == rec["metadata_sha256"]
        meta = read_json(side)
        identity = artifact_identity("calibration", graph, dict(graph_identity=case["graph_identity"],
            split_sha256=manifest["calibration"][graph]["split_sha256"], dtype="float32", classes=case["classes"]))
        required = dict(held_nodes=np.asarray(split["selected_ids"], np.int64),
                        held_labels=np.asarray(split["selected_labels"], np.int64),
                        context=np.asarray(inp["context_ids"], np.int64), yc=np.asarray(inp["labels_context"], np.int64),
                        unused_nodes=np.asarray(split["unused_ids"], np.int64))
        meta, data = saved_pair(path, side, identity, required, True)
        assert meta["branch_forwards"] == split["m"]*15 and meta["neural_optimizer_updates"] == 0
        assert meta["split"] == split
        validate_p(data["p"], (15, split["points"], case["classes"]))
        cfg = config()
        aa = np.repeat(np.asarray(cfg["a"], np.float64), 3)
        bb = np.tile(np.asarray(cfg["b"], np.float64), 5)
        assert np.allclose(data["a"], aa, atol=1e-15, rtol=0) and np.array_equal(data["b"], bb)
        scores, weights, stats = _numpy_calibration(data["p"], data["held_labels"])
        score_err = float(np.max(abs(scores-data["scores"])))
        weight_err = float(np.max(abs(weights-data["weights"])))
        assert score_err < 1e-12 and weight_err < 1e-12
        assert meta["map_index"] == stats["map_index"]
        max_score_error, max_weight_error = max(max_score_error, score_err), max(max_weight_error, weight_err)
        stats.update(weights=data["weights"].tolist(), scores=data["scores"].tolist(),
                     split=split, calibration_branch_forwards=meta["branch_forwards"])
        calibrations[graph], cal_data[graph], cal_metadata[graph] = stats, data, meta
        track(path, "calibration_probability_arrays"); track(side, "calibration_metadata")
    aggregates, candidate_metrics, table, final_meta, oracle = [], [], {}, [], []
    for case in manifest["rows"]:
        graph, mode = case["row"]["graph"], case["row"]["mode"]
        inp, targets = P.episode(case["row"])
        assert arrays_digest(inp) == case["input_sha256"] and arrays_digest(targets) == case["target_arrays_sha256"]
        assert not np.asarray(inp["observed"])[targets["query"]].any()
        cp, _ = pair_paths("calibration", graph)
        data_c = cal_data[graph]
        identity = artifact_identity("final", case["case_id"], dict(input_sha256=case["input_sha256"],
            target_arrays_sha256=case["target_arrays_sha256"], bank_sha256=case["bank_sha256"],
            calibration_sha256=sha(cp), split_sha256=manifest["calibration"][graph]["split_sha256"],
            graph_identity=case["graph_identity"], classes=case["classes"], dtype="float32"))
        path, side = pair_paths("predictions", case["case_id"])
        meta, data = saved_pair(path, side, identity, dict(**targets, weights=data_c["weights"], scores=data_c["scores"]), True)
        assert np.array_equal(data["weights"], data_c["weights"]), "None/95 weights must be bit-identical"
        rec = final_records[case["case_id"]]
        assert Path(rec["path"]).resolve() == path.resolve() and Path(rec["metadata"]).resolve() == side.resolve()
        assert rec["sha256"] == sha(path) and rec["metadata_sha256"] == sha(side)
        assert meta["branch_forwards"] == 15 and meta["neural_optimizer_updates"] == meta["target_gradient_updates"] == 0
        raw = data["candidate_p"]
        validate_p(raw, (15, case["queries"], case["classes"]))
        assert np.max(abs(raw[4] - 1/case["classes"])) < 2e-6
        predicted = _numpy_controls(raw, data["weights"], int(np.argmax(data["scores"])))
        for arm in ARMS:
            validate_p(data[arm], (case["queries"], case["classes"]))
            error = float(np.max(abs(predicted[arm]-data[arm])))
            assert error <= 1e-7, (case["case_id"], arm, error)
            max_control_error = max(max_control_error, error)
            metrics = _numpy_metrics(data[arm], data["y"])
            table[graph, mode, arm] = metrics
            aggregates.append(dict(graph=graph, mode=mode, arm=arm, metrics=metrics))
        diagnostics = []
        for g in range(15):
            metrics = _numpy_metrics(raw[g], data["y"], details=False)
            diagnostics.append(dict(candidate=g, a=float(data_c["a"][g]), b=float(data_c["b"][g]), **metrics))
        candidate_metrics.append(dict(graph=graph, mode=mode, candidates=diagnostics,
                                     disagreement=_disagreement(raw, data["weights"], data_c["a"], data_c["b"])))
        best_accuracy = max(diagnostics, key=lambda d: d["accuracy"])
        best_nll = min(diagnostics, key=lambda d: d["nll"])
        oracle.append(dict(graph=graph, mode=mode, query_truth_oracle_only=True,
                           accuracy_best_candidate=best_accuracy["candidate"], accuracy=best_accuracy["accuracy"],
                           nll_best_candidate=best_nll["candidate"], nll=best_nll["nll"],
                           used_for_prediction_or_tuning=False))
        for ref in case["references"]:
            assert sha(ref["metadata"]) == ref["metadata_sha256"]
            p = P.read_prediction(ref["path"], ref["sha256"], targets, case["classes"])
            metrics = _numpy_metrics(p, targets["y"])
            table[graph, mode, ref["arm"]] = metrics
            aggregates.append(dict(graph=graph, mode=mode, arm=ref["arm"], metrics=metrics))
            track(ref["path"], "reused_reference"); track(ref["metadata"], "reused_reference_metadata")
        track(path, "final_candidate_and_control_probabilities"); track(side, "final_metadata")
        final_meta.append(meta)
    assert len(aggregates) == len(table) == 120
    assert len(candidate_metrics) == 10 and sum(len(r["candidates"]) for r in candidate_metrics) == 150
    methods = list(ARMS)+config()["references"]
    means = {mode: {arm: _mean([table[g,mode,arm] for g in GRAPHS]) for arm in methods} for mode in ("none", "95")}
    effects = {arm: {g: {m: table[g,"95",arm][m]-table[g,"none",arm][m] for m in METRICS} for g in GRAPHS} for arm in methods}
    summary = dict(cycle="FP03", role="reused development", aggregates=aggregates,
                   means_equal_graphs=means, feature_effects_95_minus_none=effects,
                   feature_effect_means={arm: _mean(list(x.values())) for arm,x in effects.items()},
                   calibrations=calibrations, candidate_diagnostics=candidate_metrics,
                   family_headroom_query_truth_oracle_only=oracle, gate=_gate(table),
                   scope=config()["scope"], confidence_intervals=None)
    audit = dict(status="PASS", source_lock_sha256=sha(LOCK), manifest_sha256=sha(MANIFEST),
                 checks_sha256=sha(CHECKS), expected=config()["expected"],
                 independent_numpy_split_and_score_and_mixtures=True,
                 max_score_reconstruction_error=max_score_error, max_weight_reconstruction_error=max_weight_error,
                 max_control_reconstruction_error=max_control_error, bit_identical_none95_weights=True,
                 all_inputs_query_truth_context_and_hashes_verified=True, query_own_features_missing=True,
                 optimizer_updates=0, target_gradient_updates=0, statistical_parameter_fitting=False,
                 predictive_context_weighting=True, performance_gate_distinct_from_audit=True,
                 family_oracle_is_diagnostic_only=True, independent_real_ci=False)
    attempts = [dict(path=str(p.resolve()), sha256=sha(p), **read_json(p)) for p in sorted((RESULTS/"attempts").glob("*.json"))]
    assert attempts and all(a["optimizer_updates"] == 0 for a in attempts)
    budget = dict(new_neural_fits=0, optimizer_updates=0, target_gradient_updates=0,
        logical_calibration_branches=285, logical_final_branches=150,
        held_probability_vectors=1890, derived_controls=50, reused_references=70,
        calibration_seconds={g: cal_metadata[g]["inference_seconds"] for g in GRAPHS},
        calibration_setup_seconds=sum(d["setup_seconds"] for d in cal_metadata.values()),
        calibration_score_seconds=sum(d["scoring_seconds"] for d in cal_metadata.values()),
        final_seconds={r["case_id"]: r["inference_seconds"] for r in final_meta},
        final_preparation_seconds=sum(r["preparation_seconds"] for r in final_meta),
        final_aggregation_seconds=sum(r["aggregation_seconds"] for r in final_meta),
        row_read_seconds=sum(r["row_read_seconds"] for r in final_meta),
        peak_mib=max([r["peak_mib"] for r in final_meta]+[x["peak_mib"] for x in cal_metadata.values()]),
        incremental_peak_mib=max([r["incremental_peak_mib"] for r in final_meta]+[x["incremental_peak_mib"] for x in cal_metadata.values()]),
        prepare_seconds=manifest.get("preparation_seconds"), attempts=attempts,
        attempts_wall_seconds=sum(a["wall_seconds"] for a in attempts),
        pre_run_validation_seconds=sum(a["validation_seconds"] for a in attempts),
        audit_rescore_seconds=time.perf_counter()-began,
        failed_attempts=sum(a["status"] != "COMPLETE" for a in attempts),
        device="cuda", dtype="float32", workers=1, chunk=16,
        inherited_training_and_reference_cost="not zero; historical lineage in lock",
        timing_scope="calibration/final include CPU branch construction, H2D branch inputs and D2H predictions; table preparation/model load/wall separately recorded",
        abrupt_interruptions="unlogged failure cost is not inferred zero; partial pairs are never overwritten")
    write_json_new(RESULTS/"summary.json", summary)
    write_json_new(RESULTS/"audit.json", audit)
    write_json_new(RESULTS/"budget.json", budget)
    with (RESULTS/"REPORT_KO.md").open("x", encoding="utf8", newline="\n") as stream:
        stream.write(render_report(summary, audit, budget))
    for name in ("summary.json", "audit.json", "budget.json", "REPORT_KO.md", "evaluation.json", "lock.json", "checks.json"):
        track(RESULTS/name, "report_or_provenance")
    track(MANIFEST, "development_manifest")
    for a in attempts:
        track(a["path"], "execution_attempt")
    for name in read_json(LOCK)["files"]:
        track(name, "locked_lineage")
    complete = dict(status="COMPLETE", cycle="FP03", source_lock_sha256=sha(LOCK),
                    primary_development_gate_passed=summary["gate"]["passed"],
                    counts=config()["expected"], files=sorted(files.values(),key=lambda x:x["path"]),
                    self_hash="excluded to avoid circular digest", independent_real_validation=False)
    write_json_new(RESULTS/"artifact_manifest.json", complete)
    return complete


def render_report(s, a, b):
    lines = ["# FP03 context calibration: 저장 확률 재구성", "",
        "기존 개발 graph5개/none95의10episode다. 신경 학습과 optimizer update0회이며 context에서 predictive mixture weights를 계산했다. 새 PFN 또는 exact Bayes posterior 검증은 아니다.", "",
        f"감사 {a['status']}; adaptive 사전 개발 gate {'PASS' if s['gate']['passed'] else 'FAIL'}. Audit와 성능 판정은 별개다.",
        "139 original context 중126 held points/19 balanced folds,285 calibration branches 및150 final branches를 사용했다. 50 controls+70 reference=120 aggregate scores다. 독립 표본139/435/1890개로 세지 않으며 real CI가 없다.", "",
        "| Mode | 방법 | Accuracy % | BAcc % | Macro-F1 % | NLL | Brier |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for mode, arms in s["means_equal_graphs"].items():
        for arm, m in arms.items():
            lines.append(f"| {mode} | {arm} | {100*m['accuracy']:.3f} | {100*m['balanced_accuracy']:.3f} | {100*m['macro_f1']:.3f} | {m['nll']:.5f} | {m['brier']:.5f} |")
    lines += ["", "평균은5 source graph 동일 가중이다. MAP 또는 query-truth best candidate가 adaptive FAIL을 대체하지 않는다.", "",
              "| Graph | Mode | 방법 | Accuracy % | BAcc % | NLL |", "|---|---|---|---:|---:|---:|"]
    for r in s["aggregates"]:
        m = r["metrics"]
        lines.append(f"| {r['graph']} | {r['mode']} | {r['arm']} | {100*m['accuracy']:.3f} | {100*m['balanced_accuracy']:.3f} | {m['nll']:.5f} |")
    lines += ["", "## 열 가지 개발 gate", "", "| 조건 | 차이 | 사전 기준 | 판정 |", "|---|---:|---:|---|"]
    for name, g in s["gate"]["checks"].items():
        lines.append(f"| {name} | {g['value']:.6f} | {g['comparison']} {g['threshold']:.6f} | {'PASS' if g['passed'] else 'FAIL'} |")
    lines += ["", "## Calibration 진단", "", "| Graph | Context/calibration points/folds | Entropy | ESS | KL to prior | MAP index |", "|---|---|---:|---:|---:|---:|"]
    for graph, d in s["calibrations"].items():
        sp = d["split"]
        lines.append(f"| {graph} | {len(sp['original_context_ids'])}/{sp['points']}/{sp['m']} | {d['entropy']:.5f} | {d['ess']:.3f} | {d['prior_kl']:.5f} | {d['map_index']} |")
    lines += ["", "CS represented classes14/K15, selected42/unused13이며 extras는 calibration에서 모두 가리고 final에서는 원 context55개를 사용했다. Class9 query4개를 포함한 모든 minority 지표는 summary.json에 남겼다. Weights/ESS/entropy는 predictive mixture의 기술 통계이며 calibrated posterior uncertainty가 아니다. None/95 weights는 bit-identical이다.", "",
              "## 95−none 추가 피처 효과", "", "| 방법 | 평균 Accuracy 차이 %p | 평균 BAcc 차이 %p | 평균 NLL 차이 |", "|---|---:|---:|---:|"]
    for arm, m in s["feature_effect_means"].items():
        lines.append(f"| {arm} | {100*m['accuracy']:+.3f} | {100*m['balanced_accuracy']:+.3f} | {m['nll']:+.5f} |")
    lines += ["", "Graph-only calibration이므로 weight 변화가 feature mechanism adaptation을 뜻하지 않는다. 신규 shuffle이 없어 alignment 성공을 확정하지 않는다. None 근처로 수축한 harm reduction과 추가 feature 효용을 구분한다. Per-graph 효과는 summary.json에 있다.", "",
              "## Query-truth oracle: family headroom 진단 전용", "",
              "사전에 고정한 ANALYSIS_SPEC에 따른 기술 분석이다. 실제 predictor/weight/primary/tuning에는 쓰지 않았다. Accuracy/NLL 각각 다른 best candidate일 수 있다.", "",
              "| Graph | Mode | best accuracy index | Accuracy % | best NLL index | NLL |", "|---|---|---:|---:|---:|---:|"]
    for r in s["family_headroom_query_truth_oracle_only"]:
        lines.append(f"| {r['graph']} | {r['mode']} | {r['accuracy_best_candidate']} | {100*r['accuracy']:.3f} | {r['nll_best_candidate']} | {r['nll']:.5f} |")
    lines += ["", "All15 candidate 지표·weighted contrast moments·prior KL·mean pairwise TV 및 weighted TV-to-mixture는 summary.json에 모두 보존했다. Family headroom이 있어도 극소 context로 회수할 수 있다는 증명은 아니다.", "",
              "## 비용과 범위", "",
              f"Calibration inference {sum(b['calibration_seconds'].values()):.3f}s, final inference {sum(b['final_seconds'].values()):.3f}s, final preparation {b['final_preparation_seconds']:.3f}s, CUDA peak {b['peak_mib']:.2f}MiB. Execution attempts 합산 wall {b['attempts_wall_seconds']:.3f}s이며 실패 {b['failed_attempts']}개다. Cold load/lock validation/audit 및 inherited 비용은 budget.json과 lock lineage에 분리했다.",
              "Cora/CiteSeer full, 나머지 induced2000 범위이며 모두 재사용 development다. Graph별 predictor를 선택하지 않았고 no neural training/no independent real CI/no new PFN claim을 유지한다. 정상 성능 FAIL과 구현/audit 실패를 분리한다. Gate 미통과 후 a/b/score scale로 같은 bank를 구제하지 않는다.", "",
              "원자료: summary.json, audit.json, budget.json, evaluation.json, artifact_manifest.json 및 calibration/predictions의 NPZ·sidecar. 각 저장 확률과 source/checkpoint/bank/정답 identity의 hash가 연결돼 있다.", ""]
    return "\n".join(lines)


def selfcheck():
    """Tiny CPU parsing/audit tests only; no historical data or output files."""
    config()
    context = np.array([0,1,2,0,1,2,0,1,2,0,1,-1],np.int64)
    split = _numpy_split(context, 3)
    assert split["m"] == 3 and len(split["selected_ids"]) == 9 and len(split["unused_ids"]) == 2
    assert all(len(f["context_ids"]) == 6 for f in split["folds"])
    raw = np.full((15,6,3),1/3,np.float32)
    labels = np.tile(np.arange(3),2)
    scores, w, stats = _numpy_calibration(raw, labels)
    assert np.max(abs(w-1/15)) < 1e-15 and stats["map_index"] == 0
    candidate = np.full((15,2,3),1/3,np.float32)
    a = derive_predictions(candidate, w, 0)
    b = _numpy_controls(candidate,w,0)
    assert all(np.array_equal(a[k],b[k]) for k in ARMS)
    empty_scores, empty_w, _ = _numpy_calibration(np.empty((15,0,3),np.float32),np.empty(0,np.int64))
    assert not empty_scores.any() and np.max(abs(empty_w-1/15)) < 1e-15
    table = {}
    for graph in GRAPHS:
        for arm in ("adaptive","prior","fixed_edge","old_full","old_edge"):
            table[graph,"none",arm] = dict(accuracy=.5,balanced_accuracy=.5,macro_f1=.5,nll=1.,brier=.5)
    table["CoauthorPhysics","none","adaptive"]["accuracy"] = .71
    table["CoauthorCS","none","adaptive"]["accuracy"] = .56
    assert _gate(table)["passed"]
    table["CiteSeer","none","adaptive"]["nll"] = 1.06
    assert not _gate(table)["passed"]
    return dict(status="PASS", cpu_only=True, historical_data_used=False, files_written=0,
                checks=["protocol","numpy_split","score_weights_empty_uniform","mixture_reconstruction","ten_gates_pass_fail"],
                optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","run","report","selfcheck"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.resume)
        print(canonical(dict(status=result["status"],counts=result["counts"],manifest_sha256=sha(MANIFEST))))
    elif args.action == "run":
        result = run(args.resume,args.device)
        print(canonical(dict(status=result["status"],episodes=len(result["records"]))))
    elif args.action == "report":
        result = report(args.resume)
        print(canonical(dict(status=result["status"],gate=result["primary_development_gate_passed"])))
    else:
        print(canonical(selfcheck()))


if __name__ == "__main__":
    main()
