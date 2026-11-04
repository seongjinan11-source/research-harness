"""FP02 Stage0: immutable development manifest, gated inference, saved-p audit.

prepare is CPU/read-only with respect to historical evidence. run requires the
root's PASS checks and LOCKED source record; this module never creates a lock.
report rescales no model and rescores saved probabilities only. No optimizer.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows import order, before importing torch.
from sklearn.metrics import balanced_accuracy_score, f1_score
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
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
MANIFEST = ROOT / "bank/dev_manifest.json"
RESULTS = ROOT / "results"
LOCK = RESULTS / "lock.json"
CHECKS = RESULTS / "checks.json"
ARMS = ("poisson_fixed", "sbm_update", "dc_update")
GRAPHS = ("Cora", "CiteSeer", "PubMed", "CoauthorCS", "CoauthorPhysics")
MODES = ("none", "95")
EXTERNAL = ("lp_harmonic", "lp_restart", "public_graphpfn",
            "pcfi_public_graphpfn", "fisf_public_graphpfn")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
_READONLY_RUNNER = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def arrays_digest(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode("utf8"))
        if isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            if a.dtype.hasobject:
                raise TypeError("Object arrays are forbidden")
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(canonical(value).encode("utf8"))
    return h.hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(plain(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inherited_runner():
    global _READONLY_RUNNER
    if _READONLY_RUNNER is None:
        _READONLY_RUNNER = load_module("fp02_readonly_fp01_experiment",
                                       WORK / "factor_projection_cycle/experiment.py")
    return _READONLY_RUNNER


def config():
    c = read_json(ROOT / "protocol.json")
    assert c["cycle"] == "FP02-Stage0" and c["version"] == 2
    assert c["graphs"] == list(GRAPHS) and c["modes"] == list(MODES)
    assert c["arms"] == list(ARMS) and c["external_arms"] == list(EXTERNAL)
    assert c["seed"] == 11 and c["repetition"] == 0
    assert c["expected"] == dict(development_rows=10, new_predictions=30, reused_references=70)
    assert c["operator"]["iterations"] == 24 and c["operator"]["damping"] == .5
    assert c["operator"]["block"] == 6 and c["operator"]["nuisance_updates"] == 3
    assert c["operator"] == dict(iterations=24, damping=.5, block=6, nuisance_updates=3,
                                  nuisance_damping=.5, affinity_shape=5., eta=.6, theta_shape=1.,
                                  theta_rate=1., population_alpha=1., population_unary="Elogpi",
                                  density_offset="mean observed degree/(N-1)", dtype="float32")
    assert c["feature"]["grid_points"] == 41 and c["feature"]["coefficients"] == 3
    assert c["execution"]["optimizer_updates"] == c["execution"]["target_gradient_updates"] == 0
    assert not c["statistics"]["confidence_intervals"]
    assert c["execution"]["workers"] == 1 and c["execution"]["training_fits"] == 0
    assert c["gate"] == dict(mode="none", equal_graph_weight=True, physics_gain_old_full=.2,
                              cs_gain_old_full=.05, citeseer_accuracy_floor_old_full=-.01,
                              citeseer_nll_ceiling_old_full=.05, mean_balanced_accuracy_floor_old_full=0.,
                              mean_balanced_accuracy_floor_old_edge=0., mean_nll_ceiling_old_edge=.05,
                              primary_arm="dc_update", also_report="sbm_update; cannot rescue failed dc_update gate")
    return c


def case_id(row):
    return f"{row['graph']}_r0_{row['mode']}"


def episode(row):
    R = inherited_runner()
    inp, query, truth = R.G.read_real(row)
    R._validate_episode(inp, query, truth, missing_query=True)
    query, truth = np.asarray(query, np.int64), np.asarray(truth, np.int64)
    context = np.asarray(inp["context_ids"], np.int64)
    yc = np.asarray(inp["labels_context"], np.int64)
    inp = {k: v for k, v in inp.items()}  # strict inherited visible whitelist only
    assert set(inp) == {"x", "observed", "src", "dst", "context_ids",
                        "labels_context", "classes", "forward_seed"}
    arrays = dict(query=query, y=truth, context=context, yc=yc)
    return inp, arrays


def validate_probability(p, query, classes):
    p = np.asarray(p)
    assert p.shape == (len(query), int(classes)), (p.shape, len(query), classes)
    assert np.isfinite(p).all() and p.min() >= -1e-7 and p.max() <= 1 + 1e-7
    assert np.max(np.abs(p.sum(1) - 1)) <= 3e-5, "Unnormalized probability"


def read_prediction(path, expected_hash, arrays, classes):
    assert sha(path) == expected_hash, f"Prediction bytes changed: {path}"
    with np.load(path, allow_pickle=False) as saved:
        for key in ("query", "y", "context", "yc"):
            assert key in saved.files and np.array_equal(saved[key], arrays[key]), (path, key)
        p = saved["p"].copy()
    validate_probability(p, arrays["query"], classes)
    return p


def _reference(path, metadata, arm, arrays, row, input_sha, target_sha, expected=None):
    path, metadata = Path(path).resolve(), Path(metadata).resolve()
    meta = read_json(metadata)
    actual = sha(path)
    assert actual == meta["prediction_sha256"]
    if expected is not None:
        assert actual == expected
    if arm in ("old_full", "old_edge"):
        ident = meta["identity"]
        assert ident["row"] == row and ident["input_sha256"] == input_sha
        assert ident["target_arrays_sha256"] == target_sha
        assert ident["arm"] == "C_global" and ident["seed"] == 11
        assert ident["graph_arm"] == arm.removeprefix("old_")
        weights = np.asarray(meta["full_role_weights"], np.float64)
        assert weights.shape == (3,) and abs(weights[2] - 1) <= 1e-6
        assert np.max(np.abs(weights[:2])) <= 1e-6, "Legacy reference is not homophilic; alert root"
        assert ident["checkpoint_sha256"] == sha(WORK / "factor_projection_cycle/results/C_global_s11/checkpoint.pt")
    else:
        assert meta["arm"] == arm and meta["sha256"] == row["sha256"]
        for key in ("graph", "repetition", "mode", "bank_root", "path"):
            assert meta[key] == row[key], (arm, key)
    read_prediction(path, actual, arrays, int(row["classes"]))
    return dict(arm=arm, path=str(path), sha256=actual,
                metadata=str(metadata), metadata_sha256=sha(metadata),
                query_target_sha256=target_sha, identity_verified=True)


def build_manifest():
    config()
    R = inherited_runner()
    R._lock_record()  # Verifies unchanged inherited source lock/checks, no writes.
    rows = [r for r in R.G.real_rows() if r["graph"] in GRAPHS
            and int(r["repetition"]) == 0 and r["mode"] in MODES]
    rows.sort(key=lambda r: (GRAPHS.index(r["graph"]), MODES.index(r["mode"])))
    assert len(rows) == 10 and len({case_id(r) for r in rows}) == 10
    index_path = WORK / "factor_projection_cycle/results/C_global_s11/evaluation_real.json"
    index = read_json(index_path)
    assert index["status"] == "COMPLETE" and index["arm"] == "C_global" and index["seed"] == 11
    legacy = {}
    for rec in index["records"]:
        meta = read_json(rec["metadata"])
        r = meta["row"]
        key = (case_id(r), rec["graph_arm"])
        if int(r["repetition"]) == 0 and r["graph"] in GRAPHS and r["mode"] in MODES:
            assert key not in legacy
            legacy[key] = rec
    cases = []
    for row in rows:
        inp, arrays = episode(row)
        input_sha, target_sha = arrays_digest(inp), arrays_digest(arrays)
        refs = []
        for panel in ("full", "edge"):
            rec = legacy[(case_id(row), panel)]
            refs.append(_reference(rec["path"], rec["metadata"], f"old_{panel}",
                                   arrays, row, input_sha, target_sha, rec["prediction_sha256"]))
        for arm in EXTERNAL:
            path = WORK / "factor_edge_only_benchmark/results/predictions" / f"{case_id(row)}_{arm}.npz"
            refs.append(_reference(path, path.with_suffix(".json"), arm, arrays, row, input_sha, target_sha))
        cases.append(dict(case_id=case_id(row), row=row, bank_path=str((WORK / row["bank_root"] / row["path"]).resolve()),
                          bank_sha256=row["sha256"], input_sha256=input_sha,
                          target_arrays_sha256=target_sha, graph_identity=R.G.graph_identity(inp),
                          nodes=len(inp["x"]), classes=int(inp["classes"]), features=inp["x"].shape[1],
                          sources=int(np.asarray(inp["observed"]).any(1).sum()),
                          queries=len(arrays["query"]), own_features_missing=True, references=refs))
    assert sum(len(c["references"]) for c in cases) == 70
    return dict(status="PREPARED", cycle="FP02-Stage0", protocol_sha256=sha(ROOT / "protocol.json"),
                inherited_index=str(index_path), inherited_index_sha256=sha(index_path),
                rows=cases, rows_count=10, references_count=70,
                bank_arrays_rewritten=False, final_bank=False)


def prepare(resume=False):
    began = time.perf_counter()
    state = build_manifest()
    if MANIFEST.exists():
        if not resume:
            raise FileExistsError("Prepared manifest exists; use --resume for strict verification")
        old = read_json(MANIFEST)
        assert {k: v for k, v in old.items() if k != "preparation_seconds"} == state
        return old
    state["preparation_seconds"] = time.perf_counter() - began
    write_json_new(MANIFEST, state)
    return state


def verified_manifest():
    old = read_json(MANIFEST)
    current = build_manifest()
    assert {k: v for k, v in old.items() if k != "preparation_seconds"} == current, "Prepared inputs/references changed"
    return old


def source_state():
    """Return the exact root-owned lock contents; never write a source lock."""
    c = config()
    manifest = verified_manifest()
    checks = read_json(CHECKS)
    assert checks["status"] == "PASS" and checks.get("optimizer_updates", 0) == 0
    assert not checks.get("real_predictions_used", False) and not checks.get("real_bank_used", False)
    assert checks["tested_sources"]
    for name, expected in checks["tested_sources"].items():
        assert sha(name) == expected, f"Numerical source changed after checks: {name}"
    paths = [ROOT / name for name in ("run.py", "operators.py", "checks.py", "protocol.json", "PLAN_v2.md",
                                      "RESPONSE_v1.md", "IMPLEMENTATION_RUNNER.md", "IMPLEMENTATION_INFERENCE.md",
                                      "IMPLEMENTATION_CHECKS.md")]
    paths += [Path(name) for name in checks["tested_sources"]]
    paths += [ROOT / name for name in c["integrity"]["required_reviews"]]
    paths += [MANIFEST, CHECKS]
    fp = WORK / "factor_projection_cycle"
    paths += [fp / name for name in ("results/lock.json", "results/checks.json", "results/C_global_s11/checkpoint.pt",
                                     "results/C_global_s11/training.json", "results/C_global_s11/evaluation_real.json")]
    inherited = read_json(fp / "results/lock.json")
    def add_maps(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("files", "graph_checkpoints") and isinstance(item, dict):
                    for name, expected in item.items():
                        assert sha(name) == expected, f"Inherited lock changed: {name}"
                        paths.append(Path(name))
                else:
                    add_maps(item)
    add_maps(inherited)
    for case in manifest["rows"]:
        paths += [Path(case["bank_path"]), WORK / case["row"]["bank_root"] / "bank/manifest.json"]
        for ref in case["references"]:
            paths += [Path(ref["path"]), Path(ref["metadata"])]
    unique = sorted({p.resolve() for p in paths}, key=str)
    return dict(status="LOCKED", cycle="FP02-Stage0", files={str(p): sha(p) for p in unique},
                manifest_sha256=sha(MANIFEST), checks_sha256=sha(CHECKS),
                inherited_lock_sha256=sha(fp / "results/lock.json"),
                optimizer_updates=0, target_gradient_updates=0)


def require_lock():
    if not LOCK.is_file():
        raise RuntimeError("Root must write results/lock.json after numerical PASS")
    saved = read_json(LOCK)
    assert saved == source_state(), "Source/checks/bank/reference lock mismatch"
    return saved


def git_snapshot():
    result = {}
    for key, args in (("branch", ("branch", "--show-current")), ("commit", ("rev-parse", "HEAD"))):
        try:
            p = subprocess.run(["git", "-c", f"safe.directory={WORK.as_posix()}", "-C", str(WORK), *args],
                               capture_output=True, text=True, check=False)
            result[key] = p.stdout.strip() if p.returncode == 0 else None
        except OSError:
            result[key] = None
    return result


def prediction_paths(case, arm):
    p = RESULTS / "predictions" / arm / f"{case['case_id']}.npz"
    return p, p.with_suffix(".json")


def prediction_identity(case, arm):
    return dict(cycle="FP02-Stage0", case_id=case["case_id"], arm=arm, seed=11, repetition=0,
                source_lock_sha256=sha(LOCK), manifest_sha256=sha(MANIFEST),
                input_sha256=case["input_sha256"], target_arrays_sha256=case["target_arrays_sha256"],
                bank_sha256=case["bank_sha256"], graph_identity=case["graph_identity"],
                feature_checkpoint_sha256=sha(WORK / "factor_projection_cycle/results/C_global_s11/checkpoint.pt"),
                protocol_sha256=sha(ROOT / "protocol.json"), classes=case["classes"],
                dtype="float32", optimizer_updates=0, target_gradient_updates=0)


def checked_saved(case, arm, arrays, resume):
    p, side = prediction_paths(case, arm)
    if not p.exists() and not side.exists():
        return None
    if not p.is_file() or not side.is_file():
        raise RuntimeError(f"Partial prediction pair; preserve it and investigate, never overwrite: {p}")
    if not resume:
        raise FileExistsError(f"Prediction exists; use --resume: {p}")
    meta = read_json(side)
    assert meta["identity"] == prediction_identity(case, arm)
    read_prediction(p, meta["prediction_sha256"], arrays, case["classes"])
    return meta


def _small_diagnostics(info):
    """O(N+K²) statistics only; reject accidentally returned message banks."""
    def compact(v, key=""):
        if isinstance(v, dict):
            return {str(k): compact(x, str(k)) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [compact(x, key) for x in v]
        if isinstance(v, (torch.Tensor, np.ndarray)):
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            assert a.size <= 100000, f"Huge operator tensor leaked to JSON diagnostics: {key} {a.shape}"
            if a.size > 256:
                a = np.asarray(a)
                assert np.isfinite(a).all(), (key, "nonfinite diagnostics")
                return dict(shape=list(a.shape), min=float(a.min()), max=float(a.max()),
                            mean=float(a.mean()), sha256=arrays_digest({key: a}))
            return plain(a)
        return plain(v)
    return compact(info)


def _operator_arrays(info):
    """Keep every small operator statistic, including fresh/update-history arrays."""
    out = {}
    def visit(value, parts):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, parts + [str(key)])
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, parts + [str(index)])
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            assert a.size <= 100000 and not a.dtype.hasobject, (parts, a.shape)
            key = "operator__" + "__".join(parts)
            assert key not in out
            out[key] = a
    visit(info, [])
    return out


def feature_table(model, visible):
    x, obs = model._visible(visible)
    location, tau, sigma = model.hyperparameters(visible, {})
    grid = location[:, None] + math.sqrt(2) * tau[:, None] * model.gh_nodes.to(x)[None]
    table = (-.5 * math.log(2 * math.pi) - sigma.log()[None, :, None]
             - .5 * ((x[:, :, None] - grid[None]) / sigma[None, :, None]).square())
    assert model.grid_points == 41 and model.raw.numel() == 3
    return obs, table, model.gh_log_weights.to(x), dict(location=location, tau=tau, sigma=sigma)


def run(resume=False, device="cuda"):
    validation_began = time.perf_counter()
    if str(device) != "cuda":
        raise ValueError("Locked Stage0 real inference uses one cuda worker in float32")
    lock = require_lock()
    validation_seconds = time.perf_counter() - validation_began
    manifest = read_json(MANIFEST)
    RESULTS.mkdir(parents=True, exist_ok=True)
    marker = RESULTS / ".active_run.json"
    # Exclusive writer; stale marker is a root-reviewed operational recovery.
    token = str(time.time_ns())
    write_json_new(marker, dict(token=token, state="RUNNING", source_lock_sha256=sha(LOCK)))
    began = time.perf_counter()
    outcomes, completed, attempts = [], 0, RESULTS / "attempts"
    attempt_path = attempts / f"{token}.json"
    git = git_snapshot()  # Exactly one two-command metadata snapshot per invocation.
    try:
        torch.set_num_threads(4)
        assert torch.cuda.is_available(), "CUDA unavailable"
        R = inherited_runner()
        O = load_module("fp02_nuisance_operator", ROOT / "operators.py")
        load_began = time.perf_counter()
        model, fitted = R.load_model("C_global", 11, device="cuda")
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        model_digest = R.G.digest(model)
        assert model_digest == fitted["final_sha256"]
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_began
        for case in manifest["rows"]:
            row_began = time.perf_counter()
            inp, arrays = episode(case["row"])
            assert arrays_digest(inp) == case["input_sha256"]
            assert arrays_digest(arrays) == case["target_arrays_sha256"]
            row_read_seconds = time.perf_counter() - row_began
            cached = {arm: checked_saved(case, arm, arrays, resume) for arm in ARMS}
            feature_seconds = 0.
            if any(v is None for v in cached.values()):
                table_began = time.perf_counter()
                visible = R.G.tensor_input(inp, device="cuda", dtype=torch.float32)
                n, k = len(inp["x"]), int(inp["classes"])
                src = torch.as_tensor(inp["src"], device="cuda", dtype=torch.long)
                dst = torch.as_tensor(inp["dst"], device="cuda", dtype=torch.long)
                context = torch.full((n,), -1, device="cuda", dtype=torch.long)
                context[torch.as_tensor(arrays["context"], device="cuda")] = torch.as_tensor(arrays["yc"], device="cuda")
                with torch.no_grad():
                    obs, table, weights, hyper = feature_table(model, visible)
                torch.cuda.synchronize()
                feature_seconds = time.perf_counter() - table_began
            for arm in ARMS:
                meta = cached[arm]
                if meta is None:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    predict_began = time.perf_counter()
                    with torch.no_grad():
                        prob, info = O.nuisance_bp(src, dst, n, context, k, obs, table, weights,
                                                  variant=arm, iterations=24, damping=.5, block=6, details=False)
                        q = torch.as_tensor(arrays["query"], device="cuda", dtype=torch.long)
                        p = prob.index_select(0, q).detach().cpu().numpy()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - predict_began
                    peak = torch.cuda.max_memory_allocated()
                    validate_probability(p, arrays["query"], k)
                    expected_updates = 0 if arm == "poisson_fixed" else 3
                    assert int(info["nuisance_updates"]) == expected_updates
                    diagnostics = _small_diagnostics(info)
                    path, side = prediction_paths(case, arm)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    parameters = _operator_arrays(info)
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, p=p, **arrays, **parameters)
                    meta = dict(status="COMPLETE", identity=prediction_identity(case, arm),
                                prediction_path=str(path.resolve()), prediction_sha256=sha(path),
                                arm=arm, case_id=case["case_id"], graph=case["row"]["graph"], mode=case["row"]["mode"],
                                sources=case["sources"], queries=case["queries"], classes=k,
                                inference_seconds=seconds, peak_mib=peak / 2**20,
                                incremental_peak_mib=(peak - baseline) / 2**20,
                                common_feature_table_seconds=feature_seconds, common_row_read_seconds=row_read_seconds,
                                feature_hyperparameters=_small_diagnostics(hyper), diagnostics=diagnostics,
                                operator_array_keys=sorted(parameters),
                                max_rate_scope="unconstrained class-conditional pair intensity maximum, not q-weighted mean; no clipping",
                                git=git, neural_optimizer_updates=0, target_gradient_updates=0,
                                statistical_parameter_updates=expected_updates,
                                inference_scope="graph-conditioned Poisson working score; hybrid cavity BP/mean-field")
                    write_json_new(side, meta)
                    del prob, info, p, parameters
                path, side = prediction_paths(case, arm)
                outcomes.append(dict(case_id=case["case_id"], arm=arm, path=str(path.resolve()),
                                     metadata=str(side.resolve()), prediction_sha256=meta["prediction_sha256"],
                                     metadata_sha256=sha(side)))
                completed += 1
                print("STAGE0_PREDICTION", completed, "/30", case["case_id"], arm,
                      round(meta["inference_seconds"], 3), flush=True)
            if any(v is None for v in cached.values()):
                del visible, src, dst, context, obs, table, weights, hyper
            gc.collect()
            torch.cuda.empty_cache()
        assert len(outcomes) == 30
        assert R.G.digest(model) == model_digest, "Frozen feature model mutated"
        assert require_lock() == lock, "Sources changed during inference"
        index = dict(status="COMPLETE", cycle="FP02-Stage0", source_lock_sha256=sha(LOCK),
                     manifest_sha256=sha(MANIFEST), records=outcomes,
                     new_predictions=30, reused_references=70, optimizer_updates=0, target_gradient_updates=0)
        dest = RESULTS / "evaluation.json"
        if dest.exists():
            assert resume and read_json(dest) == index, "Complete evaluation index changed"
        else:
            write_json_new(dest, index)
        write_json_new(attempt_path, dict(status="COMPLETE", token=token, git=git,
                                         wall_seconds=time.perf_counter() - began,
                                         pre_run_validation_seconds=validation_seconds,
                                         model_load_seconds=model_load_seconds, verified_predictions=completed,
                                         source_lock_sha256=sha(LOCK), optimizer_updates=0))
        return index
    except Exception as exc:
        write_json_new(attempt_path, dict(status="FAILED", token=token, git=git,
                                         wall_seconds=time.perf_counter() - began, verified_predictions=completed,
                                         pre_run_validation_seconds=validation_seconds,
                                         error=repr(exc), traceback=traceback.format_exc(), optimizer_updates=0))
        raise
    finally:
        if marker.exists() and read_json(marker).get("token") == token:
            marker.unlink()


def score(p, y, classes=None):
    p = np.clip(np.asarray(p, np.float64), 1e-8, 1)
    p /= p.sum(1, keepdims=True)
    y = np.asarray(y, np.int64)
    pred = p.argmax(1)
    k = p.shape[1] if classes is None else int(classes)
    recalls, supports = [], []
    for c in range(k):
        subset = y == c
        supports.append(int(subset.sum()))
        recalls.append(float(np.mean(pred[subset] == c)) if subset.any() else None)
    return dict(accuracy=float(np.mean(pred == y)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                brier=float(np.square(p - np.eye(k)[y]).sum(1).mean()),
                class_recall=recalls, class_support=supports,
                predicted_class_mass=(np.bincount(pred, minlength=k) / len(y)).tolist(),
                true_class_mass=(np.bincount(y, minlength=k) / len(y)).tolist())


def mean_metrics(rows):
    assert rows
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def gate_from_metrics(table, arm, cfg=None):
    g = config()["gate"] if cfg is None else cfg
    candidate = {graph: table[(graph, "none", arm)] for graph in GRAPHS}
    full = {graph: table[(graph, "none", "old_full")] for graph in GRAPHS}
    edge = {graph: table[(graph, "none", "old_edge")] for graph in GRAPHS}
    cm, fm, em = (mean_metrics(list(x.values())) for x in (candidate, full, edge))
    values = dict(physics_gain_old_full=candidate["CoauthorPhysics"]["accuracy"] - full["CoauthorPhysics"]["accuracy"],
                  cs_gain_old_full=candidate["CoauthorCS"]["accuracy"] - full["CoauthorCS"]["accuracy"],
                  citeseer_accuracy_floor_old_full=candidate["CiteSeer"]["accuracy"] - full["CiteSeer"]["accuracy"],
                  citeseer_nll_ceiling_old_full=candidate["CiteSeer"]["nll"] - full["CiteSeer"]["nll"],
                  mean_balanced_accuracy_floor_old_full=cm["balanced_accuracy"] - fm["balanced_accuracy"],
                  mean_balanced_accuracy_floor_old_edge=cm["balanced_accuracy"] - em["balanced_accuracy"],
                  mean_nll_ceiling_old_edge=cm["nll"] - em["nll"])
    checks = {}
    for name, value in values.items():
        ceiling = "ceiling" in name
        # Only protect exact rational boundaries against float64 subtraction.
        passed = value <= g[name] + 1e-12 if ceiling else value >= g[name] - 1e-12
        checks[name] = dict(value=float(value), threshold=float(g[name]),
                            comparison="<=" if ceiling else ">=", passed=bool(passed))
    return dict(arm=arm, passed=all(x["passed"] for x in checks.values()), checks=checks,
                candidate_mean=cm, old_full_mean=fm, old_edge_mean=em,
                equal_graph_weight=True, independent_final_validation=False)


def report(resume=False):
    report_began = time.perf_counter()
    require_lock()
    manifest = read_json(MANIFEST)
    destination = RESULTS / "artifact_manifest.json"
    output_names = ("summary.json", "audit.json", "budget.json", "REPORT_KO.md", "artifact_manifest.json")
    present = [RESULTS / n for n in output_names if (RESULTS / n).exists()]
    if present:
        if not resume or len(present) != len(output_names):
            raise FileExistsError("Complete/partial report exists; never silently overwrite. --resume requires all report files.")
        complete = read_json(destination)
        assert complete["status"] == "COMPLETE" and complete["source_lock_sha256"] == sha(LOCK)
        for record in complete["files"]:
            assert sha(record["path"]) == record["sha256"], record["path"]
        return complete
    index = read_json(RESULTS / "evaluation.json")
    assert index["status"] == "COMPLETE" and index["source_lock_sha256"] == sha(LOCK)
    assert len(index["records"]) == 30 and index["manifest_sha256"] == sha(MANIFEST)
    indexed = {(r["case_id"], r["arm"]): r for r in index["records"]}
    assert len(indexed) == 30
    records, new_meta, files, table = [], [], [], {}
    def artifact(path, role):
        path = Path(path).resolve()
        files.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, role=role))
    for case in manifest["rows"]:
        inp, arrays = episode(case["row"])
        assert arrays_digest(inp) == case["input_sha256"] and arrays_digest(arrays) == case["target_arrays_sha256"]
        candidates = []
        for arm in ARMS:
            meta = checked_saved(case, arm, arrays, True)
            assert meta is not None and meta["neural_optimizer_updates"] == meta["target_gradient_updates"] == 0
            p, side = prediction_paths(case, arm)
            rec = indexed[(case["case_id"], arm)]
            assert rec["prediction_sha256"] == sha(p) and rec["metadata_sha256"] == sha(side)
            assert Path(rec["path"]).resolve() == p.resolve() and Path(rec["metadata"]).resolve() == side.resolve()
            candidates.append((arm, p, meta["prediction_sha256"]))
            new_meta.append(meta)
            artifact(p, "new_prediction"); artifact(side, "new_prediction_metadata")
        for ref in case["references"]:
            assert sha(ref["metadata"]) == ref["metadata_sha256"]
            candidates.append((ref["arm"], Path(ref["path"]), ref["sha256"]))
            artifact(ref["path"], "reused_reference"); artifact(ref["metadata"], "reused_reference_metadata")
        for arm, path, expected in candidates:
            p = read_prediction(path, expected, arrays, case["classes"])
            metrics = score(p, arrays["y"], case["classes"])
            graph, mode = case["row"]["graph"], case["row"]["mode"]
            table[(graph, mode, arm)] = metrics
            records.append(dict(case_id=case["case_id"], graph=graph, mode=mode, arm=arm,
                                nodes=case["nodes"], queries=case["queries"], classes=case["classes"],
                                sources=case["sources"], metrics=metrics, path=str(path), sha256=expected))
    assert len(records) == 100 and len(table) == 100
    methods = list(ARMS) + ["old_full", "old_edge"] + list(EXTERNAL)
    means = {mode: {arm: mean_metrics([table[(g, mode, arm)] for g in GRAPHS]) for arm in methods} for mode in MODES}
    effects = {arm: {graph: {m: table[(graph, "95", arm)][m] - table[(graph, "none", arm)][m]
                              for m in METRICS} for graph in GRAPHS} for arm in methods}
    effect_means = {arm: mean_metrics(list(effects[arm].values())) for arm in methods}
    gates = {arm: gate_from_metrics(table, arm) for arm in ("dc_update", "sbm_update")}
    summary = dict(cycle="FP02-Stage0", role="reused development", records=records,
                   means_equal_five_graphs=means, feature_effect_95_minus_none=effects,
                   feature_effect_equal_five_graphs=effect_means, gates=gates,
                   operator_diagnostics=[dict(case_id=m["case_id"], arm=m["arm"], diagnostics=m["diagnostics"])
                                         for m in new_meta],
                   primary_gate_passed=gates["dc_update"]["passed"], sbm_cannot_rescue_dc=True,
                   confidence_intervals=None, feature_alignment_tested=False,
                   pfn_retraining_tested=False, independent_real_holdout=False)
    audit = dict(status="PASS", source_lock_sha256=sha(LOCK), manifest_sha256=sha(MANIFEST),
                 checks_sha256=sha(CHECKS), input_hashes_and_visible_whitelist=True,
                 exact_query_truth_context_identity=True, own_query_features_missing=True,
                 legacy_homophilic_references_verified=True,
                 new_prediction_pairs=30, reused_reference_pairs=70, rescored_probability_arrays=100,
                 rows=10, graph_count=5, optimizer_updates=0, target_gradient_updates=0,
                 statistical_nuisance_updates={arm: (0 if arm == "poisson_fixed" else 3) for arm in ARMS},
                 gate_pass_is_not_audit_pass=True, final_bank_used=False)
    attempts = [dict(path=str(p.resolve()), sha256=sha(p), **read_json(p))
                for p in sorted((RESULTS / "attempts").glob("*.json"))]
    assert attempts and all(a["optimizer_updates"] == 0 for a in attempts)
    row_common = {}
    for meta in new_meta:
        # Shared table/read costs stored on all arms; count once by max per row.
        item = row_common.setdefault(meta["case_id"], dict(feature=0., read=0.))
        item["feature"] = max(item["feature"], meta["common_feature_table_seconds"])
        item["read"] = max(item["read"], meta["common_row_read_seconds"])
    budget = dict(new_training_fits=0, optimizer_updates=0, target_gradient_updates=0,
                  new_inference_calls=30, reused_inference_arrays=70, single_worker=True,
                  prediction_seconds={arm: sum(m["inference_seconds"] for m in new_meta if m["arm"] == arm) for arm in ARMS},
                  shared_feature_table_seconds=sum(x["feature"] for x in row_common.values()),
                  shared_row_read_seconds=sum(x["read"] for x in row_common.values()),
                  preparation_seconds=manifest.get("preparation_seconds"),
                  peak_mib=max(m["peak_mib"] for m in new_meta),
                  incremental_peak_mib=max(m["incremental_peak_mib"] for m in new_meta),
                  attempts=attempts, attempt_wall_seconds=sum(a["wall_seconds"] for a in attempts),
                  pre_run_validation_seconds=sum(a.get("pre_run_validation_seconds", 0.) for a in attempts),
                  audit_rescore_seconds=time.perf_counter() - report_began,
                  failed_attempts=sum(a["status"] != "COMPLETE" for a in attempts),
                  timing_scope="single CUDA worker; cold model/table/read costs separate; historical training/reference costs inherited, not zero", 
                  interrupted_attempts_without_log="not inferred zero; preserve operational logs separately")
    write_json_new(RESULTS / "summary.json", summary)
    write_json_new(RESULTS / "audit.json", audit)
    write_json_new(RESULTS / "budget.json", budget)
    text = render_report(summary, audit, budget)
    with (RESULTS / "REPORT_KO.md").open("x", encoding="utf8", newline="\n") as stream:
        stream.write(text)
    for name in ("summary.json", "audit.json", "budget.json", "REPORT_KO.md", "evaluation.json", "lock.json", "checks.json"):
        artifact(RESULTS / name, "report_or_provenance")
    artifact(MANIFEST, "immutable_development_manifest")
    for a in attempts:
        artifact(a["path"], "execution_attempt")
    for name in read_json(LOCK)["files"]:
        artifact(name, "locked_source_or_inherited_evidence")
    dedup = {x["path"]: x for x in files}
    complete = dict(status="COMPLETE", cycle="FP02-Stage0", source_lock_sha256=sha(LOCK),
                    primary_development_gate_passed=gates["dc_update"]["passed"],
                    counts=dict(new_predictions=30, reused_references=70, rescored=100),
                    files=sorted(dedup.values(), key=lambda r: r["path"]),
                    self_hash="excluded to avoid a circular digest", no_independent_final_claim=True)
    write_json_new(destination, complete)
    return complete


def render_report(summary, audit, budget):
    lines = ["# FP02 Stage 0: 저장 확률 재채점", "",
             "기존 개발 graph 5개, rep0/seed11, none/95의 고정 10개 episode다. 새 학습·신경 optimizer 업데이트는 0회이며 통계 nuisance 추론 갱신은 포함한다.", "",
             f"감사: {audit['status']}. 새 예측 30개와 재사용 비교 70개를 동일 query/truth/context로 재채점했다.",
             f"사전 DC 개발 gate: {'PASS' if summary['primary_gate_passed'] else 'FAIL'}. SBM은 별도 결과이며 DC의 실패를 대체하지 않는다.", "",
             "| 조건 | 방법 | Accuracy % | BAcc % | Macro-F1 % | NLL | Brier |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for mode, arms in summary["means_equal_five_graphs"].items():
        for arm, m in arms.items():
            lines.append(f"| {mode} | {arm} | {100*m['accuracy']:.3f} | {100*m['balanced_accuracy']:.3f} | {100*m['macro_f1']:.3f} | {m['nll']:.5f} | {m['brier']:.5f} |")
    lines += ["", "모든 평균은 source graph 5개에 동일 가중이다. Node·seed를 독립 표본으로 세거나 real CI를 붙이지 않았다.", "",
              "| Graph | 조건 | 방법 | Accuracy % | BAcc % | NLL |",
              "|---|---|---|---:|---:|---:|"]
    for r in summary["records"]:
        m = r["metrics"]
        lines.append(f"| {r['graph']} | {r['mode']} | {r['arm']} | {100*m['accuracy']:.3f} | {100*m['balanced_accuracy']:.3f} | {m['nll']:.5f} |")
    for arm, gate in summary["gates"].items():
        lines += ["", f"## {arm} 개발 gate: {'PASS' if gate['passed'] else 'FAIL'}", "",
                  "| 사전 조건 | 실제 차이 | 기준 | 판정 |", "|---|---:|---:|---|"]
        for name, check in gate["checks"].items():
            lines.append(f"| {name} | {check['value']:.6f} | {check['comparison']} {check['threshold']:.6f} | {'PASS' if check['passed'] else 'FAIL'} |")
    lines += ["", "## 피처 관측 95% − none", "",
              "이 차이는 feature가 population/affinity/degree 추론에 미친 전체 효과다. 신규 shuffle 대조가 없으므로 alignment 또는 새 피처 효용 성공을 확정하지 않는다.", "",
              "| Graph | 방법 | Accuracy 차이 %p | BAcc 차이 %p | NLL 차이 |",
              "|---|---|---:|---:|---:|"]
    for arm, graphs in summary["feature_effect_95_minus_none"].items():
        for graph, m in graphs.items():
            lines.append(f"| {graph} | {arm} | {100*m['accuracy']:+.3f} | {100*m['balanced_accuracy']:+.3f} | {m['nll']:+.5f} |")
        m = summary["feature_effect_equal_five_graphs"][arm]
        lines.append(f"| Equal-five-graph mean | {arm} | {100*m['accuracy']:+.3f} | {100*m['balanced_accuracy']:+.3f} | {m['nll']:+.5f} |")
    lines += ["", "None 근처로 수축해 기존 feature 손상이 줄어든 것을 추가 feature 정보 활용과 구분한다. Class별 recall/support와 predicted/true class mass는 summary.json에 모두 보존했다.", "",
              "## 비용과 판정 범위", "",
              f"새 forward 30회. Arm별 prediction seconds: {canonical(budget['prediction_seconds'])}. CUDA peak {budget['peak_mib']:.2f} MiB. Shared feature-table 비용 {budget['shared_feature_table_seconds']:.3f}s는 한 번씩만 집계했다.",
              f"기록된 실행 attempt {len(budget['attempts'])}개, 실패 {budget['failed_attempts']}개, 합산 wall {budget['attempt_wall_seconds']:.3f}s. 로그 없는 중단 비용을 0이라고 추정하지 않는다.", "",
              "Cora/CiteSeer는 기존 full graph, PubMed/CS/Physics는 기존 induced graph 범위다. Full/edge/DC를 graph마다 선택하지 않았다. Classical degree/affinity/population inference의 개발 실험이며 새 PFN 학습, 방법 novelty, 독립 real holdout 또는 Stage 1의 최종 성공을 뜻하지 않는다. 정상 계산의 성능 실패와 감사/구현 실패를 구분한다.", "",
              "Theta·affinity·population 및 fresh sufficient statistics는 NPZ의 operator__ 배열에 보존했다. JSON의 긴 vector는 요약/hash이며 원래 값은 NPZ에 있다. max_rate는 모든 class 상태에 대한 pair intensity 최대값이며 q-weighted 평균 rate가 아니다. Binary graph의 Poisson saturation 한계를 숨기거나 rate를 몰래 clip하지 않는다.", "",
              "원자료: summary.json, audit.json, budget.json, evaluation.json, artifact_manifest.json. Source·bank·checkpoint·예측 hash를 manifest에 연결했다. Stage 1은 별도 사전 명세와 source lock, 실제 6 fits 및 새 final 평가가 필요하다.", ""]
    return "\n".join(lines)


def selfcheck():
    # CPU-only parser/scoring/gate checks; no historical data, file writes or GPU.
    c = config()
    p = np.array([[.8, .2], [.1, .9]], np.float32)
    q, y = np.array([1, 2], np.int64), np.array([0, 1], np.int64)
    validate_probability(p, q, 2)
    m = score(p, y)
    assert m["accuracy"] == m["balanced_accuracy"] == m["macro_f1"] == 1
    assert abs(m["nll"] + np.mean(np.log([.8, .9]))) < 1e-7
    assert arrays_digest({"q": q}) != arrays_digest({"q": q[::-1]})
    table = {}
    for graph in GRAPHS:
        for arm in ("dc_update", "old_full", "old_edge"):
            table[(graph, "none", arm)] = dict(accuracy=.5, balanced_accuracy=.5, macro_f1=.5, nll=1., brier=.5)
    table[("CoauthorPhysics", "none", "dc_update")]["accuracy"] = .71
    table[("CoauthorCS", "none", "dc_update")]["accuracy"] = .56
    assert gate_from_metrics(table, "dc_update", c["gate"])["passed"]
    table[("CiteSeer", "none", "dc_update")]["nll"] = 1.06
    assert not gate_from_metrics(table, "dc_update", c["gate"])["passed"]
    return dict(status="PASS", cpu_only=True, historical_inputs_used=False,
                files_written=0, optimizer_updates=0, checks=["config", "probability", "metrics", "array_identity", "gate_pass_and_fail"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report", "selfcheck"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.resume)
        print(canonical(dict(status=result["status"], rows=result["rows_count"], references=result["references_count"], manifest_sha256=sha(MANIFEST))))
    elif args.action == "run":
        result = run(args.resume, args.device)
        print(canonical(dict(status=result["status"], predictions=len(result["records"]))))
    elif args.action == "report":
        result = report(args.resume)
        print(canonical(dict(status=result["status"], gate=result["primary_development_gate_passed"], artifact_manifest=str(RESULTS / "artifact_manifest.json"))))
    else:
        print(canonical(selfcheck()))


if __name__ == "__main__":
    main()
