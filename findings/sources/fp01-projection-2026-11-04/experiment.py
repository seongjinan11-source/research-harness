"""FP01 immutable training/evaluation runner.

This module never creates a source lock or a bank. Training requires the root's
existing lock and PASS checks. Incomplete fits/predictions are never silently
restarted. --resume verifies completed artifacts and skips them only on an exact
provenance match. Evaluation stores query probabilities; analysis owns metrics.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Windows runtime ordering: sklearn must precede torch.
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

def _local_module(name, filename):
    # Historical read-only modules use generic names (notably ``bank``) and
    # modify sys.path. Never let those imports select an FP01 dependency.
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _local_module("fp01_runner_common", "common.py")
B = _local_module("fp01_runner_bank", "bank.py")
FeatureModel = _local_module("fp01_runner_operators", "operators.py").FeatureModel

ARMS = ("A", "B", "C_global", "C")
GRAPH_ARMS = ("full", "edge")


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf8")).hexdigest()


def _arrays_digest(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode("utf8"))
        if isinstance(value, (np.ndarray, torch.Tensor)):
            a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            a = np.ascontiguousarray(a)
            if a.dtype.hasobject:
                raise TypeError("Object arrays are not valid episode inputs")
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(_canonical(value).encode("utf8"))
    return h.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(_plain(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _release(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _settings():
    c = G.cfg()
    if int(c.get("schedule_seed_offset", 94000)) != 94000:
        raise ValueError("FP01 must preserve the original seed+94000 task schedule")
    result = dict(train_tasks=int(c.get("train_tasks", 512)),
                  updates=int(c.get("updates", 256)),
                  tasks_per_update=int(c.get("tasks_per_update", 2)),
                  lr=float(c.get("lr", .001)),
                  weight_decay=float(c.get("weight_decay", .0001)),
                  grad_clip=float(c.get("grad_clip", 1.)),
                  warmup_steps=int(c.get("warmup_steps", 16)),
                  iterations=int(c.get("bp_iterations", 24)),
                  damping=float(c.get("damping", .5)),
                  grid_points=int(c.get("grid_points", 41)))
    expected = dict(train_tasks=512, updates=256, tasks_per_update=2, lr=.001,
                    weight_decay=.0001, grad_clip=1., warmup_steps=16,
                    iterations=24, damping=.5, grid_points=41)
    if result != expected:
        raise ValueError(f"FP01 PLAN_v2 settings changed: {result}")
    return result


def _lock_record():
    lock_path = G.ROOT / "results/lock.json"
    checks_path = G.ROOT / "results/checks.json"
    if not lock_path.is_file() or not checks_path.is_file():
        raise RuntimeError("Root must finish PASS checks and create the source lock first")
    if _read_json(checks_path).get("status") != "PASS":
        raise RuntimeError("FP01 correctness checks are not PASS")
    state = G.source_lock(create=False)
    return dict(source_lock=state, source_fingerprint=_fingerprint(state),
                lock_sha256=G.sha(lock_path), checks_sha256=G.sha(checks_path))


def _compact_lock(lock):
    # The immutable root lock has the full source map. Repeating that map in
    # thousands of prediction sidecars would produce very large JSON output.
    return {k: lock[k] for k in ("source_fingerprint", "lock_sha256", "checks_sha256")}


def _git_state():
    def run(*args):
        proc = subprocess.run(["git", "-c", f"safe.directory={G.WORK.as_posix()}",
                               "-C", str(G.WORK), *args],
                              capture_output=True, text=True, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else None
    try:
        return dict(branch=run("branch", "--show-current"),
                    commit=run("rev-parse", "HEAD"))
    except OSError:
        return dict(branch=None, commit=None)


def _manifest_hashes(stage, rows=None):
    if stage == "train":
        paths = [G.ROOT / "bank/train_manifest.json"]
    elif stage == "synthetic":
        paths = [G.ROOT / "bank/final_manifest.json"]
    elif stage == "real":
        roots = sorted({r["bank_root"] for r in rows})
        paths = [G.WORK / root / "bank/manifest.json" for root in roots]
    else:
        raise ValueError(stage)
    return {str(p): G.sha(p) for p in paths}


def _row_id(row):
    # Full row hash distinguishes source rates, views, repetitions and panels.
    label = "_".join(str(row.get(k, "")) for k in
                     ("graph", "base_id", "repetition", "view", "mode"))
    label = "".join(c if c.isalnum() or c in "_-" else "-" for c in label)[:100]
    return f"{label}_{_fingerprint(row)[:20]}"


def _validate_episode(inp, query, truth, missing_query=False):
    x = np.asarray(inp["x"])
    observed = np.asarray(inp["observed"], dtype=bool)
    query = np.asarray(query)
    truth = np.asarray(truth)
    context = np.asarray(inp["context_ids"])
    yc = np.asarray(inp["labels_context"])
    n, _ = x.shape
    k = int(inp["classes"])
    assert observed.shape == x.shape and np.isfinite(x).all()
    assert np.all(observed.all(1) | ~observed.any(1)), "Expected whole-node missingness"
    assert query.ndim == truth.ndim == context.ndim == yc.ndim == 1
    assert len(query) == len(truth) and len(query) > 0
    assert len(context) == len(yc) and len(np.unique(query)) == len(query)
    assert np.all((query >= 0) & (query < n))
    assert np.all((context >= 0) & (context < n))
    assert np.all((truth >= 0) & (truth < k)) and np.all((yc >= 0) & (yc < k))
    assert not np.intersect1d(query, context).size
    assert not observed[context].any(), "Context/source evidence must be disjoint"
    if missing_query:
        assert not observed[query].any(), "Final query must have no physical feature"
    return int(observed.any(1).sum())


def _make_model(arm, seed, device, settings):
    torch.manual_seed(int(seed))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    return FeatureModel(arm, seed=int(seed), iterations=settings["iterations"],
                        damping=settings["damping"], grid_points=settings["grid_points"]).to(device)


def _paths(arm, seed):
    out = G.ROOT / "results" / f"{arm}_s{seed}"
    return out, out / "checkpoint.pt", out / "training.json"


def _train_rows():
    rows = B.rows("train")
    pair = {view: {} for view in ("raw", "projected")}
    for row in rows:
        view, base = row["view"], int(row["base_id"])
        if view not in pair or base in pair[view]:
            raise ValueError("Training bank has an unexpected or duplicate view/base")
        pair[view][base] = row
    expected = set(range(512))
    if any(set(group) != expected for group in pair.values()):
        raise ValueError("Training must contain exactly base ids 0..511 in both views")
    return pair


def _training_identity(arm, seed, settings, lock, manifests, schedule, view_rows, graph_sha):
    view = "raw" if arm == "A" else "projected"
    scheduled = [view_rows[view][int(i)] for i in schedule]
    return dict(seed=int(seed), arm=arm, settings=settings, provenance=lock,
                bank_manifests=manifests, graph_checkpoint_sha256=graph_sha,
                schedule=schedule.tolist(), view=view,
                scheduled_row_ids=[_row_id(r) for r in scheduled],
                scheduled_row_sha256=[r["sha256"] for r in scheduled],
                schedule_generator=f"numpy.default_rng({seed}+94000).permutation(512)",
                train_query_rule="all original noncontext queries, including observed source nodes",
                real_gradient_updates=0)


def _complete_fit(arm, seed, identity, resume):
    out, checkpoint, report = _paths(arm, seed)
    if not out.exists():
        return None
    if not (checkpoint.is_file() and report.is_file()):
        raise RuntimeError(f"Partial fit at {out}; no silent training restart")
    if not resume:
        raise FileExistsError(f"Completed fit exists at {out}; use --resume to verify/skip")
    saved = _read_json(report)
    assert saved.get("status") == "COMPLETE"
    assert saved["identity"] == _plain(identity), "Completed fit provenance changed"
    assert saved["checkpoint_sha256"] == G.sha(checkpoint), "Checkpoint bytes changed"
    assert saved["updates"] == 256 and saved["exposures"] == 512
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert packet["identity"] == _plain(identity)
    assert packet["initial_sha256"] == saved["initial_sha256"]
    assert packet["graph_digest"] == saved["graph_digest_after"] == saved["graph_digest_before"]
    model = _make_model(arm, seed, "cpu", identity["settings"])
    assert G.digest(model) == saved["initial_sha256"], "Reset feature initialization changed"
    model.load_state_dict(packet["state_dict"], strict=True)
    assert G.digest(model) == packet["final_sha256"] == saved["final_sha256"]
    del model
    return saved


class _TrainGraphCache:
    """At most 512 small synthetic graph encodings on CPU; never a real GPU bank."""
    def __init__(self, core):
        self.core = core
        self.data = {}
        self.identities = {}
        self.seconds = 0.
        self.encodes = 0

    def get(self, base, inp, device):
        identity = G.graph_identity(inp)
        if base not in self.data:
            _sync(device)
            began = time.perf_counter()
            encoded = self.core.encode(inp, graph_arm="full")
            self.data[base] = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                               for k, v in encoded.items()}
            self.identities[base] = identity
            _sync(device)
            self.seconds += time.perf_counter() - began
            self.encodes += 1
            del encoded
        assert self.identities[base] == identity, "Raw/projected graph/context mismatch"
        return G.to_device(self.data[base], device=device)


def train(seed, arms=ARMS, device="cuda", resume=False):
    G.setup()
    settings, lock = _settings(), _lock_record()
    pair = _train_rows()
    manifests = _manifest_hashes("train")
    schedule = np.random.default_rng(int(seed) + 94000).permutation(512)
    core = G.GraphCore(seed, device=device)
    graph_initial = core.check()
    cache = _TrainGraphCache(core)
    reports = []
    for arm in arms:
        identity = _training_identity(arm, seed, settings, lock, manifests, schedule,
                                      pair, core.checkpoint_sha)
        complete = _complete_fit(arm, seed, identity, resume)
        if complete is not None:
            reports.append(complete)
            print("TRAIN_VERIFIED_COMPLETE", arm, seed, flush=True)
            continue
        out, checkpoint, report_path = _paths(arm, seed)
        out.mkdir(parents=True, exist_ok=False)
        _write_json_new(out / "started.json", dict(status="STARTED", identity=identity))
        model = _make_model(arm, seed, device, settings).train()
        initial = G.digest(model)
        git_start = _git_state()
        if arm == "B":
            a = _make_model("A", seed, "cpu", settings)
            assert G.digest(a) == initial, "A/B feature initial states must match"
            del a
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert parameters, "FeatureModel has no trainable feature parameters"
        optimizer = torch.optim.AdamW(parameters, lr=settings["lr"],
                                      weight_decay=settings["weight_decay"])
        log, exposure_ids, zero_source_tasks = [], [], 0
        cached_before = cache.encodes
        cache_seconds_before = cache.seconds
        began = time.perf_counter()
        successful_updates = 0
        try:
            with (out / "progress.jsonl").open("x", encoding="utf8") as progress:
                for step in range(settings["updates"]):
                    ids = schedule[step * 2:(step + 1) * 2]
                    lr = settings["lr"] * min((step + 1) / 16, 1) * (
                        .1 + .9 * .5 * (1 + math.cos(math.pi * step / settings["updates"])))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for base in ids:
                        row = pair[identity["view"]][int(base)]
                        inp, query, truth = B.read(row)
                        sources = _validate_episode(inp, query, truth)
                        zero_source_tasks += int(sources == 0)
                        graph = cache.get(int(base), inp, device)
                        # Forward receives no query ids, truths or generator metadata.
                        visible = G.tensor_input(inp, device=device)
                        probability = model(visible, graph)
                        q = torch.as_tensor(query, device=device, dtype=torch.long)
                        y = torch.as_tensor(truth, device=device, dtype=torch.long)
                        selected = probability.index_select(0, q)
                        loss = -selected[torch.arange(len(q), device=device), y].clamp_min(1e-10).log().mean()
                        if not loss.requires_grad:
                            if sources != 0:
                                raise RuntimeError("Nonempty feature inference lost its feature gradient graph")
                            loss = loss + sum(p.sum() * 0 for p in parameters)
                        assert bool(torch.isfinite(loss)), "Nonfinite feature CE"
                        (loss / len(ids)).backward()
                        losses.append(float(loss.detach().cpu()))
                        exposure_ids.append(int(base))
                        del graph, visible, probability, selected, loss, q, y
                    for p in parameters:
                        assert p.grad is None or bool(torch.isfinite(p.grad).all()), "Nonfinite feature gradient"
                    norm = torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"])
                    assert bool(torch.isfinite(norm)), "Nonfinite gradient norm"
                    optimizer.step()
                    assert all(bool(torch.isfinite(p).all()) for p in parameters)
                    successful_updates += 1
                    entry = dict(step=step + 1, ce=float(np.mean(losses)), lr=lr,
                                 grad_norm=float(norm.detach().cpu()), base_ids=ids.tolist(),
                                 seconds=time.perf_counter() - began)
                    log.append(entry)
                    progress.write(_canonical(entry) + "\n")
                    progress.flush()
                    if (step + 1) % 32 == 0:
                        print("TRAIN", arm, seed, step + 1, round(entry["ce"], 6),
                              round(entry["seconds"], 2), flush=True)
            _sync(device)
            assert exposure_ids == schedule.tolist(), "Task exposure order changed"
            graph_final = core.check()
            assert graph_final == graph_initial
            assert _lock_record() == lock, "Source/check lock changed during training"
            final = G.digest(model)
            packet = dict(state_dict={n: t.detach().cpu() for n, t in model.state_dict().items()},
                          identity=identity, initial_sha256=initial, final_sha256=final,
                          graph_digest=graph_final)
            with checkpoint.open("xb") as stream:
                torch.save(packet, stream)
            summary = dict(status="COMPLETE", identity=identity,
                           initial_sha256=initial, final_sha256=final,
                           graph_digest_before=graph_initial, graph_digest_after=graph_final,
                           checkpoint_sha256=G.sha(checkpoint), updates=successful_updates,
                           exposures=len(exposure_ids), exposure_order=exposure_ids,
                           zero_source_tasks=zero_source_tasks,
                           trainable_parameters=sum(p.numel() for p in parameters),
                           total_feature_parameters=sum(p.numel() for p in model.parameters()),
                           seconds=time.perf_counter() - began,
                           graph_encodes_in_this_arm=cache.encodes - cached_before,
                           graph_encoding_seconds_in_this_arm=cache.seconds - cache_seconds_before,
                           graph_cache_shared_across_arms=True, log=log,
                           git_start=git_start, git_end=_git_state(),
                           final_checkpoint_only=True, real_gradient_updates=0)
            _write_json_new(report_path, summary)
            reports.append(summary)
            print("TRAIN_COMPLETE", arm, seed, summary["checkpoint_sha256"], flush=True)
        except Exception as exc:
            failure = dict(status="FAILED", identity=identity,
                           successful_updates=successful_updates, exposures=len(exposure_ids),
                           exception=repr(exc), traceback=traceback.format_exc())
            _write_json_new(out / "failure.json", failure)
            raise
        finally:
            del model, optimizer, parameters
            _release(device)
    core.check()
    return reports


def load_model(arm, seed, device="cuda", lock=None, settings=None):
    settings = _settings() if settings is None else settings
    lock = _lock_record() if lock is None else lock
    out, checkpoint, report_path = _paths(arm, seed)
    report = _read_json(report_path)
    assert report["status"] == "COMPLETE" and report["updates"] == 256 and report["exposures"] == 512
    assert report["identity"]["provenance"] == lock
    assert report["identity"]["seed"] == int(seed) and report["identity"]["arm"] == arm
    assert report["identity"]["settings"] == settings
    assert report["checkpoint_sha256"] == G.sha(checkpoint)
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert packet["identity"] == report["identity"]
    model = _make_model(arm, seed, device, settings)
    model.load_state_dict(packet["state_dict"], strict=True)
    model.eval()
    assert G.digest(model) == report["final_sha256"] == packet["final_sha256"]
    return model, report


def _prediction_paths(arm, seed, stage, graph_arm, row):
    out, _, _ = _paths(arm, seed)
    path = out / "predictions" / stage / graph_arm / (_row_id(row) + ".npz")
    return path, path.with_suffix(".json")


def _check_probability(p, query, classes):
    assert p.shape == (len(query), classes)
    assert np.isfinite(p).all() and np.min(p) >= -1e-7 and np.max(p) <= 1 + 1e-7
    assert np.max(np.abs(p.sum(1) - 1)) < 2e-5, "Invalid normalized probability"


def _prediction_resume(path, sidecar, identity, query, truth, context, yc, resume):
    if not path.exists() and not sidecar.exists():
        return None
    if not (path.is_file() and sidecar.is_file()):
        raise RuntimeError(f"Partial prediction at {path}; do not silently overwrite")
    if not resume:
        raise FileExistsError(f"Prediction exists at {path}; use --resume to verify/skip")
    meta = _read_json(sidecar)
    assert meta["identity"] == _plain(identity), "Prediction provenance/row identity changed"
    assert meta["prediction_sha256"] == G.sha(path), "Prediction bytes changed"
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z["query"], query) and np.array_equal(z["y"], truth)
        assert np.array_equal(z["context"], context) and np.array_equal(z["yc"], yc)
        _check_probability(z["p"], query, int(identity["classes"]))
    return meta


def _evaluation_rows(stage):
    if stage == "synthetic":
        rows = B.rows("final")
    elif stage == "real":
        rows = G.real_rows()
        if len(rows) != 75:
            raise ValueError(f"Expected all 75 reused real development rows, got {len(rows)}")
    else:
        raise ValueError(stage)
    if not rows:
        raise ValueError(f"No rows in stage {stage}")
    ids = [_row_id(r) for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate evaluation row identifiers"
    return sorted(rows, key=lambda r: (str(r.get("graph", "")), str(r.get("base_id", "")),
                                      int(r.get("repetition", 0)), _row_id(r)))


def evaluate(seed, stages=("synthetic", "real"), arms=ARMS, device="cuda", resume=False):
    G.setup()
    settings, lock = _settings(), _lock_record()
    core = G.GraphCore(seed, device=device)
    graph_initial = core.check()
    models, fitted = {}, {}
    for arm in arms:
        models[arm], fitted[arm] = load_model(arm, seed, device, lock, settings)
        assert fitted[arm]["identity"]["graph_checkpoint_sha256"] == core.checkpoint_sha
    results = {}
    for stage in stages:
        rows = _evaluation_rows(stage)
        manifests = _manifest_hashes(stage, rows)
        outcomes = {arm: [] for arm in arms}
        for graph_arm in GRAPH_ARMS:
            cached_identity, graph = None, None
            graph_seconds = 0.
            for row in rows:
                inp, query, truth = G.read_real(row) if stage == "real" else B.read(row)
                sources = _validate_episode(inp, query, truth, missing_query=True)
                query, truth = np.asarray(query, dtype=np.int64), np.asarray(truth, dtype=np.int64)
                context = np.asarray(inp["context_ids"], dtype=np.int64)
                yc = np.asarray(inp["labels_context"], dtype=np.int64)
                gid = G.graph_identity(inp)
                if gid != cached_identity:
                    if graph is not None:
                        del graph
                        graph = None
                        _release(device)
                    _sync(device)
                    began = time.perf_counter()
                    graph = core.encode(inp, graph_arm=graph_arm)
                    _sync(device)
                    graph_seconds = time.perf_counter() - began
                    cached_identity = gid
                input_sha = _arrays_digest(inp)
                targets_sha = _arrays_digest(dict(query=query, y=truth, context=context, yc=yc))
                visible = None
                for arm in arms:
                    identity = dict(stage=stage, seed=int(seed), arm=arm, graph_arm=graph_arm,
                                    row=_plain(row), row_id=_row_id(row),
                                    row_sha256=row.get("sha256"), bank_manifests=manifests,
                                    input_sha256=input_sha, target_arrays_sha256=targets_sha,
                                    graph_identity=gid, graph_checkpoint_sha256=core.checkpoint_sha,
                                    graph_digest=graph_initial,
                                    classes=int(inp["classes"]), sources=sources,
                                    model_sha256=fitted[arm]["final_sha256"],
                                    checkpoint_sha256=fitted[arm]["checkpoint_sha256"],
                                    provenance=_compact_lock(lock), target_gradient_updates=0,
                                    feature_role_protocol="frozen common graph-only weights")
                    path, sidecar = _prediction_paths(arm, seed, stage, graph_arm, row)
                    meta = _prediction_resume(path, sidecar, identity, query, truth, context, yc, resume)
                    if meta is None:
                        if visible is None:
                            visible = G.tensor_input(inp, device=device)
                        _sync(device)
                        if torch.device(device).type == "cuda":
                            baseline_bytes = torch.cuda.memory_allocated(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        else:
                            baseline_bytes = None
                        began = time.perf_counter()
                        with torch.no_grad():
                            probability = models[arm](visible, graph)
                            if sources == 0:
                                # Verify the actual algorithm before adopting the p0 shortcut.
                                if not torch.allclose(probability, graph["p0"], atol=2e-6, rtol=0):
                                    raise AssertionError("Source0 algorithm differs from cached graph-only p0")
                                probability = graph["p0"]
                            p = probability.index_select(0, torch.as_tensor(query, device=device,
                                                                          dtype=torch.long)).cpu().numpy()
                        _sync(device)
                        seconds = time.perf_counter() - began
                        peak_bytes = (torch.cuda.max_memory_allocated(device)
                                      if torch.device(device).type == "cuda" else None)
                        _check_probability(p, query, int(inp["classes"]))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, p=p, y=truth, query=query,
                                                context=context, yc=yc)
                        meta = dict(identity=identity, prediction_sha256=G.sha(path),
                                    stage=stage, seed=int(seed), arm=arm, graph_arm=graph_arm,
                                    row=_plain(row), row_id=identity["row_id"],
                                    path=str(path), prediction_path=str(path.relative_to(G.ROOT)),
                                    checkpoint_sha256=fitted[arm]["checkpoint_sha256"],
                                    source_lock_sha256=lock["lock_sha256"],
                                    sources=sources, none=bool(sources == 0),
                                    seconds=seconds, inference_seconds=seconds,
                                    peak_mib=None if peak_bytes is None else peak_bytes / 2**20,
                                    incremental_peak_mib=None if peak_bytes is None else (peak_bytes - baseline_bytes) / 2**20,
                                    common_graph_encoding_seconds=graph_seconds,
                                    graph_weights=graph["weights"].detach().cpu().tolist(),
                                    full_role_weights=_plain(graph.get("full_role_weights", graph["weights"])),
                                    active_role_ids=_plain(graph.get("active_role_ids")),
                                    git=_git_state(),
                                    source0_p0_cache=bool(sources == 0))
                        _write_json_new(sidecar, meta)
                        del probability, p
                    outcomes[arm].append(dict(path=str(path), metadata=str(sidecar),
                                              row_id=identity["row_id"], graph_arm=graph_arm,
                                              prediction_sha256=meta["prediction_sha256"],
                                              seconds=meta["seconds"]))
                if visible is not None:
                    del visible
                print("EVAL_ROW", stage, seed, graph_arm, _row_id(row), flush=True)
            if graph is not None:
                del graph
                _release(device)
        assert core.check() == graph_initial
        assert _lock_record() == lock
        for arm in arms:
            assert G.digest(models[arm]) == fitted[arm]["final_sha256"], "Evaluation changed a feature model"
            assert len(outcomes[arm]) == 2 * len(rows)
            out, _, _ = _paths(arm, seed)
            index = dict(status="COMPLETE", stage=stage, seed=int(seed), arm=arm,
                         provenance=_compact_lock(lock), bank_manifests=manifests,
                         rows_per_graph_arm=len(rows), graph_arms=list(GRAPH_ARMS),
                         checkpoint_sha256=fitted[arm]["checkpoint_sha256"],
                         records=outcomes[arm], target_gradient_updates=0,
                         real_data_role="reused development evidence" if stage == "real" else None)
            dest = out / f"evaluation_{stage}.json"
            if dest.exists():
                assert resume and _read_json(dest) == _plain(index), "Evaluation index changed"
            else:
                _write_json_new(dest, index)
            results[(stage, arm)] = index
            print("EVALUATION_COMPLETE", stage, arm, seed, len(outcomes[arm]), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--seed", type=int, choices=(11, 12, 13), required=True)
    parser.add_argument("--arm", choices=ARMS, help="Default: all four arms with a common graph cache")
    parser.add_argument("--stage", choices=("synthetic", "real", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Verify/skip complete matching outputs only")
    args = parser.parse_args()
    arms = (args.arm,) if args.arm else ARMS
    if args.action == "train":
        train(args.seed, arms=arms, device=args.device, resume=args.resume)
    else:
        stages = ("synthetic", "real") if args.stage == "all" else (args.stage,)
        evaluate(args.seed, stages=stages, arms=arms, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
