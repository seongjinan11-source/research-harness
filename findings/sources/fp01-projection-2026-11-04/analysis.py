"""FP01 independent saved-probability audit and prespecified statistics.

No model forward, optimizer, bank generation, seed selection or probability
ensemble is performed here. Run only after root finishes all twelve fits and
both evaluation stages. ``selfcheck`` is a small statistics/parser check;
``readiness`` reports missing artifacts without writing any results.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Windows: must precede any common.py / torch import.
from sklearn.metrics import balanced_accuracy_score, f1_score
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier")
ARMS = ("A", "B", "C_global", "C")
GRAPH_ARMS = ("full", "edge")
SEEDS = (11, 12, 13)
BASELINES = ("lp_harmonic", "lp_restart", "public_graphpfn",
             "pcfi_public_graphpfn", "fisf_public_graphpfn")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def arrays_digest(values):
    """Independent implementation of the runner's documented array identity."""
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode("utf8"))
        if isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            assert not a.dtype.hasobject
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(canonical(value).encode("utf8"))
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def row_id(row):
    label = "_".join(str(row.get(k, "")) for k in
                     ("graph", "base_id", "repetition", "view", "mode"))
    label = "".join(c if c.isalnum() or c in "_-" else "-" for c in label)[:100]
    return f"{label}_{fingerprint(row)[:20]}"


def metrics(probability, truth):
    # Matches the audited historical scoring convention. Probability validation
    # is separate; clipping is only the finite NLL/Brier scoring convention.
    p = np.clip(np.asarray(probability, np.float64), 1e-8, 1)
    p /= p.sum(1, keepdims=True)
    y = np.asarray(truth, np.int64)
    predicted = p.argmax(1)
    return dict(accuracy=float(np.mean(predicted == y)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, average="macro", zero_division=0)),
                nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                brier=float(np.square(p - np.eye(p.shape[1])[y]).sum(1).mean()))


def mean_metrics(records):
    records = list(records)
    assert records, "Cannot average an empty metric group"
    return {m: float(np.mean([r[m] for r in records])) for m in METRICS}


def difference(a, b):
    return {m: float(a[m] - b[m]) for m in METRICS}


def descriptive(values):
    values = [float(v) for v in values]
    return dict(mean=float(np.mean(values)), values=values,
                min=float(np.min(values)), max=float(np.max(values)))


def _local(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def readiness():
    needed = [ROOT / "results" / name for name in ("lock.json", "checks.json")]
    needed += [ROOT / "bank" / f"{split}_manifest.json" for split in ("train", "dev", "final")]
    for arm in ARMS:
        for seed in SEEDS:
            out = ROOT / "results" / f"{arm}_s{seed}"
            needed += [out / name for name in ("training.json", "checkpoint.pt",
                                               "evaluation_synthetic.json", "evaluation_real.json")]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    return dict(ready=not missing, required_files=len(needed), missing=missing)


def require_ready():
    state = readiness()
    if not state["ready"]:
        raise FileNotFoundError(
            f"FP01 results incomplete: {len(state['missing'])} required artifacts missing; "
            f"first: {', '.join(state['missing'][:6])}. No analysis outputs were written.")


class Auditor:
    def __init__(self):
        require_ready()
        self.G = _local("fp01_analysis_common", "common.py")
        self.B = _local("fp01_analysis_bank", "bank.py")
        self.config = read_json(ROOT / "protocol.json")
        self.lock = self.G.source_lock(create=False)
        checks = read_json(ROOT / "results/checks.json")
        assert checks["status"] == "PASS" and checks.get("optimizer_updates", 0) == 0
        assert not checks.get("final_bank_used", False)
        self.provenance = dict(source_lock=self.lock,
                               source_fingerprint=fingerprint(self.lock),
                               lock_sha256=sha(ROOT / "results/lock.json"),
                               checks_sha256=sha(ROOT / "results/checks.json"))
        self.compact = {k: self.provenance[k] for k in
                        ("source_fingerprint", "lock_sha256", "checks_sha256")}
        self.checked_hashes = {}
        self._hash_maps(self.lock)
        self.training, self.records, self.artifacts = [], [], []
        self.none_predictions, self.role_weights = {}, {}
        self.none_max_error = 0.
        self.singleton_max_error = 0.
        self.graph_digests = {}
        self.episode_cache = {}
        self.index_rows = []
        self._prepare_banks()

    def check_file(self, path, expected=None):
        path = Path(path).resolve()
        actual = self.checked_hashes.get(str(path))
        if actual is None:
            actual = sha(path)
            self.checked_hashes[str(path)] = actual
        if expected is not None:
            assert actual == expected, f"Digest changed: {path}"
        return actual

    def _hash_maps(self, value):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if key in {"files", "graph_checkpoints"} and isinstance(child, dict):
                for path, digest in child.items():
                    assert isinstance(digest, str) and len(digest) == 64
                    self.check_file(path, digest)
            elif isinstance(child, dict):
                self._hash_maps(child)

    def _prepare_banks(self):
        for split in ("train", "dev", "final"):
            manifest = self.B._validate_manifest(ROOT / "bank" / f"{split}_manifest.json")
            assert manifest["protocol_sha256"] == sha(ROOT / "protocol.json")
            assert manifest["generator_sha256"] == sha(ROOT / "bank.py")
        self.train = {view: {r["base_id"]: r for r in self.B.train_rows(view)}
                      for view in ("raw", "projected")}
        final = self.B.rows("final")
        assert len(final) == self.config["bank_layout"]["final_rows"] == 696
        bases = {int(r["base_id"]): (r["nodes"], r["classes"]) for r in final}
        assert set(bases) == set(range(48))
        strata = defaultdict(list)
        for bid, shape in sorted(bases.items()):
            strata[shape].append(bid)
        assert len(strata) == 12 and all(len(ids) == 4 for ids in strata.values())
        self.strata = dict(strata)
        self.eval_rows = {"synthetic": final, "real": self.G.real_rows()}
        assert len(self.eval_rows["real"]) == 75
        self.rows_by_id = {stage: {row_id(r): r for r in rows}
                           for stage, rows in self.eval_rows.items()}
        for stage, rows in self.eval_rows.items():
            assert len(self.rows_by_id[stage]) == len(rows)
        self.manifests = {
            "train": {str(ROOT / "bank/train_manifest.json"): sha(ROOT / "bank/train_manifest.json")},
            "synthetic": {str(ROOT / "bank/final_manifest.json"): sha(ROOT / "bank/final_manifest.json")},
            "real": {str(WORK / root / "bank/manifest.json"): sha(WORK / root / "bank/manifest.json")
                     for root in sorted({r["bank_root"] for r in self.eval_rows["real"]})}}

    def episode(self, stage, row):
        key = (stage, row_id(row))
        if key not in self.episode_cache:
            inp, query, truth = (self.G.read_real(row) if stage == "real" else self.B.read(row))
            query, truth = np.asarray(query, np.int64), np.asarray(truth, np.int64)
            context = np.asarray(inp["context_ids"], np.int64)
            yc = np.asarray(inp["labels_context"], np.int64)
            assert not inp["observed"][query].any()
            assert not inp["observed"][context].any()
            assert not np.intersect1d(query, context).size
            assert len(query) == len(truth) > 0
            targets = dict(query=query, y=truth, context=context, yc=yc)
            # Cache only small visible identity/target arrays, not large feature matrices.
            self.episode_cache[key] = dict(targets=targets, classes=int(inp["classes"]),
                sources=int(inp["observed"].any(1).sum()), input_sha256=arrays_digest(inp),
                target_arrays_sha256=arrays_digest(targets), graph_identity=self.G.graph_identity(inp))
        return self.episode_cache[key]

    def audit_training(self):
        expected_params = dict(A=721, B=721, C_global=3, C=751)
        for seed in SEEDS:
            schedule = np.random.default_rng(seed + 94000).permutation(512).tolist()
            for arm in ARMS:
                out = ROOT / "results" / f"{arm}_s{seed}"
                report = read_json(out / "training.json")
                checkpoint = out / "checkpoint.pt"
                assert report["status"] == "COMPLETE"
                assert report["updates"] == 256 and report["exposures"] == 512
                assert report["exposure_order"] == schedule
                assert report["real_gradient_updates"] == 0 and report["final_checkpoint_only"]
                identity = report["identity"]
                assert identity["seed"] == seed and identity["arm"] == arm
                assert identity["provenance"] == self.provenance
                assert identity["bank_manifests"] == self.manifests["train"]
                assert identity["schedule"] == schedule
                assert identity["real_gradient_updates"] == 0
                view = "raw" if arm == "A" else "projected"
                assert identity["view"] == view
                scheduled = [self.train[view][bid] for bid in schedule]
                assert identity["scheduled_row_ids"] == [row_id(r) for r in scheduled]
                assert identity["scheduled_row_sha256"] == [r["sha256"] for r in scheduled]
                settings = identity["settings"]
                for key in ("train_tasks", "updates", "tasks_per_update", "lr", "weight_decay",
                            "grad_clip", "warmup_steps", "grid_points", "damping"):
                    assert settings[key] == self.config[key], (key, arm, seed)
                assert settings["iterations"] == self.config["bp_iterations"]
                logs = report["log"]
                assert len(logs) == 256
                for step, entry in enumerate(logs):
                    assert entry["step"] == step + 1
                    assert entry["base_ids"] == schedule[step * 2:(step + 1) * 2]
                    expected_lr = .001 * min((step + 1) / 16, 1) * (
                        .1 + .9 * .5 * (1 + math.cos(math.pi * step / 256)))
                    assert abs(entry["lr"] - expected_lr) < 1e-14
                    assert math.isfinite(entry["ce"]) and math.isfinite(entry["grad_norm"])
                assert report["graph_digest_before"] == report["graph_digest_after"]
                self.graph_digests.setdefault(seed, report["graph_digest_before"])
                assert self.graph_digests[seed] == report["graph_digest_before"]
                inherited = WORK / f"factor_real_graph_pilot/results/factor_s{seed}/checkpoint.pt"
                assert identity["graph_checkpoint_sha256"] == self.check_file(inherited)
                self.check_file(checkpoint, report["checkpoint_sha256"])
                packet = self.G.torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert packet["identity"] == identity
                assert packet["graph_digest"] == report["graph_digest_before"]
                assert packet["initial_sha256"] == report["initial_sha256"]
                assert state_digest(packet["state_dict"]) == packet["final_sha256"] == report["final_sha256"]
                parameter_count = 0
                for name, tensor in packet["state_dict"].items():
                    assert name in {"gh_nodes", "gh_log_weights", "raw"} or name.startswith("feature.")
                    if not name.startswith("gh_"):
                        parameter_count += tensor.numel()
                assert parameter_count == report["trainable_parameters"] == expected_params[arm]
                assert report["total_feature_parameters"] == parameter_count
                self.training.append(dict(arm=arm, seed=seed, report=report,
                                          path=str(out / "training.json"), checkpoint=str(checkpoint)))
                self.artifacts.append(dict(kind="checkpoint", path=str(checkpoint),
                                           sha256=report["checkpoint_sha256"]))
                del packet
            a, b = [r["report"] for r in self.training if r["seed"] == seed and r["arm"] in {"A", "B"}]
            assert a["initial_sha256"] == b["initial_sha256"], "A/B reset initial state differs"
        assert sum(r["report"]["updates"] for r in self.training) == 3072
        assert sum(r["report"]["exposures"] for r in self.training) == 6144

    def _prediction(self, path, expected, tolerance=2e-5):
        with np.load(path, allow_pickle=False) as saved:
            for key, value in expected["targets"].items():
                assert np.array_equal(saved[key], value), (str(path), key)
            p = saved["p"].copy()
        assert p.shape == (len(expected["targets"]["y"]), expected["classes"])
        assert np.isfinite(p).all() and p.min() >= -1e-7 and p.max() <= 1 + 1e-7
        assert float(np.max(np.abs(p.sum(1) - 1))) < tolerance
        return p

    def audit_predictions(self):
        seen = set()
        for fitted in self.training:
            arm, seed, fit = fitted["arm"], fitted["seed"], fitted["report"]
            out = ROOT / "results" / f"{arm}_s{seed}"
            for stage in ("synthetic", "real"):
                index_path = out / f"evaluation_{stage}.json"
                index = read_json(index_path)
                assert index["status"] == "COMPLETE" and index["stage"] == stage
                assert index["arm"] == arm and index["seed"] == seed
                assert index["provenance"] == self.compact
                assert index["bank_manifests"] == self.manifests[stage]
                assert index["graph_arms"] == list(GRAPH_ARMS)
                assert index["rows_per_graph_arm"] == len(self.eval_rows[stage])
                assert index["checkpoint_sha256"] == fit["checkpoint_sha256"]
                assert index["target_gradient_updates"] == 0
                expected_ids = {(graph_arm, rid) for graph_arm in GRAPH_ARMS
                                for rid in self.rows_by_id[stage]}
                index_ids = set()
                for item in index["records"]:
                    graph_arm, rid = item["graph_arm"], item["row_id"]
                    assert (graph_arm, rid) not in index_ids
                    index_ids.add((graph_arm, rid))
                    row = self.rows_by_id[stage][rid]
                    path = out / "predictions" / stage / graph_arm / f"{rid}.npz"
                    sidecar = path.with_suffix(".json")
                    assert Path(item["path"]).resolve() == path.resolve()
                    assert Path(item["metadata"]).resolve() == sidecar.resolve()
                    meta = read_json(sidecar)
                    identity = meta["identity"]
                    assert identity["row"] == row and meta["row"] == row
                    assert identity["row_id"] == meta["row_id"] == rid
                    assert identity["stage"] == meta["stage"] == stage
                    assert identity["arm"] == meta["arm"] == arm
                    assert identity["seed"] == meta["seed"] == seed
                    assert identity["graph_arm"] == meta["graph_arm"] == graph_arm
                    assert identity["row_sha256"] == row["sha256"]
                    assert identity["bank_manifests"] == self.manifests[stage]
                    assert identity["provenance"] == self.compact
                    assert identity["target_gradient_updates"] == 0
                    assert identity["feature_role_protocol"] == "frozen common graph-only weights"
                    assert identity["checkpoint_sha256"] == meta["checkpoint_sha256"] == fit["checkpoint_sha256"]
                    assert identity["model_sha256"] == fit["final_sha256"]
                    assert identity["graph_checkpoint_sha256"] == fit["identity"]["graph_checkpoint_sha256"]
                    assert identity["graph_digest"] == self.graph_digests[seed]
                    assert meta["source_lock_sha256"] == self.provenance["lock_sha256"]
                    assert Path(meta["path"]).resolve() == path.resolve()
                    assert Path(meta["prediction_path"]) == path.relative_to(ROOT)
                    expected = self.episode(stage, row)
                    for field in ("classes", "sources", "input_sha256", "target_arrays_sha256", "graph_identity"):
                        assert identity[field] == expected[field], (rid, field)
                    assert meta["sources"] == expected["sources"]
                    assert meta["none"] == (expected["sources"] == 0)
                    digest = self.check_file(path, meta["prediction_sha256"])
                    assert item["prediction_sha256"] == digest
                    p = self._prediction(path, expected)
                    uid = (stage, graph_arm, rid, arm, seed)
                    assert uid not in seen
                    seen.add(uid)
                    key = (stage, graph_arm, expected["graph_identity"], seed)
                    weights = np.asarray(meta["full_role_weights"], np.float64)
                    assert weights.shape == (3,) and np.isfinite(weights).all()
                    assert abs(weights.sum() - 1) < 1e-10
                    self.role_weights.setdefault(key, weights)
                    np.testing.assert_allclose(weights, self.role_weights[key], atol=1e-12, rtol=0)
                    active = np.asarray(meta["active_role_ids"], np.int64)
                    np.testing.assert_array_equal(np.asarray(meta["graph_weights"]), weights[active])
                    if row["mode"] == "none":
                        if key in self.none_predictions:
                            error = float(np.max(np.abs(p - self.none_predictions[key])))
                            self.none_max_error = max(self.none_max_error, error)
                            assert error <= 2e-6, "Common graph-only none prediction changed across feature arms"
                        else:
                            self.none_predictions[key] = p.copy()
                    record = dict(stage=stage, graph_arm=graph_arm, arm=arm, seed=seed,
                                  row_id=rid, graph=row["graph"], base_id=row.get("base_id"),
                                  repetition=row.get("repetition"), view=row.get("view", "projected"),
                                  mode=row["mode"], rate=row.get("rate"), classes=expected["classes"],
                                  nodes=row["nodes"], sources=expected["sources"], queries=len(p),
                                  panel=row.get("panel", "synthetic"), primary=bool(row.get("primary", False)),
                                  graph_identity=expected["graph_identity"],
                                  prediction_path=str(path), prediction_sha256=digest,
                                  sidecar_path=str(sidecar), sidecar_sha256=sha(sidecar),
                                  seconds=float(meta["inference_seconds"]), peak_mib=meta.get("peak_mib"),
                                  incremental_peak_mib=meta.get("incremental_peak_mib"),
                                  **metrics(p, expected["targets"]["y"]))
                    self.records.append(record)
                assert index_ids == expected_ids, (stage, arm, seed, len(index_ids), len(expected_ids))
                directory = out / "predictions" / stage
                assert len(list(directory.glob("*/*.npz"))) == len(expected_ids)
                assert len(list(directory.glob("*/*.json"))) == len(expected_ids)
                self.artifacts.append(dict(kind="evaluation_index", path=str(index_path), sha256=sha(index_path)))
        expected_total = sum(len(rows) * len(ARMS) * len(SEEDS) * len(GRAPH_ARMS)
                             for rows in self.eval_rows.values())
        assert len(seen) == expected_total == 18504
        # Singletons are boundary invariances; use graph-matched none predictions.
        for r in self.records:
            if r["sources"] <= 1:
                key = (r["stage"], r["graph_arm"], r["graph_identity"], r["seed"])
                with np.load(r["prediction_path"], allow_pickle=False) as z:
                    error = float(np.max(np.abs(z["p"] - self.none_predictions[key])))
                self.singleton_max_error = max(self.singleton_max_error, error)
                assert error <= 1e-5, "Source0/1 exchangeability violated"

    def audit_baselines(self):
        oldroot = WORK / "factor_edge_only_benchmark"
        oldaudit = read_json(oldroot / "results/audit.json")
        assert oldaudit["status"] == "PASS"
        self._hash_maps(read_json(oldroot / "results/lock.json"))
        records = []
        lp = {}
        for row in self.eval_rows["real"]:
            expected = self.episode("real", row)
            for arm in BASELINES:
                basename = f"{row['graph']}_r{row['repetition']}_{row['mode']}_{arm}"
                sidecar = oldroot / "results/predictions" / (basename + ".json")
                meta = read_json(sidecar)
                for key in ("graph", "repetition", "mode", "path", "sha256", "bank_root", "panel"):
                    assert meta[key] == row[key], (basename, key)
                assert meta["arm"] == arm and meta["seed"] is None
                assert meta["metadata"]["target_updates"] == 0
                path = oldroot / meta["prediction"]
                digest = self.check_file(path, meta["prediction_sha256"])
                p = self._prediction(path, expected, tolerance=3e-5)
                if arm.startswith("lp_"):
                    assert meta["metadata"]["features_used"] is False
                    assert meta["metadata"]["max_equation_residual"] < 1e-7
                    key = (row["graph"], row["repetition"], arm)
                    if key in lp:
                        assert np.array_equal(lp[key], p), "LP must be feature-intervention invariant"
                    else:
                        lp[key] = p.copy()
                else:
                    source = Path(meta["metadata"]["reused_from"])
                    self.check_file(source, meta["metadata"]["reused_sha256"])
                    with np.load(source, allow_pickle=False) as saved:
                        assert np.array_equal(saved["p"], p)
                records.append(dict(stage="real", graph_arm="reference", arm=arm, seed=None,
                    graph=row["graph"], repetition=row["repetition"], mode=row["mode"],
                    panel=row["panel"], view="historical_native_or_graph_only",
                    sources=expected["sources"], classes=expected["classes"], nodes=row["nodes"],
                    queries=len(p), prediction_path=str(path), prediction_sha256=digest,
                    sidecar_path=str(sidecar), sidecar_sha256=sha(sidecar),
                    **metrics(p, expected["targets"]["y"])))
        assert len(records) == 75 * 5
        self.records.extend(records)
        self.artifacts.append(dict(kind="historical_baseline_audit", path=str(oldroot / "results/audit.json"),
                                   sha256=sha(oldroot / "results/audit.json")))

    def real_source_coverage(self):
        """Post-hoc labels are scoring-only; never returned to a model."""
        labels = {}
        output = []
        for row in self.eval_rows["real"]:
            graph, rep = row["graph"], row["repetition"]
            path = WORK / row["bank_root"] / row["path"]
            if graph not in labels:
                if graph in {"Cora", "CiteSeer"}:
                    raw = self.G.E.D.real(graph)
                else:
                    raw = self.G.E.H.B.load_raw(graph)
                labels[graph] = raw["y"]
                del raw
            with np.load(path, allow_pickle=False) as z:
                y = labels[graph]
                if "original_ids" in z:
                    y = y[z["original_ids"]]
                assert np.array_equal(y[z["context_ids"]], z["labels_context"])
                assert np.array_equal(y[z["query"]], z["labels_query"])
                counts = np.bincount(y[z["observed"].any(1)], minlength=int(z["classes"]))
            output.append(dict(graph=graph, repetition=rep, mode=row["mode"], panel=row["panel"],
                               source_class_counts=counts.tolist(), source_classes=int((counts > 0).sum()),
                               sources=int(counts.sum()), posthoc_scoring_only=True))
        return output


def stratified_bootstrap(values, strata, resamples=5000, seed=6046110400, level=.975):
    ids = sorted(values)
    assert len(ids) == 48
    rng = np.random.default_rng(seed)
    groups = [np.asarray([values[i] for i in group], np.float64)
              for _, group in sorted(strata.items())]
    assert len(groups) == 12 and all(len(group) == 4 for group in groups)
    draws = np.zeros(resamples, np.float64)
    for group in groups:
        draws += group[rng.integers(0, len(group), size=(resamples, len(group)))].sum(1)
    draws /= len(ids)
    alpha = (1 - level) / 2
    return dict(mean=float(np.mean([values[i] for i in ids])),
                ci=[float(v) for v in np.quantile(draws, [alpha, 1 - alpha])],
                ci_level=level, resamples=resamples, base_graphs=48,
                strata=12, graphs_per_stratum=4, fixed_training_seeds=3,
                by_base_graph={str(i): float(values[i]) for i in ids})


def synthetic_results(records, config, strata):
    synthetic = [r for r in records if r["stage"] == "synthetic"]
    lookup = {(r["graph_arm"], r["base_id"], r["view"], r["mode"], r["arm"], r["seed"]): r
              for r in synthetic}
    assert len(lookup) == len(synthetic) == 16704
    tables = []
    groups = defaultdict(list)
    for r in synthetic:
        groups[(r["graph_arm"], r["view"], r["mode"], r["arm"])].append(r)
    for (graph_arm, view, mode, arm), selected in sorted(groups.items()):
        tables.append(dict(graph_arm=graph_arm, view=view, mode=mode, arm=arm,
                           base_graphs=len({r["base_id"] for r in selected}), **mean_metrics(selected)))
    by_class = []
    groups = defaultdict(list)
    for r in synthetic:
        groups[(r["graph_arm"], r["view"], r["mode"], r["classes"], r["arm"])].append(r)
    for (graph_arm, view, mode, classes, arm), selected in sorted(groups.items()):
        by_class.append(dict(graph_arm=graph_arm, view=view, mode=mode, classes=classes, arm=arm,
                             base_graphs=len({r["base_id"] for r in selected}), **mean_metrics(selected)))
    base_scores = {}
    base_none = {}
    for graph_arm in GRAPH_ARMS:
        for arm in ARMS:
            for bid in range(48):
                selected = [lookup[(graph_arm, bid, "projected", mode, arm, seed)]
                            for mode in ("50", "75", "90", "95") for seed in SEEDS]
                assert all(r["primary"] and r["sources"] >= 2 for r in selected)
                base_scores[(graph_arm, arm, bid)] = mean_metrics(selected)
                base_none[(graph_arm, arm, bid)] = mean_metrics(
                    lookup[(graph_arm, bid, "projected", "none", arm, seed)] for seed in SEEDS)
    panels, paired = {}, {}
    for graph_arm in GRAPH_ARMS:
        panels[graph_arm] = {
            arm: mean_metrics(base_scores[(graph_arm, arm, bid)] for bid in range(48)) for arm in ARMS}
        paired[graph_arm] = {}
        for comparator in ("B", "C_global"):
            pair = {}
            for metric in METRICS:
                values = {bid: base_scores[(graph_arm, "C", bid)][metric] -
                               base_scores[(graph_arm, comparator, bid)][metric] for bid in range(48)}
                pair[metric] = stratified_bootstrap(values, strata, config["bootstrap_resamples"],
                    config["statistics_seed"], config["primary_ci_level_each"])
            paired[graph_arm]["C-minus-" + comparator] = pair
        feature = {}
        for arm in ARMS:
            feature[arm] = {metric: stratified_bootstrap(
                {bid: base_scores[(graph_arm, arm, bid)][metric] -
                      base_none[(graph_arm, arm, bid)][metric] for bid in range(48)}, strata,
                config["bootstrap_resamples"], config["statistics_seed"], .95)
                for metric in METRICS}
        paired[graph_arm]["observed-minus-none"] = feature
    gates = {}
    for comparator in ("B", "C_global"):
        values = paired["full"]["C-minus-" + comparator]
        gates[comparator] = dict(accuracy_lower_positive=values["accuracy"]["ci"][0] > 0,
            accuracy_mean_at_least_one_percentage_point=values["accuracy"]["mean"] >= config["primary_min_gain"],
            nll_guard=values["nll"]["mean"] <= config["nll_guard"])
    absolute = paired["full"]["observed-minus-none"]["C"]["accuracy"]["mean"]
    success = all(all(fields.values()) for fields in gates.values()) and absolute > 0
    gate = dict(adaptive_success=bool(success), comparisons=gates,
                absolute_C_feature_accuracy_gain=absolute, absolute_C_feature_gain_positive=absolute > 0,
                primary="full/projected;50/75/90/95;own-feature-missing;source>=2",
                probability_quality_guard="mean NLL C-comparator<=.01 nats; no calibration claim",
                intervals="97.5% each of two co-primary accuracy contrasts; Bonferroni familywise95%",
                secondary_intervals="other metrics and observed-none95% descriptive/unadjusted",
                edge_panel_is_secondary=True)
    primary_by_seed = []
    for graph_arm in GRAPH_ARMS:
        for arm in ARMS:
            for seed in SEEDS:
                selected = [r for r in synthetic if r["graph_arm"] == graph_arm and r["arm"] == arm
                            and r["seed"] == seed and r["primary"]]
                primary_by_seed.append(dict(graph_arm=graph_arm, arm=arm, seed=seed, **mean_metrics(selected)))
    return dict(primary_panels=panels, gate=gate, all_rates_views=tables,
                by_class_rate_view=by_class, primary_by_seed=primary_by_seed,
                orthogonal_scope="first base graph per12N,Kcells,75/95%;diagnostic only"), paired


def real_results(records, config):
    current = [r for r in records if r["stage"] == "real" and r["graph_arm"] in GRAPH_ARMS]
    baseline = [r for r in records if r["stage"] == "real" and r["graph_arm"] == "reference"]
    assert len(current) == 1800 and len(baseline) == 375
    per_rep, summary, averaged, paired, feature = [], [], [], [], []
    rep_lookup = {}
    for graph_arm in GRAPH_ARMS:
        for graph in config["real_datasets"]:
            panel = "full" if graph in {"Cora", "CiteSeer"} else "induced2000"
            for mode in config["real_modes"]:
                for arm in ARMS + BASELINES:
                    source = current if arm in ARMS else baseline
                    selected = [r for r in source if r["graph"] == graph and r["mode"] == mode
                                and r["arm"] == arm and (arm in BASELINES or r["graph_arm"] == graph_arm)]
                    assert len(selected) == (9 if arm in ARMS else 3)
                    for rep in config["real_repetitions"]:
                        values = [r for r in selected if r["repetition"] == rep]
                        assert len(values) == (3 if arm in ARMS else 1)
                        scored = mean_metrics(values)
                        rep_lookup[(graph_arm, graph, mode, arm, rep)] = scored
                        per_rep.append(dict(graph_arm=graph_arm, graph=graph, panel=panel,
                                            mode=mode, arm=arm, repetition=rep, **scored))
                    summary.append(dict(graph_arm=graph_arm, graph=graph, panel=panel,
                                        mode=mode, arm=arm, **mean_metrics(selected)))
            for arm in ARMS + BASELINES:
                averaged.append(dict(graph_arm=graph_arm, graph=graph, panel=panel, arm=arm,
                    **mean_metrics(rep_lookup[(graph_arm, graph, mode, arm, rep)]
                                   for mode in ("95", "99", "995") for rep in config["real_repetitions"])))
            for comparator in ("B", "C_global"):
                for mode in config["real_modes"] + ["mean95_99_995"]:
                    modes = ("95", "99", "995") if mode == "mean95_99_995" else (mode,)
                    item = dict(graph_arm=graph_arm, graph=graph, panel=panel, mode=mode,
                                contrast="C-minus-" + comparator)
                    for metric in METRICS:
                        item[metric] = descriptive([
                            np.mean([rep_lookup[(graph_arm, graph, m, "C", rep)][metric] -
                                     rep_lookup[(graph_arm, graph, m, comparator, rep)][metric] for m in modes])
                            for rep in config["real_repetitions"]])
                    paired.append(item)
            for arm in ARMS + BASELINES:
                for left, right in (("95", "none"), ("99", "none"), ("995", "none"), ("95", "shuffle95")):
                    item = dict(graph_arm=graph_arm, graph=graph, panel=panel, arm=arm,
                                contrast=left + "-minus-" + right)
                    for metric in METRICS:
                        item[metric] = descriptive([
                            rep_lookup[(graph_arm, graph, left, arm, rep)][metric] -
                            rep_lookup[(graph_arm, graph, right, arm, rep)][metric]
                            for rep in config["real_repetitions"]])
                    feature.append(item)
    seed_scores = []
    for graph_arm in GRAPH_ARMS:
        for graph in config["real_datasets"]:
            for arm in ARMS:
                for seed in SEEDS:
                    selected = [r for r in current if r["graph_arm"] == graph_arm and r["graph"] == graph
                                and r["arm"] == arm and r["seed"] == seed and r["mode"] in {"95", "99", "995"}]
                    seed_scores.append(dict(graph_arm=graph_arm, graph=graph, arm=arm, seed=seed, **mean_metrics(selected)))
    return dict(per_rate=summary, mean_over_missing=averaged, per_repetition=per_rep,
                by_training_seed=seed_scores, scope="all5 reused development graphs; no independent real CI"), {
                    "contrasts": paired, "feature_effect": feature,
                    "statistics": "Fixed3seed metrics averaged within each graph/mask; three descriptive repetitions. No node bootstrap, real population significance, winner selection or probability ensemble."}


def budget_results(auditor):
    reports = auditor.training
    out = dict(model_fits=len(reports), updates=sum(r["report"]["updates"] for r in reports),
        task_presentations=sum(r["report"]["exposures"] for r in reports), target_updates=0,
        shared_inherited=auditor.config["budget"],
        fit_seconds_sum=float(sum(r["report"]["seconds"] for r in reports)),
        timing_scope="Sum of fit worker wall times, not experiment elapsed wall time or speedup benchmark.",
        fits=[dict(arm=r["arm"], seed=r["seed"], updates=r["report"]["updates"],
                   exposures=r["report"]["exposures"], parameters=r["report"]["trainable_parameters"],
                   seconds=r["report"]["seconds"], graph_encodes=r["report"]["graph_encodes_in_this_arm"],
                   graph_encode_seconds=r["report"]["graph_encoding_seconds_in_this_arm"],
                   checkpoint_path=r["checkpoint"], checkpoint_sha256=r["report"]["checkpoint_sha256"])
              for r in reports], evaluation=[])
    for stage in ("synthetic", "real"):
        for graph_arm in GRAPH_ARMS:
            for arm in ARMS:
                selected = [r for r in auditor.records if r["stage"] == stage and r["graph_arm"] == graph_arm
                            and r["arm"] == arm]
                times = [r["seconds"] for r in selected]
                peaks = [r["peak_mib"] for r in selected if r["peak_mib"] is not None]
                out["evaluation"].append(dict(stage=stage, graph_arm=graph_arm, arm=arm, cases=len(selected),
                    seconds_sum=float(np.sum(times)), median_seconds=float(np.median(times)),
                    max_peak_mib=None if not peaks else float(max(peaks)),
                    includes_cpu_probability_return=True, common_graph_encoding_excluded=True,
                    cold_mixed_shapes_not_competitive_latency_benchmark=True))
    failures = []
    for path in sorted((ROOT / "results").rglob("failure.json")):
        value = read_json(path)
        failures.append(dict(path=str(path), sha256=sha(path),
                             successful_updates=value.get("successful_updates"), exposures=value.get("exposures"),
                             exception=value.get("exception")))
    out["discovered_failed_attempt_logs"] = failures
    out["preflight_optimizer_updates"] = read_json(ROOT / "results/checks.json").get("optimizer_updates", 0)
    return out


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def report_text(summary, paired, audit, budget):
    syn, real = summary["synthetic"], summary["real"]
    lines = ["# FP01 audited saved-probability results", "", "This report is generated from saved probabilities; final scientific interpretation is written separately.", "",
        "Synthetic final is a new-seed, same-family stratified bank. Real graphs are all reused development evidence. Seeds are averaged as metrics, never as probability ensembles. Full and edge are separate panels; no per-graph winner is selected.", "",
        f"Audit: {audit['status']}; twelve fits / {budget['updates']} new successful updates / {budget['task_presentations']} task presentations / target updates0. Synthetic {audit['synthetic_predictions']}, real {audit['real_predictions']}, reused baselines {audit['baseline_predictions']}.", "",
        "## Prespecified synthetic primary", "",
        "Full/projected, missing50/75/90/95%, source>=2, common own-feature-missing queries. Average rates and three fixed training seeds within each base graph. Resample four base graphs within each of twelve N×K strata,5000times. Two accuracy contrasts have97.5%two-sided percentile intervals each (Bonferroni familywise95%).", "",
        table(["Arm", "ACC%", "BAcc%", "macroF1", "NLL", "Brier"],
              [[arm, f"{100*v['accuracy']:.3f}", f"{100*v['balanced_accuracy']:.3f}",
                f"{v['macro_f1']:.4f}", f"{v['nll']:.5f}", f"{v['brier']:.5f}"]
               for arm, v in syn["primary_panels"]["full"].items()]), "",
        table(["Contrast", "ACC difference%p [97.5%CI]", "NLL difference"],
              [[name, f"{100*v['accuracy']['mean']:+.3f} [{100*v['accuracy']['ci'][0]:+.3f},{100*v['accuracy']['ci'][1]:+.3f}]",
                f"{v['nll']['mean']:+.5f}"] for name, v in paired["synthetic"]["full"].items()
               if name.startswith("C-minus-")]), "",
        f"Prespecified adaptive-success gate: **{syn['gate']['adaptive_success']}**. Both comparisons require positive interval lower bound and mean>=1%p, C−none positive, and C NLL at most .01nats worse than either comparator. No subgroup rescues this gate.", "",
        table(["Arm", "Primary observed−none accuracy%p"],
              [[arm, f"{100*v['accuracy']['mean']:+.3f}"] for arm, v in
               paired["synthetic"]["full"]["observed-minus-none"].items()]), "",
        "The shared none prediction makes C−B identical to a difference in feature utility. These are one comparison, not independent confirmations. Source0/1 are structural boundary checks. All raw/projected/orthogonal views, rates and class counts are preserved in summary.json; orthogonal first-cell75/95 diagnostics do not replace the primary.", "",
        "## Real development: separate full and edge panels", "",
        "The following rows average95/99/99.5% equally. Cora/CiteSeer are full graphs; PubMed/Coauthors are fixed2000-node induced subgraphs. Public pipelines retain historical input interfaces; LP and public predictions are reused once, then displayed beside both graph panels. There is no independent-real confidence interval.", ""]
    for graph_arm in GRAPH_ARMS:
        lines += ["### " + graph_arm, "", table(["Graph", "Scope", "Arm", "ACC%", "BAcc%", "macroF1", "NLL", "Brier"],
            [[r["graph"], r["panel"], r["arm"], f"{100*r['accuracy']:.3f}", f"{100*r['balanced_accuracy']:.3f}",
              f"{r['macro_f1']:.4f}", f"{r['nll']:.4f}", f"{r['brier']:.4f}"]
             for r in real["mean_over_missing"] if r["graph_arm"] == graph_arm]), "",
            table(["Graph", "Arm", "Feature contrast", "ACC difference%p [3reps]", "NLL difference"],
                [[r["graph"], r["arm"], r["contrast"],
                  f"{100*r['accuracy']['mean']:+.3f} [" + ",".join(f"{100*v:+.2f}" for v in r["accuracy"]["values"]) + "]",
                  f"{r['nll']['mean']:+.5f}"] for r in paired["real"]["feature_effect"]
                 if r["graph_arm"] == graph_arm and r["arm"] in ARMS and r["contrast"] in {"95-minus-none", "95-minus-shuffle95"}]), ""]
    lines += ["## Audit and cost scope", "",
        f"Common none maximum probability error: {audit['none_cross_arm_max_probability_error']:.3g}; source0/1 versus matched none maximum: {audit['singleton_max_probability_error']:.3g}.", "",
        "All checkpoint/source/bank/input/query/target hashes and finite probability arrays were checked and independently rescored. Fitted feature state digests and frozen graph digests were checked; no forward pass was needed. Balanced accuracy averages recalls of classes present in each truth vector; macroF1 uses the historical sklearn label-union convention. Query/source class counts are descriptive scoring metadata.", "",
        "Inference timing is cold, mixed-shape worker timing including CPU probability return and excluding shared graph encoding; it is not a competitive latency benchmark. Fit seconds are a worker-time sum, not elapsed wall time. Inherited graph training, preflight and failed attempts are listed separately in budget.json.", "",
        "Detailed per-rate/class/view, seed/repetition, paired feature effects and all predictions are indexed in summary.json, paired.json and prediction_index.jsonl. artifact_manifest.json points to immutable heavy artifacts by path/hash. No calibrated-Bayes, published-AMP-reproduction, unseen-family or independent-real-generalization claim follows from this report.", ""]
    return "\n".join(lines)


def write_new(path, value, text=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        stream.write(value if text else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        stream.write("\n")


def run(output_dir):
    require_ready()  # No output files on incomplete evaluation.
    output_dir = Path(output_dir).resolve()
    assert output_dir.is_relative_to(ROOT), "Analysis output must stay in FP01"
    targets = ("summary.json", "paired.json", "audit.json", "budget.json", "REPORT_KO.md",
               "prediction_index.jsonl", "artifact_manifest.json")
    for name in targets:
        if (output_dir / name).exists():
            raise FileExistsError(f"Preserve existing analysis: {output_dir / name}; choose a new --output-dir")
    auditor = Auditor()
    auditor.audit_training()
    auditor.audit_predictions()
    auditor.audit_baselines()
    synthetic, syn_paired = synthetic_results(auditor.records, auditor.config, auditor.strata)
    real, real_paired = real_results(auditor.records, auditor.config)
    summary = dict(cycle="FP01", synthetic=synthetic, real=real,
                   real_source_coverage=auditor.real_source_coverage(),
                   synthetic_source_counts=[{k: r.get(k) for k in ("base_id", "view", "mode", "nodes", "classes", "sources", "queries", "source_class_counts", "query_class_counts")}
                                            for r in auditor.eval_rows["synthetic"]])
    paired = dict(synthetic=syn_paired, real=real_paired)
    budget = budget_results(auditor)
    audit = dict(status="PASS", cycle="FP01", model_fits=12, successful_updates=3072,
                 task_presentations=6144, target_gradient_updates=0,
                 synthetic_bank_rows=len(auditor.eval_rows["synthetic"]), real_bank_rows=75,
                 synthetic_predictions=sum(r["stage"] == "synthetic" for r in auditor.records),
                 real_predictions=sum(r["stage"] == "real" and r["graph_arm"] in GRAPH_ARMS for r in auditor.records),
                 baseline_predictions=sum(r["graph_arm"] == "reference" for r in auditor.records),
                 input_bank_target_prediction_hashes=True, complete_prediction_grid=True,
                 independently_rescored_saved_probabilities=True, checkpoint_state_digests=True,
                 frozen_graph_digests=True, source_lock_unchanged=True,
                 own_feature_missing_queries=True, context_source_disjoint=True,
                 common_graph_only_role_weights=True, probability_ensemble=False,
                 none_cross_arm_max_probability_error=auditor.none_max_error,
                 singleton_max_probability_error=auditor.singleton_max_error,
                 synthetic_bootstrap_unit="base graph within fixed N,K strata; fixed training seeds averaged",
                 real_independent_CI=False, scientific_gate=synthetic["gate"]["adaptive_success"])
    assert audit["synthetic_predictions"] == 16704 and audit["real_predictions"] == 1800
    # Re-check locked files at completion rather than relying on the hash cache.
    assert auditor.G.source_lock(create=False) == auditor.lock
    for path, digest in auditor.checked_hashes.items():
        assert sha(path) == digest, f"Artifact mutated during analysis: {path}"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "prediction_index.jsonl"
    with index_path.open("x", encoding="utf8", newline="\n") as stream:
        for row in auditor.records:
            stream.write(canonical(row) + "\n")
    for name, value in (("summary.json", summary), ("paired.json", paired), ("audit.json", audit), ("budget.json", budget)):
        write_new(output_dir / name, value)
    write_new(output_dir / "REPORT_KO.md", report_text(summary, paired, audit, budget), text=True)
    artifact_manifest = dict(cycle="FP01", source_lock_path=str(ROOT / "results/lock.json"),
        source_lock_sha256=auditor.provenance["lock_sha256"],
        summary_artifacts=[dict(path=str(output_dir / name), sha256=sha(output_dir / name))
                           for name in targets if name not in {"artifact_manifest.json", "prediction_index.jsonl"}],
        heavy_prediction_index=dict(path=str(index_path), sha256=sha(index_path), records=len(auditor.records),
                                    track_in_git=False),
        bank_manifests=[dict(path=str(ROOT / "bank" / f"{split}_manifest.json"),
                             sha256=sha(ROOT / "bank" / f"{split}_manifest.json")) for split in ("train", "dev", "final")],
        artifacts=auditor.artifacts,
        storage_policy="Track code+small summary artifacts; banks/checkpoints/predictions remain shared on disk, identified by immutable path/hash.")
    write_new(output_dir / "artifact_manifest.json", artifact_manifest)
    print(json.dumps(dict(audit=audit, output_dir=str(output_dir),
                          artifact_manifest_sha256=sha(output_dir / "artifact_manifest.json")), indent=2), flush=True)


def selfcheck():
    y = np.array([0, 1], np.int64)
    p = np.array([[.8, .2], [.25, .75]])
    value = metrics(p, y)
    assert value["accuracy"] == value["balanced_accuracy"] == value["macro_f1"] == 1
    assert abs(value["nll"] + np.log(.6) / 2) < 1e-14
    assert abs(value["brier"] - .1025) < 1e-14
    strata = {(n, k): list(range(cell * 4, cell * 4 + 4))
              for cell, (n, k) in enumerate((n, k) for n in (64, 96, 128) for k in (2, 3, 5, 7))}
    result = stratified_bootstrap({i: .02 for i in range(48)}, strata, resamples=50)
    np.testing.assert_allclose(result["ci"], [.02, .02], atol=1e-15)
    assert result["base_graphs"] == 48
    # Empty/missing artifacts must raise before an output or model is created.
    state = readiness()
    if not state["ready"]:
        try:
            require_ready()
        except FileNotFoundError as exc:
            assert "No analysis outputs were written" in str(exc)
        else:
            raise AssertionError("Missing-artifact guard failed")
    print(json.dumps(dict(status="PASS", metrics=True, stratified_constant_contrast=True,
                          missing_artifact_guard=True, no_results_analyzed=True,
                          current_ready=state["ready"], missing_files=len(state["missing"])), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "readiness", "selfcheck"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    if args.action == "run":
        run(args.output_dir)
    elif args.action == "readiness":
        print(json.dumps(readiness(), indent=2))
    else:
        selfcheck()
