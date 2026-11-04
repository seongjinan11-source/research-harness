"""FP01 immutable visible-input banks; no imputation or model dependency.

``prepare()`` replays the 512 historical raw tasks and creates dev only.
``finalize()`` is gated by the root source lock and PASS numerical checks.
Generator-only diagnostics live in separate audit files and never enter read().
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
INPUT_KEYS = ("x", "observed", "src", "dst", "context_ids",
              "labels_context", "classes", "forward_seed")
MODES = {0.5: "50", 0.75: "75", 0.9: "90", 0.95: "95",
         0.99: "99", 0.995: "995"}


def cfg():
    return json.loads((ROOT / "protocol.json").read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _npz_new(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def observed_scale(full, observed):
    """Exactly the historical observed_scale, without importing any imputer."""
    safe = np.where(observed, full, 0.).astype(np.float64)
    count = observed.sum(0)
    rms = np.sqrt(np.divide((safe * safe).sum(0), count,
                           out=np.ones(full.shape[1]), where=count > 0))
    rms = np.where(rms >= 1e-8, rms, 1.)
    return np.where(observed, np.clip(safe / rms, -10, 10), 0).astype(np.float32)


def _base(seed, index, *, split, nodes=None, classes=None):
    """Replay the old RNG verbatim for train, override only new-bank strata."""
    c = cfg()
    rng = np.random.default_rng(seed)
    n = int(rng.choice(c["nodes"]))
    k = int(rng.choice(c["classes"]))
    n = n if nodes is None else int(nodes)
    k = k if classes is None else int(classes)
    f = c["features"]
    y = rng.permutation(np.arange(n) % k).astype("int64")
    role = int(rng.integers(3))
    if split != "train":
        role = index % 3
    strength = rng.uniform(.35, .8)
    rolevalue = [-1 / (k - 1), 0, 1][role]
    degree = int(rng.choice([4, 8, 12]))
    if split != "train":
        degree = [4, 8, 12][(index + index // 3) % 3]
    density = degree / (n - 1)
    prob = density * np.where(
        y[:, None] == y[None], 1 + (k - 1) * strength * rolevalue,
        1 - strength * rolevalue)
    edges = np.triu(rng.random((n, n)) < np.minimum(prob, .98), 1)
    edges |= edges.T
    src, dst = np.nonzero(edges)
    z = rng.integers(k, size=f)
    mean = np.where(y[:, None] == z, .8, -.8 / (k - 1))
    noise = .6 * rng.normal(size=(n, f))
    full = (mean + noise).astype("float32")
    per = int(rng.choice(c["context_per_class_train"]))
    context = np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:per]
        for label in range(k)])
    context = rng.permutation(context).astype("int64")
    candidates = np.setdiff1d(np.arange(n), context)
    reserved = None
    if split != "train":
        reserved = int(rng.choice(candidates))
        candidates = candidates[candidates != reserved]
    source_order = rng.permutation(candidates)
    raw_means = np.where(np.arange(k)[:, None] == z, .8, -.8 / (k - 1))
    offset = c["stream_offsets"][split]
    projection_seed = c["projection_seed_start"] + offset + index
    rrng = np.random.default_rng(projection_seed)
    # Keep the historical float32 Rademacher/sqrt-dimension expression.
    projection = (rrng.choice([-1., 1.], size=(f, f)).astype("float32") /
                  np.sqrt(f))
    return dict(seed=int(seed), base_id=index, split=split, nodes=n, classes=k,
                features=f, y=y, full=full, raw_means=raw_means, noise=noise,
                src=src.astype("int64"), dst=dst.astype("int64"),
                context=context, source_order=source_order, reserved=reserved,
                role=role, strength=float(strength), degree_parameter=degree,
                density_parameter=float(density), projection=projection,
                projection_seed=projection_seed)


def _rotation(base):
    c = cfg()
    seed = c["orthogonal_seed_start"] + c["stream_offsets"][base["split"]] + base["base_id"]
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(
        size=(base["features"], base["features"])))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype("float64"), seed


def _source_rows(base, rate):
    obsrows = np.zeros(base["nodes"], bool)
    count = min(round(base["nodes"] * (1 - rate)), len(base["source_order"]))
    obsrows[base["source_order"][:count]] = True
    return obsrows


def _visible(base, values, obsrows):
    observed = np.broadcast_to(obsrows[:, None], values.shape).copy()
    ctx = base["context"]
    return dict(x=observed_scale(values, observed), observed=observed,
                src=base["src"], dst=base["dst"], context_ids=ctx,
                labels_context=base["y"][ctx], classes=np.int64(base["classes"]),
                forward_seed=np.int64(base["seed"]))


def _view_audit(base, transform):
    f = base["features"]
    matrix = np.eye(f) if transform is None else np.asarray(transform, np.float64)
    means = base["raw_means"] @ matrix
    noise = base["noise"] @ matrix
    covariance = .36 * matrix.T @ matrix
    offdiag = covariance - np.diag(np.diag(covariance))
    distances = np.sqrt(((means[:, None] - means[None]) ** 2).sum(-1))
    upper = distances[np.triu_indices(base["classes"], 1)]
    singular = np.linalg.svd(matrix, compute_uv=False)
    return dict(rank=int(np.linalg.matrix_rank(matrix)),
                singular_min=float(singular.min()), singular_max=float(singular.max()),
                covariance_offdiag_rms=float(np.sqrt(np.mean(offdiag ** 2))),
                mean_distance_min=float(upper.min()), mean_distance_mean=float(upper.mean()),
                class_mean_sum_maxabs=float(np.abs(means.sum(0)).max()),
                mean_squared=float(np.mean(means ** 2)),
                noise_empirical_second_moment=float(np.mean(noise ** 2))), {
                    "class_centroids": means, "noise_covariance": covariance,
                    "noise_empirical_covariance": np.cov(noise, rowvar=False)}


def _write_generator_audit(base, transforms):
    scalars, arrays = {}, {}
    for view, transform in transforms.items():
        scalars[view], detail = _view_audit(base, transform)
        arrays.update({view + "_" + key: value for key, value in detail.items()})
    arrays["projection"] = base["projection"]
    if "orthogonal" in transforms:
        arrays["orthogonal"] = transforms["orthogonal"]
    dest = ROOT / "bank" / base["split"] / "generator_audit" / f"{base['base_id']:04d}.npz"
    _npz_new(dest, **arrays)
    return dict(base_id=base["base_id"], seed=base["seed"],
                nodes=base["nodes"], classes=base["classes"],
                role=base["role"], signal=base["strength"],
                degree_parameter=base["degree_parameter"],
                density_parameter=base["density_parameter"],
                projection_seed=base["projection_seed"],
                raw_values_sha256=_array_sha(base["full"]),
                labels_sha256=_array_sha(base["y"]),
                raw_class_counts=np.bincount(base["y"], minlength=base["classes"]).tolist(),
                diagnostic_only=True, views=scalars,
                path=dest.relative_to(ROOT).as_posix(), sha256=sha(dest))


def _save_episode(base, view, mode, rate, values, obsrows, query, *, audit):
    inp = _visible(base, values, obsrows)
    truth = base["y"][query]
    assert not inp["observed"][base["context"]].any()
    assert not np.intersect1d(query, base["context"]).size
    if base["split"] != "train":
        assert not inp["observed"][query].any()
    assert np.isfinite(inp["x"]).all()
    assert not inp["x"][~inp["observed"]].any()
    dest = ROOT / "bank" / base["split"] / f"{base['base_id']:04d}_{view}_{mode}.npz"
    _npz_new(dest, **inp, query=query.astype("int64"), labels_query=truth.astype("int64"))
    # Statistics below are audit/scoring-only and never returned in inp.
    safe = np.where(inp["observed"], values, 0.).astype(np.float64)
    count = inp["observed"].sum(0)
    rms = np.sqrt(np.divide((safe * safe).sum(0), count,
                           out=np.ones(values.shape[1]), where=count > 0))
    rms = np.where(rms >= 1e-8, rms, 1.)
    total = int(inp["observed"].sum())
    clipped = int(((np.abs(safe / rms) > 10) & inp["observed"]).sum())
    source_count = int(obsrows.sum())
    row = dict(split=base["split"], base_id=base["base_id"],
               graph=f"FP01_{base['split']}_{base['base_id']:04d}",
               view=view, mode=mode, rate=float(rate), seed=base["seed"],
               nodes=base["nodes"], classes=base["classes"], features=base["features"],
               sources=source_count, context=len(base["context"]), queries=len(query),
               role=base["role"], signal=base["strength"], degree_parameter=base["degree_parameter"],
               path=dest.relative_to(ROOT).as_posix(), sha256=sha(dest),
               primary=bool(base["split"] != "train" and view == "projected" and
                            mode in {"50", "75", "90", "95"} and source_count >= 2),
               query_features_missing=bool(not inp["observed"][query].any()),
               source_class_counts=np.bincount(base["y"][obsrows], minlength=base["classes"]).tolist(),
               query_class_counts=np.bincount(truth, minlength=base["classes"]).tolist(),
               observed_rms=rms.tolist(), clipping_fraction=clipped / max(total, 1),
               generator_audit=audit["path"], generator_audit_sha256=audit["sha256"])
    return row


def _old_train_row(index):
    path = WORK / "factor_real_graph_pilot" / "bank" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    matches = [r for r in data["episodes"] if r["split"] == "train" and r["graph"] == index]
    assert len(matches) == 1
    return matches[0]


def _historical_parity(index, newrow):
    old = _old_train_row(index)
    oldpath = WORK / "factor_real_graph_pilot" / old["path"]
    assert sha(oldpath) == old["sha256"], f"Historical bank digest mismatch {index}"
    with np.load(oldpath, allow_pickle=False) as expected, np.load(ROOT / newrow["path"], allow_pickle=False) as given:
        for key in INPUT_KEYS + ("query", "labels_query"):
            assert np.array_equal(given[key], expected[key]), f"Replay mismatch task {index}, {key}"
        missing = ~given["observed"][given["query"]].any(-1)
        assert missing.any(), f"No own-feature-missing train query {index}"
    return dict(base_id=index, original_path=oldpath.relative_to(WORK).as_posix(),
                original_sha256=old["sha256"], exact_visible_input_and_truth_parity=True)


def _generate(split):
    c = cfg()
    manifest = ROOT / "bank" / f"{split}_manifest.json"
    assert not manifest.exists(), f"Never overwrite {manifest}"
    total = c["train_tasks"] if split == "train" else c[split + "_graphs"]
    episodes, audits, parity = [], [], []
    for index in range(total):
        options = {}
        if split != "train":
            repetition_count = 1 if split == "dev" else 4
            cell = index // repetition_count
            options = dict(nodes=c["nodes"][cell // len(c["final_classes"])],
                           classes=c["final_classes"][cell % len(c["final_classes"])])
        base = _base(c[split + "_start"] + index, index, split=split, **options)
        transforms = dict(raw=None, projected=base["projection"])
        orthogonal = split != "train" and (split == "dev" or index % 4 == 0)
        if orthogonal:
            transforms["orthogonal"], base["orthogonal_seed"] = _rotation(base)
        audit = _write_generator_audit(base, transforms)
        if orthogonal:
            audit["orthogonal_seed"] = base["orthogonal_seed"]
        audits.append(audit)
        if split == "train":
            query = np.setdiff1d(np.arange(base["nodes"]), base["context"])
            rates = [float(c["missing_train"][index % len(c["missing_train"])])]
        else:
            largest = _source_rows(base, .5)
            query = np.setdiff1d(np.arange(base["nodes"]),
                                 np.r_[base["context"], np.flatnonzero(largest)]).astype("int64")
            assert len(query) and base["reserved"] in query
            rates = list(c["missing_final"])
        for view, transform in transforms.items():
            values = base["full"] if transform is None else base["full"] @ transform
            viewrates = c["orthogonal_missing"] if view == "orthogonal" else rates
            for rate in viewrates:
                row = _save_episode(base, view, MODES[float(rate)], rate, values,
                                    _source_rows(base, rate), query, audit=audit)
                episodes.append(row)
                if split == "train" and view == "raw":
                    parity.append(_historical_parity(index, row))
            if split != "train" and view == "projected":
                episodes.append(_save_episode(base, view, "none", 1., values,
                                              _source_rows(base, 1.), query, audit=audit))
                obsrows = _source_rows(base, .95)
                ids = np.flatnonzero(obsrows)
                rng = np.random.default_rng(c["shuffle_seed_start"] + c["stream_offsets"][split] + index)
                shuffled = values.copy()
                shuffled[ids] = values[rng.permutation(ids)]
                episodes.append(_save_episode(base, view, "shuffle95", .95, shuffled,
                                              obsrows, query, audit=audit))
        if (index + 1) % 128 == 0 or index + 1 == total:
            print("PREPARE", split, index + 1, "episodes", len(episodes), flush=True)
    expected = c["bank_layout"][split + "_rows"]
    assert len(episodes) == expected, (split, len(episodes), expected)
    result = dict(split=split, base_graphs=total, episodes=episodes,
                  generator_audits=audits, historical_replay=parity,
                  protocol_sha256=sha(ROOT / "protocol.json"),
                  generator_sha256=sha(Path(__file__)),
                  inputs=list(INPUT_KEYS), no_imputation=True,
                  final_query_policy="Common Q excludes context and 50%-source set" if split != "train" else
                  "Historical all noncontext CE queries; observed sources included")
    _json_new(manifest, result)
    return result


def _validate_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    assert data["protocol_sha256"] == sha(ROOT / "protocol.json"), "Protocol changed after bank creation"
    for row in data["episodes"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    for row in data["generator_audits"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    return data


def prepare():
    """Create train/dev only. Repeated calls verify completed immutable banks."""
    data = {}
    for split in ("train", "dev"):
        path = ROOT / "bank" / f"{split}_manifest.json"
        data[split] = _validate_manifest(path) if path.exists() else _generate(split)
    combined = ROOT / "bank" / "manifest.json"
    state = dict(episodes=data["train"]["episodes"] + data["dev"]["episodes"],
                 manifests={split: {"path": f"bank/{split}_manifest.json",
                                    "sha256": sha(ROOT / "bank" / f"{split}_manifest.json")}
                            for split in ("train", "dev")},
                 protocol_sha256=sha(ROOT / "protocol.json"), final_generated=False)
    if combined.exists():
        assert json.loads(combined.read_text(encoding="utf-8-sig")) == state
    else:
        _json_new(combined, state)
    return state


def finalize():
    """Root-only post-lock final generation. Never called from prepare()."""
    lock = ROOT / "results" / "lock.json"
    checkpath = ROOT / "results" / "checks.json"
    assert lock.exists(), "Final bank requires root source lock"
    assert checkpath.exists(), "Final bank requires numerical checks"
    checks = json.loads(checkpath.read_text(encoding="utf-8-sig"))
    assert checks.get("status") == "PASS", "Final bank requires checks PASS"
    spec = importlib.util.spec_from_file_location("fp01_bank_lock_verifier", ROOT / "common.py")
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    common.source_lock(create=False)
    for split in ("train", "dev"):
        _validate_manifest(ROOT / "bank" / f"{split}_manifest.json")
    path = ROOT / "bank" / "final_manifest.json"
    if path.exists():
        return _validate_manifest(path)
    return _generate("final")


def rows(split):
    if split not in {"train", "dev", "final"}:
        raise ValueError(f"Unknown bank split {split!r}")
    path = ROOT / "bank" / f"{split}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))["episodes"]


def train_rows(view):
    if view not in {"raw", "projected"}:
        raise ValueError(f"Unknown training view {view!r}")
    selected = sorted((row for row in rows("train") if row["view"] == view),
                      key=lambda row: row["base_id"])
    assert [row["base_id"] for row in selected] == list(range(cfg()["train_tasks"]))
    return selected


def read(row):
    """Deterministic visible whitelist; hidden generator diagnostics stay out."""
    path = (ROOT / row["path"]).resolve()
    assert path.is_relative_to((ROOT / "bank").resolve()), "Bank path leaves FP01 bank"
    assert sha(path) == row["sha256"], row["path"]
    with np.load(path, allow_pickle=False) as data:
        inp = {key: data[key].copy() for key in INPUT_KEYS}
        query, truth = data["query"].copy(), data["labels_query"].copy()
    assert not inp["observed"][inp["context_ids"]].any()
    if row["split"] != "train":
        assert not inp["observed"][query].any()
    return inp, query, truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize", "verify"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "finalize":
        finalize()
    else:
        for split in ("train", "dev", "final"):
            path = ROOT / "bank" / f"{split}_manifest.json"
            if path.exists():
                data = _validate_manifest(path)
                print("VERIFIED", split, data["base_graphs"], len(data["episodes"]), flush=True)
