"""Locked frozen TSFM synergy/nuisance comparison. No TSFM parameter updates."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
torch.zeros(1, device="cuda")  # Initialize CUDA before transformers, as in prior local runs.
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / "protocol.json").read_text(encoding="utf8"))
VIEWS = P["views"]
LAGS = np.array(P["lags"])
L = P["context_length"]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")

def make_bank():
    dest = ROOT / "bank.npz"
    if dest.exists(): return
    rng = np.random.default_rng(P["seed"])
    n = sum(P["tasks"].values())
    origins = L - 1 + np.arange(P["origins"]) * P["origin_stride"]
    T = int(origins[-1] + 2)
    x = rng.choice([-1., 1.], size=(n, P["max_covariates"], T)).astype("float32")
    # Channel identity, sign and lag are randomized independently of model input ordering.
    pairs = np.stack([rng.choice(P["max_covariates"], 2, replace=False) for _ in range(n)])
    lag = rng.choice(LAGS, size=(n, 2))
    signs = rng.choice([-1., 1.], size=(n, 2)).astype("float32")
    families = np.array(["product"] * P["tasks"]["product"] + ["additive_control"] * P["tasks"]["additive_control"])
    y = rng.normal(0, P["noise_sd"], size=(n, T)).astype("float32")
    mean = np.zeros_like(y)
    orders = []
    for i in range(n):
        t = np.arange(4, T - 1)
        a = signs[i, 0] * x[i, pairs[i, 0], t - lag[i, 0]]
        b = signs[i, 1] * x[i, pairs[i, 1], t - lag[i, 1]]
        signal = a * b if families[i] == "product" else (a + b) / np.sqrt(2)
        mean[i, t + 1] = signal
        y[i, t + 1] += signal
        orders.append(rng.permutation(P["max_covariates"]))
    np.savez_compressed(dest, x=x, y=y, mean=mean, pairs=pairs, lag=lag, signs=signs,
                        family=families, order=np.stack(orders), origins=origins)
    write_json(ROOT / "bank_manifest.json", {"bank_sha256": sha(dest), "protocol_sha256": sha(ROOT / "protocol.json"),
        "n_tasks": n, "n_product": P["tasks"]["product"], "origins": origins.tolist(),
        "causal_definition": "y[t+1] uses x[:,t-lag] only; forecast context ends at t, no future covariates supplied",
        "marginal_matching": "Every useful/nuisance channel iid balanced Rademacher; exact population marginals match, empirical counts fluctuate",
        "split_unit": "Entire independently generated base task; all views/origins remain paired"})

def view_ids(B, i, view):
    pair = B["pairs"][i]
    order = B["order"][i]
    nuisance = [int(j) for j in order if j not in pair]
    selected = {"history": [], "u": [int(pair[0])], "v": [int(pair[1])], "uv": pair.tolist(),
                "uvN3": pair.tolist() + nuisance[:3], "uvN8": list(range(P["max_covariates"]))}[view]
    return [int(j) for j in order if j in selected]

def screen_ids(B, i, ids):
    if len(ids) <= 2: return ids
    # Correlations use only labels already observed at the first forecast origin.
    ts = np.arange(4, L - 1)
    y = B["y"][i, ts + 1]
    scores = []
    for j in ids:
        a = B["x"][i, j, ts[:, None] - LAGS]
        a = a - a.mean(0)
        yc = y - y.mean()
        scores.append(np.max(np.abs(a.T @ yc) / np.maximum(np.linalg.norm(a, axis=0) * np.linalg.norm(yc), 1e-12)))
    best = set(np.argsort(scores, kind="stable")[-2:])
    return [j for k, j in enumerate(ids) if k in best]

def features(B, i, ids, ts, interactions):
    ts = np.asarray(ts)
    raw = B["x"][i][np.array(ids, dtype=int)[:, None, None], ts[None, :, None] - LAGS[None, None, :]]
    # covariate x sample x lag -> sample x flattened columns
    out = [B["y"][i, ts[:, None] - LAGS]]
    if len(ids): out.append(raw.transpose(1, 0, 2).reshape(len(ts), -1))
    if interactions:
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                out.append((raw[a, :, :, None] * raw[b, :, None, :]).reshape(len(ts), -1))
    return np.concatenate(out, 1).astype("float64")

def classic(B, quantiles):
    n, no = len(B["y"]), len(B["origins"])
    outputs = {m: np.zeros((n, len(VIEWS), no, len(quantiles)), "float32")
               for m in ["raw_ridge", "interaction_ridge", "interaction_lasso"]}
    meta, start = [], time.perf_counter()
    ts = np.arange(4, L - 1)  # y[t+1] <= y[511] is known at origin511.
    for i in range(n):
        for vi, view in enumerate(VIEWS):
            ids = view_ids(B, i, view)
            yy = B["y"][i, ts + 1].astype("float64")
            for m in outputs:
                xx = features(B, i, ids, ts, m != "raw_ridge")
                test = features(B, i, ids, B["origins"], m != "raw_ridge")
                sc = StandardScaler().fit(xx[:320])
                tr, va = sc.transform(xx[:320]), sc.transform(xx[320:416])
                make = (lambda a: Lasso(alpha=a, max_iter=4000, tol=1e-5)) if m.endswith("lasso") else (lambda a: Ridge(alpha=a))
                alphas = P["lasso_alphas"] if m.endswith("lasso") else P["ridge_alphas"]
                vals = []
                for alpha in alphas:
                    fit = make(alpha).fit(tr, yy[:320])
                    vals.append(float(np.mean((fit.predict(va) - yy[320:416]) ** 2)))
                alpha = alphas[int(np.argmin(vals))]
                sc = StandardScaler().fit(xx[:416])
                fit = make(alpha).fit(sc.transform(xx[:416]), yy[:416])
                residual = yy[416:] - fit.predict(sc.transform(xx[416:]))
                forecasts = fit.predict(sc.transform(test))[:, None] + np.quantile(residual, quantiles)[None]
                outputs[m][i, vi] = forecasts
                meta.append({"task": i, "view": view, "model": m, "n_features": xx.shape[1], "alpha": alpha})
        if (i + 1) % 8 == 0: print(f"classical tasks {i+1}/{n}, {time.perf_counter()-start:.1f}s", flush=True)
    np.savez_compressed(ROOT / "results/classical.npz", **outputs)
    write_json(ROOT / "results/classical_fit.json", {"fits": meta, "elapsed_seconds": time.perf_counter() - start})

class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.zeros(4, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, 4, device=base.weight.device))
    def forward(self, x):
        return self.base(x) + 2 * (x @ self.A.T) @ self.B.T

def group_forward(model, groups):
    sizes = [len(g) for g in groups]
    ctx = torch.as_tensor(np.concatenate(groups), device="cuda")
    gids = torch.as_tensor(np.repeat(np.arange(len(groups)), sizes), device="cuda")
    first = np.cumsum([0] + sizes[:-1])
    return model(context=ctx, group_ids=gids, num_output_patches=1).quantile_preds[first, :, 0]

def chronos(B, pipe, name):
    model = pipe.model.eval()
    n, no = len(B["y"]), len(B["origins"])
    out = np.zeros((n, len(VIEWS), no, len(pipe.quantiles)), "float32")
    selected, group_rows, seconds = [], 0, 0.
    for i in range(n):
        groups, locs = [], []
        for vi, view in enumerate(VIEWS):
            ids = view_ids(B, i, view)
            if name == "correlation_chronos2": ids = screen_ids(B, i, ids)
            selected.append({"task": i, "view": view, "ids": ids})
            for oi, t in enumerate(B["origins"]):
                g = np.concatenate([B["y"][i, None, t-L+1:t+1], B["x"][i, ids, t-L+1:t+1]], axis=0)
                groups.append(g)
                locs.append((vi, oi))
        # Limit rows, not number of groups, for comparable memory use.
        j = 0
        with torch.inference_mode():
            while j < len(groups):
                k, rows = j, 0
                while k < len(groups) and rows + len(groups[k]) <= 96:
                    rows += len(groups[k]); k += 1
                torch.cuda.synchronize(); st = time.perf_counter()
                pr = group_forward(model, groups[j:k]).float().cpu().numpy()
                torch.cuda.synchronize(); seconds += time.perf_counter() - st
                for p, (vi, oi) in zip(pr, locs[j:k]): out[i, vi, oi] = p
                group_rows += rows; j = k
        if (i + 1) % 8 == 0:
            np.savez_compressed(ROOT / f"results/{name}_partial.npz", predictions=out, completed_tasks=i+1)
            print(f"{name} tasks {i+1}/{n}, forward seconds={seconds:.1f}", flush=True)
    np.savez_compressed(ROOT / f"results/{name}.npz", predictions=out)
    write_json(ROOT / f"results/{name}_meta.json", {"selected_ids": selected, "forward_seconds": seconds,
        "group_rows": group_rows, "forecasts": n * len(VIEWS) * no, "parameters_updated": 0,
        "quantiles": pipe.quantiles, "context_length": L, "future_covariates": None})

def smoke(pipe, B):
    groups = [np.stack([B["y"][0, :L], *B["x"][0, :3, :L]])]
    with torch.inference_mode():
        a = group_forward(pipe.model, groups).cpu().numpy()
        b = pipe.predict(groups, prediction_length=1, context_length=L, batch_size=96)[0][0, :, 0].numpy()[None]
        rev = [groups[0][[0, 3, 2, 1]]]
        c = group_forward(pipe.model, rev).cpu().numpy()
        two = group_forward(pipe.model, groups + [B["y"][1, None, :L]]).cpu().numpy()[0]
    check = {"direct_vs_pipeline_max_abs": float(np.max(np.abs(a-b))),
             "covariate_permutation_max_abs": float(np.max(np.abs(a-c))),
             "separate_task_group_isolation_max_abs": float(np.max(np.abs(a[0]-two)))}
    assert max(check.values()) < 2e-4, check
    # Simultaneous train/test alignment and useful-pair feature membership.
    for i in range(len(B["y"])):
        ts = np.array(B["origins"])
        a = B["signs"][i, 0] * B["x"][i, B["pairs"][i, 0], ts-B["lag"][i, 0]]
        b = B["signs"][i, 1] * B["x"][i, B["pairs"][i, 1], ts-B["lag"][i, 1]]
        expected = a*b if B["family"][i] == "product" else (a+b)/np.sqrt(2)
        assert np.max(np.abs(expected-B["mean"][i, ts+1])) < 1e-6
        ff = features(B, i, view_ids(B, i, "uvN8"), ts, True)
        if B["family"][i] == "product": assert np.min(np.max(np.abs(ff - expected[:, None]), axis=0)) < 1e-6 or np.min(np.max(np.abs(ff + expected[:, None]), axis=0)) < 1e-6
    check.update({"causal_signal_alignment": True, "all_pair_true_term_included_without_selection": True,
                  "classical_training_max_target_time": L-1, "first_evaluation_target_time": L})
    write_json(ROOT / "results/input_audit.json", check)
    print("input audit", check, flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["bank", "classical", "chronos", "lora", "screen", "audit"])
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    make_bank()
    B = dict(np.load(ROOT / "bank.npz"))
    if args.stage == "bank": return
    torch.set_num_threads(4)
    if args.stage == "classical":
        with threadpool_limits(limits=4): classic(B, json.loads((Path(P["chronos_model"]) / "config.json").read_text())["chronos_config"]["quantiles"])
        return
    pipe = Chronos2Pipeline.from_pretrained(P["chronos_model"], device_map="cuda", dtype=torch.float32, local_files_only=True)
    pipe.model.eval()
    if args.stage == "audit": smoke(pipe, B); return
    name = {"chronos": "chronos2", "lora": "old_null_lora", "screen": "correlation_chronos2"}[args.stage]
    if args.stage == "lora":
        for block in pipe.model.encoder.block:
            att = block.layer[1].self_attention
            att.q, att.k = LoRA(att.q), LoRA(att.k)
        state = torch.load(P["old_lora"], map_location="cuda", weights_only=True)
        missing, unexpected = pipe.model.load_state_dict(state, strict=False)
        expected = {k for k in pipe.model.state_dict() if k.endswith(".A") or k.endswith(".B")}
        assert set(state) == expected and not unexpected
        write_json(ROOT / "results/lora_load_audit.json", {"sha256": sha(P["old_lora"]), "loaded_keys": len(state),
            "adapter_parameters": sum(v.numel() for v in state.values()), "exact_key_match": True,
            "source_training": "Previous Alibaba null-invariance + teacher real-context preservation; not this synthetic task"})
    chronos(B, pipe, name)

if __name__ == "__main__": main()
