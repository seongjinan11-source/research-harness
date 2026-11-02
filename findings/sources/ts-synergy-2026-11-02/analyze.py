import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
from scipy.stats import norm

R = Path(__file__).resolve().parent
P = json.loads((R / "protocol.json").read_text())
B = dict(np.load(R / "bank.npz"))
Q = np.array(json.loads((Path(P["chronos_model"]) / "config.json").read_text())["chronos_config"]["quantiles"])
Y = B["y"][:, B["origins"] + 1]
V = P["views"]

def loss(pred):
    diff = Y[:, None, :, None] - pred
    return 2 * np.maximum(Q * diff, (Q-1) * diff).mean(axis=(2, 3))

def ci(x):
    x = np.asarray(x)
    rng = np.random.default_rng(14011)
    boot = x[rng.integers(len(x), size=(10000, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(), "n_tasks": len(x)}

def main():
    forecasts = dict(np.load(R / "results/classical.npz"))
    for m in ["chronos2", "old_null_lora", "correlation_chronos2"]:
        forecasts[m] = np.load(R / f"results/{m}.npz")["predictions"]
    oracle = B["mean"][:, B["origins"]+1, None] + P["noise_sd"] * norm.ppf(Q)[None, None]
    oloss = loss(np.repeat(oracle[:, None], len(V), axis=1))
    losses = {m: loss(f) for m, f in forecasts.items()}
    summary = {}
    for fam in ["product", "additive_control"]:
        ix = B["family"] == fam
        rows = {}
        for m, f in forecasts.items():
            ll = losses[m][ix]
            rows[m] = {"loss": {v: float(ll[:, vi].mean()) for vi, v in enumerate(V)},
                "G_joint": ci(np.minimum(ll[:, 1], ll[:, 2])-ll[:, 3]),
                "P_null3": ci(ll[:, 4]-ll[:, 3]), "P_null8": ci(ll[:, 5]-ll[:, 3]),
                "history_to_clean_gain": ci(ll[:, 0]-ll[:, 3]),
                "nuisance_forecast_L1": float(np.mean(np.abs(f[ix, 5]-f[ix, 3]))),
                "median_mse": {v: float(np.mean((f[ix, vi, :, np.where(Q==.5)[0][0]]-Y[ix])**2)) for vi, v in enumerate(V)}}
        summary[fam] = {"models": rows, "privileged_oracle_loss": float(oloss[ix, 3].mean())}
    prod = B["family"] == "product"
    comp = {}
    for a, b in [("old_null_lora", "chronos2"), ("correlation_chronos2", "chronos2"),
                 ("interaction_ridge", "chronos2"), ("interaction_lasso", "chronos2"),
                 ("interaction_lasso", "old_null_lora"), ("interaction_lasso", "raw_ridge")]:
        comp[f"{a} - {b}"] = {v: ci(losses[a][prod, vi]-losses[b][prod, vi]) for vi, v in enumerate(V)}
    c = summary["product"]["models"]["chronos2"]
    gj, pn = c["G_joint"], c["P_null8"]
    gate_j = gj["mean"] > .01 and gj["ci95"][0] > 0
    gate_n = pn["mean"] > .01 and pn["ci95"][0] > 0
    base = summary["product"]["models"]["raw_ridge"]["loss"]["history"]
    oracle_l = summary["product"]["privileged_oracle_loss"]
    gap = base - oracle_l
    recoveries = {m: (base-summary["product"]["models"][m]["loss"]["uvN8"])/gap for m in ["interaction_ridge", "interaction_lasso"]}
    floor = any(recoveries[m] >= .8 and comp[f"{m} - chronos2"]["uvN8"]["ci95"][1] < 0 for m in recoveries)
    # Lag-0 and lagged primary-bank strata are descriptive, not new tests/gates.
    strata = {}
    for label, sel in {"both_lag0": prod & (B["lag"] == 0).all(1), "any_lagged": prod & (B["lag"] > 0).any(1)}.items():
        if sel.any(): strata[label] = {m: {v: float(losses[m][sel, vi].mean()) for vi,v in enumerate(V)} for m in losses}
    out = {"metric": "mean twice pinball over21quantiles", "summary": summary, "product_comparisons": comp,
        "interaction_gap_recovery": recoveries, "gates": {"joint": bool(gate_j), "nuisance": bool(gate_n),
            "interaction_floor_solves_toy": bool(floor), "run_2x2": bool(gate_j and gate_n and not floor)},
        "lag_strata_descriptive": strata,
        "scope": "Frozen synthetic comparison; no new TSFM training. 2x2 is untested if gate false; old LoRA is transfer from Alibaba, not trained on this bank."}
    (R / "results/statistics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
    np.savez_compressed(R / "results/task_losses.npz", **losses, privileged_oracle=oloss)
    print("PRODUCT", json.dumps(summary["product"], indent=2))
    print("GATES", out["gates"], "RECOVERY", recoveries)

if __name__ == "__main__": main()
