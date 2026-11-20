"""Dev-only pre-check (DIFFERENTIATION_KO.md §4.2; design use only, no confirm data).

Question: on the existing dev pool, how often do the pair decisions of
  R2  fixed robust z-score   mean_s (g_s - med_s)/MAD_s   (med/MAD from dev heuristics in bands 0-3)
  R2g MO-GRPO-style          per-stratum z-score with the std of the group's children (G = children)
  R3  per-stratum win rate   mean_s [P_i(c<p) + .5 P(=)] - .5
  R0  mixed mean gap
disagree? If R2 ~ R3 nearly always, comparing prescriptions adds little beyond MO-GRPO's Pre-Norm
finding. Pairs whose ref behaviour is identical are dropped. Lower gap = better.

Run from co_rl/:  python 2026-11-20_strata_mixed_reward/precheck_r2_r3.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2026-11-19_reward_noise_diagnostic"))
import numpy as np  # noqa: E402

from analyze import load_matrix  # noqa: E402

D = Path(__file__).resolve().parent.parent / "2026-11-19_reward_noise_diagnostic" / "results"
OUT = Path(__file__).resolve().parent / "results"
PER = 50  # instances per stratum (plan §3/§4)


def strata(pool):
    res = D / pool / "dev"
    heur = json.loads((res / "heuristics.json").read_text())
    hids = [h["hid"] for h in heur]
    if pool.startswith("obp"):
        G, _ = load_matrix(res / "ref_in.jsonl", hids, "gaps")
        S = {s: G[:, i * PER:(i + 1) * PER] for i, s in enumerate(["5k_c100", "5k_c500", "1k_c100", "1k_c500"])}
    else:
        S = {}
        for n, f in ((50, "ref_in"), (100, "shift_n100"), (200, "shift_n200")):
            S[f"N{n}"] = load_matrix(res / f"{f}.jsonl", hids, "gaps")[0][:, :PER]
    return heur, S


def run(pool):
    heur, S = strata(pool)
    names = list(S)
    valid = np.all([np.isfinite(S[s]).all(1) for s in names], axis=0)
    idx = {h["hid"]: i for i, h in enumerate(heur)}
    band = np.array([h["band"] for h in heur])
    g = {s: S[s].mean(1) for s in names}
    ref03 = valid & (band <= 3)
    med = {s: np.median(g[s][ref03]) for s in names}
    mad = {s: np.median(np.abs(g[s][ref03] - med[s])) * 1.4826 for s in names}
    groups, parent = defaultdict(list), {}
    for h in heur:
        (groups[h["group"]].append(idx[h["hid"]]) if h["role"] == "child" else parent.__setitem__(h["group"], idx[h["hid"]]))
    rows = []
    for gname, cs in groups.items():
        p = parent[gname]
        cs = [c for c in cs if valid[c]]
        if not valid[p] or not cs:
            continue
        gsd = {s: np.std([g[s][c] for c in cs], ddof=1) if len(cs) > 1 else np.nan for s in names}
        for c in cs:
            if all(np.allclose(S[s][c], S[s][p]) for s in names):
                continue
            d = {s: g[s][c] - g[s][p] for s in names}
            r0 = -np.sign(np.mean([d[s] for s in names]))
            r2 = -np.sign(sum(d[s] / mad[s] for s in names if mad[s] > 0))
            r2g_terms = [d[s] / gsd[s] for s in names if np.isfinite(gsd[s]) and gsd[s] > 0]
            r2g = -np.sign(sum(r2g_terms)) if r2g_terms else 0.0
            wr = np.mean([np.mean(S[s][c] < S[s][p]) + .5 * np.mean(S[s][c] == S[s][p]) for s in names]) - .5
            r3 = np.sign(wr)
            signs = {np.sign(d[s]) for s in names} - {0}
            rows.append(dict(group=gname, band=int(band[p]), conflict=len(signs) > 1, G=len(cs),
                             R0=r0, R2=r2, R2g=r2g, R3=r3))
    return rows, dict(med=med, mad=mad)


def rate(rows, a, b):
    return float(np.mean([r[a] != r[b] for r in rows])) if rows else float("nan")


def boot(rows, a, b, n=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r[a] != r[b])
    keys = list(by)
    rng = np.random.default_rng(seed)
    bs = [np.mean(np.concatenate([by[keys[i]] for i in rng.integers(len(keys), size=len(keys))])) for _ in range(n)]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    OUT.mkdir(exist_ok=True)
    rep = {}
    for pool in ("obp_v2", "tsp_v3"):
        rows, stats = run(pool)
        sub = {"all": rows, "bands0_3": [r for r in rows if r["band"] <= 3],
               "conflict_bands0_3": [r for r in rows if r["band"] <= 3 and r["conflict"]]}
        rep[pool] = {"stats": {k: {s: float(v) for s, v in d.items()} for k, d in stats.items()},
                     "median_children_per_group": float(np.median([r["G"] for r in rows])) if rows else None}
        for nm, rr in sub.items():
            e = {"pairs": len(rr), "groups": len({r["group"] for r in rr}),
                 "R3_ties": int(sum(r["R3"] == 0 for r in rr))}
            for a, b in (("R2", "R3"), ("R2g", "R3"), ("R0", "R3"), ("R0", "R2"), ("R2", "R2g")):
                e[f"{a}_vs_{b}"] = {"disagree": rate(rr, a, b), "ci": boot(rr, a, b) if rr else None}
            rep[pool][nm] = e
    (OUT / "precheck_dev.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
