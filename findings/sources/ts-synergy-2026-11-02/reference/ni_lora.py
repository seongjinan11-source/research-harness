"""NI-LoRA kill test: null-invariance LoRA on Chronos-2 cross-series (group) attention, Alibaba development services only.

Backbone frozen; LoRA (r = 4, alpha = 8) on W_Q and W_K of the GroupSelfAttention of all 12 encoder blocks.
Teacher f_0 = same model with LoRA disabled (stop-gradient). No future Y in the objective.
Loss (all weights 1, D = MSE over all 21 quantiles x horizon, target row):
  L_inv   D(f_phi(X,R,N_1..K), f_0(X,R))          null type ~ U{white, self-shuffle, self-phase, dupX, dupR1}, K ~ U{1,3}
  L_keep  D(f_phi(X,R), f_0(X,R))
  L_keep4 D(f_phi(X,R,R4), f_0(X,R,R4))
  L_ctx   D(f_phi(X,R,R4) - f_phi(X,R), f_0(X,R,R4) - f_0(X,R))
Data: the 50 pilot services split by service, 30 train / 20 validation (seed 0). Training and early-stopping windows
use the selection period only (context ends at t in 240..360, step 5). Final evaluation on the 20 validation services at the
12 Web test origins (dev units; held-out services are never touched). AdamW lr 1e-4, wd 1e-4, eval every 25 steps,
patience 75, max 300 steps.
Outputs: ni_lora_curve.csv, ni_lora_eval.csv, ni_lora_summary.csv, ni_lora_state.pt
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_web as rw  # noqa: E402

L, H, T_TR = rw.L, rw.H, rw.T_TR
RANK, ALPHA, LR, WD, MAXSTEP, EVAL_EVERY, PATIENCE, BATCH = 4, 8, 1e-4, 1e-4, 300, 25, 75, 24
NULLS = ["white", "shuffle", "phase", "dupX", "dupR1"]
QMED = None


class LoRA(nn.Module):
    on = True

    def __init__(self, base, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.s = alpha / r

    def forward(self, x):
        y = self.base(x)
        return y + self.s * (x @ self.A.T) @ self.B.T if LoRA.on else y


def load_series():
    P = pd.read_pickle(os.path.join(HERE, "panel.pkl"))
    Man = pd.read_csv(os.path.join(HERE, "manifest.csv"))
    Man = Man[Man.pilot].reset_index(drop=True)
    arr = P.values.astype(np.float64)
    arr = (arr - arr[:, :T_TR].mean(1, keepdims=True)) / arr[:, :T_TR].std(1, keepdims=True)
    row = {f"{a}|{b}": i for i, (a, b) in enumerate(P.index)}
    ser = []
    for _, m in Man.iterrows():
        d = {k: arr[row[m[k]]] for k in ["R1", "R2", "R3", "R4"]}
        d["x"] = arr[row[f"{m.target}|providerRPC_RT"]]
        ser.append({k: v.astype(np.float32) for k, v in d.items()})
    return ser


def windows(ser, idx, origins, with_y=False):
    W = {k: np.stack([ser[i][k][t - L:t] for i in idx for t in origins]) for k in ["x", "R1", "R2", "R3", "R4"]}
    if with_y:
        W["y"] = np.stack([ser[i]["x"][t:t + H] for i in idx for t in origins])
    return W


def null_streams(x, r1, kind, K, rng):
    if kind == "white":
        return x.mean() + x.std() * rng.standard_normal((K, len(x))).astype(np.float32)
    if kind == "shuffle":
        return np.stack([rng.permutation(x) for _ in range(K)])
    if kind == "phase":
        return np.stack([rw.phase_self(x, rng) for _ in range(K)]).astype(np.float32)
    return np.repeat((x if kind == "dupX" else r1)[None], K, 0)


def forward(model, groups):
    """groups: list of (n_i, L) arrays, target row first -> (len(groups), 21, H) quantile forecasts of the target rows."""
    ctx = torch.tensor(np.concatenate(groups), device="cuda")
    gid = torch.tensor(np.repeat(np.arange(len(groups)), [len(g) for g in groups]), device="cuda")
    first = torch.tensor(np.cumsum([0] + [len(g) for g in groups[:-1]]), device="cuda")
    q = model(context=ctx, group_ids=gid, num_output_patches=math.ceil(H / 16)).quantile_preds
    return q[first][..., :H]


def predict(model, groups, teacher=False, bs=256):
    LoRA.on = not teacher
    out = []
    with torch.no_grad():
        for c in range(0, len(groups), bs):
            out.append(forward(model, groups[c:c + bs]).float().cpu().numpy())
    LoRA.on = True
    return np.concatenate(out)


def batch_loss(model, W, ids, rng):
    base = [np.stack([W["x"][i], W["R1"][i], W["R2"][i], W["R3"][i]]) for i in ids]
    with4 = [np.concatenate([b, W["R4"][i][None]]) for b, i in zip(base, ids)]
    withN = [np.concatenate([b, null_streams(W["x"][i], W["R1"][i], NULLS[rng.integers(5)], int(rng.choice([1, 3])), rng)])
             for b, i in zip(base, ids)]
    with torch.no_grad():
        LoRA.on = False
        t = forward(model, base + with4)
        LoRA.on = True
    t3, t4 = t[:len(ids)], t[len(ids):]
    s = forward(model, base + with4 + withN)
    s3, s4, sN = s[:len(ids)], s[len(ids):2 * len(ids)], s[2 * len(ids):]
    D = lambda a, b: ((a - b) ** 2).mean()
    parts = dict(inv=D(sN, t3), keep=D(s3, t3), keep4=D(s4, t4), ctx=D(s4 - s3, t4 - t3))
    return sum(parts.values()), {k: float(v) for k, v in parts.items()}


def shift_stats(P, ref, y, W):
    """Algorithm 1: P (M, n, H) or (n, H) target medians vs reference ref (n, H); unit = service (W windows)."""
    P = P[None] if P.ndim == 2 else P
    D = P - ref[None]
    M = len(D)
    db = D.mean(0)
    b = (db ** 2).sum(-1) - (((D - db) ** 2).sum(-1).sum(0) / (M * (M - 1)) if M > 1 else 0)
    s = lambda v: v.reshape(-1, W).sum(1)
    return float(np.sqrt(max(np.median(s(b) / s(((y - ref) ** 2).sum(-1))), 0)))


def evaluate(model, ser, val_idx):
    E = windows(ser, val_idx, rw.ORIGINS, with_y=True)
    n, W = len(E["x"]), len(rw.ORIGINS)
    y = E["y"]
    base = [np.stack([E["x"][i], E["R1"][i], E["R2"][i], E["R3"][i]]) for i in range(n)]
    rows, res = [], {}
    for who, tch in [("teacher", True), ("student", False)]:
        med = lambda groups: predict(model, groups, teacher=tch)[:, QMED]
        F3 = med(base)
        F4 = med([np.concatenate([b, E["R4"][i][None]]) for i, b in enumerate(base)])
        FU = med([E["x"][i][None] for i in range(n)])
        r = dict(F3=F3, F4=F4, FU=FU)
        for kind in NULLS:
            for K in [1, 3, 7, 15]:
                rng = np.random.default_rng(1000 + K)
                Md = 1 if kind.startswith("dup") else 5
                P = np.stack([med([np.concatenate([b, null_streams(E["x"][i], E["R1"][i], kind, K, rng)]) for i, b in enumerate(base)])
                              for _ in range(Md)])
                rows.append(dict(model=who, null=kind, K=K, R=shift_stats(P, F3, y, W)))
        rng = np.random.default_rng(77)
        r["UN3"] = np.stack([med([np.concatenate([E["x"][i][None], null_streams(E["x"][i], E["R1"][i], "white", 3, rng)]) for i in range(n)])
                             for _ in range(5)])
        res[who] = r
    T = pd.DataFrame(rows)
    umse = lambda P: ((P - y) ** 2).mean(-1).reshape(-1, W).mean(1)
    summ = {}
    for who in ["teacher", "student"]:
        r = res[who]
        mU, m3, m4 = umse(r["FU"]), umse(r["F3"]), umse(r["F4"])
        mN = np.mean([umse(p) for p in r["UN3"]], 0)
        g_std, g_nccu = (mU - m3) / mU, (mN - m3) / mN
        summ[who] = dict(mse_R3=float(m3.mean()), mse_R4=float(m4.mean()), mse_U=float(mU.mean()),
                         R4_gain_pct=float(100 * np.median((m3 - m4) / m3)),
                         g_std_pct=float(100 * np.median(g_std)), g_nccu_pct=float(100 * np.median(g_nccu)),
                         nccu_gap_pct=float(100 * np.median(np.abs(g_std - g_nccu))),
                         ctx_move=float(np.linalg.norm(r["F3"] - r["FU"])))
    t, s = res["teacher"], res["student"]
    d0, d1 = t["F4"] - t["F3"], s["F4"] - s["F3"]
    cos = (d0 * d1).sum(-1) / (np.linalg.norm(d0, axis=-1) * np.linalg.norm(d1, axis=-1) + 1e-12)
    extra = dict(R4_effect_norm_retention=float(np.linalg.norm(d1) / np.linalg.norm(d0)),
                 R4_effect_cos_mean=float(cos.mean()), R4_effect_cos_median=float(np.median(cos)),
                 R4_forecast_preservation=float(1 - np.linalg.norm(s["F4"] - t["F4"]) / np.linalg.norm(d0)),
                 R3_student_vs_teacher_rel=float(np.linalg.norm(s["F3"] - t["F3"]) / np.linalg.norm(t["F3"] - t["FU"])),
                 ctx_move_ratio=summ["student"]["ctx_move"] / summ["teacher"]["ctx_move"],
                 mse_R3_change_pct=100 * (summ["student"]["mse_R3"] / summ["teacher"]["mse_R3"] - 1),
                 mse_R4_change_pct=100 * (summ["student"]["mse_R4"] / summ["teacher"]["mse_R4"] - 1))
    return T, summ, extra


def main():
    global QMED
    torch.manual_seed(0)
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
    QMED = pipe.quantiles.index(0.5)
    model = pipe.model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    ser = load_series()
    # sanity: direct forward == pipeline median (before LoRA)
    E = windows(ser, [0, 1], rw.ORIGINS[:3])
    g = [np.stack([E["x"][i], E["R1"][i], E["R2"][i], E["R3"][i]]) for i in range(len(E["x"]))]
    ref = np.stack([o[0, QMED].numpy() for o in pipe.predict(list(g), prediction_length=H, context_length=L)])
    with torch.no_grad():
        mine = forward(model, g)[:, QMED].cpu().numpy()
    err = np.abs(mine - ref).max() / np.abs(ref).max()
    print(f"direct-forward check: max rel diff {err:.2e}", flush=True)
    assert err < 1e-4
    for blk in model.encoder.block:
        att = blk.layer[1].self_attention
        att.q, att.k = LoRA(att.q), LoRA(att.k)
    lora = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad]
    n_tr, n_all = sum(p_.numel() for p_ in lora), sum(p_.numel() for p_ in model.parameters())
    print(f"trainable {n_tr} of {n_all} ({100 * n_tr / n_all:.3f}%)", flush=True)

    perm = np.random.default_rng(0).permutation(len(ser))
    tr_idx, va_idx = perm[:30], perm[30:]
    sel = np.arange(L, T_TR + 1, 5)
    Wtr, Wva = windows(ser, tr_idx, sel), windows(ser, va_idx, sel)
    opt = torch.optim.AdamW(lora, lr=LR, weight_decay=WD)
    rng = np.random.default_rng(1)
    curve, best, best_step, best_state = [], np.inf, 0, None

    def val_loss():
        vr = np.random.default_rng(2)
        tot, parts = 0.0, {}
        with torch.no_grad():
            for c in range(0, len(Wva["x"]), 64):
                ids = np.arange(c, min(c + 64, len(Wva["x"])))
                l_, p_ = batch_loss(model, Wva, ids, vr)
                tot += float(l_) * len(ids)
                for k, v in p_.items():
                    parts[k] = parts.get(k, 0) + v * len(ids)
        n = len(Wva["x"])
        return tot / n, {k: v / n for k, v in parts.items()}

    v0, vp0 = val_loss()
    curve.append(dict(step=0, split="val", loss=v0, **vp0))
    print(f"step 0 val {v0:.5f} {vp0}", flush=True)
    for step in range(1, MAXSTEP + 1):
        ids = rng.integers(0, len(Wtr["x"]), BATCH)
        loss, parts = batch_loss(model, Wtr, ids, rng)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(dict(step=step, split="train", loss=float(loss), **parts))
        if step % EVAL_EVERY == 0:
            v, vp = val_loss()
            curve.append(dict(step=step, split="val", loss=v, **vp))
            print(f"step {step} train {float(loss):.5f} val {v:.5f} {vp}", flush=True)
            if v < best:
                best, best_step = v, step
                best_state = {k: t_.detach().clone() for k, t_ in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
            elif step - best_step >= PATIENCE:
                print(f"early stop at {step}, best {best_step}", flush=True)
                break
    pd.DataFrame(curve).to_csv(os.path.join(HERE, "ni_lora_curve.csv"), index=False)
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        torch.save(best_state, os.path.join(HERE, "ni_lora_state.pt"))
    T, summ, extra = evaluate(model, ser, va_idx)
    T["suppression"] = np.nan
    piv = T.pivot_table(index=["null", "K"], columns="model", values="R").reset_index()
    piv["suppression"] = 1 - piv.student / piv.teacher
    piv.to_csv(os.path.join(HERE, "ni_lora_eval.csv"), index=False)
    S = pd.DataFrame([dict(who=k, **v) for k, v in summ.items()])
    S.to_csv(os.path.join(HERE, "ni_lora_summary.csv"), index=False)
    pd.Series(dict(trainable=n_tr, trainable_pct=100 * n_tr / n_all, best_step=best_step, **extra)).to_csv(
        os.path.join(HERE, "ni_lora_extra.csv"))
    pd.set_option("display.width", 200)
    print(piv.round(4).to_string(index=False))
    print(S.round(4).to_string(index=False))
    print(pd.Series(extra).round(4).to_string())


if __name__ == "__main__":
    main()
