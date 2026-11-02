import sys
sys.dont_write_bytecode = True
import json, hashlib
from pathlib import Path
import run as r  # Reuse the working sklearn -> torch -> CUDA -> chronos import order.
import numpy as np
import torch
from chronos import Chronos2Pipeline
from analyze import loss

R = Path(__file__).resolve().parent

def main():
    print("audit: imports ready", flush=True)
    B = dict(np.load(R / "bank.npz"))
    # All cached metrics recomputed from forecasts, not manually copied numbers.
    cached = dict(np.load(R / "results/task_losses.npz"))
    forecasts = dict(np.load(R / "results/classical.npz"))
    for m in ["chronos2", "old_null_lora", "correlation_chronos2"]:
        forecasts[m] = np.load(R / f"results/{m}.npz")["predictions"]
    metric_err = {m: float(np.max(np.abs(loss(f)-cached[m]))) for m,f in forecasts.items()}
    assert max(metric_err.values()) == 0
    assert all(np.isfinite(f).all() for f in forecasts.values())
    print("audit: metrics verified", flush=True)
    # Counterfactual future-value mutation: no predictor construction sees future truth.
    t = int(B["origins"][0])
    altered = {k:v.copy() for k,v in B.items()}
    altered["x"][:, :, t+1:] = 9000
    altered["y"][:, t+1:] = -9000
    for i in range(len(B["y"])):
        ids = r.view_ids(B, i, "uvN8")
        assert r.screen_ids(B,i,ids) == r.screen_ids(altered,i,ids)
        for inter in [True,False]:
            assert np.array_equal(r.features(B,i,ids,[t],inter),r.features(altered,i,ids,[t],inter))
        assert np.array_equal(B["y"][i,None,t-r.L+1:t+1],altered["y"][i,None,t-r.L+1:t+1])
        assert np.array_equal(B["x"][i,ids,t-r.L+1:t+1],altered["x"][i,ids,t-r.L+1:t+1])
    # Independent prefix replay on two task families. No fitting/update is run.
    torch.set_num_threads(4)
    print("audit: causal construction verified; loading checkpoint", flush=True)
    torch.zeros(1, device="cuda")
    pipe = Chronos2Pipeline.from_pretrained(r.P["chronos_model"], device_map="cuda", dtype=torch.float32, local_files_only=True)
    pipe.model.eval()
    replay = {}
    for m in ["chronos2", "correlation_chronos2", "old_null_lora"]:
        if m == "old_null_lora":
            for block in pipe.model.encoder.block:
                a = block.layer[1].self_attention
                a.q, a.k = r.LoRA(a.q), r.LoRA(a.k)
            pipe.model.load_state_dict(torch.load(r.P["old_lora"],map_location="cuda",weights_only=True),strict=False)
        errors = []
        for i in [0,64]:
            for vi,view in enumerate(r.VIEWS):
                ids = r.view_ids(B,i,view)
                if m == "correlation_chronos2": ids = r.screen_ids(B,i,ids)
                groups = [np.concatenate([B["y"][i,None,o-r.L+1:o+1], B["x"][i,ids,o-r.L+1:o+1]]) for o in B["origins"][:3]]
                with torch.inference_mode(): p = r.group_forward(pipe.model,groups).cpu().numpy()
                errors.append(float(np.max(np.abs(p-forecasts[m][i,vi,:3]))))
        replay[m] = max(errors)
    assert max(replay.values()) < 2e-5, replay
    # Population null matching does not imply exact finite-sample matching.
    moments = {}
    for family in ["product","additive_control"]:
        a,b = [],[]
        for i in np.where(B["family"]==family)[0]:
            pair = B["pairs"][i]
            a.extend(B["x"][i,pair,:r.L].flatten().tolist())
            b.extend(B["x"][i,[j for j in range(10) if j not in pair],:r.L].flatten().tolist())
        moments[family] = {"useful_mean":float(np.mean(a)),"nuisance_mean":float(np.mean(b)),
                           "useful_second_moment":float(np.mean(np.array(a)**2)),"nuisance_second_moment":float(np.mean(np.array(b)**2))}
    source = Path("C:/Users/seongjin/Desktop/code/SD-SDGL/alibaba_ms2021")
    assets = [Path(r.P["chronos_model"])/"model.safetensors",Path(r.P["chronos_model"])/"config.json",Path(r.P["old_lora"]),
              source/"ni_lora.py",source/"ni_lora_extra.csv",Path(R/"protocol.json"),Path(R/"bank.npz")]
    out = {"status":"PASS", "metric_recompute_max_abs":metric_err,"replay_max_abs":replay,
        "replayed_forecasts_per_model":36,"future_mutation_construction_check":"PASS for96tasks, screening/raw/interaction/context",
        "marginal_moments":moments,"input_audit":json.loads((R/"results/input_audit.json").read_text()),
        "asset_hashes":{str(p):r.sha(p) for p in assets},"no_tsfm_updates":True,
        "total_tsfm_forecasts":int(3*len(B["y"])*len(r.VIEWS)*len(B["origins"])),
        "classical_fits":int(len(B["y"])*len(r.VIEWS)*3),
        "limitations":"Frozen-stage only; origin-dependent history legitimately includes earlier observed test outcomes; classical coefficients fixed at first origin; LoRA is one historical Alibaba checkpoint, not3 newly trained seeds."}
    (R/"results/audit.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print("AUDIT PASS",replay,"metrics",metric_err)

if __name__ == "__main__": main()
