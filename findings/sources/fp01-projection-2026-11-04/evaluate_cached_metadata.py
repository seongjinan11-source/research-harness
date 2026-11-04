"""Operational FP01 wrapper: cache descriptive Git metadata, no predictor edits.

Reviewed after source lock. Existing complete outputs retain strict runner
identity/hash checks. Only interrupted unpaired files may be quarantined.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(11, 12, 13), required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fp01_unchanged_evaluation", ROOT / "experiment.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.G.setup()
    original_git = runner._git_state
    snapshot = original_git()
    lock = runner._lock_record()
    packet = dict(seed=args.seed, started=time.time(), wrapper_sha256=sha(__file__),
                  runner_sha256=sha(ROOT / "experiment.py"),
                  lock_sha256=sha(ROOT / "results/lock.json"),
                  git_start=snapshot, git_scope="evaluation-start descriptive metadata only",
                  patch_scope="runner._git_state only; no arithmetic, identity, hash or input patch",
                  gpu_concurrency=1, resume=True, orphans=[], replay=None)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_s{args.seed}"
    runroot = ROOT / "runtime" / "metadata_cache" / run_id
    runroot.mkdir(parents=True, exist_ok=False)
    # Root stops the old worker and waits for its owning shell before invoking.
    # Never fabricate an absent sidecar or weaken strict completed-pair checks.
    for arm in runner.ARMS:
        predroot = ROOT / "results" / f"{arm}_s{args.seed}" / "predictions"
        if not predroot.exists():
            continue
        stems = {p.with_suffix("") for p in predroot.rglob("*") if p.suffix in (".npz", ".json")}
        for stem in sorted(stems):
            npz, sidecar = stem.with_suffix(".npz"), stem.with_suffix(".json")
            if npz.is_file() and sidecar.is_file():
                continue
            for source in (npz, sidecar):
                if not source.is_file():
                    continue
                source = source.resolve()
                assert source.is_relative_to((ROOT / "results").resolve())
                dest = (runroot / "orphans" / source.relative_to(ROOT)).resolve()
                assert dest.is_relative_to(runroot.resolve()) and not dest.exists()
                record = dict(original=str(source), preserved=str(dest), sha256=sha(source),
                              reason="Unpaired artifact after explicit operational worker interruption")
                dest.parent.mkdir(parents=True, exist_ok=True)
                source.rename(dest)
                assert sha(dest) == record["sha256"]
                packet["orphans"].append(record)
    write_new(runroot / "started.json", packet)
    runner._git_state = lambda: dict(snapshot)
    try:
        if args.replay:
            import numpy as np
            import torch
            # An existing nondegenerate row is replayed solely for execution
            # invariance, without selecting/changing any scientific settings.
            candidates = sorted((ROOT / "results" / f"C_s{args.seed}" / "predictions" / "synthetic" / "full").glob("*.json"))
            chosen = next(p for p in candidates if json.loads(p.read_text(encoding="utf8"))["sources"] >= 2)
            meta = json.loads(chosen.read_text(encoding="utf8"))
            inp, query, truth = runner.B.read(meta["row"])
            assert runner._arrays_digest(inp) == meta["identity"]["input_sha256"]
            model, report = runner.load_model("C", args.seed)
            core = runner.G.GraphCore(args.seed)
            graph = core.encode(inp, graph_arm="full")
            with torch.no_grad():
                probability = model(runner.G.tensor_input(inp), graph)[query].cpu().numpy()
            saved_path = chosen.with_suffix(".npz")
            with np.load(saved_path, allow_pickle=False) as saved:
                assert np.array_equal(saved["query"], query) and np.array_equal(saved["y"], truth)
                assert np.array_equal(saved["context"], inp["context_ids"])
                assert np.array_equal(saved["yc"], inp["labels_context"])
                error = float(np.max(np.abs(probability - saved["p"])))
                assert error <= 2e-6
            assert core.check() == meta["identity"]["graph_digest"]
            assert runner.G.digest(model) == report["final_sha256"] == meta["identity"]["model_sha256"]
            packet["replay"] = dict(metadata=str(chosen), prediction_sha256=sha(saved_path), max_probability_error=error)
            del model, core, graph, probability
            runner._release("cuda")
        runner.evaluate(args.seed, resume=True)
        assert runner._lock_record() == lock
        packet["status"] = "COMPLETE"
    except BaseException:
        packet["status"] = "ERROR"
        packet["traceback"] = traceback.format_exc()
        raise
    finally:
        packet["finished"] = time.time()
        packet["git_end"] = original_git()
        write_new(runroot / "finished.json", packet)
        print("OPERATIONAL_WRAPPER", str(runroot / "finished.json"), flush=True)


if __name__ == "__main__":
    main()
