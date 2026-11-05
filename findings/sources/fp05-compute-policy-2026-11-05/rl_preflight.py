"""Discarded development updates and score-free native-size resource smoke."""
import sklearn
import gc
import json
import time
import torch
import numpy as np
from rl_util import ROOT, cfg, setup, dump, sha, digest
import rl_data as D
from rl_run import model_class, set_arm, make_optimizer, fit_group, predict


def main():
    setup(); output = ROOT / 'results/preflight.json'
    assert not output.exists(), 'Preserve previous preflight; explicit repair required'
    trials = []; starts = {}; samples = D.rows('dev')
    for arm in cfg()['arms']:
        model = set_arm(model_class()(999), arm)
        opt = make_optimizer(model); scaler = torch.amp.GradScaler('cuda', init_scale=64.)
        starts[arm] = digest(model)
        frozen = model.frozen_digest(); torch.cuda.reset_peak_memory_stats()
        for update in range(cfg()['dev']['smoke_updates']):
            inp, y = D.read(samples[update]); inp = dict(inp, forward_seed=91053000 + update)
            result = fit_group(model, arm, inp, y, 91054000 + update, opt, scaler, update)
            trials.append(dict(arm=arm, update=update + 1, input_sha256=samples[update]['sha256'], **result))
        assert model.frozen_digest() == frozen
        trials[-1]['peak_mib'] = torch.cuda.max_memory_allocated() / 2**20
        trials[-1]['model_parameters_changed'] = starts[arm] != digest(model)
        assert trials[-1]['model_parameters_changed']
        del model, opt, scaler; gc.collect(); torch.cuda.empty_cache()
    model = model_class()(999).eval(); resources = []
    rows = [r for r in D.rows('real') if r['graph'] == 'CiteSeer' and r['setting'] == '95'][:1]
    assert rows
    inp, _ = D.read(rows[0])
    for arm in ['fixed', 'uniform', 'group_pg']:
        torch.cuda.synchronize(); baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        begun = time.perf_counter()
        p, timing = predict(model, inp, arm)
        torch.cuda.synchronize()
        resources.append(dict(arm=arm, nodes=len(inp['values']), features=inp['values'].shape[1],
            queries=len(inp['query_ids']), actual_forwards=timing['forwards'],
            seconds=time.perf_counter() - begun, total_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            additional_peak_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20,
            finite_probability=bool(np.isfinite(p).all()), scores_inspected=False))
    assert len(set(starts.values())) == 1, 'Same public parameters/seed must initialize all4arms identically'
    # Exclude policy/trainability flags: digest includes parameter values, not flags.
    result = dict(status='PASS', discarded_task_groups=len(trials),
                  discarded_logical_trace_slots=len(trials) * cfg()['train']['traces'],
                  initial_digests=starts, discarded_trials=trials, resources=resources,
                  final_bank_used=False, real_test_scores_inspected=False,
                  notes='Partial internal timers only; cold samples, no latency superiority claim')
    dump(output, result)
    print(json.dumps(dict(status=result['status'], discarded_groups=len(trials), resources=resources)), flush=True)


if __name__ == '__main__':
    main()
