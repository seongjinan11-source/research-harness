"""FP05 independent toy oracles and tiny released-model correctness checks.

Importing this module does not construct a model or run a GPU operation.  Only
the root orchestrator should execute the full checks.  --toy-only is CPU-only
and deliberately cannot produce the PASS receipt used by the scientific lock.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import sklearn  # noqa: F401; Windows environment requires this before torch.
import argparse
import copy
import io
import itertools
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import rl_util as B


TRACES = tuple(itertools.product((0, 1), repeat=3))
WHITELIST = {
    "values", "observed", "src", "dst", "context_ids", "context_labels",
    "query_ids", "classes", "forward_seed",
}


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def _toy_features(stage, prefix):
    return np.asarray([1., stage / 2, sum(prefix), prefix[-1] if prefix else 0,
                       prefix[0] if prefix else 0, int(tuple(prefix) == (1, 1))], np.float64)


def _toy_logp(theta, trace):
    total = theta.new_zeros(())
    for stage, bit in enumerate(trace):
        feature = torch.as_tensor(_toy_features(stage, trace[:stage]), dtype=theta.dtype)
        logit = (theta * feature).sum()
        total = total + F.logsigmoid(logit if bit else -logit)
    return total


def toy_checks():
    """Independent float64 enumeration, finite difference and score-function tests."""
    theta = torch.tensor([.2, -.4, .3, .1, -.2, .05], dtype=torch.float64, requires_grad=True)
    losses = torch.tensor([1.2, .9, 1.4, .6, .8, 1.1, .5, .4], dtype=torch.float64)
    costs = torch.tensor([sum(t) for t in TRACES], dtype=torch.float64)
    logps = torch.stack([_toy_logp(theta, t) for t in TRACES])
    prob = logps.exp()
    _require(abs(float(prob.sum().detach()) - 1) < 1e-12, "toy leaf probabilities do not sum to one")
    manual = []
    for trace in TRACES:
        p = 1.
        for stage, bit in enumerate(trace):
            z = float(theta.detach().numpy() @ _toy_features(stage, trace[:stage]))
            q = 1 / (1 + math.exp(-z))
            p *= q if bit else 1 - q
        manual.append(p)
    enumeration_error = float(np.max(np.abs(np.asarray(manual) - prob.detach().numpy())))
    _require(enumeration_error < 1e-14, "NumPy/Torch conditional probability mismatch")
    objective = (prob * (losses + .01 * costs)).sum()
    direct = torch.autograd.grad(objective, theta, retain_graph=True)[0]
    score_objective = (prob.detach() * (losses + .01 * costs) * logps).sum()
    score = torch.autograd.grad(score_objective, theta)[0]
    score_error = float((direct - score).abs().max())
    fd = []
    eps = 1e-6
    for i in range(theta.numel()):
        plus, minus = theta.detach().clone(), theta.detach().clone()
        plus[i] += eps
        minus[i] -= eps
        def value(t):
            return float(sum(_toy_logp(t, a).exp() * (losses[j] + .01 * costs[j])
                             for j, a in enumerate(TRACES)))
        fd.append((value(plus) - value(minus)) / (2 * eps))
    fd_error = float(np.max(np.abs(np.asarray(fd) - direct.numpy())))
    _require(score_error < 1e-12 and fd_error < 2e-9, "exact policy-gradient oracle failed")

    # This is a separate scalar Bernoulli example, not the production loss code.
    z = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
    reward_source = torch.tensor([-1., -1., 1., 1.], dtype=torch.float64, requires_grad=True)
    reward = reward_source.detach()
    advantage = (reward - reward.mean()) / (reward.std(unbiased=False) + 1e-6)
    logp = actions * F.logsigmoid(z) + (1 - actions) * F.logsigmoid(-z)
    actor = -(advantage.detach() * logp).mean()
    actor_grad = torch.autograd.grad(actor, z)[0]
    _require(float(actor_grad) < 0, "rewarded action has wrong policy-gradient direction")
    p_before = float(z.detach().sigmoid())
    p_after = float((z.detach() - .1 * actor_grad).sigmoid())
    _require(p_after > p_before and reward_source.grad is None, "reward detach/direction failed")
    equal = torch.ones(8, dtype=torch.float64)
    zero_adv = (equal - equal.mean()) / (equal.std(unbiased=False) + 1e-6)
    z_equal = torch.tensor(.31, dtype=torch.float64, requires_grad=True)
    equal_loss = -(zero_adv * F.logsigmoid(z_equal)).mean()
    equal_gradient = float(torch.autograd.grad(equal_loss, z_equal)[0])
    _require(float(equal_loss) == 0 and equal_gradient == 0, "equal-reward group is not exactly zero")
    return dict(status="PASS", dtype="float64", probability_sum=float(prob.sum().detach()),
                independent_probability_error=enumeration_error,
                direct_vs_score_gradient_max=score_error, direct_vs_finite_difference_max=fd_error,
                rewarded_action_gradient=float(actor_grad), rewarded_probability_before=p_before,
                rewarded_probability_after=p_after, equal_group_loss=float(equal_loss),
                equal_group_gradient=equal_gradient, reward_gradient_detached=True)


def _summary_oracle(h, labels, stage, observed, indegree):
    """NumPy implementation, independent of model.policy_state."""
    h = np.asarray(h, np.float64)
    labels = np.asarray(labels)
    n, d = h.shape
    c = len(labels)
    def cosine(x, y):
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return float(np.dot(x, y) / denom) if denom > 1e-24 else 0.
    context_mean = h[:c].mean(0) if c else np.zeros(d)
    other_mean = h[c:].mean(0) if c < n else np.zeros(d)
    same, different = [], []
    for i in range(c):
        for j in range(i + 1, c):
            if np.linalg.norm(h[i]) <= 1e-12 or np.linalg.norm(h[j]) <= 1e-12:
                continue
            (same if labels[i] == labels[j] else different).append(cosine(h[i], h[j]))
    value = np.asarray([
        stage / 3, np.mean(observed) if np.size(observed) else 0.,
        np.any(observed, axis=1).mean(), c / n, math.log(n) / 10,
        math.log(max(np.shape(observed)[1], 1)) / 10, np.mean(indegree) / n,
        np.linalg.norm(h, axis=1).mean() / 20, np.var(h, axis=0).mean() / 10,
        cosine(context_mean, other_mean) if np.linalg.norm(context_mean) * np.linalg.norm(other_mean) > 1e-12 else 0., np.mean(same) if same else 0.,
        np.mean(different) if different else 0., float(bool(same)), float(bool(different)),
    ], np.float64)
    value[1:] = np.clip(value[1:], -10., 10.)
    return value


def _fixture():
    values = np.asarray([[.2, -.7, .4, 1.1], [.3, -.4, .5, .8], [.25, -.5, .45, .9],
                         [-.8, .6, -.2, .3], [-.5, .9, -.1, .4], [-.6, .8, -.15, .35]], np.float32)
    observed = np.asarray([[1] * 4, [1] * 4, [0] * 4, [1] * 4, [1] * 4, [0] * 4], bool)
    # GraphPFN's local input convention includes self edges.
    src = np.r_[np.arange(6), [0, 1, 1, 2, 3, 4, 4, 5]].astype(np.int64)
    dst = np.r_[np.arange(6), [1, 0, 2, 1, 4, 3, 5, 4]].astype(np.int64)
    inp = dict(values=values, observed=observed, src=src, dst=dst,
               context_ids=np.asarray([0, 3], np.int64), context_labels=np.asarray([0, 1], np.int64),
               query_ids=np.asarray([2, 5], np.int64), classes=2, forward_seed=51060917)
    _require(set(inp) == WHITELIST, "fixture violates model input whitelist")
    return inp


def _params(model, name):
    items = getattr(model, name)
    return list(items() if callable(items) else items)


def _zero(model):
    model.base.zero_grad(set_to_none=True)
    model.policy.zero_grad(set_to_none=True)


def _autocast(device):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                          enabled=torch.device(device).type == "cuda")


def _forward(model, inp, trace=None, mode="greedy", action_rng=None):
    with _autocast(next(model.policy.parameters()).device):
        return model.forward(inp, trace=trace, mode=mode, action_rng=action_rng)


def _finite_gradients(parameters):
    present = [p.grad for p in parameters if p.grad is not None]
    _require(present and all(torch.isfinite(g).all() for g in present), "missing/nonfinite gradients")
    norm = math.sqrt(sum(float(g.detach().float().square().sum()) for g in present))
    return dict(parameters=len(parameters), gradients_present=len(present), l2=norm)


def _summary_checks(module):
    helper = getattr(module, "policy_state", None)
    _require(helper is not None, "model must expose policy_state for an independent summary oracle")
    rng = np.random.default_rng(51060901)
    h = rng.normal(size=(7, 192)).astype(np.float32)
    observed = rng.random((7, 4)) < .45
    degree = np.asarray([2, 3, 3, 3, 3, 3, 2], np.float32)
    labels = np.asarray([0, 1, 0], np.int64)
    def actual(hidden, label, obs=observed, deg=degree):
        return helper(torch.from_numpy(hidden), torch.from_numpy(label), 2,
                      torch.from_numpy(obs), torch.from_numpy(deg)).detach().cpu().numpy()
    got = actual(h, labels)
    expected = _summary_oracle(h, labels, 2, observed, degree)
    error = float(np.max(np.abs(got - expected)))
    _require(got.shape == (14,) and error < 3e-6, "independent 14-statistic oracle mismatch")
    renamed = actual(h, np.asarray([7, 2, 7], np.int64))
    _require(np.max(np.abs(got - renamed)) < 1e-7, "policy summary depends on raw class names")
    order = np.asarray([2, 0, 1, 6, 4, 3, 5])
    reordered = actual(h[order], labels[order[:3]], observed[order], degree[order])
    _require(np.max(np.abs(got - reordered)) < 3e-6, "policy summary is not block-node invariant")
    zeros = h.copy(); zeros[:3] = 0
    empty = actual(zeros, np.zeros(0, np.int64))
    _require(np.isfinite(empty).all(), "empty context summary is nonfinite")
    zero_context = actual(zeros, labels)
    _require(zero_context[12] == 0 and zero_context[13] == 0, "zero-norm context pairs must be invalid")
    return dict(numpy_oracle_max=error, class_renaming_max=float(np.max(np.abs(got - renamed))),
                block_node_permutation_max=float(np.max(np.abs(got - reordered))),
                empty_context_finite=True, zero_norm_pair_flags_zero=True)


def model_checks(device="cuda", progress=None):
    """Only root executes this function; one tiny disposable Model instance."""
    progress = progress if progress is not None else {"optimizer_step_attempts": 0, "successful_optimizer_updates": 0}
    B.setup()
    import rl_model as M
    _require(torch.device(device).type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    model = M.Model(11, device=device)
    model.base.eval(); model.policy.eval()
    inp = _fixture()
    target = torch.tensor([0, 1], device=device)
    adapters = _params(model, "adapter_parameters")
    policies = _params(model, "policy_parameters")
    _require(adapters and policies and all(p.requires_grad for p in adapters + policies), "trainable groups invalid")
    _require(all(p.dtype == torch.float32 for p in adapters + policies), "AdamW trainable parameters must be FP32")
    frozen_before = B.digest(model.base, trainable=False)
    initial_adapter = B.digest(model.base, trainable=True)
    initial_policy = B.digest(model.policy)

    # All-zero routing should be original GraphPFN with identical feature RNG.
    with _autocast(device):
        original = model.original(inp)
    ref_gradient = torch.autograd.grad(F.cross_entropy(original.float(), target), adapters, allow_unused=True)
    ref_gradient = [None if g is None else g.detach().cpu().float() for g in ref_gradient]
    routed = _forward(model, inp, (0, 0, 0))
    forward_error = float((original.detach().float() - routed["logits"].detach().float()).abs().max())
    calls_before = model.hook_calls_total
    got_gradient = torch.autograd.grad(F.cross_entropy(routed["logits"].float(), target), adapters, allow_unused=True)
    _require(model.hook_calls_total == calls_before, "activation checkpoint backward repeated policy decisions")
    gradient_error = 0.
    for expected, got in zip(ref_gradient, got_gradient):
        _require((expected is None) == (got is None), "000 adapter-gradient connectivity differs")
        if got is not None:
            got = got.detach().cpu().float()
            gradient_error = max(gradient_error, float((expected - got).abs().max()))
            _require(torch.allclose(expected, got, atol=3e-3, rtol=1e-2), "000 adapter-gradient mismatch")
    _require(forward_error <= 2e-3, "000 original-forward mismatch")
    _require(routed["trace"] == (0, 0, 0), "forced trace not respected")
    _require(not routed["states"].requires_grad, "policy states are not detached")
    greedy_initial = _forward(model, inp)
    _require(greedy_initial["trace"] == (0, 0, 0), "zero-logit policy greedy tie must choose 0")

    # Nonzero final policy weights make tree normalization nontrivial.
    policy_state_initial = copy.deepcopy(model.policy.state_dict())
    final = [layer for layer in model.policy.modules() if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.weight.copy_(torch.linspace(-.2, .2, final.weight.numel(), device=final.weight.device).view_as(final.weight))
        final.bias.fill_(.13)
    leaves, prefix, ce, logps = [], {}, [], []
    for trace in TRACES:
        out = _forward(model, inp, trace)
        _require(tuple(out["trace"]) == trace and out["cost"] == sum(trace), "trace or cost mismatch")
        _require(out["action_calls"] == 3, "forward did not make exactly three decisions")
        _require(tuple(out["states"].shape) == (3, 14) and tuple(out["action_logits"].shape) == (3,), "policy metadata shapes")
        _require(torch.isfinite(out["states"]).all() and torch.isfinite(out["action_logits"]).all(), "nonfinite policy metadata")
        _require(not out["states"].requires_grad, "hidden-policy detach violated")
        expected_logp = sum(F.logsigmoid(out["action_logits"][j] if bit else -out["action_logits"][j]) for j, bit in enumerate(trace))
        _require(torch.allclose(out["logp"], expected_logp, atol=2e-6, rtol=0), "conditional logp mismatch")
        _require(torch.allclose(out["states"][:, 0], torch.tensor([0., 1/3, 2/3], device=device), atol=1e-7, rtol=0), "stage coordinate mismatch")
        for stage in range(3):
            key = trace[:stage]
            state = out["states"][stage].detach().cpu()
            if key in prefix:
                _require(torch.allclose(prefix[key], state, atol=2e-6, rtol=0), "same prefix changed under common feature RNG")
            else:
                prefix[key] = state
        leaves.append(out["logits"].detach().float().cpu())
        ce.append(F.cross_entropy(out["logits"].float(), target).detach())
        logps.append(out["logp"])
    tree_prob = torch.stack(logps).exp()
    _require(abs(float(tree_prob.detach().sum()) - 1) < 3e-6, "actual 8-leaf probability sum !=1")
    variation = max(float((p - leaves[0]).abs().max()) for p in leaves)
    _require(variation > 0, "all traces have identical predictions on nonconstant fixture")
    _zero(model)
    actor = (tree_prob * (torch.stack(ce) + .01 * torch.tensor([sum(t) for t in TRACES], device=device))).sum()
    calls_before = model.hook_calls_total
    actor.backward()
    _require(model.hook_calls_total == calls_before, "actor backward repeated policy decisions")
    actor_stats = _finite_gradients(policies)
    _require(actor_stats["l2"] > 0 and all(p.grad is None for p in adapters), "actor gradient leaks into adapters or is absent")
    _zero(model)
    out = _forward(model, inp, (1, 1, 1))
    calls_before = model.hook_calls_total
    F.cross_entropy(out["logits"].float(), target).backward()
    _require(model.hook_calls_total == calls_before, "CE backward repeated policy decisions")
    adapter_stats = _finite_gradients(adapters)
    _require(adapter_stats["l2"] > 0 and all(p.grad is None for p in policies), "CE gradient leaks into policy or adapters are disconnected")

    # One discarded actual AdamW/GradScaler step, with synthetic positive actor advantage.
    _zero(model)
    step_adapter_before = B.digest(model.base, trainable=True)
    step_policy_before = B.digest(model.policy)
    optimizer = torch.optim.AdamW([{"params": adapters, "lr": 3e-5}, {"params": policies, "lr": 1e-3}], weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=64., enabled=torch.device(device).type == "cuda")
    out = _forward(model, inp, (1, 1, 1))
    smoke_loss = F.cross_entropy(out["logits"].float(), target) - .1 * out["logp"]
    calls_before = model.hook_calls_total
    scaler.scale(smoke_loss).backward()
    _require(model.hook_calls_total == calls_before, "optimizer smoke backward repeated policy decisions")
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(adapters, 1., error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(policies, 1., error_if_nonfinite=True)
    scale_before = float(scaler.get_scale())
    progress["optimizer_step_attempts"] += 1
    scaler.step(optimizer); scaler.update()
    if float(scaler.get_scale()) >= scale_before:
        progress["successful_optimizer_updates"] += 1
    _require(float(scaler.get_scale()) >= scale_before, "discarded optimizer smoke overflowed/skipped")
    _require(all(torch.isfinite(p).all() for p in adapters + policies), "optimizer created nonfinite parameters")
    _require(B.digest(model.base, trainable=True) != step_adapter_before and B.digest(model.policy) != step_policy_before,
             "optimizer did not update both trainable groups")
    _require(B.digest(model.base, trainable=False) == frozen_before, "frozen pretrained parameters changed")
    _require(all(p.grad is None for p in model.base.parameters() if not p.requires_grad), "frozen gradient unexpectedly exists")

    # Real serialization, but no file/checkpoint is created or overwritten.
    state = model.trainable_state()
    stream = io.BytesIO(); torch.save(state, stream); stream.seek(0)
    decoded = torch.load(stream, map_location="cpu", weights_only=False)
    before_logits = _forward(model, inp, (1, 0, 1))["logits"].detach().cpu()
    before_adapter, before_policy = B.digest(model.base, trainable=True), B.digest(model.policy)
    with torch.no_grad():
        adapters[0].add_(.01); policies[-1].add_(.01)
    model.load_trainable_state(decoded)
    after_logits = _forward(model, inp, (1, 0, 1))["logits"].detach().cpu()
    _require(B.digest(model.base, trainable=True) == before_adapter and B.digest(model.policy) == before_policy, "save/load parameter digest mismatch")
    replay_error = float((before_logits.float() - after_logits.float()).abs().max())
    _require(replay_error <= 2e-3, "save/load forward replay mismatch")

    # Independent policy RNG sequence: model's feature-seed reset must not reset it.
    model.policy.load_state_dict(policy_state_initial)
    generator = np.random.default_rng(51060933)
    oracle = np.random.default_rng(51060933)
    sampled, expected = [], []
    for _ in range(4):
        sampled.append(tuple(_forward(model, inp, mode="sample", action_rng=generator)["trace"]))
        expected.append(tuple(int(v < .5) for v in oracle.random(3)))
    _require(sampled == expected, "policy actions do not use the independent NumPy RNG")

    # The upstream target encoder ranks represented context labels.  Check the
    # explicit physical-class map and exchangeability of absent context classes.
    absent = dict(inp, classes=4, context_labels=np.asarray([0, 2], np.int64))
    absent_out = _forward(model, absent, (0, 0, 0))
    _require(tuple(absent_out["decoder_class_order"]) == (0, 2, 1, 3), "decoder physical-class map is wrong")
    absent_prob = absent_out["logits"].float().softmax(-1)
    absent_error = float((absent_prob[:, 1] - absent_prob[:, 3]).abs().max())
    _require(absent_error < 2e-7, "unrepresented classes do not share their total probability mass")

    # Sensitivities of the released PFN are descriptive; no exact equivariance
    # is promised by the pretrained target/feature embeddings.
    baseline = _forward(model, inp, (0, 0, 0))["logits"].detach().float().softmax(-1)
    renamed = dict(inp, context_labels=1-inp["context_labels"])
    class_pred = _forward(model, renamed, (0, 0, 0))["logits"].detach().float().softmax(-1)[:, [1, 0]]
    cp = np.asarray([2, 0, 3, 1])
    columns = dict(inp, values=inp["values"][:, cp], observed=inp["observed"][:, cp])
    column_pred = _forward(model, columns, (0, 0, 0))["logits"].detach().float().softmax(-1)
    order = np.asarray([3, 0, 4, 1, 5, 2]); inverse = np.argsort(order)
    nodes = dict(inp, values=inp["values"][order], observed=inp["observed"][order],
                 src=inverse[inp["src"]], dst=inverse[inp["dst"]],
                 context_ids=inverse[inp["context_ids"]], query_ids=inverse[inp["query_ids"]])
    node_pred = _forward(model, nodes, (0, 0, 0))["logits"].detach().float().softmax(-1)
    sensitivity = dict(class_rename_probability_max=float((baseline-class_pred).abs().max()),
                       column_order_probability_max=float((baseline-column_pred).abs().max()),
                       node_order_probability_max=float((baseline-node_pred).abs().max()),
                       scope="diagnostic only; not an exact pretrained equivariance assertion")
    _require(B.digest(model.base, trainable=False) == frozen_before, "frozen parameters changed after replay")

    return dict(status="PASS", device=str(device), scope="tiny synthetic fixture only; no bank/real/final reads",
                input_whitelist=sorted(WHITELIST), original000_forward_max=forward_error,
                original000_adapter_gradient_max=gradient_error,
                tolerance=dict(forward_absolute=2e-3, gradient_absolute=3e-3, gradient_relative=1e-2),
                all_trace_probability_sum=float(tree_prob.detach().sum()), prefix_states=len(prefix),
                trace_logit_variation=variation, detached_actor_gradient=actor_stats,
                live_adapter_ce_gradient=adapter_stats, discarded_optimizer_updates=1,
                frozen_digest_before=frozen_before, frozen_digest_after=B.digest(model.base, trainable=False),
                save_load_replay_max=replay_error, independent_action_rng=True,
                checkpoint_hashes=dict(model.checkpoint_hashes), upstream_commit=model.upstream_commit,
                initial_adapter_digest=initial_adapter, initial_policy_digest=initial_policy,
                absent_context_class_probability_error=absent_error,
                released_predictor_sensitivity=sensitivity,
                routed_forward_calls=model.hook_calls_total//3, original_reference_forward_calls=1,
                policy_statistics=_summary_checks(M),
                raw_hidden_poison_scope="Data pipeline checks; represented PCFI values at M=0 are legitimate inputs")


def run(device="cuda", toy_only=False):
    """Return a JSON-serializable result, including a preserved failure payload."""
    start = time.perf_counter()
    result = dict(cycle="FP05", status="RUNNING", optimizer_updates=0,
                  real_bank_used=False, final_bank_used=False, training_bank_used=False,
                  protocol_version=B.cfg()["version"], toy_only=bool(toy_only))
    progress = {"optimizer_step_attempts": 0, "successful_optimizer_updates": 0}
    result["execution_progress"] = progress
    paths = [Path(__file__).resolve(), B.ROOT / "rl_model.py", B.ROOT / "rl_util.py",
             B.ROOT / "protocol.json", B.ROOT / "PLAN_v2.md"]
    before = {str(p): B.sha(p) for p in paths if p.exists()}
    result["tested_sources"] = before
    try:
        if not toy_only:
            before = B.sources()
            result["tested_sources"] = before
        result["toy"] = toy_checks()
        if not toy_only:
            result["model"] = model_checks(device, progress)
            result["status"] = "PASS"
        else:
            result["status"] = "TOY_PASS"
        after = ({str(p): B.sha(p) for p in paths if p.exists()} if toy_only else B.sources())
        _require(before == after, "scientific files changed during checks")
        result["tested_sources_after"] = after
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    result["optimizer_updates"] = progress["successful_optimizer_updates"]
    result["seconds"] = time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--toy-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or B.ROOT / "results" / ("checks_toy.json" if args.toy_only else "checks.json")
    if output.exists():
        raise FileExistsError(f"Preserve prior receipt: {output}; choose a new --output path for a retry")
    result = run(args.device, args.toy_only)
    B.dump(output, result)  # create-only for successful and failed receipts alike
    print(json.dumps(dict(status=result["status"], output=str(output), seconds=result["seconds"])), flush=True)
    if result["status"] not in ("PASS", "TOY_PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
