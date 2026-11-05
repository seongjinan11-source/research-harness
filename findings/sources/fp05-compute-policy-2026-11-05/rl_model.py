"""FP05 controller on the released GraphPFNLight; imports start no model job.

The caller supplies represented PCFI values, never hidden raw values/targets.
All pretrained forward/preprocessing code remains unmodified on disk.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows DLL import order.
from contextlib import contextmanager
import math
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_util import setup, cfg, sha, digest


INPUT_KEYS = frozenset({'values', 'observed', 'src', 'dst', 'context_ids',
                        'context_labels', 'query_ids', 'classes', 'forward_seed'})


@contextmanager
def paired_rng(seed: int, device: torch.device):
    """Restore CPU/the selected CUDA generator; never touch action NumPy RNG."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


def policy_state(label_states: torch.Tensor, context_labels: torch.Tensor,
                 stage: int, observed: torch.Tensor, indegree: torch.Tensor) -> torch.Tensor:
    """Fourteen FP32 detached statistics in V2 order, context nodes first.

    Cosine pairs are unordered i<j, excluding zero-norm vectors. Relabeling
    context IDs preserves equality groups. Whole-model class equivariance is
    not implied because the released model has fixed target embeddings.
    """
    if label_states.ndim != 2 or label_states.shape[0] < 1:
        raise ValueError('label_states must be [N,D], N>=1')
    if stage not in (0, 1, 2):
        raise ValueError('stage must be 0,1,2')
    with torch.autocast(device_type=label_states.device.type, enabled=False):
        h = label_states.detach().float()
        obs = torch.as_tensor(observed, device=h.device, dtype=torch.bool)
        labels = torch.as_tensor(context_labels, device=h.device, dtype=torch.long)
        degree = torch.as_tensor(indegree, device=h.device, dtype=torch.float32)
        n, _ = h.shape
        if obs.ndim != 2 or obs.shape[0] != n or obs.shape[1] < 1:
            raise ValueError('observed must be [N,F], F>=1')
        if degree.shape != (n,) or labels.ndim != 1 or labels.numel() > n:
            raise ValueError('invalid indegree/context shape')
        if not bool(torch.isfinite(h).all()):
            raise FloatingPointError('nonfinite policy hidden state')
        c, features = labels.numel(), obs.shape[1]
        zero = h.new_zeros(())
        eps = 1e-12
        if c and c < n:
            left, right = h[:c].mean(0), h[c:].mean(0)
            denominator = left.norm() * right.norm()
            contrast = torch.where(denominator > eps,
                                   (left * right).sum()/denominator.clamp_min(eps), zero)
        else:
            contrast = zero
        same_cos = different_cos = same_valid = different_valid = zero
        if c >= 2:
            hc = h[:c]
            norms = hc.norm(dim=-1)
            unit = hc / norms.clamp_min(eps).unsqueeze(-1)
            cosines = unit @ unit.T
            pairs = torch.triu(torch.ones((c, c), device=h.device, dtype=torch.bool), diagonal=1)
            pairs = pairs & (norms[:, None] > eps) & (norms[None, :] > eps)
            equal = labels[:, None] == labels[None, :]
            same, different = pairs & equal, pairs & ~equal
            same_count, different_count = same.sum(), different.sum()
            same_cos = torch.where(same_count > 0, cosines.masked_fill(~same, 0).sum()/same_count.clamp_min(1), zero)
            different_cos = torch.where(different_count > 0, cosines.masked_fill(~different, 0).sum()/different_count.clamp_min(1), zero)
            same_valid = (same_count > 0).float()
            different_valid = (different_count > 0).float()
        state = torch.stack((
            h.new_tensor(stage/3),
            obs.float().mean(), obs.any(1).float().mean(), h.new_tensor(c/n),
            h.new_tensor(math.log(n)/10), h.new_tensor(math.log(features)/10),
            degree.mean()/n, h.norm(dim=-1).mean()/20,
            h.var(dim=0, correction=0).mean()/10,
            contrast, same_cos, different_cos, same_valid, different_valid,
        ))
        state = torch.cat((state[:1], state[1:].clamp(-10., 10.)))
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError('nonfinite policy statistics')
        return state.detach()


class Model(nn.Module):
    """Adapters receive live CE gradients; only the separate policy receives PG."""
    summary = staticmethod(policy_state)

    def __init__(self, seed: int, device: str = 'cuda'):
        super().__init__()
        self.seed = int(seed)
        self.device = torch.device(device)
        self.config = cfg()
        self.U = setup()
        # The public predictor is identical across train seeds; only phi init
        # varies. Forking also prevents construction from consuming run RNG.
        with paired_rng(0, self.device):
            self.base, loaded_names = self.U.load_light(device=str(self.device), always_checkpoint=True)
        self.loaded_checkpoint_names = tuple(sorted(loaded_names))
        layers = self.base.tfm.module.transformer_encoder.layers
        if len(layers) != 12:
            raise ValueError('the locked released predictor must retain all12 wrappers')
        chosen = tuple(int(i) for i in self.config['layers'])
        if chosen != (2, 5, 8):
            raise ValueError('FP05 action layers are locked to2/5/8')
        self.action_layers = chosen
        adapter_ids = set()
        for layer in layers:
            if not hasattr(layer, 'conv') or not hasattr(layer, 'mlp'):
                raise TypeError('expected released GraphPFNLayerWrapper')
            adapter_ids.update(id(p) for p in layer.conv.parameters())
            adapter_ids.update(id(p) for p in layer.mlp.parameters())
        for _, parameter in self.base.named_parameters():
            parameter.requires_grad_(id(parameter) in adapter_ids)
        self.adapter_names = tuple(name for name, parameter in self.base.named_parameters() if parameter.requires_grad)
        if not self.adapter_names:
            raise RuntimeError('no trainable released adapters')
        with paired_rng(self.seed, torch.device('cpu')):
            self.policy = nn.Sequential(nn.Linear(14, 32), nn.Tanh(), nn.Linear(32, 1))
            nn.init.zeros_(self.policy[-1].weight)
            nn.init.zeros_(self.policy[-1].bias)
        self.policy.to(device=self.device, dtype=torch.float32)
        self.checkpoint_hashes = {str(self.U.ADAPTERS): sha(self.U.ADAPTERS), str(self.U.LIMIX): sha(self.U.LIMIX)}
        self.upstream_commit = self.U.COMMIT
        self._active = None
        self._handles = []
        self._hook_calls_total = 0
        owner = weakref.ref(self)
        for stage, layer_idx in enumerate(self.action_layers):
            def hook(module, args, output, *, decision_stage=stage, model_ref=owner):
                model = model_ref()
                if model is None or model._active is None:
                    return None
                return model._decision(module, output, decision_stage)
            self._handles.append(layers[layer_idx].register_forward_hook(hook))
        self.eval()

    def adapter_parameters(self):
        return (parameter for parameter in self.base.parameters() if parameter.requires_grad)

    def policy_parameters(self):
        return self.policy.parameters()

    def frozen_digest(self):
        return digest(self.base, trainable=False)

    @property
    def hook_calls_total(self):
        """Useful to assert backward/checkpointing causes no new decisions."""
        return self._hook_calls_total

    def _prepare(self, inp):
        if set(inp) != INPUT_KEYS:
            raise ValueError(f'input whitelist mismatch: missing={INPUT_KEYS-set(inp)}, extra={set(inp)-INPUT_KEYS}')
        parameter = next(self.base.parameters())
        device = parameter.device
        values = torch.as_tensor(inp['values'], device=device, dtype=torch.float32)
        observed = torch.as_tensor(inp['observed'], device=device, dtype=torch.bool)
        if values.ndim != 2 or values.shape != observed.shape or min(values.shape) < 1:
            raise ValueError('values/observed must be the same nonempty [N,F]')
        n = values.shape[0]
        src = torch.as_tensor(inp['src'], device=device, dtype=torch.long)
        dst = torch.as_tensor(inp['dst'], device=device, dtype=torch.long)
        ids = torch.as_tensor(inp['context_ids'], device=device, dtype=torch.long)
        labels = torch.as_tensor(inp['context_labels'], device=device, dtype=torch.long)
        queries = torch.as_tensor(inp['query_ids'], device=device, dtype=torch.long)
        classes = int(inp['classes'])
        if src.ndim != 1 or src.shape != dst.shape or ids.ndim != 1 or labels.shape != ids.shape or queries.ndim != 1:
            raise ValueError('invalid edge/context/query vector shapes')
        if not (0 < ids.numel() < n) or classes < 1:
            raise ValueError('released LimiX requires at least one context and one noncontext node')
        for name, indices in [('src', src), ('dst', dst), ('context_ids', ids), ('query_ids', queries)]:
            if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= n):
                raise ValueError(f'{name} outside node vocabulary')
        if ids.unique().numel() != ids.numel() or queries.unique().numel() != queries.numel():
            raise ValueError('context/query node IDs must be unique')
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= classes):
            raise ValueError('context label outside supplied class vocabulary')
        order = torch.argsort(ids, stable=True)
        ids, labels = ids[order], labels[order]
        train_mask = torch.zeros(n, device=device, dtype=torch.bool)
        train_mask[ids] = True
        if queries.numel() and bool(train_mask[queries].any()):
            raise ValueError('query/context overlap would include dummy context outputs in loss')
        permutation = torch.argsort((~train_mask).float(), stable=True)
        indegree = torch.bincount(dst, minlength=n).float()
        # Upstream MulticlassTargetEncoder ranks context labels. Recover the
        # physical IDs; unrepresented classes use remaining head slots in order.
        represented = sorted(set(labels.detach().cpu().tolist()))
        decoder_order = represented + [c for c in range(classes) if c not in represented]
        inverse_class_order = torch.argsort(torch.tensor(decoder_order, device=device), stable=True)
        return dict(values=values, observed=observed, src=src, dst=dst, ids=ids, labels=labels,
                    queries=queries, classes=classes, train_mask=train_mask, permutation=permutation,
                    indegree=indegree, inverse_class_order=inverse_class_order,
                    decoder_class_order=tuple(decoder_order), represented_count=len(represented),
                    forward_seed=int(inp['forward_seed']))

    def _decision(self, layer, output, stage):
        active = self._active
        if stage != len(active['states']):
            raise RuntimeError('duplicate/out-of-order policy hook; check activation checkpoint boundaries')
        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError('released wrapper must return hidden and two attention entries')
        x, feature_attention, sample_attention = output
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError('expected released [1,N,T,192] hidden tensor')
        prepared = active['prepared']
        state = policy_state(x[0, :, -1, :], prepared['labels'], stage,
                             prepared['observed'][prepared['permutation']],
                             prepared['indegree'][prepared['permutation']])
        with torch.autocast(device_type=x.device.type, enabled=False):
            action_logit = self.policy(state.float()).squeeze(-1)
        if not bool(torch.isfinite(action_logit)):
            raise FloatingPointError('nonfinite policy logit')
        forced = active['trace']
        if forced is not None:
            action = forced[stage]
        elif active['mode'] == 'greedy':
            action = int(float(action_logit.detach()) > 0.)  # Equal binary probs tie to0.
        else:
            probability = float(action_logit.detach().sigmoid())
            action = int(active['action_rng'].random() < probability)
        logterm = F.logsigmoid(action_logit if action else -action_logit)
        active['states'].append(state)
        active['action_logits'].append(action_logit)
        active['logterms'].append(logterm)
        active['actions'].append(action)
        self._hook_calls_total += 1
        if action:
            graph = layer._graph_holder.graph
            if graph is None:
                raise RuntimeError('upstream graph holder cleared before action')
            x = layer.mlp(graph, layer.conv(graph, x))
        return x, feature_attention, sample_attention

    def _execute(self, inp, active=None):
        import dgl
        if self._active is not None:
            raise RuntimeError('one Model instance cannot run simultaneous forwards')
        prepared = self._prepare(inp)
        if active is not None:
            active['prepared'] = prepared
        self.base.eval()
        self.policy.eval()
        self._active = active
        try:
            graph = dgl.graph((prepared['src'], prepared['dst']), num_nodes=prepared['values'].shape[0],
                              device=prepared['values'].device)
            with paired_rng(prepared['forward_seed'], prepared['values'].device):
                pred = self.base(graph=graph, features=prepared['values'], y_train=prepared['labels'],
                                 train_mask=prepared['train_mask'],
                                 task_type='binclass' if prepared['classes'] <= 2 else 'multiclass')
            if pred.shape[-1] < prepared['classes']:
                raise ValueError('supplied K exceeds the released decoder capacity')
            scores = F.log_softmax(pred[:, :prepared['classes']].float(), dim=-1)
            represented = prepared['represented_count']
            missing = prepared['classes'] - represented
            if missing > 1:
                # Native target encoding compresses represented classes. No
                # context establishes names for the remaining head slots: keep
                # their total probability, distribute it exchangeably, then
                # map represented/remaining slots to supplied physical IDs.
                equal_logmass = torch.logsumexp(scores[:, represented:], dim=-1, keepdim=True) - math.log(missing)
                scores = torch.cat((scores[:, :represented], equal_logmass.expand(-1, missing)), dim=-1)
            logits = scores[:, prepared['inverse_class_order']][prepared['queries']]
            return logits, prepared
        finally:
            self._active = None
            self.base._graph_holder.graph = None

    def original(self, inp):
        """Unmodified released forward, bypass policy hooks; same query/K map."""
        return self._execute(inp, active=None)[0]

    def forward(self, inp, trace=None, mode='greedy', action_rng=None):
        if mode not in ('greedy', 'sample'):
            raise ValueError('mode must be greedy or sample')
        if trace is not None:
            trace = tuple(trace)
            if len(trace) != 3 or any(action not in (0, 1) for action in trace):
                raise ValueError('trace must contain exactly three binary actions')
            trace = tuple(int(action) for action in trace)
        if trace is None and mode == 'sample' and not isinstance(action_rng, np.random.Generator):
            raise ValueError('sampling requires an explicit independent NumPy Generator')
        active = dict(trace=trace, mode=mode, action_rng=action_rng,
                      states=[], action_logits=[], logterms=[], actions=[])
        logits, prepared = self._execute(inp, active)
        if len(active['states']) != 3:
            raise RuntimeError('not all three original wrapper decisions were observed')
        return dict(logits=logits, logp=torch.stack(active['logterms']).sum(),
                    states=torch.stack(active['states']).detach(),
                    action_logits=torch.stack(active['action_logits']),
                    trace=tuple(active['actions']), cost=sum(active['actions']), action_calls=3,
                    decoder_class_order=prepared['decoder_class_order'])

    def trainable_state(self):
        return dict(schema='FP05-trainable-v1', seed=self.seed,
                    adapters={name: parameter.detach().cpu().clone() for name, parameter in self.base.named_parameters()
                              if parameter.requires_grad},
                    policy={name: value.detach().cpu().clone() for name, value in self.policy.state_dict().items()},
                    frozen_digest=self.frozen_digest(), checkpoint_hashes=dict(self.checkpoint_hashes))

    def load_trainable_state(self, state):
        if state.get('schema') != 'FP05-trainable-v1' or set(state['adapters']) != set(self.adapter_names):
            raise ValueError('trainable checkpoint schema/adapter names mismatch')
        if state['checkpoint_hashes'] != self.checkpoint_hashes or state['frozen_digest'] != self.frozen_digest():
            raise ValueError('public frozen checkpoint changed')
        with torch.no_grad():
            for name, parameter in self.base.named_parameters():
                if parameter.requires_grad:
                    value = state['adapters'][name]
                    if parameter.shape != value.shape:
                        raise ValueError(f'adapter shape mismatch: {name}')
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.policy.load_state_dict(state['policy'], strict=True)
        self.seed = int(state['seed'])
        return self


__all__ = ['Model', 'policy_state', 'paired_rng', 'INPUT_KEYS']
