"""Separate-development numerical checks and discarded smoke steps, then freeze design."""
import sys
sys.dont_write_bytecode = True
import sklearn
import copy
import itertools
import json
import math
import numpy as np
import torch
from common import ROOT, WORK, OLD_MODEL, C, D, cfg, dump, read, rows, sources, sha
from operators import CellFactorPFN, FrozenGraph, make_model
from reference import Reference, logpdf


def main():
    assert not (ROOT / 'results/design_lock.json').exists()
    C.setup()
    torch.set_num_threads(2)
    errors, metadata = {}, []
    dev = [read(r) for r in rows('dev')]
    inp, q, y = dev[0]
    b = D.tensors([inp], dtype=torch.float64)
    graph = FrozenGraph().cuda().double().eval()
    original = OLD_MODEL.FactorPFN().cuda().double()
    original.load_state_dict(torch.load(WORK / cfg()['graph_checkpoint'], weights_only=False)['state_dict'])
    with torch.inference_mode():
        gj, gc, _ = graph(b)
        oldj, _, _, oldc = original.potentials(b)
        errors['frozen_graph_j_parity'] = float(abs(gj - oldj[:, :, :12, :12]).max())
        errors['frozen_graph_constant_parity'] = float(abs(gc - oldc).max())
    assert max(errors.values()) < 1e-10
    n = 3
    engine = Reference('cpu', n=3, nt=2, nh=1)
    tiny = {'x': torch.tensor([[[0.], [.8], [-1.1]]], dtype=torch.float64), 'observed': torch.tensor([[[False], [True], [True]]]),
            'context': torch.tensor([[0, -1, -1]]), 'prior': torch.tensor([[1., 0, 0]], dtype=torch.float64),
            'pair_valid': torch.zeros(1, 2, 3, 3, dtype=torch.float64)}
    tiny['pair_valid'][:, :, 0, 1] = tiny['pair_valid'][:, :, 1, 0] = 1
    j = torch.zeros(1, 4, n, n, dtype=torch.float64)
    j[0, :, 0, 1] = j[0, :, 1, 0] = torch.tensor([.1, .2, .3, .4])
    constant = torch.tensor([[.03, -.1, .2, .0]], dtype=torch.float64)
    gp = (j, constant, None)
    exact = engine.point_exact(tiny, gp)
    scores, states = [], []
    for s in itertools.product([-1., 1.], repeat=3):
        if s[0] != -1:
            continue
        s = torch.tensor(s, dtype=torch.float64)
        for h in range(4):
            for law, aa, bb in engine.candidates:
                for z in [-1., 1.]:
                    log = .5 * s @ j[0, h] @ s + constant[0, h] - math.log(4 * 12 * 2)
                    for i in [1, 2]:
                        log = log + logpdf(tiny['x'][0, i, 0], law, s.new_tensor(aa), s.new_tensor(bb), s[i] * z)
                    scores.append(log)
                    states.append((s + 1) / 2)
    independent = torch.stack(scores).softmax(0) @ torch.stack(states)
    errors['exact_latent_sum_independent'] = float(abs(exact[0, :, 1] - independent).max())
    bp = engine.fixed_bp(tiny, gp, 80)
    errors['fixed_bp_tree_parity'] = float(abs(bp - exact).max())
    assert errors['exact_latent_sum_independent'] < 1e-10 and errors['fixed_bp_tree_parity'] < 5e-5, errors
    a, bb = torch.tensor(.35), torch.tensor(1.)
    x = torch.tensor([-1.5, 1.5, 0.])
    ratio = .5 * (logpdf(x, 'variance', a, bb, x.new_tensor(1)) - logpdf(x, 'variance', a, bb, x.new_tensor(-1)))
    errors['variance_evidence_even'] = float(abs(ratio[0] - ratio[1]))
    assert ratio[0] > 0 and ratio[2] < 0
    with torch.inference_mode():
        ref = Reference()
        p = ref.native_exact(b)
        poison = copy.deepcopy(b)
        poison['x'][~poison['observed']] = 123456.
        errors['reference_hidden_poison'] = float(abs(ref.native_exact(poison) - p).max())
        flip = copy.deepcopy(b)
        known = flip['context'] >= 0
        flip['context'][known] = 1 - flip['context'][known]
        errors['reference_class_flip'] = float(abs(ref.native_exact(flip) - p.flip(-1)).max())
        initial = []
        for arm in ['original', 'static', 'adaptive']:
            m = CellFactorPFN(arm, 101).cuda().double()
            initial.append(m(b))
        errors['matched_initial_predictions'] = float(max(abs(initial[i] - initial[0]).max() for i in range(1, 3)))
    assert max(errors[k] for k in ['reference_hidden_poison', 'reference_class_flip', 'matched_initial_predictions']) < 1e-6
    frozen_digests = set()
    for arm in cfg()['arms']:
        m = make_model(arm, 101)
        before = C.parameter_digest(m, trainable=False)
        count = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if arm in ['original', 'static', 'adaptive']:
            frozen_digests.add(before)
            batch = D.tensors([dev[i][0] for i in [0, 12, 24, 36]])
            m.train()
            prob = m(batch)
            terms = []
            for i, k in enumerate([0, 12, 24, 36]):
                _, query, labels = dev[k]
                terms.append(-prob[i, query, torch.tensor(labels, device='cuda')].clamp_min(1e-10).log().mean())
            loss = torch.stack(terms).mean()
            loss.backward()
            params = [p for p in m.parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert any(bool(p.grad.abs().max() > 0) for p in params)
            optimizer = torch.optim.AdamW(params, lr=.001)
            optimizer.step()
            assert C.parameter_digest(m, trainable=False) == before
            m.eval()
            with torch.inference_mode():
                p0 = m(D.tensors([inp]))
                poisoned = {k: v.copy() for k, v in inp.items()}
                poisoned['x'][~poisoned['observed']] = 123456.
                errors[arm + '_hidden_poison'] = float(abs(m(D.tensors([poisoned])) - p0).max())
                flipped = {k: v.copy() for k, v in inp.items()}
                kk = flipped['context'] >= 0
                flipped['context'][kk] = 1 - flipped['context'][kk]
                errors[arm + '_class_flip'] = float(abs(m(D.tensors([flipped])) - p0.flip(-1)).max())
                bb = D.tensors([inp])
                for key in ['incidence', 'adj', 'pair_valid']:
                    bb[key] = bb[key].flip(1)
                errors[arm + '_relation_permutation'] = float(abs(m(bb) - p0).max())
                bb = D.tensors([inp])
                bb['x'], bb['observed'] = bb['x'].flip(-1), bb['observed'].flip(-1)
                errors[arm + '_column_permutation'] = float(abs(m(bb) - p0).max())
            assert max(value for key, value in errors.items() if key.startswith(arm + '_')) < 5e-5
        else:
            p = C.predict(m, D.public_input(inp, q, 4101))
            assert p.shape == (len(q), 2) and np.isfinite(p).all()
        metadata.append({'arm': arm, 'trainable_parameters': count, 'frozen_sha256': before})
        del m
        torch.cuda.empty_cache()
    assert len(frozen_digests) == 1
    static = next(r['trainable_parameters'] for r in metadata if r['arm'] == 'static')
    adaptive = next(r['trainable_parameters'] for r in metadata if r['arm'] == 'adaptive')
    assert abs(static - adaptive) / max(static, adaptive) < .02
    for r in rows('train') + rows('dev'):
        inp, query, labels = read(r)
        assert not inp['observed'][inp['context'] >= 0].any() and (~inp['observed'][query].any(-1)).any()
        assert sha(ROOT / 'bank' / r['path']) == r['sha256']
    dump(ROOT / 'results/preflight.json', {'errors': errors, 'models': metadata, 'discarded_development_optimizer_steps': 3,
         'same_frozen_graph_digest': True, 'all_dev_and_train_masks_and_hashes_checked': True,
         'scope': 'Development numerics only. No final graph generated or accuracy-based model selection. Three smoke weights discarded.'})
    dump(ROOT / 'results/design_lock.json', {'sources': sources(), 'development_manifest': sha(ROOT / 'bank/development_manifest.json'),
         'preflight': sha(ROOT / 'results/preflight.json'), 'scope': 'Architecture/protocol fixed before final bank generation'})
    print('DEVELOPMENT_PASSED_DESIGN_LOCKED', errors, metadata, flush=True)


if __name__ == '__main__':
    main()
