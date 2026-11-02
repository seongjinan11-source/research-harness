"""Secondary latent-law accuracy: avoid assigning meaning to roundoff at q=.5."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results/weights.json').read_text(encoding='utf8'))
tolerance=json.loads((root/'runtime/latent_accuracy_addendum.json').read_text())['tolerance']
output={}
for k in [1,4]:
    for budget in [0,1,8,'full']:
        rr=[r for r in rows if r['k']==k and r['sources']==budget]
        item={}
        for field,target in [('feature_g1','feature_law'),('joint_g1','feature_law'),('label_t1','label_law'),('joint_t1','label_law')]:
            q=np.asarray([r[field] for r in rr]);y=np.asarray([r[target] for r in rr])
            tied=abs(q-.5)<=tolerance
            score=np.where(tied,.5,((q>.5)==y).astype(float))
            item[field]={'fractional_tie_accuracy':float(score.mean()),'tie_fraction':float(tied.mean()),
                'max_abs_from_half':float(abs(q-.5).max()),'brier':float(((q-y)**2).mean())}
        output[f'k{k}_o{budget}']=item
(root/'results/latent_accuracy_audit.json').write_text(json.dumps({'tolerance':tolerance,
    'scope':'Only secondary latent g/t diagnostics. Numerical ties receive expected .5 accuracy. Original statistics and all query classification metrics unchanged.',
    'groups':output},indent=2),encoding='utf8')
print('LATENT_TIES_AUDITED',flush=True)
