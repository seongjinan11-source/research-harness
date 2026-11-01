"""Record an inspected secondary effect without changing either registered gate."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiment as E
import yaml

paired=json.loads((ROOT/'results/paired.json').read_text())
selected=[r for r in paired if r['group'] in ['controlled:offdiagonal:clustered','controlled:offdiagonal:node']
    and r['contrast']=='crossed_mask:standard - coupled_mask:standard' and r['metric']=='accuracy']
assert len(selected)==2 and all(r['low']>0 for r in selected)
E.write(ROOT/'results/secondary_accuracy.json',{'contrasts':selected,
    'scope':'Inspected secondary results selected for preservation after the primary No-Go. No multiplicity correction, gate change, new training, or independent validation.'})
harness=ROOT.parent/'research-harness';folder=harness/'findings/sources/mechanism-crossed-2026-11-02'
manifest=json.loads((folder/'manifest.json').read_text())
for alias,original in [('preserve_accuracy_effect.py',Path(__file__).resolve()),('secondary_accuracy.json',ROOT/'results/secondary_accuracy.json')]:
    target=folder/alias;shutil.copyfile(original,target)
    manifest['sources'].append({'alias':alias,'original':str(original),'snapshot':target.relative_to(harness).as_posix(),
        'sha256':E.C.sha(target),'bytes':target.stat().st_size})
E.write(folder/'manifest.json',manifest)
path=harness/'findings/mechanism-crossed-prior.yaml';doc=yaml.safe_load(path.read_text(encoding='utf8'))
assert not any(e['id']=='MCP-P02' for e in doc['entries'])
obs='; '.join(f"{r['group']}: accuracy difference {100*r['delta']:+.3f}%p, CI [{100*r['low']:+.3f}, {100*r['high']:+.3f}]" for r in selected)
doc['entries'].append({'id':'MCP-P02','kind':'preserved','claim':'Constructed off-diagonal 조건의 Coupled Mask 대비 정확도 신호',
    'observation':obs,
    'do_not_claim':'사후 선별해 보존한 보조 metric 결과이며 다중 비교 보정과 독립 재검증은 없다. LP·PCFI·Official Mask 대비 같은 정확도 우위, NLL 우위 또는 실제 graph 전이로 확대하지 않고 원래 두 gate의 No-Go를 유지한다.',
    'tags':['prior-mismatch','distribution-shift','eval-protocol'],'keywords':['mechanism crossing','off diagonal','secondary accuracy','no multiplicity correction'],
    'source_refs':[(folder/'secondary_accuracy.json').relative_to(harness).as_posix(),(folder/'paired.json').relative_to(harness).as_posix()],
    'evidence':'원래 분석에 포함된 secondary accuracy contrast의 사후 보존'})
path.write_text(yaml.safe_dump(doc,allow_unicode=True,sort_keys=False),encoding='utf8')
print('PRESERVED_SECONDARY_ACCURACY',obs)
