"""Execution-only restart: identical operations, disjoint checkpoint workers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as E
root=E.ROOT
old=E.json.loads((root/'runtime/lock_before_parallel.json').read_text())
new=E.copy.deepcopy(old);new['sources']=E.sources()
assert old['sources']['protocol.json']==new['sources']['protocol.json']
assert old['sources']['reference_common']==new['sources']['reference_common']
assert old['bank_manifest_sha256']==E.C.sha(root/'bank/manifest.json')
files=list((root/'results/s11').glob('*.npz'));episodes={}
for path in files:
    episode=tuple(path.stem.split('_')[:2]);episodes.setdefault(episode,[]).append(path)
archive=root/'runtime/interrupted_predictions';archive.mkdir(exist_ok=True)
moved=[];kept=[]
for key,paths in episodes.items():
    complete=len(paths)==len(E.config()['arms'])
    for path in paths:
        try:
            with E.np.load(path) as data:
                assert data['views'].shape[0]==E.config()['views']
                assert E.np.isfinite(data['views']).all()
        except Exception:complete=False
    for path in paths:
        if complete:kept.append({'path':path.relative_to(root).as_posix(),'sha256':E.C.sha(path)})
        else:
            target=archive/path.name
            assert root in path.resolve().parents and root in target.resolve().parents
            path.rename(target);moved.append(target.relative_to(root).as_posix())
E.write(root/'results/lock.json',new)
E.write(root/'runtime/parallel_restart.json',{
    'change':'Only optional --seed dispatch, one independent process per predeclared checkpoint; no mathematical, data, model, loss or evaluation change.',
    'previous_lock':old,'new_lock':new,'preserved_completed_predictions':kept,
    'archived_incomplete_episode_predictions':moved,
    'note':'A whole interrupted episode is replayed so full-observed arm equality is checked within one process.'})
print('Preserved',len(kept),'predictions; archived',len(moved),'incomplete episode files')
