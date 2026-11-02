"""Append this experiment without revising any prior finding or source."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil
from collections import Counter
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
root=Path(__file__).resolve().parents[1];repo=root.parent/'research-harness'
prefix='findings/sources/internal-pcfi-2026-11-02/'
dest=repo/prefix;assert not dest.exists();dest.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=list(root.glob('*.py'))+[root/p for p in ['protocol.json','METHOD_KO.md','REPORT_KO.md','bank/generator.json','bank/development.json','bank/manifest.json']]
paths+=list((root/'results').rglob('*.json'))+list((root/'logs').glob('*.log'))+list((root/'runtime').glob('*.py'))
manifest=[]
for p in sorted(set(paths)):
    q=dest/p.relative_to(root);q.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,q)
    manifest.append({'alias':p.relative_to(root).as_posix(),'original':str(p),'snapshot':q.relative_to(repo).as_posix(),'sha256':sha(q),'bytes':q.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'date':'2026-11-02','scope':'Literal PCFI inside a public PFN; sources copied verbatim. Checkpoints/raw input and probability banks stay in workspace.','sources':manifest},ensure_ascii=False,indent=2),encoding='utf8')
s=json.loads((root/'results/statistics.json').read_text(encoding='utf8'));p=s['subsets']['primary'];cost=json.loads((root/'results/cost.json').read_text(encoding='utf8'))
bank=json.loads((root/'bank/manifest.json').read_text(encoding='utf8'))['episodes']
size=f"nodes{min(r['nodes'] for r in bank)}~{max(r['nodes'] for r in bank)}, physical features{min(r['features'] for r in bank)}~{max(r['features'] for r in bank)}"
def diff(name,m='accuracy',subset=p):
    d=subset['contrasts'][name][m];scale=100 if m=='accuracy' else 1
    a,l,h=[x*scale for x in [d['delta'],*d['ci95']]];return f'{a:+.6f} CI[{l:+.6f},{h:+.6f}]'
numbers='; '.join(f"{a}:accuracy{v['accuracy']*100:.6f}%/NLL{v['nll']:.6f}/Brier{v['brier']:.6f}" for a,v in p['means'].items())
contrasts='; '.join(a+' accuracy%p '+diff(a)+', NLL '+diff(a,'nll') for a in p['contrasts'])
refs=[prefix+q for q in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/cost.json','results/audit.json','results/preflight.json','results/development_actual_pcfi_parity.json']]
claim='실제 PCFI 전파를 공개 PFN 내부에 배치하고 context/PFN 상태로 전파 가중치를 조절하면 외부 PCFI·내부 고정 PCFI·label-blind router보다 분류가 좋아진다.'
if s['expand']:
    core={'id':'IPCFI-P01','kind':'preserved','claim':claim,'observation':numbers+'. '+contrasts+f". Primary={s['primary_gate']}, full2%p guard={s['full_guard']}.",
          'do_not_claim':'같은 size-restricted official generator의 새 graph 결과다. 실제 graph·unseen schema 전이, labels-only 효과, calibrated feature uncertainty 또는 paper novelty가 아니다. Context 대조는 PFN feedback 경로 전체이며 label-blind는 그 hidden 표현까지 제거한다. 비용과 NLL/Brier를 별도로 본다.'}
else:
    core={'id':'IPCFI-F01','kind':'failure','claim':claim,'observation':numbers+'. '+contrasts+f". Primary={s['primary_gate']}, full2%p guard={s['full_guard']}, expand=False.",
          'verdict':'현재100-step literal PCFI/25-step interleaving/513-parameter router/256-update 구현의 메인 방법 확대를 보류한다. 이 final bank에서 head/loss/temperature/전파 횟수/학습량을 바꿔 구제하지 않는다.',
          'failure_reason':'사전 비교를 모두 만족하는 분류 이득을 확인하지 못했다. 전파가 이미 충분한지, context 요약·bounded router·frozen backbone/input interface·prior·최적화·학습량 중 무엇 때문인지는 분리하지 않아 고유 원인은 미상이다. 상수 신호를 row-stochastic 전파로 바꿀 수 없다는 수학적 제약을 전체 실패의 단일 원인으로 쓰지 않는다.',
          'reason_status':'unknown','excluded_explanations':['이전 generic MLP refiner가 아니라 원래 PCFI의 거리 confidence/관측값 clamp/100회 전파/1회 correlation을 구현했고 독립 SciPy 기준과 비교했다.',
          '네 모델은 같은 공개 초기화·train tasks·seed별 schedule·256 successful CE updates·100 propagation steps·1 correlation·12PFN layers다. Context/label-blind의513개 router 파라미터와 초기 예측도 같다. 고정 대조는 이513개가 없으므로 모든 arm의 정확한 동일 파라미터 비교라고 주장하지 않는다.',
          'Label-blind router는 labels가 섞인 PFN hidden까지 읽지 않는다. 실제 optimizer smoke, router/hidden gradient, 저장/복원, poison/clamp와 source/bank/prediction hash/replay를 확인했다.'],
          'closed_scope':'Official graph/SCM small family의 '+size+', classes2~4. 새64 train/4dev/64final graphs, uniform/clustered node90/99.5 missing, native pair/RMS/mask input,4 arms×3seeds×256 CE updates, 자기 피처 없는 query accuracy primary.',
          'not_closed':'JIMP-F01의 generic refiner와 별개 실행이다. 외부 PCFI의 실제 그래프 성과, 모든 내부 PCFI 방식, 다른 전파/표현/학습 원리, native NaN interface, 실제 heterograph·unseen schema 일반을 기각하지 않는다.',
          'reopen_when':'현재 동일한 numeric propagation을 다시 배치하거나 bounded weight만 조절하는 것을 넘어, 새 그래프의 context가 전달해야 할 정보를 바꾸는 구체적인 연산이 강한 외부 PCFI/PFN보다 유리할 근거가 생길 때.',
          'comparison_floor':'동일 actual PCFI/100steps/1correlation/12PFN layers의 외부 PCFI, 내부 고정, context와 strict label-blind router, 같은 train tasks/CE/initialization/updates. Context-frequency/uniform prior와 NLL/Brier/CPU→CPU end-to-end 비용도 보고한다.',
          'recheck':'본 final bank는 tuning하지 않는다. 변경할 정보·연산과 사전 primary를 먼저 정하고 독립 graph/schema에서 node-level 대신 base-graph paired 분석으로 한 번 검증한다. Labels-only 효과와 PFN-feedback 전체 효과를 구분한다.',
          'reopen_tags':['observation-mask','graph-structure-use','tokenization'],'reopen_keywords':['context conditioned propagation','nonconstant feature evidence','native PFN feature state','feature propagation bottleneck'],'revisit_status':'open'}
core.update(tags=['observation-mask','graph-structure-use','tokenization','eval-protocol','inference-cost'],keywords=['literal PCFI','internal diffusion','PFN interleaving','context feedback','strict label blind','confidence weighted propagation','observed clamp','same propagation budget'],source_refs=refs,evidence='새12모델3072 successful CE updates. Dev4 smoke optimizer steps 폐기 후 design lock, 이후64개 final graphs를 생성했다. Primary는4 node-missing episodes×3fixed seeds를 graph 안에서 평균하고5000 paired bootstrap. 실제 graph/새 schema가 아니다.')
lesson={'id':'IPCFI-L01','kind':'lesson','claim':'실제 PCFI 내부화와 이전 MLP refiner, strict router label-blindness와 labels-only 효과를 분리한다.',
 'observation':'새 구현은 PCFI의 공간 전파와 채널 상관 보정 연산을 유지했고 초기 context/label-blind 예측은 내부 고정과 같았다. 작은 그래프의 SciPy 차이는 최대4.43e-7, 실제 dev FP32 kernel/cache 차이는8.54e-5였다. Label-blind router에 PFN states와 labels를 오염시켜도 routing은 같았다.',
 'interpretation':'PFN 중간 feature tokens도 이미 context labels를 담을 수 있어 명시적 label input만0으로 두면 label-blind가 아니다. 본 대조는 hidden 경로 전체를 없애므로 PFN feedback+context의 묶음 효과이고 labels-only 원인분해가 아니다. 고정 PCFI를25step씩 끼우는 것과 앞에서100step을 완료하는 것은 PFN의 비선형 층 때문에 일반적으로 동치가 아니다. Row-stochastic 연산은 상수 신호를 유지하며 zero-source/zero-start에서 새 값을 만들지 못한다. 이를 바로 전체 failure 원인으로 단정하지 않는다.',
 'tags':['observation-mask','tokenization','graph-structure-use','eval-protocol'],'keywords':['PCFI versus MLP refiner','hidden label leakage','strict label blind routing','propagation placement','constant signal invariance'],
 'source_refs':[prefix+q for q in ['METHOD_KO.md','REPORT_KO.md','results/preflight.json','results/development_numeric_repair.json']],'evidence':'실제 source/수치검사와 미분·정규화 연산의 범위다. 최초 atomic scatter의 반복 차이는 final 생성 전에 dense 정규화/bmm로 보완했고 네 dev smoke weights는 폐기했다.'}
finding={'project':'Literal PCFI inside public GraphPFN','date':'2026-11-02','source':prefix+'REPORT_KO.md','setting':'Same public initialization/frozen policy.4 arms×3seeds×256 updates, actual PCFI100steps/1correlation/one12-layer PFN. Primary fresh64 graphs의 자기 피처 없는 query accuracy.','related':['GraphPFN'],'entries':[core,lesson]}
(repo/'findings/internal-pcfi.yaml').write_text(yaml.safe_dump(finding,allow_unicode=True,sort_keys=False,width=170),encoding='utf8')
old=json.loads((root/'runtime/previous_harness.json').read_text(encoding='utf8'));entries=[];projects=0
for path in (repo/'findings').glob('*.yaml'):
    if path.name.startswith('_'):continue
    projects+=1;entries+=(yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('entries',[])
hashes={e['id']:hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for e in entries}
assert all(hashes[k]==v for k,v in old['entries'].items())
assert all(sha(repo/p)==h for p,h in old['sources'].items())
counts=Counter(e['kind'] for e in entries)
dump={'old_records_unchanged':len(old['entries']),'old_sources_unchanged':len(old['sources']),'new_ids':[core['id'],lesson['id']],'counts':dict(counts),'projects':projects,'findings':len(entries)}
(root/'results/archive_integrity.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2),encoding='utf8')
section='## 최신 실제 PCFI 내부화 실험\n\n공개 GraphPFN에서 실제 PCFI를100회 전파+1회 상관 보정으로 구현하고, 외부 처리·내부 고정·context router·strict label-blind router를 각각3seeds/256 updates로 새 학습했다. 이전 generic MLP refiner 재실행이 아니다. Fresh64 final graphs의 uniform/clustered90/99.5%에서 자기 피처 없는 query가 primary다.\n\n| 방법 | Accuracy % | NLL |\n|---|---:|---:|\n'
for a,v in p['means'].items():section+=f"| {a} | {v['accuracy']*100:.5f} | {v['nll']:.6f} |\n"
section+='\n'+contrasts+f". 사전 primary={s['primary_gate']}, full2%p guard={s['full_guard']}, 확대 기준={s['expand']}. "+('이 구현의 확대는 보류하며, 고유 실패 원인은 미상이다. ' if not s['expand'] else '이 small-family 비교 효과는 보존하되 실제 graph/schema 전이는 별도다. ')
section+='Context 대조는 labels-only가 아니라 PFN feedback 경로 전체다. [보고서]('+prefix+'REPORT_KO.md), [계산 정의]('+prefix+'METHOD_KO.md), [구조화 판정](findings/internal-pcfi.yaml).\n\n'
for name in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
    path=repo/name;body=path.read_text(encoding='utf8');anchor='## 최신 feature-law Factor 실험';assert anchor in body
    body=body.replace(anchor,section+anchor,1)
    body=body.replace('30개 프로젝트 묶음의133항목',f'{projects}개 프로젝트 묶음의{len(entries)}항목').replace('연구 기록 133항목',f'연구 기록 {len(entries)}항목')
    body=body.replace('실패·중단39개, 보존29개, 교훈43개, 미실행22개',f"실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개")
    body=body.replace('30개 프로젝트 묶음이며 실패·중단39, 보존29, 교훈43, 미실행22항목',f"{projects}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목")
    body=body.replace('133번의 독립 실험',f'{len(entries)}번의 독립 실험').replace('[133개 전체 기록]',f'[{len(entries)}개 전체 기록]')
    body=body.replace('graphpfn_review_packet_2026-11-02_v5.zip','graphpfn_review_packet_2026-11-02_v6.zip')
    body=body.replace('v5는 feature-law Factor 최종 결과까지 포함한다.','v6는 실제 PCFI 내부화까지 포함한다. v5는 직전 feature-law Factor snapshot이다.').replace('과거 v1/v2/v3/v4 ZIP','과거 v1/v2/v3/v4/v5 ZIP')
    body=body.replace('최근 업데이트: feature-law Factor cell 교체 결과까지 포함한','최근 업데이트: 실제 PCFI 내부화 결과까지 포함한')
    path.write_text(body,encoding='utf8')
print('ARCHIVED',dump)
