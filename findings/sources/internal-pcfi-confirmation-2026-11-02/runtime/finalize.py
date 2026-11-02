"""Write result interpretation and append-only research records after all checks."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import shutil
from collections import Counter
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml

root = Path(__file__).resolve().parents[1]; repo = root.parent / 'research-harness'
def read(p): return json.loads((root / p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
s = read('results/statistics.json'); audit = read('results/audit.json'); cost = read('results/cost.json')
complete = read('results/complete.json'); p = s['subsets']['primary']; contrast = p['contrast']
go = s['continue_to_FP_FISF']
def diff(d, scale=1):
    a, lo, hi = [x * scale for x in [d['delta'], *d['ci95']]]
    return f'{a:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]'

verdict = ('고정 내부 PCFI의 정확도 이득이 새 graph bank에서도 사전 기준을 통과했다. FP/FISF 내부화 비교는 별도 통제 프로토콜로 진행할 수 있다.' if go else
    '새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다.')
report = '# 내부 고정 PCFI — 독립 graph bank 재검증\n\n'
report += verdict + '\n\n'
report += '기존 개발 bank에서 선택한 후보를 확인하는 실험이다. 기존 checkpoint 6개를 그대로 재사용했고 새 모델·학습 update는 모두 0개다. 같은 학습 seed 11/12/13과 같은 official generator family이므로 학습 seed 집단·실제 graph·미지 schema 일반화 검증은 아니다.\n\n'
report += '## Primary\n\n새 base graph 128개에서 uniform/clustered whole-node 90/99.5% missing 네 조건, 클래스당 최대 context 4개를 평가했다. 자기 피처가 없는 query의 episode 평균을 구한 뒤 네 조건과 세 고정 학습 seed를 graph 안에서 평균했다. 아래 CI는 graph 128개에 대한 paired bootstrap 5,000회이며, bank 생성 전에 단일 primary 기준과 full-feature 2%p guard를 고정했다.\n\n'
report += '| 방법 | Accuracy % | NLL | Brier |\n|---|---:|---:|---:|\n'
for arm, v in p['means'].items(): report += f"| {arm} | {100*v['accuracy']:.6f} | {v['nll']:.6f} | {v['brier']:.6f} |\n"
report += '\n내부 고정−외부 차이:\n\n'
for metric in ['accuracy', 'nll', 'brier']:
    report += '- ' + metric + (' %p' if metric == 'accuracy' else '') + ': ' + diff(contrast[metric], 100 if metric == 'accuracy' else 1) + '\n'
report += f"\nPrimary 확인={s['primary_confirmed']}, full-feature guard={s['full_guard']}, FP/FISF 후속 진행 기준={go}. Full-feature accuracy 차이는 {diff(s['subsets']['full']['contrast']['accuracy'],100)}%p다. CI가 0을 포함한 차이를 동등성이나 효과 없음으로 해석하지 않는다.\n\n"
report += '## 사전 보조 분할\n\n| 분할 | External accuracy % | Fixed accuracy % | Fixed−external %p [95% CI] |\n|---|---:|---:|---|\n'
labels = {'setting_1': 'uniform90%', 'setting_2': 'uniform99.5%', 'setting_3': 'clustered90%', 'setting_4': 'clustered99.5%', 'cell': 'cell99.5%', 'k1': 'uniform99.5%, context1/class', 'full': 'full, 전체 query'}
for key, label in labels.items():
    v = s['subsets'][key]
    report += f"| {label} | {v['means']['external']['accuracy']*100:.6f} | {v['means']['internal_fixed']['accuracy']*100:.6f} | {diff(v['contrast']['accuracy'],100)} |\n"
report += '\nFull을 제외한 표는 자기 피처 없는 query이며, 모든 query 결과는 statistics.json에 별도로 있다. 개별 분할은 다중비교 미보정 보조 결과로 primary를 대체하지 않는다.\n\n| 고정 학습 seed | External accuracy % | Fixed accuracy % | 차이 %p |\n|---|---:|---:|---:|\n'
for seed in ['11','12','13']:
    a = p['per_seed']['external'][seed]['accuracy']; b = p['per_seed']['internal_fixed'][seed]['accuracy']
    report += f'| {seed} | {a*100:.6f} | {b*100:.6f} | {(b-a)*100:+.6f} |\n'
report += '\n독립된 새 학습 seed를 뽑은 결과가 아니므로 세 행을 독립 graph 표본처럼 bootstrap하지 않았다.\n\n'
report += '보조 prior 기준은 ' + '; '.join(f"{a}: accuracy {v['accuracy']*100:.6f}%, NLL {v['nll']:.6f}" for a,v in s['secondary_priors'].items()) + '다. Context 빈도만 사용하는 기준은 동률의 argmax 규칙에 민감할 수 있다.\n\n'
report += '## 원 결과와의 관계\n\n원 bank의 fixed−external 차이는 +0.65743%p, 95% CI [−0.11984,+1.37358]였다. 네 primary 조건과 세 학습 seed에서 점추정이 모두 양수였지만 이 배치 비교는 원 context-router primary가 아닌 보조 관찰이었다. 그 관찰을 근거로 후보를 선택한 뒤 이번 새 bank의 기준을 잠갔다. 두 bank의 절대 accuracy를 빼서 개선폭으로 쓰거나, 합친 CI를 독립 재현으로 제시하지 않는다.\n\n'
report += '내부 고정은 PCFI 전파 계수가 고정된 방법이다. 전체 모델이 학습되지 않았다는 뜻이 아니다. 원 실험에서 공개 GraphPFN adapter/encoder/decoder를 각 seed당 256회 학습한 checkpoint를 사용했다. 두 방법 모두 실제 PCFI 100회·correlation 1회·PFN 12층 한 번이며, 내부는 25회씩 layer0/3/6/9에 배치한다. 원 코드·원 checkpoint·학습 bank를 수정하지 않았다.\n\n'
report += '## 검증과 비용\n\n'
report += f"원 bank replay 18건을 design lock 전에 확인했다. 새 bank는 {audit['nodes_range'][0]}~{audit['nodes_range'][1]} nodes, physical features {audit['raw_features_range'][0]}~{audit['raw_features_range'][1]}이며 raw graph fingerprint가 원 train/dev/test와 겹치지 않는다. Primary 512개 episode 중 실제 관측 cell이 0개인 {audit['zero_observation_primary_episodes']}개도 유지했다. Odd padding은 primary membership과 관측 수에서 제외했다.\n\n"
report += f"저장 예측 {audit['predictions_rescored']}건을 독립 재채점하고 source/config/bank/checkpoint/prediction SHA와 관측값 clamp, 100-step/1-correlation/12-layer 연산 수를 확인했다. 새 bank replay {len(audit['replays'])}건의 최대 확률 차이는 {max(r['max_abs_error'] for r in audit['replays']):.9g}였다. 평가 전후 모델 파라미터 해시는 같았다.\n\n"
report += '| 방법 | 네 조건 median latency의 평균 ms | 추가 peak work memory MiB 범위 |\n|---|---:|---:|\n'
for arm, v in cost['methods'].items():
    mem = [r['additional_peak_mib'] for r in v['conditions']]
    report += f"| {arm} | {v['mean_condition_median_ms']:.6f} | {min(mem):.3f}~{max(mem):.3f} |\n"
report += '\n같은 RTX3090에서 seed11/첫 graph 네 primary 조건, 5회 warmup·30회 반복의 CPU 입력→CPU 확률 시간이다. Confidence/거리 계산·PCFI·PFN을 포함하고 로드/파일 I/O를 제외했다. 총 GPU 메모리가 아닌 추가 작업 메모리다. 외부 GPU 부하는 통제하지 않았다.\n\n'
report += '## 판정 범위\n\n' + verdict + '\n\n실패·불확실한 원인이 graph 변동, 제한된 학습 bank/seed, backbone/input interface, placement 자체 중 무엇인지 분리하지 않았으므로 원인 미상으로 남긴다. 동일 bank를 보고 학습량·온도·router·전파 횟수를 변경하는 구제 실험은 하지 않는다. 관측값을 전달하는 연산의 구현 자산과 원 개발 결과는 그대로 보존한다.\n\n'
report += '실행 원본: protocol.json, confirmation.py, METHOD_KO.md, bank/manifest.json, results/design.json, results/bank_lock.json, results/statistics.json, results/audit.json, results/cost.json. Raw graph와 checkpoint 및 개별 확률은 workspace에 유지하고 공유용 source snapshot에는 넣지 않는다.\n'
(root/'REPORT_KO.md').write_text(report, encoding='utf8')

prefix = 'findings/sources/internal-pcfi-confirmation-2026-11-02/'
dest = repo / prefix; assert not dest.exists(); dest.mkdir(parents=True)
paths = list(root.glob('*.py')) + [root/p for p in ['protocol.json','METHOD_KO.md','REPORT_KO.md','bank/generator.json','bank/manifest.json']]
paths += list((root/'results').rglob('*.json')) + list((root/'logs').glob('*.log')) + list((root/'runtime').glob('*.py'))
manifest = []
for path in sorted(set(paths)):
    q = dest / path.relative_to(root); q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,q)
    manifest.append({'alias':path.relative_to(root).as_posix(),'original':str(path),'snapshot':q.relative_to(repo).as_posix(),'sha256':sha(q),'bytes':q.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'date':'2026-11-02','scope':'Frozen six-checkpoint confirmation on new128 graph bank. Zero training updates. Raw banks/checkpoints remain local.','sources':manifest},ensure_ascii=False,indent=2),encoding='utf8')
numbers = '; '.join(f"{a}:accuracy {v['accuracy']*100:.6f}%/NLL {v['nll']:.6f}/Brier {v['brier']:.6f}" for a,v in p['means'].items())
numbers += '; fixed-external accuracy%p '+diff(contrast['accuracy'],100)+', NLL '+diff(contrast['nll'])+', Brier '+diff(contrast['brier'])
refs = [prefix + q for q in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/design.json','results/statistics.json','results/audit.json','results/cost.json']]
core = {'id':'IPCFIC-P01' if go else 'IPCFIC-F01','kind':'preserved' if go else 'failure',
 'claim':'개발 bank에서 선택한 내부 고정 PCFI의 외부 PCFI 대비 정확도 이득이 추가 학습 없이 새 graph bank에서 재현된다.',
 'observation':numbers+f". Primary={s['primary_confirmed']}, full2pp guard={s['full_guard']}, continuation={go}.",
 'tags':['observation-mask','graph-structure-use','tokenization','eval-protocol','inference-cost'],
 'keywords':['fixed internal PCFI','independent graph confirmation','development selected candidate','frozen checkpoints','no new training','placement replication'],
 'source_refs':refs,'evidence':'동일 official graph/SCM family, 새128 base graphs×7settings×6 reused checkpoints=5376 predictions. 새 training updates0. Graph 단위 paired bootstrap5000회; 고정 training seed3개 평균. 원64graph bank와 합치지 않음.'}
if go:
    core['do_not_claim']='같은 고정 training checkpoints와 생성 family의 새 graph 재현이다. 실제 graph, unseen schema, 새 training-seed population, paper novelty, context router의 성공이 아니다. 후속 FP/FISF 내부화는 아직 실행하지 않았다.'
else:
    core.update(verdict=verdict,
      failure_reason='사전에 고정한 새 bank accuracy 확인 기준을 통과하지 못했다. 고유 원인은 분리하지 않아 미상이며, CI가0을 포함한 경우 효과 없음/동등성을 증명한 것으로 쓰지 않는다.',reason_status='unknown',
      excluded_explanations=['새 학습·checkpoint 선택·온도 tuning 없이 기존6개 checkpoint를 그대로 평가했다. 원 입력/연산/loader 및 저장 예측 replay와 hash를 확인했다.','원 graph와 requested/actual seed 및 raw graph fingerprint가 겹치지 않는다. Zero-source mask를 제외하거나 성공할 때까지 재추출하지 않았다.'],
      closed_scope='같은 official graph/SCM family와 training seeds11/12/13에서, 현재100-step literal PCFI의 fixed25-step interleaving을 원 checkpoint 그대로 사용한 독립128graph accuracy confirmation 및 이를 근거로 한 FP/FISF 자동 확장.',
      not_closed='IPCFI-F01은 context router의 원 판정으로 유지한다. 개발 bank의 fixed 양의 점추정은 삭제하지 않는다. 모든 내부화, native missing interface, 다른 학습/관측 연산, 실제 graph 또는 unseen schema 일반을 기각하지 않는다.',
      reopen_when='같은 배치를 반복하거나 bank를 tuning하는 대신, 현재 checkpoint/입력/연산의 한계를 바꾸는 구체적인 근거가 생기고 강한 외부 PCFI 대조와 새 training seeds 및 독립 graph 검증을 사전 설계할 수 있을 때.',
      comparison_floor='동일 public PFN 초기화/학습 budget의 외부 actual PCFI, 내부 fixed, context/strict label-blind 개발 대조. Primary missing-query accuracy와 NLL/Brier/full guard 및 CPU→CPU 비용을 분리한다.',
      recheck='현재 두 bank를 재선택에 쓰지 않고, 변경할 원리·single primary와 충분한 training-seed/graph 단위를 먼저 잠근다. 원 CI0을 equivalence로 쓰지 않는다.',
      reopen_tags=['graph-structure-use','observation-mask','tokenization'],reopen_keywords=['internal imputation inference','PCFI placement reproducibility','feature evidence bottleneck'],revisit_status='open')
lesson = {'id':'IPCFIC-L01','kind':'lesson','claim':'Context router 실패, 내부 고정의 개발 신호, 새 graph 확인, 새 training-seed 일반화를 구분한다.',
 'observation':'원64graph bank에서 fixed−external accuracy +0.65743%p CI[-0.11984,+1.37358]였고 세 seed·네 primary cell의 점추정은 모두 양수였다. 그 후보를 선택한 뒤 새128graph bank의 정확도 기준을 잠갔다. 새 결과: '+numbers+'.',
 'interpretation':'같은 bank의 여러 양의 점추정은 독립 재현을 대신하지 않는다. 반대로 원 context-router No-Go만으로 fixed 내부화를 이미 실패했다고 부를 수도 없다. 이번 결과는 기존6 checkpoints의 새 graph 검증이며 새 학습 seed 집단의 신뢰구간이 아니다. 원·새 bank의 절대 정확도를 빼거나 합친 결과를 독립 confirmation으로 부르지 않는다.',
 'tags':['eval-protocol','observation-mask','tokenization'],'keywords':['replication','graph bootstrap','fixed training seeds','candidate selection','independent confirmation'],
 'source_refs':refs,'evidence':'New128graph bank 생성 전 protocol/source/checkpoint hash lock. 원135개 finding과540개 source 파일은 그대로 유지한다.'}
finding = {'project':'Fixed internal PCFI independent-graph confirmation','date':'2026-11-02','source':prefix+'REPORT_KO.md',
 'setting':'Reused six checkpoints;0 updates; same official generator family;128 new base graphs; missing-query accuracy primary.','related':['GraphPFN'],'entries':[core,lesson]}
(repo/'findings/internal-pcfi-confirmation.yaml').write_text(yaml.safe_dump(finding,allow_unicode=True,sort_keys=False,width=170),encoding='utf8')
old = read('runtime/previous_harness.json'); entries=[]; projects=0
for path in (repo/'findings').glob('*.yaml'):
    if path.name.startswith('_'):continue
    projects+=1; entries+=(yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('entries',[])
hashes={e['id']:hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for e in entries}
assert all(hashes[k]==v for k,v in old['entries'].items())
assert all(sha(repo/p)==h for p,h in old['sources'].items())
counts=Counter(e['kind'] for e in entries)
(root/'results/archive_integrity.json').write_text(json.dumps({'old_records_unchanged':len(old['entries']),'old_sources_unchanged':len(old['sources']),'new_ids':[core['id'],lesson['id']],'counts':dict(counts),'projects':projects,'findings':len(entries)},ensure_ascii=False,indent=2),encoding='utf8')
section='## 내부 고정 PCFI의 새 graph 재검증\n\n기존 checkpoint6개를 새128graph bank에서 추가 학습 없이 확인했다. 원64graph bank를 보고 선택한 후보이며, 생성 family와 학습 seeds11/12/13은 같다.\n\n| 방법 | Missing-query accuracy % | NLL |\n|---|---:|---:|\n'
for arm,v in p['means'].items():section+=f"| {arm} | {v['accuracy']*100:.6f} | {v['nll']:.6f} |\n"
section+='\nFixed−external accuracy '+diff(contrast['accuracy'],100)+'%p. '+verdict+' NLL/Brier와 비용은 별도로 보고한다. [보고서]('+prefix+'REPORT_KO.md), [사전 프로토콜]('+prefix+'protocol.json), [구조화 판정](findings/internal-pcfi-confirmation.yaml).\n\n'
for name in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
    path=repo/name;body=path.read_text(encoding='utf8');anchor='## 최신 실제 PCFI 내부화 실험';assert anchor in body
    body=body.replace(anchor,section+anchor,1)
    body=body.replace('31개 프로젝트 묶음의135항목',f'{projects}개 프로젝트 묶음의{len(entries)}항목').replace('연구 기록 135항목',f'연구 기록 {len(entries)}항목')
    body=body.replace('실패·중단40개, 보존29개, 교훈44개, 미실행22개',f"실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개")
    body=body.replace('31개 프로젝트 묶음이며 실패·중단40, 보존29, 교훈44, 미실행22항목',f"{projects}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목")
    body=body.replace('135번의 독립 실험',f'{len(entries)}번의 독립 실험').replace('[135개 전체 기록]',f'[{len(entries)}개 전체 기록]')
    body=body.replace('graphpfn_review_packet_2026-11-02_v6.zip','graphpfn_review_packet_2026-11-02_v7.zip')
    body=body.replace('v6는 실제 PCFI 내부화까지 포함한다.','v7는 내부 고정 PCFI의 독립 graph 재검증까지 포함한다. v6는 직전 내부화 학습 snapshot이다.').replace('과거 v1/v2/v3/v4/v5 ZIP','과거 v1/v2/v3/v4/v5/v6 ZIP')
    body=body.replace('최근 업데이트: 실제 PCFI 내부화 결과까지 포함한','최근 업데이트: 내부 고정 PCFI 재검증까지 포함한')
    if name=='RESEARCH_HANDOFF_KO.md':
        oldprompt='모든 내부화 방법을 기각한 결과는 아니야.'
        extra=f" 이어 기존 external/fixed checkpoint6개를 새128graph에서 추가 학습 없이 검증했어. Fixed−external은 {diff(contrast['accuracy'],100)}%p였어. "+verdict
        assert oldprompt in body;body=body.replace(oldprompt,oldprompt+extra,1)
        body=body.replace('1. [최신 실제 PCFI 내부화 보고서]', '1. [내부 고정 PCFI 재검증]('+prefix+'REPORT_KO.md)을 먼저 확인하고, [직전 실제 PCFI 내부화 보고서]',1)
    path.write_text(body,encoding='utf8')
path=repo/'README.md';body=path.read_text(encoding='utf8')
body=body.replace('실데이터, 최근 여섯 학습 실험, 이전 시도·중단 범위, 보존한 연산, 확인된 병목과 미실행을 구분했다.','실데이터, 최근 여섯 학습 실험과 내부 고정 PCFI의 추가 학습 없는 독립 graph 재검증, 이전 시도·중단 범위를 구분했다.')
path.write_text(body,encoding='utf8')
path=repo/'tools/build_review_packet.py';body=path.read_text(encoding='utf8');assert 'VERSION = 6' in body
path.write_text(body.replace('VERSION = 6','VERSION = 7',1),encoding='utf8')
print('REPORT_AND_ARCHIVE_READY',core['id'],counts,flush=True)
