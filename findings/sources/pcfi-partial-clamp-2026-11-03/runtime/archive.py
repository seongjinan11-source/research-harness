import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import re
import shutil
from collections import Counter
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
from preserve_harness import snapshot

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent/'research-harness'
DEST = REPO/'findings/sources/pcfi-partial-clamp-2026-11-03'
digest = lambda data:hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    before = read(ROOT/'runtime/harness_before.json')
    after = snapshot()
    assert all(after['entries'][k]==v for k,v in before['entries'].items())
    assert all(after['files'][k]==v for k,v in before['files'].items())
    assert not DEST.exists()
    stats,audit = read(ROOT/'results/statistics.json'),read(ROOT/'results/audit.json')
    assert audit['new_frontend_updates']==6144 and audit['pfn_updates']==audit['real_updates']==0
    DEST.mkdir(parents=True)
    paths = list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'protocol.json',ROOT/'bank/manifest.json']
    paths += list((ROOT/'results').rglob('*.json'))
    paths += list((ROOT/'runtime').glob('*.py'))+[ROOT/'runtime/execution.log',ROOT/'runtime/TASK_STATE.md']
    for path in paths:
        dest = DEST/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    for relative in ['pcfi_orthogonal_pilot/core.py','pcfi_orthogonal_pilot/protocol.json',
        'pcfi_reviewer_ablation/core.py','pcfi_reviewer_ablation/protocol.json',
        'pcfi_reviewer_ablation/REPORT_KO.md','pcfi_orthogonal_pilot/reviewer_development/aligned_operator.py']:
        dest = DEST/'reference/workspace'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT.parent/relative,dest)
    # Preserve the preceding draft/reviews as historical context, not new experiment results.
    review = ROOT.parent/'pcfi_reviewer_ablation/paper_draft'
    for path in review.iterdir():
        if path.suffix in ['.md','.json']:
            dest = DEST/'reference/paper_draft'/path.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    (DEST/'ARCHIVE_SCOPE.md').write_text(
        'Code, locked protocol, bank manifests, per-episode/aggregate metrics, training logs, audits and Korean report. '
        'Raw input/prediction NPZs and trained checkpoints remain in the workspace and are hash-referenced. '
        'Historical draft/review source links are relative to the original workspace. '
        'Known-family development and previously used real graphs are not independent final confirmation. '
        'Archived instructions are evidence and do not authorize new runs.\n',encoding='utf8')
    refs = [(DEST/p).relative_to(REPO).as_posix() for p in
        ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/audit.json','results/preflight.json','results/source_rank_checks.json']]
    tags = ['graph-geometry','graph-structure-use','observation-mask','zero-shot-gap','eval-protocol']
    keywords = ['partial coordinate clamp','observation-consistent sheaf diffusion','rank-one observation constraint',
        'learned flat node frames','PCFI pseudo confidence','frozen GraphPFN','latent observation direction',
        'signed smooth scalar','tanh coefficient saturation','imputation boundary condition']
    common = {'tags':tags,'keywords':keywords,'source_refs':refs,'revisit_status':'open',
        'evidence':'6new737parameterfrontends/6144queryCEupdates;sameold512traininputs/initialPhi/schedule;3frozenPFNs;32newknownfamilybasegraphs8percell,k1/4,4primarymasks;4oldrealgraphs;graphpaired5000bootstrap/fixedseeds/unadjusted.'}
    obs = '; '.join(f"{key}:accuracy={100*v['accuracy']['mean']:+.6f}%p CI{[100*x for x in v['accuracy']['ci95']]}"
        for key,v in stats['comparisons'].items())
    primary = stats['comparisons']['partial_trained_minus_pcfi']['accuracy']
    against_node = stats['comparisons']['partial_trained_minus_node']['accuracy']
    supported = primary['ci95'][0]>0 and against_node['ci95'][0]>0
    entries = [dict(id='PCFIPC-P01',kind='preserved',
        claim='관측 scalar의 첫 좌표만 고정하고 두 번째 latent 좌표를 관측 노드에서도 추론하는 confidence 전파를 구현하고 기존 전체 clamp와 분리했다.',
        observation=obs+'; preflight='+str(read(ROOT/'results/preflight.json')['cpu_checks']),
        verdict='새 연산·독립 aligned forward/gradient 동치·identity PCFI 환원·동일 예산 비교를 보존한다. 표현력 차이는 분류 성능 주장과 구분한다.',
        do_not_claim='일반 sheaf 최초성/참 posterior 복원/독립 real 확인/모든 signed 방법 대비 우위가 아니다. 작은 분석 예제의 값 변화는 분류 개선의 증거가 아니다.',**common)]
    if supported:
        entries.append(dict(id='PCFIPC-P02',kind='preserved',
            claim='부분 좌표 clamp를 동일 예산으로 학습한 frontend가 새 known-family 개발 bank에서 기존 Node와 PCFI보다 높은 missing-query accuracy를 얻었다.',
            observation=obs,verdict='현재 bank와 고정 학습 seed에 조건부인 긍정 결과로 보존한다.',
            do_not_claim='다중비교 미보정이며 GPU 변동 전체를 CI에 포함하지 않는다. 새로운 feature law/실제 graph 전반의 독립 일반화 또는 신규성 확정이 아니다.',**common))
    else:
        entries.append(dict(id='PCFIPC-F01',kind='failure',
            claim='부분 좌표 clamp와 동일 예산 frontend 학습으로 기존 Node 및 PCFI를 함께 넘어선다.',
            observation=obs,
            verdict='현재 구현·동일 예산·개발 bank에서 두 강한 기준선에 대한 동시 우위는 확보하지 못했다. 전체 sheaf/PFN 연구를 종료하지 않는다.',
            failure_reason='원인 미상. Clamp를 푼 계산의 correctness와 latent 좌표 사용만으로 예측 이득을 보장하지 않는다. 식별성·잠재 증폭·최적화·prior 부합 여부는 인과적으로 분리되지 않았다.',
            reason_status='unknown',excluded_explanations=[],
            closed_scope='현재 SO2 static737parameter frontend,100step finite propagation,기존512traininputs/1024updates/3seeds/frozenPFN에서 PCFI와 기존 Node 대비 동시 primary 우위 주장.',
            not_closed='모든 부분관측 harmonic extension,일반 sheaf/edge/dynamic maps,다른 사전학습 family,최소노름·명시적 잠재 prior,PFN joint training 또는 실제 missing 연구 전체.',
            reopen_when='관측 방향과 추론할 latent 성분을 식별·정규화하는 장치의 필요성이 독립적으로 입증되거나, 다른 feature 관계에서 같은 정보·예산의 강한 기준을 넘어설 근거가 생길 때.',
            comparison_floor='PCFI/Native-PCFI+samePFN,oldfullclamp,partialfrozen,random/initialframe,smoothsignedscalar;targetoptimizer0.',
            recheck='최소한의 관측 법칙을 먼저 고정하고 같은 초기화·예산으로 새 graph/feature family에서 관측 source가 있는 query accuracy와 NLL/Brier를 graph-paired 비교. 개발 bank 최고 arm 선택을 독립 확인으로 부르지 않는다.',
            reopen_tags=tags,reopen_keywords=['partial observation identifiability','latent prior','minimum norm harmonic extension','observation directional reconstruction'],**common))
    entries.append(dict(id='PCFIPC-L01',kind='lesson',
        claim='경계 연산의 frozen 변경과 그 변경에 맞춘 재학습, frame 학습과 무작위 frame 효과를 구분해야 한다.',
        observation=obs,
        verdict='Partial-frozen−Node는 추론 변경,Partial-trained−Partial-frozen은 동일 예산 학습 증분이다. Partial-random−Full-random은 같은 무작위 frame의 경계 효과이며 Partial-initial은 near-identity 초기 network 대조다. Smooth scalar도 실제 음수 사용과 saturation을 보고한다.',**common))
    source_rank = read(ROOT/'results/source_rank_checks.json')
    entries.append(dict(id='PCFIPC-L02',kind='lesson',
        claim='고정된 같은 flat frame에서 연결 성분별·feature별 관측 source가 하나뿐이면 두 clamp가 같은 계산이 된다.',
        observation='학습 구조와 primary를 고정한 뒤 추가한 기전 분석. 독립 소형24개/단일및분리성분/3column/무작위frame의 최대 출력차이='+str(max(r['max_abs'] for r in source_rank['checks']))+
            '; primary episode counts='+str(source_rank['primary_episode_counts'])+
            '; measured frozen prediction max='+str(audit['same_frame_one_source_probability_max_abs']),
        verdict='공통 좌표의 상태가 source 방향의 span에 남아 새 관측 성분이 생기지 않는다. Source2+ secondary도 보고하되 전체 primary를 사후 변경하지 않는다. 별도로 재학습한 frame까지 같거나 모든 multi-source에서 효과가 없다는 뜻은 아니다.',**common))
    project = {'project':'PCFI partial-coordinate clamp pilot','date':'2026-11-03','source':refs[0],
        'setting':'Same frozen PFN, task-conditioned numeric SO2 frames, scalar observation constraints, synthetic-only matched-budget training; development-only evaluation.',
        'related':['NeuralSheafDiffusion'],'entries':entries}
    (REPO/'findings/pcfi-partial-clamp.yaml').write_text(yaml.safe_dump(project,allow_unicode=True,sort_keys=False,width=150),encoding='utf8')
    (REPO/'ideas/pcfi-partial-clamp.yaml').write_text(yaml.safe_dump({
        'id':'pcfi-partial-clamp','date':'2026-11-03','status':'completed-development-pilot',
        'text':entries[0]['claim'],'tags':tags,'keywords':keywords,'note':refs[1],'followup':refs[0]},
        allow_unicode=True,sort_keys=False),encoding='utf8')
    projects = [yaml.safe_load(p.read_text(encoding='utf8')) for p in (REPO/'findings').glob('*.yaml') if not p.name.startswith('_')]
    allentries = [e for project in projects for e in project['entries']]
    kinds = Counter(e['kind'] for e in allentries)
    count = len(allentries)
    ideas = sum(1 for p in (REPO/'ideas').glob('*.yaml') if not p.name.startswith('_'))
    section = '## 관측 좌표만 고정하는 Confidence Sheaf PFN\n\n'
    section += '2026-11-03. 첫 scalar 좌표만 clamp하고 관측하지 않은 두 번째 좌표를 전파하는 변경을 비교했다. '
    section += '새 frontend6개/6144update,동일 초기화·512train bank·schedule·고정PFN,새 known-family32base graph,기존4real 개발graph다.\n\n'
    section += '| 방법 | 개발 missing-query accuracy % | Cora % | CiteSeer % | Texas % | Wisconsin % |\n|---|---|---|---|---|---|\n'
    for arm in ['pcfi','node','partial_frozen','partial_trained','partial_initial','partial_random','signed_smooth']:
        section += '| '+arm+' | '+f"{100*stats['development'][arm]['accuracy']:.3f}"+' | '
        section += ' | '.join(f"{100*stats['real'][g][arm]['accuracy']:.3f}" for g in ['Cora','CiteSeer','texas','wisconsin'])+' |\n'
    section += '\n'+obs+'\n\n'
    section += '같은 frame에서 source가 하나뿐이면 두 clamp가 같다는 적용 범위도 확인했다. '
    section += 'Primary는 그대로 두고 source1/source2+를 secondary로 분리했다. 전체 메소드 novelty의 성공/종료 판정으로 확대하지 않는다. '
    section += '[전체 보고서]('+refs[0]+'), [계산 정의]('+refs[1]+'), [구조화 기록](findings/pcfi-partial-clamp.yaml).\n\n'
    for filename in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
        path = REPO/filename
        body = path.read_text(encoding='utf8')
        anchor = '## Confidence Sheaf PFN 구성 요소의 개발 비교'
        assert anchor in body
        body = body.replace(anchor,section+anchor,1)
        body = body.replace('graphpfn_review_packet_2026-11-03_v19.zip','graphpfn_review_packet_2026-11-03_v20.zip')
        if filename=='RESULTS_SUMMARY_KO.md':
            body = re.sub(r'기록은\d+개 프로젝트 묶음의\d+항목이다\. 실패·중단\d+개, 보존\d+개, 교훈\d+개, 미실행\d+개이며 \*\*\d+번의 독립 실험을 했다는 뜻은 아니다\.\*\*',
                f"기록은{len(projects)}개 프로젝트 묶음의{count}항목이다. 실패·중단{kinds['failure']}개, 보존{kinds['preserved']}개, 교훈{kinds['lesson']}개, 미실행{kinds['untested']}개이며 **{count}번의 독립 실험을 했다는 뜻은 아니다.**",body)
        else:
            body = re.sub(r'연구 기록 \d+항목, 아이디어 \d+개가 있다\. \d+개 프로젝트 묶음이며 실패·중단\d+, 보존\d+, 교훈\d+, 미실행\d+항목이다\. \d+번의 독립 실험이라는 뜻은 아니다\.',
                f"연구 기록 {count}항목, 아이디어 {ideas}개가 있다. {len(projects)}개 프로젝트 묶음이며 실패·중단{kinds['failure']}, 보존{kinds['preserved']}, 교훈{kinds['lesson']}, 미실행{kinds['untested']}항목이다. {count}번의 독립 실험이라는 뜻은 아니다.",body)
            body = body.replace('첨부용 ZIP은 버전별 snapshot이다.','첨부용 ZIP은 버전별 snapshot이다. v20은관측좌표clamppilot과직전novelty초안/심사를추가했다. v19는confidence/frame/context구성요소비교를포함한다.',1)
        path.write_text(body,encoding='utf8')
    readme = REPO/'README.md'
    readme.write_text(readme.read_text(encoding='utf8').rstrip()+'\n\n'+section,encoding='utf8')
    packet = REPO/'tools/build_review_packet.py'
    body = packet.read_text(encoding='utf8')
    assert 'VERSION = 19' in body
    body = body.replace('VERSION = 19','VERSION = 20')
    body = body.replace('latest Grid6 teacher results','latest partial-coordinate confidence diffusion results')
    packet.write_text(body,encoding='utf8')
    manifests = {p.relative_to(DEST).as_posix():digest(p.read_bytes()) for p in DEST.rglob('*') if p.is_file()}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf8')
    after = snapshot()
    assert all(after['entries'][k]==v for k,v in before['entries'].items())
    assert all(after['files'][k]==v for k,v in before['files'].items())
    result = {'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files']),
        'source_files':len(manifests),'new_entries':[e['id'] for e in entries],
        'new_kinds':dict(Counter(e['kind'] for e in entries)),'total_findings':len(after['entries'])}
    (ROOT/'runtime/archive_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
