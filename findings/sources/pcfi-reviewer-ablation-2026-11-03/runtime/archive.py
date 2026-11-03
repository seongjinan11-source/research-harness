import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,shutil,re
from collections import Counter
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent/'research-harness'
DEST=REPO/'findings/sources/pcfi-reviewer-ablation-2026-11-03'
digest=lambda data:hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def preserve():
    before=read(ROOT/'runtime/harness_before.json')
    entries={}
    for path in (REPO/'findings').glob('*.yaml'):
        if path.name.startswith('_'):continue
        for row in yaml.safe_load(path.read_text(encoding='utf8'))['entries']:
            entries[row['id']]=digest(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())
    assert all(entries[key]==value for key,value in before['entries'].items())
    assert all(digest((REPO/path).read_bytes())==value for path,value in before['files'].items())
    return {'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files'])}


def main():
    preserve()
    assert not DEST.exists()
    DEST.mkdir(parents=True)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'protocol.json',ROOT/'bank/manifest.json']
    files+=list((ROOT/'results').rglob('*.json'))
    files+=list((ROOT/'runtime').glob('*.py'))
    for path in files:
        target=DEST/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    references=['pcfi_orthogonal_pilot/core.py','pcfi_orthogonal_pilot/protocol.json',
                'nsd_token_pfn/core.py','nsd_token_pfn/bank.py','joint_imputation_pfn/common.py',
                'sheaf_pcfi_pilot/core.py','sheaf_pcfi_pilot/data.py','regret_prior_pilot/core.py',
                'fisf_graphpfn/imputation.py']
    references += ['pcfi_orthogonal_pilot/reviewer_development/'+name for name in
                   ['REVIEW_KO.md','aligned_operator.py','verify_aligned.py','development_plan.json','results/node_alignment.json']]
    for relative in references:
        path=ROOT.parent/relative
        target=DEST/'reference/workspace'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    (DEST/'ARCHIVE_SCOPE.md').write_text('Code, protocol, reports, manifests, aggregate/per-episode metrics, audits and exact-node reparameterization checks. Raw numerical banks, per-query probability NPZs and trained checkpoints remain in the workspace. Snapshot paths retain workspace-relative dependencies and historical instructions do not authorize new runs. Known-family development and previously used real inputs are not independent final confirmation.\n',encoding='utf8')
    stats=read(ROOT/'results/statistics.json')
    audit=read(ROOT/'results/audit.json')
    aligned=read(ROOT.parent/'pcfi_orthogonal_pilot/reviewer_development/results/node_alignment.json')
    refs=[(DEST/path).relative_to(REPO).as_posix() for path in
          ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/audit.json','results/preflight.json']]
    refs.append((DEST/'reference/workspace/pcfi_orthogonal_pilot/reviewer_development/REVIEW_KO.md').relative_to(REPO).as_posix())
    tags=['graph-geometry','graph-structure-use','observation-mask','zero-shot-gap','eval-protocol']
    keywords=['confidence weighted sheaf imputation','geometry-only context shuffle','label-free node frames',
              'spatial confidence factorial','learned signed scalar baseline','static flat connection',
              'aligned two-channel diffusion','frozen downstream PFN','component ablation',
              'scalar gain clipping','no negative coefficients learned']
    common={'tags':tags,'keywords':keywords,'source_refs':refs,'revisit_status':'open',
            'evidence':'9new737parameterfrontends/9216queryCEupdates;3oldPFNcheckpoints fixed;old512traininputs,exactsameinitialPhiandschedule;32newknownfamilydevelopmentgraphs8perfactorialcell,k1/4perclass,4primarymasks;4oldrealgraphs;5000basegraphpairedbootstrap/fixedseeds/unadjustedmultiplecontrasts.'}
    differences=[]
    for key,value in stats['comparisons'].items():
        score=value['accuracy']
        differences.append(f"{key}={100*score['mean']:+.6f}%p CI{[100*v for v in score['ci95']]}")
    entries=[
        dict(id='PCFIRA-P01',kind='preserved',
             claim='고정 PFN의 수치 Node-frame에서 context label 통계, 공간 confidence, 직접 signed scalar의 기여를 동일 학습 예산으로 분리했다.',
             observation='; '.join(differences)+'; signed diagnostics='+str({split:values['signed'] for split,values in stats['diagnostics'].items()})+
                 ';geometryshuffleCIincludes0;GPUstoredprobmax='+str(max(r['stored_probability_max_abs'] for r in audit['replay']))+
                 ';storedargmaxchangemax='+str(max(r['stored_argmax_change'] for r in audit['replay'])),
             verdict='구성 요소별 차이를 개발 결과로 보존하고 유효하지 않은 부품은 단순화 또는 주장 축소 대상으로 남긴다. 일부 ablation이 우수하다는 이유로 전체 메소드의 독립 검증 성공이나 종료로 바꾸지 않는다.',
             do_not_claim='Known-family새개발graph와기존real재사용이다. Scalar는음수gain0/clip약67%여서충분히최적화된signed함수족을대표한다고볼수없다. Geometryshuffle의효과가없는데label-free재학습차이만으로올바른class의미를읽었다고하지않는다. 양의interaction은uniformNode의손실도포함하며PCFI대비우위확인은아니다. 모든scalar방법/학습된진짜geometry/새featurelaw또는미사용real의일반화우위를뜻하지않는다.',**common),
        dict(id='PCFIRA-P02',kind='preserved',
             claim='Static Node-frame의 숫자 전파를 좌표 정렬된 두 scalar 채널의 confidence diffusion으로 동치 구현했다.',
             observation=f"69forward/6parametergradient/18dense-sourcekernelchecks;maxforward={max(r['aligned_max_abs'] for r in aligned['forward_checks'])};gradient={max(r['maximum_parameter_gradient_error'] for r in aligned['gradient_checks'])};frontendCPUmedians="+str({str(row['graph']):{v['backend']:v['median_ms'] for v in row['timings']} for row in aligned['frontend_cost']})+'. Prior22filesunchanged;optimizer0/PFNforward0.',
             verdict='같은 계산을 더 단순하게 구현한 정확한 재표현과 제한된 CPU frontend 비용 개선으로 보존한다.',
             do_not_claim='일반edge-incidence/dynamicSheaf의동치나새정리,추가분류정확도,전체pipeline/GPU속도우위가아니다. Sparse경로는float32전이계수를재계산하므로반올림차이가있다.',**common),
        dict(id='PCFIRA-L01',kind='lesson',
             claim='Geometry의 context 효과는 PFN에 들어가는 context label 효과와 분리해야 하며 k1 class-permutation 대칭을 고려해야 한다.',
             observation='이전shuffle은frontend와PFN의label을함께바꿨다. 이번은geometry에만assignmentshuffle하고PFN에는원label을유지했다. 별도로label통계를제거한geometry를같은초기화부터학습했다. 클래스당1context일때assignmentshuffle은class이름의순열과같을수있다.',
             verdict='k1의불변성을context무효의증거로쓰지않고k4및label-free재학습을함께본다. Frozen입력개입과제거후재학습을같은실험으로부르지않는다.',**common),
        dict(id='PCFIRA-L02',kind='lesson',
             claim='공간 confidence와 Node-frame은 factorial로 분리하고 label heterophily와 feature law도 독립적으로 다룬다.',
             observation='I/Node×weighted/uniform공간전파를비교했고confidence는descriptor와채널보정에유지했다. 기존dev2graph는모두label-homo였으며새32개개발graph는label-homo/hetero×feature-smooth/rough네cell에8개씩배정했다. 원test/realbank를독립최종검증으로재명명하지않았다.',
             verdict='공간가중치interaction을confidence전체효과로확대하지않고,라벨heterophily를모든numericfeature의부호반전과같게보지않는다. 한부품이효과없으면그부품의필요성을좁히고전체연구를자동중단하지않는다.',**common)
    ]
    project={'project':'Confidence Sheaf PFN reviewer ablations','date':'2026-11-03',
             'source':refs[0],'setting':'Same frozen PFN, numeric missing-feature frontend components, synthetic-only matched-budget training; development-only evaluation.',
             'related':['NeuralSheafDiffusion'],'entries':entries}
    (REPO/'findings/pcfi-reviewer-ablation.yaml').write_text(yaml.safe_dump(project,allow_unicode=True,sort_keys=False,width=150),encoding='utf8')
    idea={'id':'pcfi-reviewer-ablation','date':'2026-11-03','status':'completed-development-ablations',
          'text':entries[0]['claim'],'tags':tags,'keywords':keywords,'note':refs[1],'followup':refs[0]}
    (REPO/'ideas/pcfi-reviewer-ablation.yaml').write_text(yaml.safe_dump(idea,allow_unicode=True,sort_keys=False),encoding='utf8')
    projects=[];allentries=[]
    for path in (REPO/'findings').glob('*.yaml'):
        if path.name.startswith('_'):continue
        projects.append(path)
        allentries+=yaml.safe_load(path.read_text(encoding='utf8'))['entries']
    kinds=Counter(entry['kind'] for entry in allentries)
    count=len(allentries);ideas=len(list((REPO/'ideas').glob('*.yaml')))
    section='## Confidence Sheaf PFN 구성 요소의 개발 비교\n\n2026-11-03. Context의geometry전용개입,공간confidence×Nodeframe,직접signedscalar를비교했다. 기존과같은초기Phi·512train입력·1024update·고정PFN을유지했고새frontend9개를학습했다. 새32개knownfamily개발graph는label/feature네cell에8개씩배정하고k1/4를함께평가했다. 기존4real은개발재사용이다.\n\n'
    section+='| 방법 | 새 개발 accuracy % | Cora % | CiteSeer % | Texas % | Wisconsin % |\n|---|---:|---:|---:|---:|---:|\n'
    for arm in ['pcfi','uniform','node','node_nolabel','node_uniform','signed','node_geometry_shuffle']:
        section+='| '+arm+' | '+f"{100*stats['development'][arm]['accuracy']:.3f}"+' | '+' | '.join(f"{100*stats['real'][g][arm]['accuracy']:.3f}" for g in ['Cora','CiteSeer','texas','wisconsin'])+' |\n'
    section+='\n'+ '; '.join(differences)+'\n\n구성요소의효과를분리해보존하며각비교를전체프로젝트의자동종료기준으로쓰지않는다. Native-PCFI/LPfloor와수치변동을별도보고했다. StaticNode의동치두채널계산과CPUfrontend비용개선도보존한다. [전체보고서]('+refs[0]+'), [계산정의]('+refs[1]+'), [구조화기록](findings/pcfi-reviewer-ablation.yaml).\n\n'
    for filename in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
        path=REPO/filename
        body=path.read_text(encoding='utf8')
        anchor='## PCFI의 confidence 가중 공간 확산을 Sheaf로 교체'
        assert anchor in body
        body=body.replace(anchor,section+anchor,1)
        body=body.replace('graphpfn_review_packet_2026-11-03_v18.zip','graphpfn_review_packet_2026-11-03_v19.zip')
        if filename=='RESULTS_SUMMARY_KO.md':
            body=re.sub(r'기록은\d+개 프로젝트 묶음의\d+항목이다\. 실패·중단\d+개, 보존\d+개, 교훈\d+개, 미실행\d+개이며 \*\*\d+번의 독립 실험을 했다는 뜻은 아니다\.\*\*',
                f"기록은{len(projects)}개 프로젝트 묶음의{count}항목이다. 실패·중단{kinds['failure']}개, 보존{kinds['preserved']}개, 교훈{kinds['lesson']}개, 미실행{kinds['untested']}개이며 **{count}번의 독립 실험을 했다는 뜻은 아니다.**",body)
        else:
            body=body.replace('confidence 가중 Sheaf-PCFI 공간 교체 pilot까지 포함한','Confidence Sheaf PFN 구성 요소 비교까지 포함한',1)
            body=re.sub(r'연구 기록 \d+항목, 아이디어 \d+개가 있다\. \d+개 프로젝트 묶음이며 실패·중단\d+, 보존\d+, 교훈\d+, 미실행\d+항목이다\. \d+번의 독립 실험이라는 뜻은 아니다\.',
                f"연구 기록 {count}항목, 아이디어 {ideas}개가 있다. {len(projects)}개 프로젝트 묶음이며 실패·중단{kinds['failure']}, 보존{kinds['preserved']}, 교훈{kinds['lesson']}, 미실행{kinds['untested']}항목이다. {count}번의 독립 실험이라는 뜻은 아니다.",body)
        path.write_text(body,encoding='utf8')
    path=REPO/'README.md'
    path.write_text(path.read_text(encoding='utf8').rstrip()+'\n\n'+section,encoding='utf8')
    path=REPO/'tools/build_review_packet.py'
    body=path.read_text(encoding='utf8')
    assert 'VERSION = 18' in body
    path.write_text(body.replace('VERSION = 18','VERSION = 19'),encoding='utf8')
    preservation=preserve()
    dump_path=ROOT/'results/harness_preservation.json'
    dump_path.write_text(json.dumps(preservation,indent=2),encoding='utf8')
    shutil.copyfile(dump_path,DEST/'results/harness_preservation.json')
    snapshot={p.relative_to(DEST).as_posix():digest(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf8')
    print('ARCHIVED',count,ideas,len(snapshot),preservation,flush=True)


if __name__=='__main__':
    main()
