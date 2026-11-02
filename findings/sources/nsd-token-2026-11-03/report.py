import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/'results'/n).read_text(encoding='utf8'))
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    s=read('statistics.json');a=read('audit.json');cost=read('cost.json');records=read('records.json');controls=read('controls.json');arms=['mask','pcfi100','identity','static','dynamic'];datasets=['Cora','CiteSeer','texas','wisconsin']
    bank=json.loads((ROOT/'bank/manifest.json').read_text(encoding='utf8'))['episodes']
    counts={d:{case:[r['observed_nodes'] for r in bank if r['split']=='real' and r['graph']==d and r['case']==case] for case in ['uniform90','uniform995','cluster90','cluster995']} for d in datasets}
    contrast={}
    for ds in datasets:
        contrast[ds]={}
        for arm in ['pcfi100','dynamic']:
            original=[r['accuracy'] for r in records if r['split']=='real' and r['graph']==ds and r['seed']==11 and r['arm']==arm]
            contrast[ds][arm]={'original':float(np.mean(original))}
            for control in ['no_features','shuffle_features','shuffle_context']+(['freeze_geometry'] if arm=='dynamic' else []):
                alt=[r['original_missing']['accuracy'] for r in controls if r['graph']==ds and r['arm']==arm and r['control']==control]
                contrast[ds][arm][control]=float(np.mean(alt))
    (ROOT/'results/control_summary.json').write_text(json.dumps(contrast,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# NSD token GraphPFN — synthetic 학습·실제 고정 평가','',
        '2026-11-03. 판정: **'+('제한된 Go, 추가 독립 검증 필요' if s['gates']['expand'] else '현재 pilot 확대 기준 미통과')+'**.','',
        '관측값·mask·context에서 바로 시작하고 PFN hidden state와 restriction map을 함께 갱신하는 모델을 실제 학습했다. PCFI로 먼저 채우거나 scalar completion으로 되돌리지 않는다. 추가 학습은 synthetic query CE만 사용했고 실제 graph optimizer update는0회다. 이 구성은 NSD의 연산을 PFN에 넣은 SO(2) adapter이며, 원본 NSD의 전체 아키텍처·target-graph 학습 재현이나 새로운 원리라고 주장하지 않는다.','',
        f"핵심결과:synthetic dynamic-PCFI는{100*s['comparisons_dynamic_minus']['pcfi100']['accuracy']['mean']:+.3f}%p,95%CI{[round(100*x,3) for x in s['comparisons_dynamic_minus']['pcfi100']['accuracy']['ci95']]}다. 그러나identity대비{100*s['comparisons_dynamic_minus']['identity']['accuracy']['mean']:+.3f}%p/static대비{100*s['comparisons_dynamic_minus']['static']['accuracy']['mean']:+.3f}%p의CI는0을포함한다. 실제heterophily평균dynamic-PCFI는{100*s['real_hetero_differences']['pcfi100']:+.3f}%p다. 따라서합성에서의개선과learned/refreshedgeometry의독립가치를구분한다.",'',
        '## 계산과 비교','',
        'PFN의192차원 feature/label token을2×96으로 해석한다.0/3/6/9층에서 ordered endpoint restriction F_i,e/F_j,e를 만들고, weighted augmented-normalized sheaf Laplacian과 학습된 좌우 mixing·ELU residual update를 적용한다. Dynamic은 현재 token에서 map을4번 계산하고 Static은 첫 map을4층에서 재사용한다. Identity는 map만I다. 세 모델의 학습 가능한 구조는 같으며, 관측 feature-pair token/context label token은 NSD 단계에서 직전 상태로 고정한다. 일반 PFN 층은 이후 latent state를 갱신할 수 있다.','',
        'Mask/PCFI100도 같은 public initialization·동일 common modules·synthetic task 순서·1,024updates로 새로 학습했다. Plain arm의 미사용 NSD container가 있다는 사실을 동일 유효 용량으로 해석하지 않는다. **이전384update pilot과 절대 점수를 빼서 구조의 효과라고 쓰지 않는다.**','',
        '64개 기존 corrected train graph에8조건을 부여한512episode를2pass 사용했다. 원래6조건의 관측값·mask·context·query 배열은 그대로 보존했고, PCFI cache는 원래100step+channel correction을 완료했다. Source2/8을 추가한 것은 informative geometry 조건도 학습하기 위한 사전 결정이며, 원래의 zero-source99.5% 조건을 제거하지 않았다.','',
        '## 새로운 synthetic graph: primary','',
        '새64base graph,128nodes,16features,2~7classes,기존과같은 label-SBM×feature-filter family다. Primary는90/99.5%·uniform/cluster4조건의 자기 피처가 없는 query accuracy다. 조건·3고정학습seed를graph내평균하고base graph paired bootstrap5000회를계산했다.','',
        table(['방법','Accuracy %','NLL','Brier'],[[arm,f"{100*s['synthetic'][arm]['accuracy']:.3f}",f"{s['synthetic'][arm]['nll']:.5f}",f"{s['synthetic'][arm]['brier']:.5f}"] for arm in arms]),'',
        table(['Dynamic − 비교군','Accuracy 차이 %p','95% graph-paired CI'],[[arm,f"{100*s['comparisons_dynamic_minus'][arm]['accuracy']['mean']:+.3f}",'['+', '.join(f'{100*x:+.3f}' for x in s['comparisons_dynamic_minus'][arm]['accuracy']['ci95'])+']'] for arm in ['mask','pcfi100','identity','static']]),'',
        '## 실제 graph: 개발용 고정 평가','',
        '이전4dataset/3context-mask 반복을 재사용했다. 기존 train mask에서class당최대4개context를선택하고공식test mask에서평가한다. 나머지label과validationlabel은입력/선택/학습에사용하지않았다. 모든학습된arm은3고정trainingseed평균이다. 실제graph는선정된developmentbank이며새독립realtest로부르지않는다.','',
        table(['방법']+datasets,[[arm]+[f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms+['native_pcfi100','public_native_pcfi100','lp','negative_lp','context_prior']]),'',
        '단위%. 주 입력은 fixed RP16·관측값만의RMS/clip이다. Native PCFI는원본physicalfeature에서전파한뒤같은RP16을적용해새pcfi100checkpoint로예측했다. Native는전파때원본고차원관측정보를사용하고주모델은미리압축된RP16을받으므로동일표현정보의geometry-only대조가아니라강한전체pipelinefloor다. Public은동일wrapper의singleforward초기가중치대조이며공식full전처리/ensemble재현이아니다. 고정positive/negativeLP는동일입력의이전확률artifact를hash확인해재사용했고CLP/CPGNN의재현이아니다. Context prior동률은첫class선택으로낮은accuracy가발생할수있다.','',
        table(['Heterophily mean: Dynamic − 비교군','%p'],[[arm,f"{100*value:+.3f}"] for arm,value in s['real_hetero_differences'].items()]),'',
        f"Homophily mean dynamic−pcfi100은{100*s['real_homo_difference']:+.3f}%p다.4dataset의노드를독립표본처럼bootstrap하거나전체graph모집단의유의성으로확대하지않는다.",'',
        '## 실제로 geometry를 갱신했는가?','',
        table(['방법/평가','stage별 cos-negative 비율','stage별 첫map과의 차이','최대anchor변경'],[[arm+'/'+split,','.join(f'{x:.4f}' for x in stats['negative_cos_fraction_by_stage']),','.join(f'{x:.4f}' for x in stats['refresh_difference_from_first_by_stage']),f"{stats['max_anchor_change']:.2g}"] for arm,modes in s['geometry'].items() for split,stats in modes.items()]),'',
        '여기서negative는SO(2)transport의cos항(T00=T11)이음수인비율이다. 행렬모든성분이음수라는뜻도,그러한상태가정답geometry라는뜻도아니다. Refresh차이는첫단계map과의차이다. Map이달라졌다는사실은갱신연산이작동했다는검증이며그갱신의예측적유용성은static대비성능으로판단해야한다.','',
        '## 고정 source 수와 no-source: secondary','',
        table(['조건']+arms,[[case]+[f"{100*s['synthetic_by_case'][case][arm]['accuracy']:.2f}" for arm in arms] for case in ['uniform90','uniform995','cluster90','cluster995','sources2','sources8']]),'',
        table(['Primary partition']+arms,[[part]+[f"{100*s['zero_source_partition'][part][arm]['accuracy']:.2f}" for arm in arms] for part in ['zero','nonzero']]),'',
        'Source2/8은별도mask/context/query설정이며다른case의절대차이를source수하나의인과효과로쓰지않는다. No-source여도label token과graph로계산이진행되지만새로운관측feature정보를만들었다고해석하지않는다. Primary를secondary로바꾸거나좋은case만고르지않는다.','',
        table(['실제graph','관측조건','관측노드수(3반복)'],[[d,case,','.join(map(str,v))] for d in datasets for case,v in counts[d].items()]),'',
        '관측수가0인경우도유지했다. 위횟수는3개의mask/context반복이며학습seed3개로복제해독립관측수로세지않는다.','',
        '## Feature/context 개입: seed11 진단','',
        table(['Graph','방법','원래입력 %','Feature제거 %','Feature셔플 %','Context셔플 %','고정map전환 %'],[[d,arm,f"{100*contrast[d][arm]['original']:.2f}",f"{100*contrast[d][arm]['no_features']:.2f}",f"{100*contrast[d][arm]['shuffle_features']:.2f}",f"{100*contrast[d][arm]['shuffle_context']:.2f}",f"{100*contrast[d][arm]['freeze_geometry']:.2f}" if arm=='dynamic' else '—'] for d in datasets for arm in ['pcfi100','dynamic']]),'',
        '각개입은원래missingquery집합을고정한다. Feature셔플은관측source내순열,context셔플은제공된label의대응만섞는다. No-feature에서PCFIcache도재계산/제거해잔존정보를차단했다. 고정map전환은같은dynamiccheckpoint의추론시freeze진단이며추가학습된static모델과구분한다. 가장좋은개입을새방법으로선택하지않았다.','',
        '## Full-feature와 확률 품질','',
        table(['Full-feature accuracy %','Synthetic']+datasets,[[arm,f"{100*s['full_feature']['test']['all'][arm]['accuracy']:.2f}"]+[f"{100*s['full_feature']['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms]),'',
        'Full-feature는all-queryreference로missingprimary와query집합이다르다. Dynamic은fullfeature에서도querylabeltoken을갱신하므로원래scalarimputer실험과다르다. Syntheticfull과realfull의차이는전이의단서이며특정원인의증명은아니다.','',
        table(['Graph','방법','NLL','Brier','Macro-F1 %'],[[d,arm,f"{s['real'][d][arm]['nll']:.5f}",f"{s['real'][d][arm]['brier']:.5f}",f"{100*s['real'][d][arm]['macro_f1']:.2f}"] for d in datasets for arm in ['pcfi100','identity','static','dynamic','native_pcfi100','negative_lp']]),'',
        '## 비용과 검증','',
        f"총{s['training']['checkpoints']}checkpoints,{s['training']['valid_updates']}validupdates,학습시간합{s['training']['training_seconds_sum']:.1f}초다.3process병렬실행의벽시계시간과다르다. Devsmoke5steps는전부폐기했다. 실제학습0회다.",'',
        table(['방법','Graph','median ms','p10/p90 ms','추가GPU MiB'],[[r['arm'],r['graph'],f"{r['median_ms']:.2f}",f"{r['p10_ms']:.2f}/{r['p90_ms']:.2f}",f"{r['peak_additional_gpu_mib']:.2f}"] for r in cost['records']]),'',
        'CPU의visibleRP16입력에서CPU확률까지측정했다. PCFI는spatial100+channelcorrection을다시계산하고NSD는confidence/map/hiddenupdate를포함한다. Checkpointload·최초RP·원본physicalfeaturePCFI는제외한다. Warmup3/repeat10이며다른사용자GPU활동은통제하지못했다.','',
        f"Audit상태:{a['status']}. 입력{a['input_files_hashed']}개와예측{a['prediction_files_hashed']}개hash,15checkpoint와동일초기화/schedule/고정가중치를검사했다.10selectedforecast재실행최대확률차이는{max(r['max_probability_difference'] for r in a['replay']):.7f},같은session반복차이는{max(r['within_session_max_difference'] for r in a['replay']):.7f},최대argmax변경은{100*max(r['argmax_change_fraction'] for r in a['replay']):.2f}%다. .1assertion은손상검사용일뿐엄격재현기준이아니다. CI는저장예측에조건부이며이수치변동전체를포함하지않는다.",'',
        '명시적denseincidence일치·LaplacianPSD/spectrum·nodepermutation·공급된map하의gaugeidentity·mapgradientfinite-difference·negativecos표현가능성을검증했다. Gauge검증은학습mapencoder전체의gaugeequivariance주장이아니다. Anchor고정·static/dynamicrefresh·미사용PCFIcachepoison·mapgradientsmoke도통과했다.','',
        '## 판정 범위','',f"사전기준: {json.dumps(s['gates'],ensure_ascii=False)}",'',
        'PCFI대비이득,추가hidden diffusion효과(identity),geometry효과(static),갱신효과(dynamic)를별도로읽는다. 일부accuracy/NLL/source조건의개선만으로모든gate성공으로묶지않는다. 기준미통과시새loss/head/추가update로현재구성을구제하지않는다. 결과의고유한실패원인은이비교만으로확정할수없다.','',
        '새syntheticgraph도동일knownfamily이며실제graph는RP16의기존developmentbank다. 원본NSD나모든sheaf의실패/성공,unseenheterogeneousschema일반화,완전한missingmarginalization을입증하지않는다. [METHOD_KO.md](METHOD_KO.md),[원점수](results/statistics.json),[audit](results/audit.json),[개입](results/control_summary.json)을함께보면된다.','']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8');print('REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
