import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/'results'/name).read_text(encoding='utf8'))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    s=read('statistics.json');u=read('supplement.json');a=read('audit.json');cost=read('cost.json')
    arms=['mask','pcfi100','pcfi200','weighted','signed','sheaf'];datasets=['Cora','CiteSeer','texas','wisconsin']
    passed=s['gates']['expand']
    lines=['# Sheaf-PCFI GraphPFN — synthetic 학습·실제 graph 고정 평가',
        '', '2026-11-03. 판정: **'+('Go' if passed else '현재 구현 확대 No-Go')+'**.', '',
        '**추가 학습은 synthetic task에서만 했고 실제 graph optimizer update는0회다.** 공개 GraphPFN 초기화에서6arm×3seed를 실제 학습했다. 이 보고서는 특정2D transport·RP16·384updates pilot의 판정이며 모든 sheaf 방법의 실패를 뜻하지 않는다.', '',
        '## 주 비교: 원래의 양의 PCFI 전파를2D transport로 바꾸면 좋아지는가?', '',
        'Confidence·관측 source clamp·마지막1회 inter-channel correction은 유지했다. 학습된 공간 operator를 PFN 앞의 미분 가능한 계산에 넣고 query CE로 함께 학습했다. Router459개 파라미터를 포함한 전체 trainable container는4,012,718개이며 static arm의 미사용 front weight는 gradient를 받지 않는다. 유효 용량이 같다는 주장은 하지 않는다.', '',
        '주 지표는 자기 피처가 없는 query accuracy다. 90/99.5% whole-node missing, uniform/clustered4조건을 평균했다. 아래 실제 graph 수치는3context/mask 반복×3고정학습seed 평균이다. Context는기존 train mask에서class당최대4개이며 validation label은사용하지않았다.', '',
        table(['방법']+datasets,[[arm]+[f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms+['native_pcfi100','public_native_pcfi100','lp','negative_lp','context_prior']]), '',
        '단위는%. native_pcfi100은 **원본 고차원 피처에서PCFI100→같은고정RP16**을 사용하고 pcfi100과같은 synthetic-trained PFN으로 예측한 강한입력대조다. public_native_pcfi100은 같은wrapper·RP16·singleforward에서 public 초기가중치를 유지한 대조로, 공식 full전처리·ensemble 재현 점수는아니다. lp/negative_lp는고정100step 양/음전파이며 CLP/CPGNN의 재현이아니다. context_prior는context 빈도이며 동률이면첫class를선택해class불균형query에서매우낮을수있다.', '',
        f"Heterophily 두 graph 평균에서 sheaf−pcfi100은{100*s['real_hetero_differences']['pcfi100']:+.3f}%p, sheaf−signed는{100*s['real_hetero_differences']['signed']:+.3f}%p, sheaf−원본피처PCFI는{100*s['real_hetero_differences']['native_pcfi100']:+.3f}%p다. Homophily 두 graph 평균차이는{100*s['real_homo_difference']:+.3f}%p다. 이4개는개발용graph이며 dataset-population 유의성을주장하지않는다.", '',
        f"강한고정대조도남긴다. Feature를쓰지않는negative_lp는Texas{100*s['real']['texas']['negative_lp']['accuracy']:.2f}%,Wisconsin{100*s['real']['wisconsin']['negative_lp']['accuracy']:.2f}%였다. 현재PFN구성이이구조/context기준선도넘지못했다. 이것은signed label전파와node-feature전파가동일한연산이라는뜻이나새PFN의성과라는뜻이아니며,관측된성능격차만보존한다.", '',
        '## 새 synthetic graph에서의 결과', '',
        '64train graphs×6episodes=384task를seed당1회노출했다. 새32test basegraphs,128nodes,16features,2~7classes다. Label-SBM의homo/hetero와feature filter의+/−law를교차했다. Class수는독립RNG추출이다. 이작은bank가class별정확한factorial matching인것은아니다.', '',
        table(['방법','Accuracy %','NLL','Brier'],[[arm,f"{100*s['synthetic'][arm]['accuracy']:.3f}",f"{s['synthetic'][arm]['nll']:.5f}",f"{s['synthetic'][arm]['brier']:.5f}"] for arm in arms]), '',
        table(['Sheaf − 비교군','Accuracy 차이 %p','95% graph-paired CI'],[[arm,f"{100*s['synthetic_comparisons']['sheaf - '+arm]['accuracy']['mean']:+.3f}",'['+', '.join(f'{100*x:+.3f}' for x in s['synthetic_comparisons']['sheaf - '+arm]['accuracy']['ci95'])+']'] for arm in ['mask','pcfi100','pcfi200','weighted','signed']]), '',
        'CI는32basegraph paired bootstrap5000회다. 같은graph의4조건과3고정학습seed를먼저평균했다. 노드나seed를독립graph처럼세지않았다. PCFI100/200/weighted대비accuracy CI가0을포함하므로증분우위는미확인이다. Signed보다높다는것만으로cross-stalk메소드의성공으로판정하지않는다. Mask보다낮은결과도함께보존한다.', '',
        table(['Label mode / feature filter']+arms,[[key]+[f"{100*vals[arm]['accuracy']:.2f}" for arm in arms] for key,vals in u['synthetic_factorial'].items()]), '',
        'Feature filter 이름smooth/rough는생성계수의부호를뜻하며 달성된featureenergy와labelagreement는bank에별도기록했다. Label heterophily를피처이웃불일치와동일시하지않는다.', '',
        '학습 후 transport 진단에서는weighted/signed/sheaf 모두synthetic missing test의첫대각성분 a<0비율이0이었다. Sheaf의평균절대cross-stalk는'+f"{u['transport_diagnostics_synthetic_missing']['sheaf']['mean_abs_cross_stalk']:.5f}"+'였다. **첫좌표의직접적인scalar부호전환은관측되지않았다. 이는2D행렬의모든성분이나고유값이양수라는뜻은아니다.** 다른좌표와반복경로의signed효과까지없었다고단정하지않으며현재결과를음의전파나표준NSD원리의반증으로쓰지않는다.', '',
        '## 원본 피처에서 전파한 추가 진단', '',
        '주 비교결과를읽기전에추가범위를정했다. Seed11·첫context/mask반복·4결측조건만사용한다. 같은합성학습router를각physicalchannel에공유하고,spatial propagation을128channel씩처리한뒤전체channel의correlation을한번계산하고RP16으로압축한다. 추가학습·실제정답tuning은없다. 공간계산chunking검증오차는4.2e−7이하다. 이진단은원본피처에서학습한모델이아니며primary대체나최고조합선택에쓰지않는다.', '',
        table(['입력/방법']+datasets,[[arm]+[f"{100*u['native_first_repetition_seed11'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in ['pcfi100','native_pcfi100','sheaf','native_pcfi200','native_weighted','native_signed','native_sheaf']]), '',
        '첫반복에서native_sheaf는Texas/Wisconsin의native_pcfi100보다높지만,Cora/CiteSeer에서는일관되게높지않고Texas에서는native_signed가더높다. 이는입력좌표순서에따른개발용신호이며3seed×3반복검증이나원본피처에서의학습성과로승격하지않는다.', '',
        '## 피처 효용과 관측 source 수', '',
        '아래는seed11의3반복·4조건평균이다. 피처제거·source-row shuffle에도원래결측query집합을그대로유지했다. 관측수0/1이면shuffle이아무변화를못주는경우가있으므로분리해서해석해야한다.', '',
        table(['Graph','방법','Full 입력 %','No-feature %','Shuffled %'],[[d,arm,f"{100*u['feature_intervention_seed11'][d][arm]['full_accuracy']:.2f}",f"{100*u['feature_intervention_seed11'][d][arm]['no_features_accuracy']:.2f}",f"{100*u['feature_intervention_seed11'][d][arm]['shuffled_sources_accuracy']:.2f}"] for d in datasets for arm in ['pcfi100','sheaf']]), '',
        table(['Graph','결측조건','관측 노드 수(3반복)'],[[d,case,','.join(map(str,counts))] for d in datasets for case,counts in u['real_sources_by_case'][d].items()]), '',
        '관측수가0인episode도재추출하지않았다. 모든missing을복구할정보가있다고가정하지않는다. Reconstruction은hidden truth를예측후에만읽은RP16좌표의MSE다. 원래raw-feature공간의MSE가아니며정확도성공기준을대체하지않는다.', '',
        table(['RP16 missing MSE']+datasets,[[arm]+[f"{u['numeric_reconstruction_rp16_mse'][d][arm]:.3f}" for d in datasets] for arm in arms]), '',
        '## Full-feature reference와 확률 점수', '',
        table(['Full-feature accuracy %']+datasets,[[arm]+[f"{100*u['full_feature_reference']['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms]), '',
        'Full-feature에서는모든cell을clamp하므로동일checkpoint내부에서front의전파효과가사라진다. 모델별차이는학습경로가달랐던결과다. Missing primary와query집합도달라직접적인missing손실차이의분해로사용하지않는다.', '',
        table(['Graph','방법','NLL','Brier','Macro-F1 %'],[[d,arm,f"{s['real'][d][arm]['nll']:.5f}",f"{s['real'][d][arm]['brier']:.5f}",f"{100*s['real'][d][arm]['macro_f1']:.2f}"] for d in datasets for arm in ['pcfi100','weighted','signed','sheaf','native_pcfi100']]), '',
        '## 비용·검증·설계 교정', '',
        f"유효학습은18checkpoints/{s['training']['valid_updates']}updates,학습시간합{s['training']['valid_training_seconds']:.1f}초다(3process 병렬실행의elapsed time과다름). 초기생성기의class수–featurelaw결합을성능표판독전에발견해mask3개의1152updates를폐기하고모든arm을public초기화에서다시학습했다. 원본은별도폴더에보존했다. Smoke optimizer step8회도폐기했다. [교정기록](DESIGN_CORRECTION_KO.md)을참조한다.", '',
        table(['방법','Graph','CPU 입력→확률 median ms','p10/p90 ms','추가 GPU MiB'],[[r['arm'],r['graph'],f"{r['median_ms']:.2f}",f"{r['p10_ms']:.2f}/{r['p90_ms']:.2f}",f"{r['peak_additional_gpu_mib']:.2f}"] for r in cost['records']]), '',
        '비용은RTX3090·현재공통구현에서warmup3/repeat10으로측정했다. 매번RP16입력으로geometry·warm100을다시계산하고이후전파와PFN1회를포함한다. Mask에도불필요한공통geometry비용이포함된다. Checkpoint loading·최초RP·원본피처PCFI비용은제외되어native-PCFI와속도비교한표가아니다. 다른사용자의GPU활동은통제하지못했다.', '',
        f"Audit {a['status']}: bank{a['bank_files_hashed']}개와저장예측{a['saved_predictions_hashed']}개hash,18checkpoint,동일초기화·schedule·고정가중치,12forecast재실행을확인했다. 최대checkpoint재실행확률차이는{max(r['max_probability_difference'] for r in a['replay']):.7f},같은session반복차이는{max(r['same_session_repeat_max_difference'] for r in a['replay']):.7f}다. 초기.005재현기준은weighted/test에서.00749716으로실패했고,뒤의audit는차이를숨기지않고argmax변경과평균차이까지기록했다. .1은손상검사용상한일뿐엄격재현PASS기준이아니다. 표는원래저장예측을모두다시scoring한값이며재학습·예측교체는없다. PCFI/SciPy,identity환원,source clamp,zero-source,hidden값·querylabel poison,class-name permutation 및router gradient preflight는통과했다. 구현/정보경계검증과엄격수치재현을구분한다.", '',
        f"재실행한episode의argmax변경은최대{100*max(r['replay_argmax_change_fraction'] for r in a['replay']):.2f}%였으며,같은session반복에서도차이가관측됐다. Front수치차이는최대{max(r['front_same_session_max_difference'] for r in a['replay']):.2g}였다. 작은입력차이의증폭가능성은있지만원인을단정하지않는다. Bootstrap CI는저장된예측에조건부이며이수치변동까지포함한CI가아니다. 특히+0.13%p같은작은차이를재현된방법우위로해석하지않는다.", '',
        '## 판정과 닫는 범위', '',
        f"사전에정한gate: synthetic={s['gates']['synthetic']},real-development={s['gates']['real_development']},native-PCFI floor={s['gates']['real_native_floor']},확대={s['gates']['expand']}.", '',
        '**현재의2D symmetric-contraction Sheaf-PCFI를메인메소드로확대하지않는다.** PCFI100/200/양수weighted보다합성accuracy증분이확인되지않았고실제heterophily평균에서도개선하지못했다. 일부CiteSeer증분이나signed대비개선을전체성공으로묶지않는다. 지금결과를살리기위한loss·head·step증가는실행하지않았다.', '',
        '실패원인의고유한인과분해는미확인이다. Warm positive100,RP16,제한된transport함수족,384updates,synthetic–real차이는설정상의제한이며확인된원인이라고쓰지않는다. 이실험은원본NSD·FSD-CAP·CLP/CPGNN재현,원본피처공간합성학습,일반적sheaf표현력,heterogeneous graph의실패를입증하지않는다. 재개하려면현재한계중무엇을어떤연산으로바꾸며왜강한native-PCFI와양수weighted를넘을지를새근거로명시해야한다.', '',
        '계산정의는[METHOD_KO.md](METHOD_KO.md),원점수는[statistics.json](results/statistics.json),추가진단은[supplement.json](results/supplement.json),정보경계/재실행은[audit.json](results/audit.json)에있다.', '',
        '문헌: [PCFI](https://arxiv.org/abs/2305.16618)의confidence/채널전파와 [Neural Sheaf Diffusion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html)의학습가능transport를출발점으로삼았다. 원래논문의heterophily학습결과를이번의synthetic-only전이성과로간주하지않는다.', '']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8')
    print('REPORT WRITTEN',flush=True)
if __name__=='__main__':main()
