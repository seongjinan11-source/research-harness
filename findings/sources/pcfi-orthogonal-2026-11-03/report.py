import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/'results'/n).read_text(encoding='utf8'))
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    s=read('statistics.json');a=read('audit.json');p=read('preflight.json');cost=read('cost.json');records=read('records.json');controls=read('controls.json')
    arms=['pcfi','scalar','edge','node'];datasets=['Cora','CiteSeer','texas','wisconsin'];cs={}
    for d in datasets:
        cs[d]={}
        for arm in ['pcfi','edge','node']:
            cs[d][arm]={'original':float(np.mean([r['accuracy'] for r in records if r['split']=='real' and r['graph']==d and r['arm']==arm and r['seed']==11]))}
            for control in ['no_features','shuffle_features','shuffle_context']+(['identity_transport','drop_cross_stalk'] if arm!='pcfi' else []):
                cs[d][arm][control]=float(np.mean([r['original_missing']['accuracy'] for r in controls if r['graph']==d and r['arm']==arm and r['control']==control]))
    (ROOT/'results/control_summary.json').write_text(json.dumps(cs,ensure_ascii=False,indent=2),encoding='utf8')
    go=[arm for arm in ['edge','node'] if s['gates'][arm]['expand']]
    lines=['# Confidence 가중 PCFI 공간 전파의 직교 Sheaf 교체 결과','',
           '2026-11-03. 판정: **'+('제한된 확대 기준 통과: '+', '.join(go) if go else '현재 구성의 확대 기준 미통과')+'**.','',
           '관측 숫자 피처를 confidence로 가중한 graph sheaf diffusion으로 결측값을 추정하고, 같은 PFN에 입력했다. PCFI의 거리 confidence·관측값 고정·self-loop·100회 전파·마지막1회 채널 상관 보정을 유지했다. Hidden token을 확산하거나 PCFI100 뒤에 Sheaf100을 더한 실험과 구분한다.','',
           '## 계산과 학습 범위','',
           '관측 cell을(x,0), 결측 cell을(0,0)으로 놓고 z_i←sum_j P_ij T_ij z_j를100회 계산한다. 관측값은 매번(x,0)으로 고정한다. Edge는 ordered endpoint로부터 T_ij=R(theta_j,e−theta_i,e)를, Node는 공유 frame으로부터 T_ij=Q_i^T Q_j를 계산한다. Scalar는 같은 edge angle의 cos 계수만 사용하고 cross-stalk를 제거한다.','',
           '각 seed의 기존 synthetic-trained PCFI-PFN checkpoint를 모든 arm에서 동일하게 고정했다. 새 학습 대상은737-parameter 대치 연산뿐이고 query CE gradient가 PFN을 통과한다. PFN은 학습 때도 eval mode이며 parameter update가0회다. 따라서 이번 결과는 고정된 분류기에서 전파 연산을 바꾼 효과이며, joint PFN 재학습의 성과가 아니다.','',
           'Scalar/Edge/Node를 각각3seed·1,024updates로 학습했다. 새 frontend9개/9,216updates이며 모든 arm의 공통 PFN 사전 학습 이력은 별도다. 같은512개 synthetic train episode와 순서를 썼다. 새로운 teacher·auxiliary loss·실제 graph 학습은 없다.','',
           'Map descriptor는 관측값·mask·confidence·degree와 context-only 관계 통계다. 숨은 피처, query 정답, 참 feature law/homophily, PFN hidden feedback, 피처 warm-up을 사용하지 않는다. Maps는100회 동안 고정한다. 정확한I에서 첫 좌표만 읽으면 angle gradient가 사라질 수 있어 작은 비영 초기값으로 학습하고 I 환원은 별도로 검사했다.','',
           '## 새 합성 graph의 primary','',
           '동일 label-SBM×feature-filter family의 새64graph다. Uniform/clustered whole-node90/99.5%에서 자기 피처가 없는 query accuracy가 primary다. 4조건·3고정학습seed를 base graph 안에서 평균하고5,000회 graph-paired bootstrap으로 차이를 계산했다.','',
           table(['방법','Accuracy %','NLL','Brier'],[[arm,f"{100*s['synthetic'][arm]['accuracy']:.3f}",f"{s['synthetic'][arm]['nll']:.5f}",f"{s['synthetic'][arm]['brier']:.5f}"] for arm in arms]),'',
           table(['차이','Accuracy 차이 %p','95% CI'],[[candidate+' − '+other,f"{100*s['comparisons'][candidate][other]['accuracy']['mean']:+.3f}",'['+', '.join(f'{100*v:+.3f}' for v in s['comparisons'][candidate][other]['accuracy']['ci95'])+']'] for candidate in ['edge','node'] for other in ['pcfi','scalar']]),'',
           'Edge와 Node는 사전 지정한 두 후보이며 모두 보고한다. 여러 비교의 CI는 다중비교 미보정이다. CI가0을 포함한 차이를 동등성이나 효과 없음으로 바꾸지 않고, 이 bank에서 좋은 후보만 골라 독립 확인으로 부르지 않는다.','',
           '## 실제 graph의 개발용 평가','',
           table(['방법']+datasets,[[arm]+[f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms+['pcfi_numpy','native_pcfi','mask_reference','lp','negative_lp']]),'',
           '단위%. 기존4개 graph·3개 context/mask 반복을 재사용한 개발 평가다. 기존 train mask에서class당최대4개context만 제공했다. 실제 optimizer update와 query/validation label을 이용한 선택은0회다. 네 arm의 확률을 새로 계산했고 native-PCFI/LP의 이전 같은 입력 예측은 hash를 검증해 유지했다.','',
           table(['후보','Heterophily−PCFI %p','Heterophily−Scalar %p','Heterophily−Native %p','Homophily−PCFI %p'],[[arm]+[f"{100*s['real_hetero_differences'][arm][x]:+.3f}" for x in ['pcfi','scalar','native_pcfi']]+[f"{100*s['real_homo_difference'][arm]:+.3f}"] for arm in ['edge','node']]),'',
           'RP16 입력 인터페이스와 관측값만의 RMS 전처리는 기존과 같다. Native-PCFI는 원본 고차원 피처에서 전파한 뒤 같은RP16에 넣는 더 강한 전체 pipeline floor다. 이 차이를 동일표현의 geometry-only 효과로 해석하지 않는다. Negative LP는 고정된 signed label propagation이며 CPGNN/CLP 재현은 아니다.','',
           'Pcfi는 이번 동일Torch kernel의 I대조이고 pcfi_numpy는 같은 입력·같은 고정PFN의 기존 NumPy PCFI 확률이다. GPU I환원 검사에서 미세한 반복 차이가 관측되어 원CPU 경로의 실제 점수도 보조로 함께 보존했다. 이 추가는 학습이나 primary 성공 기준을 바꾸지 않는다.','',
           'Mask reference는 같은 synthetic prior에서 별도로 학습한 기존 zero-fill/mask PFN checkpoint의 pipeline 대조다. 이번 네 arm의 동일 고정 PCFI-PFN과는 checkpoint가 다르므로 공간 전파만의 효과로 해석하지 않는다.','',
           '## 결측 조건과 source 수','',
           table(['조건']+arms,[[case]+[f"{100*s['synthetic_by_case'][case][arm]['accuracy']:.2f}" for arm in arms] for case in ['uniform90','uniform995','cluster90','cluster995','sources2','sources8']]),'',
           table(['Primary source 분할']+arms,[[part]+[f"{100*s['zero_source_partition'][part][arm]['accuracy']:.2f}" for arm in arms] for part in ['zero','nonzero']]),'',
           'Source0은 그대로 유지했다. 관측 숫자가 하나도 없으면 이 연산은 새 피처를 만들지 않으며 동일 PFN의 graph/context 예측으로 환원된다. Source2/8은 secondary이며 primary를 대체하지 않는다.','',
           '## Transport와 개입','',
           table(['방법/입력','cos 음수 비율','sin 절댓값 평균','둘째 좌표 절댓값 평균','관측값 변화'],[[arm+'/'+split,f"{v['negative_cos_fraction']:.5f}",f"{v['mean_abs_sin']:.5f}",f"{v['second_coordinate_mean_abs']:.5f}",f"{v['observed_change']:.2g}"] for arm,sp in s['geometry'].items() for split,v in sp.items()]),'',
           'Cos의 부호와 cross-stalk 사용은 계산의 작동 여부를 나타내며 올바른 geometry나 정확도 우위를 증명하지 않는다. 전체 행렬의 모든 성분이 음수라는 뜻도 아니다.','',
           table(['Graph','방법','원본 %','Feature 제거 %','Feature 셔플 %','Context 셔플 %','T=I %','Cross-stalk 제거 %'],[[d,arm]+[f"{100*cs[d][arm][k]:.2f}" for k in ['original','no_features','shuffle_features','shuffle_context']]+[f"{100*cs[d][arm][k]:.2f}" if k in cs[d][arm] else '—' for k in ['identity_transport','drop_cross_stalk']] for d in datasets for arm in ['pcfi','edge','node']]),'',
           '학습 seed11 하나의 개입 진단이다. 모든 조건에서 원래 결측 query집합을 고정했다. Feature 제거는 관측값과 mask를 제거하고 geometry도 다시 계산한다. Context 셔플은 frontend와 PFN에 제공되는 label 대응을 함께 바꾼다. T=I와 cross-stalk 제거는 같은 checkpoint의 추론 조작이며 새로 학습한 후보를 대체하지 않는다.','',
           '## Full feature와 확률 품질','',
           table(['Full accuracy %','Synthetic']+datasets,[[arm,f"{100*s['full_feature']['test']['all'][arm]['accuracy']:.2f}"]+[f"{100*s['full_feature']['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms]),'',
           'Full은 전체 query이며 primary와 query집합이 다르다. 같은 checkpoint를 사용하고 관측값을 고정하므로 full 입력은 원 PCFI와 동일해야 한다. Source0 역시 동일한0 입력으로 환원되어야 하며 audit에서 저장 확률을 비교했다.','',
           table(['Graph','방법','NLL','Brier','Macro F1 %'],[[d,arm,f"{s['real'][d][arm]['nll']:.5f}",f"{s['real'][d][arm]['brier']:.5f}",f"{100*s['real'][d][arm]['macro_f1']:.2f}"] for d in datasets for arm in arms]),'',
           '## 검증과 비용','',
           f"CPU 개발 검사에서 NumPy PCFI와의 최대 출력 차이는{p['pcfi_numpy_max_abs']:.3g}, I 강제 환원 차이는0이었다. Frontend gradient와 실제 optimizer step, 관측값 고정, source0, node permutation과 역방향 변환을 확인했다. Dev smoke3steps는 폐기했다.",'',
           f"Audit: {a['status']}. 입력{a['input_files_hashed']}개와 예측{a['prediction_files_hashed']}개를 hash로 확인했다. PFN 고정과9frontend checkpoint/9,216updates 및 기존train/dev/real입력 보존을 확인했다. Full/source0 저장 확률{a['full_and_zero_source_reduction']['cases']}건 비교의 최대 차이는{a['full_and_zero_source_reduction']['max_probability_error']:.3g}다.",'',
           f"선택된 예측 재실행에서 저장 확률 대비 최대차이는{max(r['saved_max_abs'] for r in a['replay']):.7f}, 같은 session 반복 최대차이는{max(r['repeat_max_abs'] for r in a['replay']):.7f}, 최대argmax변경은{100*max(r['saved_argmax_change_fraction'] for r in a['replay']):.2f}%다. CI는 저장 예측에 조건부이며 전체 수치 변동을 포함하지 않는다. 0.1assertion은 큰 손상 검사이지 엄격 재현 기준은 아니다.",'',
           '초기 audit의 GPU I환원 bitwise equality 검사는 실패했다. 별도 probe에서 I환원 두 경로의 차이는2.38e-7~4.77e-7, 같은I경로 반복 차이는최대7.15e-7이었고 CPU bitwise 환원은 일치했다. 원 assertion 실패와 probe를 그대로 보존한다. 후속 audit은 GPU 차이를 측정값으로 보고하며 exact GPU equality 통과로 바꾸지 않았다. 모델·primary 예측·성공 기준은 변경하지 않았다.','',
           table(['구현','Graph','median ms','p10/p90 ms','추가 GPU MiB'],[[r['arm'],r['graph'],f"{r['median_ms']:.2f}",f"{r['p10_ms']:.2f}/{r['p90_ms']:.2f}",f"{r['peak_additional_gpu_mib']:.2f}"] for r in cost['records']]),'',
           '자체 GPU 작업 종료 후 직렬로3warmup·10repeat를 측정했다. CPU의 visible RP16 입력부터 CPU 확률까지이며 confidence/추가 geometry/100회전파/채널보정/PFN을 포함한다. PCFI-NumPy는 기존 CPU imputer이고 Torch PCFI에는 불필요한 LP descriptor 계산을 넣지 않았다. Model/data load·처음RP는 제외한다. 추가 memory와 전체VRAM은 다르며 다른 사용자의GPU활동은 통제하지 못했다.','',
           '## 판정 범위','',json.dumps(s['gates'],ensure_ascii=False),'',
           '판정 대상은 현재737-parameter static SO(2) 수치 전파가 고정된 synthetic-trained PFN에서 PCFI 및 scalar 전파를 넘어서는가다. 새 합성 graph도 같은knownfamily이고 실제4graph는 개발 bank다. Joint PFN 학습, 일반O(d), 원본feature공간 학습, 원NSD재현 또는 새로운 schema 일반화의 결과로 확대하지 않는다. 고유 실패 원인은 분리되지 않았다. 이번 bank에서 새loss/차원/예산을 조정해 구제하지 않는다.','',
           '[계산 정의](METHOD_KO.md), [원점수](results/statistics.json), [검증](results/audit.json), [개입](results/control_summary.json).','']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8');print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
