import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(n):return json.loads((ROOT/'results'/n).read_text(encoding='utf8'))
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    s=read('statistics.json');a=read('audit.json');cost=read('cost.json');records=read('records.json');controls=read('controls.json');pre=read('preflight.json')
    arms=['mask','pcfi100','identity','static','dynamic','node_frame'];datasets=['Cora','CiteSeer','texas','wisconsin']
    contrast={d:{'original':float(np.mean([r['accuracy'] for r in records if r['split']=='real' and r['arm']=='node_frame' and r['seed']==11 and r['graph']==d]))} for d in datasets}
    for d in datasets:
        for ctrl in ['no_features','shuffle_features','shuffle_context','freeze_frames','identity_frames']:
            contrast[d][ctrl]=float(np.mean([r['original_missing']['accuracy'] for r in controls if r['graph']==d and r['control']==ctrl]))
    (ROOT/'results/control_summary.json').write_text(json.dumps(contrast,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 노드 직교 frame GraphPFN 실험 결과','',
           '2026-11-03. 판정: **'+('제한된 확대 기준 통과' if s['gates']['expand'] else '현재 구성의 확대 기준 미통과')+'**.','',
           'Edge마다 restriction을 만드는 대신 노드마다 SO(2) frame 하나를 만들고 모든 incidence에 공유했다. 입력·전파 위치·학습 예산은 유지했다. 새 모델은3개, query CE update는3,072회이며 실제 graph 학습은0회다.','',
           '## 계산과 비교 조건','',
           'T_ij=Q_i^T Q_j로 구성해 단계별 순환 경로의 변환 곱을I로 제한한다. 노드 자신의 descriptor와 closed-neighborhood descriptor 평균을 같은72→32→1Phi에 넣는다. 기존 edge 모델은 두 endpoint의 descriptor를 넣었다. 동일 파라미터 수와 초기화는 확인했지만 두 함수족과 conditioning은 다르다.','',
           '기존 GraphPFN의 모든 feature/label token에 같은 geometry를 쓰며0/3/6/9층에서 갱신한다. Confidence·degree+1 정규화·좌우 mixer·ELU·step·관측 토큰 anchor도 유지했다. PCFI를 추가 입력하거나 라벨 전파만 분리하는 변경은 하지 않았다.','',
           '기존5개 비교군은 같은1,024update를 끝낸 checkpoint를 재사용했다. 기존64train graph·512episodes·seed별taskorder와 초기화가 정확히 같다. 새 합성64graph에서는 비교군도 모두 다시 추론했다. 실제4graph의 비교군 확률은 같은 입력의 기존 artifact다.','',
           '## 새 합성 graph의 primary 결과','',
           '동일 known family의 새64base graph다. Uniform/clustered90/99.5% whole-node missing의 결측 query accuracy를 평가했다. 4조건·3고정학습seed를 graph 안에서 평균하고 graph-paired bootstrap5,000회를 계산했다.','',
           table(['모델','Accuracy %','NLL','Brier'],[[arm,f"{100*s['synthetic'][arm]['accuracy']:.3f}",f"{s['synthetic'][arm]['nll']:.5f}",f"{s['synthetic'][arm]['brier']:.5f}"] for arm in arms]),'',
           table(['Node-frame − 비교군','Accuracy 차이 %p','95% CI'],[[arm,f"{100*s['comparisons_node_minus'][arm]['accuracy']['mean']:+.3f}",'['+', '.join(f'{100*v:+.3f}' for v in s['comparisons_node_minus'][arm]['accuracy']['ci95'])+']'] for arm in ['identity','dynamic','static','mask','pcfi100']]),'',
           'CI가0을 포함한 비교를 동등성이나 효과 없음으로 해석하지 않는다. 이전 test bank의 절대 accuracy와 이번 accuracy를 빼서 모델 개선으로 쓰지 않는다.','',
           '## 실제 graph의 개발용 평가','',
           table(['모델']+datasets,[[arm]+[f"{100*s['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms+['native_pcfi100','public_native_pcfi100','lp','negative_lp']]),'',
           '단위%. 기존4graph·3context/mask반복을 사용했다. 가중치는 고정이고 query/validation labels로 선택하지 않았다. Node-frame만 새로 실제 추론했으며 비교군은 hash 검증된 기존 확률이다. 실제 graph 수가4개이므로 노드 bootstrap으로 임의 graph 모집단의 유의성을 주장하지 않는다.','',
           f"Heterophily 평균 Node-frame−PCFI는{100*s['real_hetero_differences']['pcfi100']:+.3f}%p, −Dynamic은{100*s['real_hetero_differences']['dynamic']:+.3f}%p다. Homophily 평균 Node-frame−PCFI는{100*s['real_homo_difference']:+.3f}%p다.",'',
           'Native-PCFI는 원본 고차원 공간에서 전파한 뒤 같은RP16으로 압축한다. 주모델은 먼저RP16으로 압축된 관측값을 받으므로 native 비교는 동일표현의 geometry-only 효과가 아니라 전체 pipeline floor다. Negative LP는 고정 signed LP이며 CPGNN/CLP 재현은 아니다.','',
           '## Source 수와 결측 조건별 보조 결과','',
           table(['조건']+arms,[[case]+[f"{100*s['synthetic_by_case'][case][arm]['accuracy']:.2f}" for arm in arms] for case in ['uniform90','uniform995','cluster90','cluster995','sources2','sources8']]),'',
           table(['Primary source 분할']+arms,[[part]+[f"{100*s['zero_source_partition'][part][arm]['accuracy']:.2f}" for arm in arms] for part in ['zero','nonzero']]),'',
           'Source2/8은 별도 mask/context/query 조건이며 primary를 대체하지 않는다. 관측 피처가0인 episode도 유지했고 graph/context로 예측하는 것을 새로운 피처 정보를 복원한 것으로 해석하지 않는다.','',
           '## Frame이 실제로 작동했는가','',
           table(['모델/평가','stage별 cos-negative 비율','첫 map과의 차이','anchor 최대변경'],[[arm+'/'+split,', '.join(f'{x:.4f}' for x in d['negative_cos_fraction_by_stage']),', '.join(f'{x:.4f}' for x in d['refresh_from_first_by_stage']),d['max_anchor_change']] for arm in ['identity','dynamic','node_frame'] for split,d in s['geometry'][arm].items()]),'',
           'Negative는 SO(2) transport의 cos항이 음수인 비율이며 행렬 전체의 부호나 올바른 geometry를 의미하지 않는다. 위 통계는 full을 제외한 secondary source조건까지 포함한다.','',
           f"구현 검사에서 선형 공통좌표 항등식 오차는{pre['aligned_linear_operator_error']:.3g}, 학습 map의 triangle cycle 오차는{pre['learned_triangle_cycle_error']:.3g}, 공유 incidence 오차는{pre['shared_incidence_error']:.3g}였다. 이는 고정 frame 연산의 검사이며 전체 비선형 PFN의 gauge-equivariance나 성능 보장은 아니다.",'',
           '## 동일 checkpoint의 개입 결과','',
           table(['Graph','원본 %','Feature 제거 %','Feature 셔플 %','Context 셔플 %','Frame 고정 %','Frame I %'],[[d]+[f"{100*contrast[d][k]:.2f}" for k in ['original','no_features','shuffle_features','shuffle_context','freeze_frames','identity_frames']] for d in datasets]),'',
           '학습 seed11 하나의 진단이다. 모든 개입에서 원래 결측 query집합을 고정했다. Frame 고정은 같은 checkpoint에서 첫 단계 map을 재사용하는 조작이고, 별도로 학습한 static 모델과 다르다. 좋은 개입을 골라 새 최종 방법으로 사용하지 않았다.','',
           '## 확률 품질과 full feature','',
           table(['Graph','모델','NLL','Brier','Macro F1 %'],[[d,arm,f"{s['real'][d][arm]['nll']:.5f}",f"{s['real'][d][arm]['brier']:.5f}",f"{100*s['real'][d][arm]['macro_f1']:.2f}"] for d in datasets for arm in ['pcfi100','identity','dynamic','node_frame']]),'',
           table(['Full accuracy %','Synthetic']+datasets,[[arm,f"{100*s['full_feature']['test']['all'][arm]['accuracy']:.2f}"]+[f"{100*s['full_feature']['real'][d][arm]['accuracy']:.2f}" for d in datasets] for arm in arms]),'',
           'Full은 전체 query이며 primary와 query집합이 다르므로 절대 차이를 missing의 인과효과로 해석하지 않는다.','',
           '## 비용과 재현 검사','',
           table(['모델','Graph','median ms','p10/p90 ms','추가 GPU MiB'],[[r['arm'],r['graph'],f"{r['median_ms']:.2f}",f"{r['p10_ms']:.2f}/{r['p90_ms']:.2f}",f"{r['peak_additional_gpu_mib']:.2f}"] for r in cost['records']]),'',
           '자체 GPU 작업 종료 후 직렬로3회 warmup·10회 측정했다. CPU의 visible RP16 입력부터 CPU 확률까지이며 confidence/frame/전파를 포함한다. 모델 load·최초RP는 제외한다. 다른 사용자 부하는 통제하지 못했다. 이번 구현은 기존 edge sparse 연산을 유지해 공통좌표 전파로 계산을 줄이는 최적화는 하지 않았다.','',
           f"Audit: {a['status']}. 입력{a['inputs_hashed']}개, 새 예측{a['new_prediction_files_hashed']}개와3checkpoint를 검증했다. 선택된 재실행의 저장 확률 최대차이는{max(r['saved_max_abs'] for r in a['replay']):.7f}, 동일 session 반복 최대차이는{max(r['repeat_max_abs'] for r in a['replay']):.7f}, 저장 예측 대비 최대argmax변경은{100*max(r['saved_argmax_change_fraction'] for r in a['replay']):.2f}%다.",'',
           '0.1 검사는 큰 손상만 거르는 기준이며 엄격한 수치 재현 통과 기준이 아니다. CI는 저장 예측에 조건부이며 전체 수치 변동을 포함하지 않는다. 초기화·동일taskorder·frozen weights·기존 입력 보존·새test graph 중복 여부도 확인했다.','',
           '## 판정 범위','',json.dumps(s['gates'],ensure_ascii=False),'',
           '이번 비교가 판단하는 것은 현재SO(2)/RP16/공유token geometry/1,024updates에서 node-frame 제약이 기존 edge map과 Identity를 넘어서는가다. 실패 원인을 고유하게 분리한 결과도, 모든 orthogonal frame의 실패/성공도 아니다. 미통과하면 이번 bank를 보고 loss·frame 차원·학습량을 바꾸어 구제하지 않는다.','',
           '[계산 정의](METHOD_KO.md), [원점수](results/statistics.json), [검증](results/audit.json), [개입](results/control_summary.json).','']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8');print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
