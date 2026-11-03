import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent


def main():
    stats=json.loads((ROOT/'results/statistics.json').read_text(encoding='utf8'))
    audit=json.loads((ROOT/'results/audit.json').read_text(encoding='utf8'))
    names={'pcfi':'PCFI','uniform':'Uniform I','node':'Node','node_nolabel':'Node label 제거',
           'node_uniform':'Node 공간 균등','signed':'Signed scalar','node_geometry_shuffle':'Geometry label shuffle'}
    comparison_names={'node_minus_pcfi':'Node − PCFI',
        'node_minus_node_nolabel':'Node − label 제거',
        'node_minus_signed':'Node − signed scalar',
        'node_minus_node_geometry_shuffle':'Node − geometry label shuffle',
        'node_uniform_minus_uniform':'공간 균등의 Node − I',
        'node_minus_node_uniform':'Node의 confidence − 공간 균등',
        'pcfi_minus_uniform':'I의 confidence − 공간 균등',
        'signed_minus_pcfi':'Signed scalar − PCFI',
        'node_nolabel_minus_pcfi':'Label 제거 − PCFI',
        'spatial_confidence_by_node_interaction':'공간 confidence × Node interaction'}
    lines=['# Confidence Sheaf PFN의 context와 confidence 및 scalar 비교','',
        '2026년 11월 3일. 기존 Node-frame을 개발 후보로 유지한 상태에서 구성 요소를 분리했다. 새 학습은 frontend 9개와 9,216 update이며, PFN과 실제 graph의 가중치 update는 모두 0회다. 결과에 따른 추가 loss나 budget 조정은 하지 않았다.','',
        '## 새 개발 graph의 primary','',
        '동일 known family의 새 base graph 32개다. Label homophily와 feature 법칙 네 cell에 각각 8개 graph를 배정하고, 두 context 수와 네 primary 결측 조건 및 고정 학습 seed 3개를 graph 안에서 평균했다. 자기 피처가 없는 query의 accuracy가 primary다.','',
        '| 방법 | Accuracy % | NLL | Brier |','|---|---|---|---|']
    for arm,score in stats['development'].items():
        lines.append(f"| {names[arm]} | {100*score['accuracy']:.3f} | {score['nll']:.5f} | {score['brier']:.5f} |")
    lines += ['', '| 비교 | Accuracy 차이 %p | 95% CI |','|---|---|---|']
    for name,values in stats['comparisons'].items():
        value=values['accuracy'];lo,hi=value['ci95']
        lines.append(f"| {comparison_names[name]} | {100*value['mean']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines += ['', 'CI는 base graph 단위 paired bootstrap 5,000회다. 여러 비교는 다중비교 미보정이며 저장된 예측과 세 고정 학습 seed에 조건부다. 구성 요소의 ablation이므로 모든 비교를 통과해야 한다는 종료 기준을 새로 붙이지 않는다.','',
        '## 세 비교의 해석','']
    def evidence(key):
        value=stats['comparisons'][key]['accuracy'];lo,hi=value['ci95']
        return value['mean'],lo,hi
    mean,lo,hi=evidence('node_minus_node_nolabel')
    if lo>0:
        context='Label 신호를 제거해 같은 예산으로 학습한 geometry보다 원 Node가 높았다. 이 개발 bank에서는 geometry의 context-label 통계에 추가 가치가 있다.'
    elif hi<0:
        context='Label 신호를 제거해 같은 예산으로 학습한 geometry가 원 Node보다 높았다. 현재 context-label 통계의 추가 이득은 지지되지 않으며, label-free geometry로 단순화할 가능성이 남는다. 이를 새 graph의 독립 검증 결과로 부르지 않는다.'
    else:
        context='같은 예산으로 학습한 label-free geometry와의 차이 CI가 0을 포함했다. Geometry의 명시적 context-label 통계가 필요한지는 이번 비교에서 확인되지 않았다. 동등성을 입증한 것은 아니다.'
    lines += ['**Context의 독립 효과.** '+context,
              'Frozen geometry shuffle은 PFN의 label을 바꾸지 않은 별도 개입이다. Assignment 제거 후 재학습과 추론 때의 shuffle은 다른 질문이므로 함께 보고한다. 이번에는 geometry shuffle의 accuracy 차이 CI가 0을 포함했다. 따라서 재학습 제거 대조의 이득만으로 올바른 class 묶음의 의미를 읽어 geometry를 적응시켰다고 주장하지 않는다.','']
    mean,lo,hi=evidence('spatial_confidence_by_node_interaction')
    interpretation='양의 interaction을 지지한다.' if lo>0 else '음의 interaction을 지지한다.' if hi<0 else 'CI가 0을 포함해 interaction의 방향이 확인되지 않았다.'
    lines += [f'**공간 confidence와 frame의 결합.** Interaction은 (Node−PCFI)−(Node uniform−Uniform I)로 계산했다. 평균 {100*mean:+.3f}%p이며 {interpretation} Descriptor와 채널 보정의 confidence는 유지했으므로 confidence 전체의 유무로 확대하지 않는다.','']
    lines += ['이 양의 interaction에는 공간 가중치가 균등할 때 Node가 I보다 낮아진 효과도 포함된다. Weighted Node 자체의 PCFI 대비 차이 CI는 0을 포함하므로, confidence와 frame이 함께 강한 기준선을 넘는 새로운 예측 신호를 만들었다고 주장하지 않는다.','']
    mean,lo,hi=evidence('node_minus_signed')
    interpretation='현재 Node가 이 직접 scalar 대조보다 높았다.' if lo>0 else '직접 scalar 대조가 현재 Node보다 높았다.' if hi<0 else 'CI가 0을 포함해 두 함수족의 우위가 확인되지 않았다.'
    lines += ['**직접 scalar와의 비교.** '+interpretation+' 이 scalar는 edge 계수를 직접 학습하지만 clipping과 별도 초기 전파 함수를 사용한다. 가능한 모든 scalar 방법에 대한 최강 대조나 전체 함수족의 우열로 확대하지 않는다.','',
              '## Context 수와 대칭','',
              '| k per class | 방법 | Accuracy % |','|---|---|---|']
    for k,arms in stats['by_context'].items():
        for arm in ['node','node_nolabel','node_geometry_shuffle','signed']:
            lines.append(f"| {k} | {names[arm]} | {100*arms[arm]['accuracy']:.3f} |")
    lines += ['', '클래스당 context가 하나이면 assignment shuffle이 class 이름의 순열일 수 있다. 현재 geometry는 class 이름에 불변인 통계를 쓰므로 k=1의 작은 변화는 수치 변동과 함께 해석한다. k=4에서는 class별 context 묶음 자체를 바꾸므로 다른 개입이다.','',
              '## Label와 feature 법칙별 결과','',
              '| 개발 cell | PCFI % | Node % | Label 제거 % | Node 공간 균등 % | Signed % |','|---|---|---|---|---|---|']
    for cell,arms in stats['by_cell'].items():
        lines.append('| '+cell+' | '+' | '.join(f"{100*arms[arm]['accuracy']:.3f}" for arm in ['pcfi','node','node_nolabel','node_uniform','signed'])+' |')
    lines += ['', 'Label heterophily와 feature roughness는 별도의 생성 축이다. 같은 그래프에서 라벨이 다른 이웃이 많다는 이유로 모든 numeric feature의 부호를 뒤집어야 한다고 해석하지 않는다.','',
              '## 실제 graph의 개발 평가','',
              '| 방법 | Cora % | CiteSeer % | Texas % | Wisconsin % |','|---|---|---|---|---|']
    for arm in list(names)+['native_pcfi','lp','negative_lp','mask_reference']:
        lines.append('| '+names.get(arm,arm)+' | '+' | '.join(f"{100*stats['real'][graph][arm]['accuracy']:.3f}" for graph in ['Cora','CiteSeer','texas','wisconsin'])+' |')
    lines += ['', '기존 네 real graph의 동일 입력과 세 context/mask 반복을 재사용했다. 새 일곱 arm의 확률은 모두 이번 실행에서 계산했다. Native-PCFI, LP와 Mask floor는 이전 동일 입력의 hash 검증된 예측을 유지했으며 신규 학습군이 아니다. Native-PCFI는 원본 고차원에서 먼저 전파하므로 동일표현의 공간 operator 대조와 구분한다. 실제 graph 결과는 개발용 점수이며 독립 holdout이나 다수 graph에 대한 유의성 주장이 아니다.','',
              '## Source가 없는 조건과 확률 품질','',
              '| Source | 방법 | Accuracy % | NLL | Brier |','|---|---|---|---|---|']
    for source,info in stats['by_source'].items():
        for arm,score in info['scores'].items():
            lines.append(f"| {source} | {names[arm]} | {100*score['accuracy']:.3f} | {score['nll']:.5f} | {score['brier']:.5f} |")
    lines += ['', 'Source0에서는 추정기가 새 숫자를 생성하지 않고 같은 PFN의 graph/context 예측으로 돌아간다. Source0와 nonzero는 과제 구성이 다르므로 두 평균의 차이를 피처 관측의 인과적 효과로 해석하지 않는다. Source2/8과 full의 secondary 결과는 statistics.json에 별도로 보존했다.', '',
              '## 전파 계수와 검증','',
              '| Signed scalar 입력 | 음수 계수 비율 | 평균 gain | +1 clip 비율 |','|---|---|---|---|']
    for split,arms in stats['diagnostics'].items():
        d=arms['signed']
        lines.append(f"| {split} | {d['negative_gain_fraction']:.6f} | {d['gain_mean']:.6f} | {d['gain_clip_positive_fraction']:.6f} |")
    lines += ['', '직접 scalar는 이번 실행에서 음수 gain을 실제로 사용하지 않았고, +1 clipping이 약 67%였다. 이는 signed 함수족의 표현력이 부족하다는 증거가 아니다. 최적화가 부호를 활용하지 못한 이유도 분리되지 않았다. 따라서 이 대조만으로 2D transport가 signed 전파보다 필요하다는 결론은 내리지 않는다.']
    replay=audit['replay']
    max_stored=max(r['stored_probability_max_abs'] for r in replay)
    max_repeat=max(r['repeat_probability_max_abs'] for r in replay)
    max_argmax=max(r['stored_argmax_change'] for r in replay)
    lines += ['', f"Audit {audit['status']}. 기존 파일 {audit['old_files_preserved']}개와 입력 {audit['input_files_hashed']}개, 새 예측 {audit['new_predictions_hashed']}개를 hash로 확인했다. 새 frontend {audit['new_frontend_models']}개의 초기 상태와 과제 순서가 기존 Node와 일치하고 총 {audit['new_frontend_updates']} update였다. PFN과 실제 graph update는 0회다.", '',
              f"선택된 12개 예측의 replay에서 저장 확률 대비 최대 차이는 {max_stored:.7f}, 같은 session 반복 최대 차이는 {max_repeat:.7f}, 저장 예측 대비 최대 argmax 변경은 {100*max_argmax:.3f}%다. 0.1 검사는 큰 손상 검사이며 엄격한 재현 보장은 아니다. CI는 이 전체 수치 변동을 포함하지 않는다.", '',
              f"Full과 source0의 PCFI 환원 {audit['full_source0_comparisons']}건의 최대 확률 차이는 {audit['full_source0_probability_max_abs']:.7f}다. Gain의 부호와 clipping은 학습 연산의 실제 사용을 나타내며 성능의 인과적 설명을 확정하지 않는다.", '',
              '## 다음 판단의 범위','',
              '이 결과는 구성 요소의 필요성을 검토하는 개발 실험이다. 이미 검토한 실제 그래프와 같은 known family에서 좋은 arm을 고른 것을 독립 확인으로 부르지 않는다. 특정 부품이 유효하지 않다면 메소드와 주장을 단순화하고, 최종 구조를 고정한 뒤 새로운 그래프에서 확인한다. 이번 실행에서는 결과를 보고 새로운 loss, 초기화, 학습량을 추가하지 않았다.','',
              '[계산과 비교 정의](METHOD_KO.md), [원점수와 CI](results/statistics.json), [검증](results/audit.json), [고정한 프로토콜](protocol.json).','']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8')
    print('REPORT WRITTEN',ROOT/'REPORT_KO.md',flush=True)


if __name__=='__main__':
    main()
