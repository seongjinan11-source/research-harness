import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {'pcfi':'PCFI','node':'기존 Node 전체 clamp','partial_frozen':'Partial-frozen',
    'partial_trained':'Partial-trained','partial_initial':'Partial 초기 network',
    'full_random':'Full-random','partial_random':'Partial-random',
    'signed_smooth':'Smooth signed scalar','signed':'기존 clipped scalar'}


def verdict(score):
    lo,hi = score['ci95']
    return '양의 차이를 지지' if lo>0 else '음의 차이를 지지' if hi<0 else 'CI가 0 포함; 방향 미확인'


def main():
    stats = json.loads((ROOT/'results/statistics.json').read_text())
    audit = json.loads((ROOT/'results/audit.json').read_text())
    preflight = json.loads((ROOT/'results/preflight.json').read_text())
    d_node = stats['comparisons']['partial_trained_minus_node']['accuracy']
    d_pcfi = stats['comparisons']['partial_trained_minus_pcfi']['accuracy']
    if d_node['ci95'][0]>0 and d_pcfi['ci95'][0]>0:
        conclusion = '이 개발 bank에서는 기존 Node와 PCFI 대비 양의 accuracy 차이를 함께 지지했다. 독립 graph/feature family 전이와 강한 전체 pipeline 비교는 별도로 남는다.'
    else:
        conclusion = '현재 개발 bank에서 기존 Node와 PCFI를 함께 넘어선다는 주장은 확보하지 못했다. 계산의 표현력 차이는 확인됐지만, 그것을 실제 분류의 이득으로 연결했는지는 각 비교를 나누어 판단한다.'
    lines = ['# 관측 좌표만 고정하는 Sheaf-PCFI: 개발 실험 결과','',
        '2026-11-03. 실제 관측된 첫 좌표만 clamp하고 두 번째 좌표는 관측 노드에서도 추론하는 변경을 검증했다. 기존 모델의 추론 변경과 같은 예산의 새 학습을 분리했다. 기존 결과와 checkpoint는 보존했다.','',
        '**현재 판정.** '+conclusion,
        f"Partial-trained−Node {100*d_node['mean']:+.3f}%p, CI [{100*d_node['ci95'][0]:+.3f},{100*d_node['ci95'][1]:+.3f}]; Partial-trained−PCFI {100*d_pcfi['mean']:+.3f}%p, CI [{100*d_pcfi['ci95'][0]:+.3f},{100*d_pcfi['ci95'][1]:+.3f}].",'',
        '## Primary: 자기 피처가 없는 query','',
        '새 known-family base graph 32개, label homo/hetero × feature smooth/rough 각 8개다. 두 context 수·네 primary missing 조건·세 고정 학습 seed를 graph 안에서 평균했다. 실제 graph는 기존 개발 입력을 재사용했다.','',
        '| 방법 | Accuracy % | NLL | Brier |','|---|---|---|---|']
    for arm,v in stats['development'].items():
        lines.append(f"| {NAMES[arm]} | {100*v['accuracy']:.3f} | {v['nll']:.5f} | {v['brier']:.5f} |")
    lines += ['', '| 비교 | Accuracy 차이 %p | 95% CI | 판정 |','|---|---|---|---|']
    for pair,values in stats['comparisons'].items():
        a,b = pair.split('_minus_')
        v = values['accuracy'];lo,hi = v['ci95']
        lines.append(f"| {NAMES[a]} − {NAMES[b]} | {100*v['mean']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {verdict(v)} |")
    lines += ['', 'Graph 단위 paired bootstrap 5,000회이며 다중비교 미보정이다. 세 고정 seed 및 저장된 확률에 조건부로 계산한 CI이며 GPU 수치 replay 불확실성을 포함하지 않는다.','',
        '## 해석 범위','']
    for pair,title,meaning in [
        ('partial_frozen_minus_node','추론 때 경계 변경','같은 학습 frame/PFN에서 clamp만 바꾼 효과다.'),
        ('partial_trained_minus_partial_frozen','변경한 경계에 맞춘 학습','같은 원래 초기화부터 재학습한 효과이며 추가 update를 받은 fine-tuning이 아니다.'),
        ('partial_trained_minus_node','기존 Node 대비','관측 경계 변경과 그 방식의 동일 예산 학습을 합친 비교다.'),
        ('partial_trained_minus_pcfi','PCFI 대비','현재 제안이 기존 강한 scalar confidence 전파를 넘는지에 대한 직접 비교다.'),
        ('partial_trained_minus_partial_random','Frame 학습','무작위 각도 대조와 학습된 입력 조건부 각도의 차이다. 무작위 범위를 조정하지 않았다.')]:
        v = stats['comparisons'][pair]['accuracy']
        lines.append(f"**{title}.** 평균 {100*v['mean']:+.3f}%p, {verdict(v)}. {meaning}\n")
    lines += ['두 번째 좌표가 실제로 0이 아니거나 소형 예제의 표현력이 늘었다는 것만으로 분류 메소드의 성공을 선언하지 않는다. 성능 차이의 원인이 식별 불가능성·증폭·학습 부족 중 무엇인지는 이 비교만으로 확정하지 않는다. 특히 개선이 없다면 clamp 변경의 현재 효용을 보류하는 것이며 모든 sheaf/PCFI/PFN 연구의 종료 판정이 아니다.','',
        '## Source 수와 결측 조건','',
        '| Source | 방법 | Accuracy % | NLL | Brier |','|---|---|---|---|---|']
    for source,info in stats['by_source'].items():
        for arm in ['pcfi','node','partial_frozen','partial_trained','partial_random','signed_smooth']:
            v = info['scores'][arm]
            lines.append(f"| {source} | {NAMES[arm]} | {100*v['accuracy']:.3f} | {v['nll']:.5f} | {v['brier']:.5f} |")
    lines += ['', 'Source0와 nonzero는 서로 다른 과제 부분집합이다. 각 부분집합에서도 해당 조건의 seed/context/mask를 graph 안에서 먼저 평균하고, 포함된 base graph에 같은 가중치를 준다. 두 평균의 차이를 관측 피처의 인과 효과로 해석하지 않는다. Source0/full은 같은 PFN 입력으로 돌아가는 correctness 조건이다.','',
        '| Source 제어 조건 | PCFI % | Node % | Partial-trained % | Partial-random % |','|---|---|---|---|---|']
    for case,arms in stats['secondary'].items():
        lines.append('| '+case+' | '+' | '.join(f"{100*arms[a]['accuracy']:.3f}" for a in ['pcfi','node','partial_trained','partial_random'])+' |')
    source_check = json.loads((ROOT/'results/source_rank_checks.json').read_text())
    lines += ['', '**추가 기전 분석.** 학습 구조와 primary를 고정한 뒤, 같은 frame에서 연결 성분별 관측 source가 하나뿐이면 전체/부분 clamp가 같은 계산이 됨을 유도하고 소형 24개 예제로 확인했다. 공통 좌표의 상태가 그 source 방향의 span에 머물기 때문에 새 성분을 얻지 못한다. Source0/one/two+ 조건은 전체 primary를 바꾸지 않은 secondary 분석이다.',
        f"Primary 256 episode 중 source0 {source_check['primary_episode_counts']['zero']}개, source1 {source_check['primary_episode_counts']['one']}개, source2+ {source_check['primary_episode_counts']['two_or_more']}개다. 새로 학습한 frame과 기존 frame은 다를 수 있으므로, 별도 학습한 Partial-trained와 Node까지 source1에서 동치라는 뜻은 아니다."]
    lines += ['', '## 생성 조건과 학습 seed','',
        '| Label / feature | PCFI % | Node % | Partial-frozen % | Partial-trained % | Smooth signed % |','|---|---|---|---|---|---|']
    for cell,info in stats['by_cell'].items():
        arms = info['scores']
        lines.append('| '+cell+' | '+' | '.join(f"{100*arms[a]['accuracy']:.3f}" for a in ['pcfi','node','partial_frozen','partial_trained','signed_smooth'])+' |')
    lines += ['', '| Seed | Partial-trained − Node %p | Partial-trained − PCFI %p |','|---|---|---|']
    for seed,info in stats['by_seed'].items():
        lines.append('| '+seed+' | '+' | '.join(f"{100*info['comparisons'][key]['accuracy']['mean']:+.3f}" for key in ['partial_trained_minus_node','partial_trained_minus_pcfi'])+' |')
    lines += ['', 'Label heterophily와 feature roughness는 별도 축이다. Label heterophily만으로 모든 numeric feature의 부호를 반전해야 한다고 가정하지 않는다. Context별 결과와 mask별 결과는 statistics.json에 함께 보존했다.','',
        '## 실제 graph: 개발 평가','',
        '| 방법 | Cora % | CiteSeer % | Texas % | Wisconsin % |','|---|---|---|---|---|']
    for arm in list(NAMES)+['native_pcfi','negative_lp','mask_reference']:
        lines.append('| '+NAMES.get(arm,arm)+' | '+' | '.join(f"{100*stats['real'][g][arm]['accuracy']:.3f}" for g in ['Cora','CiteSeer','texas','wisconsin'])+' |')
    lines += ['', '새 아홉 arm은 동일 real 입력에 이번 실행에서 평가했다. Native-PCFI/negative-LP/Mask는 기존 동일 입력 cache 기준이다. Native-PCFI는 원본 고차원에서 먼저 전파하는 더 강한 전체 pipeline이며 geometry-only 대조와 구분한다. 네 실제 graph는 독립 holdout이 아니다. Downstream은 기존 synthetic-trained 실험용 wrapper이며 공식 GraphPFN 전체 pipeline의 재현 성과가 아니다.','',
        '## 잠재 상태와 smooth scalar','',
        '| 입력 | Partial arm | 관측 노드 두 번째 좌표 평균 절댓값 | 마지막 step 변화 최댓값 | 출력 절댓값 최댓값 |','|---|---|---|---|---|']
    for split,arms in stats['diagnostics'].items():
        for arm in ['partial_frozen','partial_trained','partial_random']:
            d = arms[arm]
            lines.append(f"| {split} | {NAMES[arm]} | {d['observed_second_mean_abs']['mean']:.6f} | {d['last_step_max_change']['max']:.6f} | {d['output_max_abs']['max']:.6f} |")
    lines += ['', '고정 100회 전파가 실제 실행한 메소드다. 마지막 변화가 작다는 사실만으로 잠재 상태의 유일성이나 정답성을 주장하지 않는다. Source가 거의 없을 때 자유로운 성분의 식별 문제가 남을 수 있지만, 성능 저하 원인으로 확인된 것은 아니다.','',
        '| 입력 | Smooth 음수 gain 비율 | 평균 gain | abs(g)>0.999 비율 | 평균 tanh 미분 |','|---|---|---|---|---|']
    for split,arms in stats['diagnostics'].items():
        d = arms['signed_smooth']
        lines.append(f"| {split} | {d['negative_gain_fraction']['mean']:.6f} | {d['gain_mean']['mean']:.6f} | {d['gain_saturation_fraction']['mean']:.6f} | {d['gain_derivative_mean']['mean']:.6f} |")
    lines += ['', 'Hard clipping은 제거했지만 smooth한 함수도 saturation될 수 있다. 계수의 실제 음수 사용과 미분을 보고하며, 이 한 parameterization을 전체 signed 함수족의 최강 대조라고 주장하지 않는다.','',
        '## 검증과 보존','',
        f"새 frontend {audit['new_frontend_models']}개, {audit['new_frontend_updates']} update. 기존과 초기 parameter·train task schedule·PFN hash가 일치한다. PFN/real update는 모두 0회다. 사전 검사 optimizer step은 폐기했다.",
        f"기존 파일 {audit['old_files_preserved']}개, 입력 {audit['input_files_hashed']}개 및 새 예측 {audit['new_predictions_hashed']}개 hash를 확인했다."]
    c = preflight['cpu_checks']
    lines += [f"전체 clamp의 기존 구현 차이 {c['full_matches_old_max_abs']:.3g}, 부분 clamp의 독립 좌표계 출력 차이 {c['partial_aligned_max_abs']:.3g}, parameter gradient 차이 {c['partial_aligned_gradient_max_abs']:.3g}. Identity는 PCFI와 정확히 같았다.",
        f"소형 분석 예제에서 기존 출력 {c['toy_full_query']:.6f}, 부분 clamp 출력 {c['toy_partial_query']:.6f}였다. 이는 다른 함수족이라는 검증이며 분류 성능 개선이 아니다.",
        f"Full/source0 확률 환원 {audit['full_source0_comparisons']}건, 최대 차이 {audit['full_source0_probability_max_abs']:.7f}.",
        f"같은 frame의 source1 Partial-frozen−Node 확률 비교 {audit['same_frame_one_source_comparisons']}건, 최대 차이 {audit['same_frame_one_source_probability_max_abs']:.7f}. 수학적 연산 동치와 실제 GPU 확률 replay 차이를 구분한다."]
    replay = audit['replay']
    lines += [f"선택된 replay에서 저장 확률 대비 최대 차이 {max(r['stored_probability_max_abs'] for r in replay):.7f}, 같은 session 반복 최대 차이 {max(r['repeat_probability_max_abs'] for r in replay):.7f}, argmax 변경 최대 {100*max(r['stored_argmax_change'] for r in replay):.3f}%. 0.1 threshold는 큰 손상 검사일 뿐 엄격한 재현 보장이 아니다.", '',
        '근거: [메소드](METHOD_KO.md), [고정 프로토콜](protocol.json), [전체 통계](results/statistics.json), [audit](results/audit.json).']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'runtime/TASK_STATE.md').write_text('# Partial-clamp experiment\n\nStatus: COMPLETED.\n\nTraining, paired analysis, audit and Korean report complete. PFN/real updates0.\n',encoding='utf-8')
    print('REPORT COMPLETE',flush=True)


if __name__=='__main__':
    main()
