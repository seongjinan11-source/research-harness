"""Korean result summaries from completed, locked experiment outputs."""
from pathlib import Path
import json
import numpy as np
WORK=Path(__file__).resolve().parents[2]
def load(folder,name):return json.loads((WORK/folder/'results'/name).read_text(encoding='utf8'))
def contrast(v,scale=1):return f"{scale*v['difference']:+.6f} [{scale*v['ci95'][0]:+.6f}, {scale*v['ci95'][1]:+.6f}]"


def memory():
    folder='feature_memory_pilot';root=WORK/folder;p=load(folder,'paired.json');audit=load(folder,'audit.json')
    rows=[]
    for arm in ['context','all_nodes','observed','pcfi']:
        for seed in [11,12,13]:rows+=json.loads((root/'results'/f'{arm}_s{seed}'/'evaluation.json').read_text())
    rows+=load(folder,'baselines.json')
    keep=lambda r:r['mode']=='standard' and r['pair'] in ('official','offdiagonal') and r['mechanism'] in ('uniform_nodes','clustered_nodes')
    lines=['# Feature memory 분리 — 완료 결과','',
        '**판정: 현재 공개 GraphPFN의 feature-group memory routing에서는 accuracy 개선을 확인하지 못했다.**',
        '기존 sample attention의 context-only key/value 제한은 확인됐지만, 이를 바꾸는 것만으로 이전 실패가 해결된다는 가설은 지지되지 않았다. 고유한 실패 원인은 여전히 미상이다.','',
        '- 새 64 train / 64 test base graphs, 3 fixed training seeds, 4 arms, arm/seed당 512 성공 updates.',
        '- 같은 public initialization·파라미터·과제 순서·query CE. 기존 transformer 고정. PCFI arm은 실제 PCFI 입력으로 학습했다.',
        '- Primary는 공식 및 controlled off-diagonal의 whole-node uniform/clustered90/99.5% accuracy. CI는 base graph64개를 resample한다.',
        '- 공개 모델의 native 두 좌표 group을 memory 단위로 썼다. 완전한 sparse cell-token 모델의 결과가 아니다.','',
        '| 모델 | Accuracy (%) | NLL | Brier |','|---|---:|---:|---:|']
    for arm in ['context','all_nodes','observed','pcfi','lp','class_prior']:
        rr=[r for r in rows if r['arm']==arm and keep(r)]
        lines.append(f"| {arm} | {100*np.mean([r['accuracy'] for r in rr]):.4f} | {np.mean([r['nll'] for r in rr]):.6f} | {np.mean([r['brier'] for r in rr]):.6f} |")
    lines+=['','| Accuracy 차이 | %p와 paired95% CI |','|---|---|']
    for ref in ['context','all_nodes','pcfi','lp']:lines.append(f"| observed − {ref} | {contrast(p[f'observed-{ref}/accuracy'],100)} |")
    lines+=['',f"Observed의 NLL은 context 대비 {contrast(p['observed-context/nll'])}, PCFI 대비 {contrast(p['observed-pcfi/nll'])} 높았다. Accuracy의 CI가0을 포함하는 것과 확률 품질 악화는 구분한다.",
        'Clustered90/off-diagonal의 observed arm은 feature를 섞었을 때 오히려 NLL이 낮아졌다. 이 실행에서는 실제 feature–node 정렬을 유익하게 활용했다고 주장하지 않는다. 이것을 모든 실제 feature의 무용함으로 확대하지 않는다.','',
        f"저장된 예측 {audit['prediction_files_rescored']:,}개를 독립 재채점했고 총 {audit['successful_updates']:,} 성공 updates와 frozen parameters 보존을 확인했다.",
        '이번 bank와 mixed-prior512-update 비교는 이전 MCP의 +3.43%p를 독립 재현한 실험이 아니다. 이전 NLL gate와 긍정 신호를 수정하지 않는다.',
        '전체 결과·사전 고정 조건: REPORT.md, protocol.json, results/paired.json, interventions.json, final_integrity.json.']
    (root/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def circuit():
    folder='direct_marginal_pilot';root=WORK/folder;s=load(folder,'summary.json');p=load(folder,'paired.json');i=load(folder,'interventions.json');d=load(folder,'decision.json');controls=load(folder,'feature_controls.json');audit=load(folder,'audit.json')
    lines=['# 직접 주변화하는 conditional graph circuit — 완료 결과','',
        '**판정: 정규화·주변화 유효성 및 두 학습된 직접 예측기 대비 개선은 확인했다. Grid6 대비 메인 메소드 경쟁력은 충족하지 못했다.**','',
        '- 새 12-node binary typed-SBM family: target8/helper4, relation2, 연속 feature2. 세 prior-family descriptor는 모든 모델에 제공했다.',
        '- Matched primary는 새48 base graphs의 whole-node uniform/clustered50/90%, 총 context1/2/4 평균. OOD까지 총192 base graphs다.',
        '- Circuit/direct graph/public Mask PFN을 3 fixed seeds로 학습했다. 모델마다 2048 task exposures와256 성공 updates, query CE만 사용했다. Parameter 수는 동일하지 않다.',
        '- Circuit은 56,385 parameters, public Mask PFN은 4,012,259 trainable parameters다. Public PFN에는 typed context support까지 제공했다.','',
        '## 1. 같은 모델 내부의 주변화는 유효하다','',
        'Graph와 알려진 family에만 조건화한 normalized tree mixture를 만들고, 관측 feature와 context label은 leaf evidence로 평가한다. 각 원본 변수 scope는 한 번씩만 사용한다.',
        '소형 열거, 전체 질량1, label 합 규칙, 연속 feature 적분, conditional ratio의 최대 오차는 약10^-15 수준이었다. 단일 class0 context에서 음의 transition을 주면 반대 class 확률0.95257을 표현할 수 있었다.',
        '이는 현재 conditional tree model 안의 정확한 계산이다. 원래 cyclic SBM의 참 posterior가 정확하다는 뜻은 아니며 GSPN/MaM 원 논문의 재현도 아니다.','',
        '## 2. 예측 성능','',
        '| 방법 | Accuracy (%) | NLL | Brier | Grid6-exact predictive KL |','|---|---:|---:|---:|---:|']
    for arm in ['mask_pfn','direct_graph','circuit','grid6_bp','grid6_exact']:
        r=next(r for r in s if r['arm']==arm and r['subset']=='primary')
        lines.append(f"| {arm} | {100*r['accuracy']:.4f} | {r['nll']:.6f} | {r['brier']:.6f} | {r['grid6_kl']:.6f} |")
    lines+=['',f"Circuit−Mask PFN accuracy: {contrast(p['circuit-mask_pfn/accuracy'],100)} %p. NLL: {contrast(p['circuit-mask_pfn/nll'])}.",
        f"Circuit−Direct graph accuracy: {contrast(p['circuit-direct_graph/accuracy'],100)} %p. NLL: {contrast(p['circuit-direct_graph/nll'])}.",
        '따라서 사전 neural-baseline gate는 통과했다. 서로 다른 과제인 feature-memory 실험의 accuracy와 절대값을 비교하지 않는다.','',
        '## 3. 실제 관측값을 사용했는가','',
        '아래 intervention은 context2/clustered50, 전체192 base graphs의 보조 비교이며 primary와 조건이 다르다.','',
        '| Intervention−원래 입력 | Accuracy %p [95% CI] | NLL [95% CI] |','|---|---|---|']
    for mode in ['no_observed','shuffled']:
        a=next(v for v in i if v['arm']=='circuit' and v['mode']==mode and v['metric']=='accuracy');b=next(v for v in i if v['arm']=='circuit' and v['mode']==mode and v['metric']=='nll')
        lines.append(f"| {mode} | {contrast(a,100)} | {contrast(b)} |")
    lines+=['','이 family에서는 실제 feature와 node 대응 관계를 예측에 활용한다. 이를 IMDB 등 기존 실제 graph에서의 feature 기여로 옮겨 주장하지 않는다.','',
        '## 4. 강한 기준선이 제한하는 주장','',
        f"Grid6-BP 대비 circuit NLL은 {contrast(p['circuit-grid6_bp/nll'])} 더 높고, accuracy는 {contrast(p['circuit-grid6_bp/accuracy'],100)} %p 낮았다.",
        'Grid6는 이전 six-support prior를 유지하되 이 family의 Gaussian feature orientation을 적분하도록 확장한 known-family baseline이다. 실제 test density/signal을 주는 별도 oracle과 구분한다.',
        'Primary 설계 고정 뒤 해석용으로 추가한 feature-only/constant-tree 대조는 아래와 같다. Gaussian magnitude0.8/sd0.6라는 정확한 training-family 상수를 아는 model-based 대조이며 학생 모델의 성과가 아니다.','',
        '| 추가 대조 | Accuracy (%) | NLL |','|---|---:|---:|']
    for r in controls['primary']:lines.append(f"| {r['arm']} | {100*r['accuracy']:.4f} | {r['nll']:.6f} |")
    v=controls['paired']['circuit-known_feature_only/nll'];a=controls['paired']['circuit-known_feature_only/accuracy']
    lines += ['',f"Circuit−feature-only NLL: {contrast(v)}; accuracy: {contrast(a,100)} %p. 추가 NLL 이득은 확인되지 않았고 accuracy CI도0을 포함한다. 이것이 두 방법의 동등성을 증명하는 것은 아니다.",
        '관측 feature가 label context 바깥에만 있는 primary 사후 분할에서는 circuit52.69%/NLL0.69955, Grid6-BP64.87%/0.62460이었다. Visible mask/context로 나눈 기술적 분석이며 새 primary gate나 독립 확인으로 쓰지 않는다. 올바른 주변화 연산이 있어도 구조를 통해 떨어진 feature와 label을 연결하는 예측이 충분하지 않을 수 있음을 보여주는 미해결 조건이다.',
        '원인으로 tree 표현력·graph-conditioned potential fit·학습량 중 하나를 확정하지 않는다.','',
        '## 5. 비용과 범위','',
        f"같은 GPU의 단일 graph visible-input→확률 median: circuit {d['circuit_median_ms']:.3f}ms, Grid6-BP {d['grid6_bp_median_ms']:.3f}ms. 약간의 측정값 차이가 있지만 사전2배 속도 기준을 충족하지 않는다. Grid6-exact predictive KL도 {d['matched_grid6_predictive_kl']:.6f}로 .01 목표보다 크다.",
        'BP-only 비용에서는 reference 생성에만 필요한 exact query marginal 출력을 제외했고 예측 차이0을 확인했다. 모델 구성·bank I/O·사전학습 비용은 이 latency 밖이며 외부 GPU 부하는 통제하지 않았다.',
        f"학습9개, 총 {audit['successful_updates']:,} 성공 updates와 {audit['task_exposures']:,} task exposures, 저장 예측 {audit['prediction_files_rescored']:,}개를 감사했다. Test4608개 중 실제 physical feature 관측이0개인 episode는 {audit['zero_observation_test_episodes']:,}개다.",
        '대형·실제 heterograph, 새 semantic schema, 가변 feature 폭, 독립적으로 학습한 MaM, full sparse cell-token 모델은 아직 실행하지 않았다. 현재 결과로 이를 해결했다고 쓰지 않는다.',
        '근거: protocol.json, results/preflight.json, paired.json, feature_controls.json, descriptive_breakdowns.json, benchmark.json, final_integrity.json.']
    (root/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':memory();circuit();print('KOREAN_REPORTS_WRITTEN')
