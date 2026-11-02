import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 s=json.loads((ROOT/'results/statistics.json').read_text());b=json.loads((ROOT/'results/benchmark.json').read_text());p=s['primary'];lines=[]
 names=['mask_pfn','pcfi_pfn','constant_bp','factor_bp','factor_exact','smc_gibbs1','smc_gibbs_cost','diffusion_only','diffusion_smc_fixed','diffusion_smc_ess','grid6_exact','grid6_bp']
 lines+=['# Diffusion + Factor correction + Rényi-2/ESS — 실제 실행 결과','',
 '2026-11-02. 기존 Factor3개를 고정하고8,476-parameter denoiser3개를 실제로 학습했다. 새 학습3,072updates,seed당1024updates/batch16이다. 새96base graph/576episodes,primary는 matched48graph의6disjoint settings와3fixed seeds 평균이다. 같은 potential의 정확한 분포·실제 query 정답·Grid6 reference를 구분한다. Missing50/75%의 작은 known-family다.','',
 '## 사전 primary','',
 '| 방법 | Accuracy % | NLL | Brier | Exact-Factor KL ↓ | Grid6 KL ↓ |','|---|---:|---:|---:|---:|---:|']
 for name in names:
  v=p[name];lines.append(f"| {name} | {100*v['accuracy']:.4f} | {v['nll']:.6f} | {v['brier']:.6f} | {v['factor_kl']:.6f} | {v['grid6_kl']:.6f} |")
 lines+=['',f"**Gate: inference={s['gates']['inference']},classification={s['gates']['classification']},ESS increment={s['gates']['ess_increment']}.** KL 개선을 accuracy 성공으로 바꾸지 않는다. 아래는64particles/8stages 구현의 pointwise95%CI이며,세 학습 seed를 고정한48base-graph paired bootstrap5000회다. 다중비교 보정은 하지 않았다.",'',
 '| 비교 (왼쪽−오른쪽) | Accuracy %p [95% CI] | Exact-Factor KL 차이 [95% CI] |','|---|---:|---:|']
 for name,mm in s['comparisons'].items():
  a=mm['accuracy'];k=mm['factor_kl'];lines.append(f"| {name} | {100*a['mean']:+.4f} [{100*a['ci95'][0]:+.4f},{100*a['ci95'][1]:+.4f}] | {k['mean']:+.6f} [{k['ci95'][0]:+.6f},{k['ci95'][1]:+.6f}] |")
 lines+=['','## 판정','',
 '**이번 learned diffusion proposal + MH/SMC + ESS controller를 메인 메소드로 확대하지 않는다.** 추론·분류·ESS 증분 gate가 모두 실패했다. Diffusion+fixed는 비용 대조 SMC보다 accuracy가 유의하게 낮았고, 같은 frozen-model marginal의 KL도 더 컸다. 더 저렴한1sweep SMC도 평균적으로 diffusion보다 좋았다. Pure diffusion보다 보정 arm의 finite-particle KL이 악화한 결과도 숨기지 않는다.',
 '', '이는 diffusion 전체, RNP objective 또는 모든 Rényi 접근의 실패가 아니다. 닫는 범위는 이번512task/1024update·width24·64particle/8stage·기존 Factor 함수족과 fixed/ESS 스케줄이다. 학습량·proposal fit·초기 particle coverage·mutation mixing·finite-particle error의 고유 원인은 분리하지 않았다. 계산적으로 유효한 kernel이라는 것과 작은 budget에서 좋은 예측을 만든다는 것은 다르다.',
 '', '## 무엇이 남았나','',
 '동일한 frozen Factor를 정확히 합산하면 Grid6 KL이0.091114에서0.011446으로 줄었다. 기존 potential이 참 생성 posterior인 것은 아니지만, 그 조건부 분포를 더 정확히 계산할 예측적 여지가 있었다. 반면 accuracy는71.4038%→71.6449%,차이+0.2411%p이고 CI는0을 포함했다. 따라서 현재 bank에서 “추론을 정확히 하면 classification이 크게 좋아진다”는 근거는 약하다. Exact 합산은 지수적 비용의 소형 진단이며 배포 메소드로 선택하지 않는다.',
 '', '## ESS와 particle 검사','',
 '아래는 전체576episode의 세 seed 평균으로 primary48graph만의 값과 다르다. Final ESS는 가중치의 균등함을 나타낼 뿐, 초기 모드 손실·particle 간 의존성·예측 정확도를 측정하지 않는다. Pure diffusion의 ESS64는 uniform weights라서 정의상 나오는 값이다. Acceptance에는 self-transition도 포함될 수 있다.','',
 '| Sampler | Final ESS /64 | MH acceptance | 고유 final state 수 /64 | Role KL |','|---|---:|---:|---:|---:|']
 for arm in ['smc_gibbs1','smc_gibbs_cost','diffusion_only','diffusion_smc_fixed','diffusion_smc_ess']:
  dd=[s['diagnostics'][f'{arm}_s{x}'] for x in [11,12,13]];u=np.mean([s['diversity'][f'{arm}_s{x}']['unique_final_states_mean'] for x in [11,12,13]])
  vals={k:float(np.mean([d[k] for d in dd])) for k in ['mean_final_ess','mean_acceptance','role_kl']}
  acc='해당 없음' if vals['mean_acceptance']<0 else f"{vals['mean_acceptance']:.4f}"
  lines.append(f"| {arm} | {vals['mean_final_ess']:.3f} | {acc} | {u:.3f} | {vals['role_kl']:.6f} |")
 lines+=['','고유 state 수가 많아도 posterior mass를 올바르게 배분했다는 뜻은 아니다. 큰 correction 오차를 “weight collapse 한 가지 원인”으로 설명하지 않는다. Production 후 별도4-spin tree/4096particle 수치 감사에서는 세 corrected sampler가 정확한 node marginal과 최대오차0.01미만으로 일치했다. 이는 production의64particles 정확성이나 일반적인 수렴 증명이 아니다.','',
 '## 비용','',
 '개발 graph의 latency만 보고 ordinary SMC 비용 대조를2Gibbs sweeps로 잠갔다. 실제 단일 graph 측정에서도 fixed diffusion45.82ms와2sweep SMC50.44ms는 비슷했으나,1sweep SMC33.67ms·원 Factor BP4.73ms·Grid6-BP15.67ms가 더 빨랐다. 동일 particle/stage를 같은 wall time으로 부르지 않는다. 아래는 seed11의 사전 primary형 입력1개,3warmup/15repeat,CPU visible input→CPU probabilities이며 factor 추출·public PCFI 재계산을 포함한다. 모델과 reference state-table loading/I/O는 제외했고 외부GPU 부하는 통제하지 않았다.','',
 '| 방법 | Median ms | p10–p90 ms |','|---|---:|---:|']
 for name,v in b['methods'].items():lines.append(f"| {name} | {v['median_ms']:.3f} | {v['p10_ms']:.3f}–{v['p90_ms']:.3f} |")
 train=[json.loads((ROOT/'results'/f'denoiser_s{x}'/'training.json').read_text()) for x in [11,12,13]]
 teach=[json.loads((ROOT/'results'/f'preparation_s{x}.json').read_text()) for x in [11,12,13]]
 lines+=['',f"실제 denoiser 학습 시간 합계{sum(x['seconds'] for x in train):.3f}s,오프라인 train-joint teacher 준비 합계{sum(x['teacher_train_seconds'] for x in teach):.3f}s였다. Teacher의 full enumeration은 학습 시 추가 비용이며 추론에는 없다. 원 console의 마지막 PREPARED 시간은 마지막 seed부터 센 timer여서 전체 준비 비용으로 사용하지 않았다. Batch16 cached-potential 비용은 statistics.json에 따로 보존했다.",'',
 '## 전체 조건·seed','',
 '원 sampled-label accuracy/NLL/Brier,exact-model KL,Grid6 KL은 모든 condition·seed에서 별도로 남겼다. 표는accuracy이며 subgroup으로 primary를 바꾸지 않는다.','',
 '| Arm | seed11 | seed12 | seed13 | 자기 피처 없는 query | Density-low | Signal-high |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in names:
  vv=[s['seed_primary'][str(x)][arm]['accuracy'] for x in [11,12,13]]+[s['secondary_missing_query'][arm]['accuracy']]+[s['secondary_ood'][case][arm]['accuracy'] for case in ['density_low','signal_high']]
  lines.append('| '+arm+' | '+' | '.join(f'{100*x:.4f}' for x in vv)+' |')
 lines+=['','## 재현 및 범위','',
 '- 모든8640sampler episode output을 다시 실행해 original probability와 최대오차0을 확인했다. 기존 Mask/PCFI6개도 각1회 replay 최대오차0,source/checkpoint/bank hash 및3,072실제 updates를 확인했다.',
 '- 기존 Factor/Mask/PCFI checkpoint9개와 원148개 findings는 바꾸지 않는다. 새 학습은denoiser3개뿐이다. Original Factor encoder·likelihood constants와 role prior는 동일하다.',
 '- 원래 node/feature latent joint의 정확한 합산으로 inference fit을 측정했다. 실제 hidden query/helper labels를 joint teacher나 sampler가 입력받지 않는다. Query 정답은 평가에서만 사용했다.',
 '- ESS는 Rényi-2 entropy identity를 사용한 기존 SMC 제어다. Rényi Neural Processes의 학습 목적, DPM-SNC/ReDiSC/RDSMC 원본 구현은 실행하지 않았다. 이를 해당 논문들의 실패로 기록하지 않는다.',
 '- 작은2-relation/2-feature/binary schema와50/75%feature missing이다. Source/context는 disjoint지만 query/source는 겹칠 수 있다. 자기 피처 없는 query 결과는 별도 secondary이며99.5%missing·실제 graph·unseen schema/feature-law의 성과가 아니다.',
 '- Density/signal OOD의 Grid6는 알려진 parameter support의 model-based reference이며 true-parameter Bayes가 아니다. 모든 CI는 pointwise,training seeds 고정이다.',
 '', '정확한 연산은 METHOD_KO.md와protocol.json,원본 수치는results/statistics.json·benchmark.json·audit.json·convergence_audit.json에 있다. Raw bank·teacher joint draws·checkpoint·모든 probabilities는 workspace에 보존하고 공유 snapshot에는 제외한다.']
 text='\n'.join(lines)+'\n'
 (ROOT/'REPORT_KO.md').write_text(text,encoding='utf8');print('REPORT_READY')
if __name__=='__main__':main()
