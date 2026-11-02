import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
 s=json.loads((ROOT/'results/statistics.json').read_text());bank=json.loads((ROOT/'bank/manifest.json').read_text())
 a=json.loads((ROOT/'results/bank_audit.json').read_text());lines=[]
 lines+=['# J / V 관측 prior — 학습 결과','',
 '2026-11-02. **두 prior를 모두 실제로 학습했다.** 새30개 공개 GraphPFN 기반 checkpoint, 총6,912 optimizer updates다. 기존 모델에 사후 mask 선택만 적용한 실험이 아니다. Primary는 새64 base graph, source4/96개, 총 context8개, 자기 피처 없는 query24개의 accuracy다. 3개 학습 seed를 graph 안에서 평균했다. 이전 192-node Gaussian/probit 결과와 생성 family·입력·context sampling이 달라 절대 정확도를 빼서 전후 개선으로 해석하지 않는다.','',
 '## Primary 결과','', '| 학습 prior | 입력 | Accuracy % | NLL | Brier |','|---|---|---:|---:|---:|']
 labels={'random':'Random','geometry':'거리/coverage','entropy':'J: entropy 감소','decision':'V: 분류 오류 감소','random_cost':'Random, 2배 updates'}
 for interface in ['mask','pcfi']:
  for p in labels:
   v=s['primary_means'][f'{p}_{interface}'];lines.append(f"| {labels[p]} | {interface} | {100*v['accuracy']:.4f} | {v['nll']:.6f} | {v['brier']:.6f} |")
 lines+=['','각 candidate와 같은 입력 방식의 대조를 짝지은 차이다. CI는64graph paired bootstrap5000회이며3개 seed를 고정한다. 개별95% CI이며 multiplicity 보정은 하지 않았다. J/V 각각 Mask에서 세 대조를 모두 넘어야 primary 통과다.','',
 '| 후보 | 대조 | Accuracy 차이 %p [95% CI] | NLL 차이 [95% CI] |','|---|---|---:|---:|']
 for arm,cs in s['comparisons'].items():
  for name,ms in cs.items():
   d=ms['accuracy'];n=ms['nll'];lines.append(f"| {arm} | {name} | {100*d['mean']:+.4f} [{100*d['ci95'][0]:+.4f}, {100*d['ci95'][1]:+.4f}] | {n['mean']:+.6f} [{n['ci95'][0]:+.6f}, {n['ci95'][1]:+.6f}] |")
 lines+=['',f"**사전 primary 판정: J={s['gates']['entropy']}, V={s['gates']['decision']}.** 정확도 gate를 NLL/Brier나 특정 subgroup으로 대체하지 않는다. CI가0을 포함하는 것은 동등성 증명이 아니다.",'',
 '## 관측 위치가 실제로 달라졌나?','',
 '두 후보는 source budget·graph·feature/label law·초기화·CE를 유지하고 학습 mask prior만 바꾼다. 각 후보는50% random과50% 점수 상위3개 mask 중 균일 선택의 mixture다. 학습에 사용한256-draw 점수는 선택 편향을 포함할 수 있으므로 아래는 별도4096-draw로 계산한 첫16 train graph primary 설계의 평균이다. 이는 새 test bank의 모델 선택 지표가 아니며 이 결과를 보고 mask를 바꾸지 않았다.','',
 '| Mask prior | 독립 감사 J | 독립 감사 V |','|---|---:|---:|']
 for p,z in a['independent_score_means'].items():lines.append(f"| {labels[p]} | {z['J']:.6f} | {z['V']:.6f} |")
 same=[v['same_entropy_decision_mask_fraction'] for v in s['design_summary'].values()]
 lines+=['',f"실제 J/V 선택 mask 일치율은6개 context×budget cell에서 {100*min(same):.2f}~{100*max(same):.2f}%였다. J와 V는 다른 정의지만 이번 finite family에서는 설계를 크게 구분하지 못했다. 이 결과로 일반적으로 J와 V가 같다고 주장하지 않는다.",'',
 '## Feature를 활용했나?','',
 '같은 primary query를 유지하고 관측값을 모두 제거하거나, 관측 행끼리 값을 순열했다. 셔플에서는 PCFI를 다시 계산하고, 모든 피처 제거에서는 all-missing PCFI와 같은0행렬을 쓴다. 다음은 standard−intervention 정확도 차이다. Primary와 독립인 메소드 성공 판정으로 승격하지 않는다.','',
 '| Arm | 피처 제거 대비 %p [95% CI] | 피처 셔플 대비 %p [95% CI] |','|---|---:|---:|']
 for arm,z in s['interventions'].items():
  ds=[z[mode]['accuracy'] for mode in ['no_features','shuffled']]
  lines.append('| '+arm+' | '+' | '.join(f"{100*d['mean']:+.4f} [{100*d['ci95'][0]:+.4f}, {100*d['ci95'][1]:+.4f}]" for d in ds)+' |')
 lines+=['','## 전체 관측 조건과 seed','',
 '다음 accuracy는 사전 고정한 모든 standard setting이다. Primary와 동일한 역할로 판정하지 않는다. NLL/Brier는 results/statistics.json의 all_cells에 함께 남겼다.']
 for interface in ['mask','pcfi']:
  lines+=['',f'### {interface}', '', '| 조건 | Random | Geometry | J | V | Random 2배 |','|---|---:|---:|---:|---:|---:|']
  for cell in s['all_cells'][f'random_{interface}']:
   if not cell.endswith('_standard'):continue
   vv=[100*s['all_cells'][f'{p}_{interface}'][cell]['accuracy'] for p in labels]
   lines.append('| '+cell+' | '+' | '.join(f'{v:.4f}' for v in vv)+' |')
 lines+=['','각 seed의 primary accuracy도 그대로 보고한다. Final checkpoint를 seed별로 고르거나 ensemble probability를 만든 값이 아니라, graph별 metric을 seed 평균한 비교다.','',
 '| Arm | seed11 | seed12 | seed13 |','|---|---:|---:|---:|']
 for arm in s['primary_means']:
  lines.append('| '+arm+' | '+' | '.join(f"{100*s['seed_primary'][arm+'_s'+str(seed)]['accuracy']:.4f}" for seed in [11,12,13])+' |')
 lines+=['','## Known-family reference','',
 'Teacher는 graph에서 정해지는64개 latent state를 모두 열거한다. Gaussian feature noise와 Bernoulli label 생성 law는 reference와 일치한다. Posterior 계산은 이 유한 family에서 exact지만 관측의 기대 가치는 Monte Carlo 추정이다. 실제 full latent realization을 reference에 주지 않는다. 별도 Gaussian/probit 논문 구현이나 Grid6의 재현이 아니다.','',
 '| Reference | Accuracy % | NLL | Brier | 평균 posterior 최대확률 % |','|---|---:|---:|---:|---:|']
 for name,v in s['reference_primary'].items():lines.append(f"| {name} | {100*v['accuracy']:.4f} | {v['nll']:.6f} | {v['brier']:.6f} | {100*v['expected_accuracy']:.4f} |")
 d=s['exact_feature_gain']['accuracy'];lines+=['',f"Exact-family의 동일-query feature gain: {100*d['mean']:+.4f} [95% CI {100*d['ci95'][0]:+.4f}, {100*d['ci95'][1]:+.4f}]%p. 관측 feature가 쓸모 있는 family라는 사실과, 학습 observation prior를 바꾸는 것이 유리하다는 주장은 구분한다.",'',
 '## 계산·재현·범위','',
 f"- Train bank:64graph×6context/source cell×4prior =1,536episodes. Test bank:64graph×8setting =512episodes. Neural standard/ablation probabilities19,200배열, reference1,536배열.",
 '- 일반 arm192updates, random_cost384updates. 두 입력 방식·다섯 prior·세 seed를 실제로 학습했다. 모델마다 trainable4,012,259개로 같다. 초기 상태·일반 arm schedule·frozen parameters·최종 checkpoint hash·실제 update 수와 저장된 모든 metric을 재검사했다. 최종 checkpoint30개를 각1회 replay했다.',
 f"- Train bank 전체 cold 생성 {bank['cost']['train']['total_seconds']:.3f}s, 그중 모든 후보 J/V scoring {bank['cost']['train']['cold_scoring_seconds']:.3f}s. 공유 계산을 한 후보에 전부 청구하는 보수적 비교도 남겼다."]
 for it,c in s['cost'].items():lines.append(f"- {it}: random 2배 학습의 추가 시간 {c['mean_extra_random_training_seconds']:.3f}s; 전체 cold bank 생성 시간을 덮는가={c['extra_random_covers_entire_bank_generation']}. Seed 작업이 같은3090에서 겹쳐 실행됐으므로 isolated latency 결과는 아니다.")
 lines+=['- Primary는95.83% whole-node missing이다. Source1/96은98.96%, source8/96은91.67% missing이다. Total context2/8은 class-balanced sampling이 아니다. Query의 실제 정답과 latent mechanism은 mask 설계기의 인자가 아니다.',
 '- 실제 X/Y draw는 코드상 graph 설계 loop 전에 생성하지만 별도 RNG이고 선택 함수에 전달되지 않아 mask와 독립이다. Protocol의 “after selection”은 시간 순서 표현이 부정확했다. 원 source/lock/bank를 바꾸지 않고 runtime/scope_errata.json에 정정했다.',
 '- Context/source/query는 full reference를 제외하고 완전히 분리한다. 각 모델은 test graph에서 학습하거나 mask를 골라주지 않는다. 최종 random·clustered·full·feature-ablation 입력은 모든 arm에 공통이다.',
 '- 새로운 graph seed의 동일 finite family다. 실제 graph·미학습 schema·feature law·무한 latent GP나 과거 공식 prior 전체에 대한 결론이 아니다. 짧은 학습·작은 family·mask score 추정·학생 fitting의 원인을 이 실험만으로 분리하지 않는다.',
 '- 이번에 실제로 재학습한 pipeline 대조는 PCFI다. FISF-PFN·다른 imputer의 동일-prior 학습이나 실제 graph 전이는 실행하지 않았다.',
 '- Source와 protocol은 final bank 생성 전에 잠갔다. 독립 MC 감사·통계·보고서 코드는 runtime/에 별도로 남겼고 학습 코드나 mask/최종 checkpoint를 바꾸지 않았다.', '',
 '## 판정','']
 if not any(s['gates'].values()):
  lines+=['**현재의 J/V mask prior를 메인 메소드로 확대하지 않는다.** 두 설계의 기대 정보 가치는 약간 높일 수 있었지만, 이것만으로 같은 PFN의 heldout accuracy가 강한 random·geometry·extra-training 대조를 안정적으로 넘는다는 기준은 통과하지 못했다. 이는 관측 가치 설계 전체의 불가능성이나 “feature가 필요 없다”는 결론이 아니다. 실제 실행한50:50 top3-mixture·finite family·192-update 범위를 닫는다.','',
  '추가 loss/head나 같은 bank에서 mask 분율·candidate 수·update 수를 바꾸는 구조 구제는 하지 않는다. 재개하려면 기존 결과와 다른 구체적인 정보/계산 병목을 제시하고, 단순 geometry와 같은 비용의 random training을 유지한 새 독립 프로토콜이 필요하다.']
 else:
  lines+=['적어도 한 후보가 사전 Mask primary를 통과했다. 같은-prior PCFI strong floor와 feature interventions, seed 안정성 및 비용을 별도로 판단해야 하며, 작은 finite-family 결과를 범용 missing/heterogeneous GraphPFN 성과나 확정 novelty로 확대하지 않는다.']
 lines+=['','원시 source·protocol·bank·30 checkpoint·모든 확률은 workspace에 보존했다. 공유 snapshot에는 raw bank와 checkpoint를 제외하고 계산 정의·source·lock·설계 감사·전체 조건·seed별 통계·원본 보고서를 포함한다.']
 (ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('REPORT_READY')
if __name__=='__main__':main()
