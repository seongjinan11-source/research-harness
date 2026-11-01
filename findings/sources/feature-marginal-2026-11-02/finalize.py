"""Verify finished fixed-bank run and write a reviewable report from saved scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as E


def main():
    root=E.ROOT;c=E.config()
    lock=E.json.loads((root/'results/lock.json').read_text())
    assert lock['sources']==E.sources()
    for s in c['checkpoint_seeds']:
        assert (root/'results'/f's{s}'/'evaluation.json').exists(),s
    E.analyze()
    summary=E.json.loads((root/'results/summary.json').read_text())
    pairs=E.json.loads((root/'results/paired.json').read_text())
    priors=E.json.loads((root/'results/priors.json').read_text())
    decision=E.json.loads((root/'results/decision.json').read_text())
    audits=E.json.loads((root/'results/audit.json').read_text())
    restart=E.json.loads((root/'runtime/parallel_restart.json').read_text())
    for item in restart['preserved_completed_predictions']:
        assert E.C.sha(root/item['path'])==item['sha256']
    manifest=E.json.loads((root/'bank/manifest.json').read_text())['episodes']
    for r in manifest: assert E.C.sha(root/'bank'/r['path'])==r['sha256']
    assert lock['bank_manifest_sha256']==E.C.sha(root/'bank/manifest.json')
    rows=[]
    for seed in c['checkpoint_seeds']:
        rows+=E.json.loads((root/'results'/f's{seed}'/'evaluation.json').read_text())
        path=E.REFERENCE/'results'/f'mask_plain_s{seed}'/'trainable.pt'
        assert E.C.sha(path)==lock['checkpoint_sha256'][str(seed)]
    probability_sum_error=0.;query_counts=[]
    for r in rows:
        with E.np.load(root/r['prediction']) as data:
            probability_sum_error=max(probability_sum_error,float(E.np.abs(data['views'].sum(-1)-1).max()))
            query_counts.append(len(data['y']))
    diagnostics={'probability_sum_max_error':probability_sum_error,
        'query_counts':[min(query_counts),max(query_counts)],
        'observed_rows_by_setting':{str(s):{'min':min(r['observed_rows'] for r in manifest if r['setting']==s),
            'max':max(r['observed_rows'] for r in manifest if r['setting']==s),
            'zero':sum(r['observed_rows']==0 for r in manifest if r['setting']==s)} for s in range(len(c['settings']))},
        'max_sample_clipping_fraction':max(r['diagnostics']['sample_clipping_fraction'] for r in rows),
        'minimum_nodes':min(r['nodes'] for r in manifest),'maximum_nodes':max(r['nodes'] for r in manifest),
        'minimum_raw_features':min(r['features'] for r in manifest),'maximum_raw_features':max(r['features'] for r in manifest),
        'parameter_updates':0,'preserved_pre_restart_files':len(restart['preserved_completed_predictions'])}
    E.write(root/'results/finalization.json',diagnostics)
    def value(g,a,m='nll'):
        return next(r[m] for r in summary if r['group']==g and r['arm']==a)
    def pair(g,a,m='nll'):
        return next(r for r in pairs if r['group']==g and r['contrast']=='joint_samples - '+a and r['metric']==m)
    def interval(g,a):
        r=pair(g,a);return f"{r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}]"
    lines=['# Frozen Graph-correlated Feature Marginalization — 결과', '',
        f"판정: **{decision['decision']}**", '',
        '이전 모델의 반복 numeric update 대신 feature completion을 여러 번 생성해 예측 확률을 평균하는 후보를 실제 구현했다. 이번 실행은 새로운 학습법이 아니라 고정 PFN의 입력·추론 연산 대조다.', '',
        '## 같은 checkpoint에서 비교한 주 결과', '',
        'NLL은 낮을수록 좋다. Uniform/Clustered는 각각 90%와 99.5% whole-node missing의 평균이다. Context 최대4/class, 각 32개 fresh graph와 고정 checkpoint 3개를 평균했다. 모든 arm이 4개 동일 feature-position RNG view를 사용한다.', '',
        '| 입력·연산 | Uniform NLL | Clustered NLL | Uniform acc | Clustered acc | 보조 k1 NLL |',
        '|---|---:|---:|---:|---:|---:|']
    for a in c['arms']:
        lines.append(f"| {a} | {value('uniform',a):.5f} | {value('clustered',a):.5f} | {value('uniform',a,'accuracy')*100:.2f}% | {value('clustered',a,'accuracy')*100:.2f}% | {value('k1',a):.5f} |")
    lines+=['','| Joint samples − 대조 NLL | Uniform 차이 [95% CI] | Clustered 차이 [95% CI] |',
        '|---|---:|---:|']
    for a in c['arms']:
        if a!='joint_samples':lines.append(f'| {a} | {interval("uniform",a)} | {interval("clustered",a)} |')
    lines+=['',
        f"주 gate 통과: **{decision['primary_passed']}**. 사전 조건은 joint가 mean·diagonal·zero·PCFI 각각을 Uniform와 Clustered에서 모두 이기는 것이다. k1이나 일부 seed를 주 조건으로 바꾸지 않았다.", '',
        '**보존할 결과:** 샘플 주변화는 mean 및 PCFI 입력보다 두 주 조건 모두 낮은 NLL을 얻었고 해당 paired CI는 0 미만이다. 따라서 이것을 단순히 전부 실패로 기록하지 않는다. 다만 covariance 보존의 추가 효과는 Clustered에서만 확인됐으며 Uniform에서는 diagonal과 구분되지 않았다. Zero 대비 Clustered CI도 0을 포함해 사전 확장 gate는 실패했다. 이것은 관측값의 실제 정보 활용이나 새로운 학습법의 증명이 아니다.', '',
        'Bootstrap은 rates와 checkpoint seeds를 graph별 평균한 뒤 동일 graph32개를 5,000회 paired resampling했다. 학습 seed 모집단 또는 completion RNG 전체의 불확실성이 아니다. 다중 비교 보정은 하지 않았다.', '',
        '## 단순 prior와 확률 점수', '',
        '| Group | Context prior NLL | Uniform prior NLL | Joint Brier | Zero Brier | PCFI Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for g in ('uniform','clustered','k1','full'):
        lines.append(f"| {g} | {priors[g]['context_prior']['nll']:.5f} | {priors[g]['uniform_prior']['nll']:.5f} | {value(g,'joint_samples','brier'):.5f} | {value(g,'zero','brier'):.5f} | {value(g,'pcfi','brier'):.5f} |")
    lines+=['',
        '각 조건·seed·single view 점수는 `results/summary.json`, accuracy/Brier paired CI는 `results/paired.json`에 보관했다. NLL 개선 여부를 accuracy 유의 개선으로 바꾸지 않는다.', '',
        '## 무엇을 바꿨는가', '',
        '- 관측값을 정확히 고정한 채 missing features를 graph Gaussian law에서 조건부 생성했다. 별도 neural head, posterior loss 또는 추가 학습은 없다.',
        '- Joint와 diagonal은 동일한 rho draws·조건부 평균·각 분산을 유지한다. Rho 내 cross-node covariance만 다르다. Rho 자체는 두 arm 모두 graph 전체에 공유된다.',
        '- Point controls도 4회 실행해 일반 feature-position ensemble 효과와 호출 횟수를 맞췄다. 샘플 평균은 평균 확률이며 평균 logit이 아니다.',
        '- Native NaN은 같은 RMS/mask interface에서 encoder의 missing vector를 쓰는 대조다. 공식 전체 preprocessing 또는 NaN-input 재학습 성과가 아니다.', '',
        '## 해석 범위', '',
        '이 covariance는 명시한 surrogate이며 공식 SCM의 feature posterior를 계산한 것이 아니다. 채널은 rho 조건부 독립이고, completion law는 context labels에 조건화하지 않는다. 완전 Bayesian posterior predictive를 구현했다고 주장하지 않는다.', '',
        'PFN weights는 zero+mask 입력으로 추가 학습했던 checkpoint다. 여기의 PCFI 역시 같은 checkpoint에 PCFI 값을 제공한 입력 대조다. 별도의 PCFI-trained 모델이나 distributional input으로 학습한 PFN의 경쟁력은 평가하지 않았다. 새 graph seeds를 사용했지만 same generator family이며 실제 heterograph 또는 unseen semantic schema 검증은 아니다.', '',
        'Primary gate 실패 시 닫는 범위는 현재 surrogate와 frozen operator의 확대다. 실패 원인을 Gaussian misspecification, frozen PFN의 활용 한계, sampling 수 또는 정보 부족 중 하나로 확정하지 않는다. 실제 hidden-feature 복원 오차도 평가하지 않았다. 다른 family나 joint training을 자동으로 구제 실험하지 않는다.', '',
        '## 재현·실행 감사', '',
        f"- {len(manifest)}개 episode, {audits['saved_probability_files']}개 확률 파일, {audits['scored_probabilities']}개 forward-view 출력을 독립 재채점했다. Query 수 범위 {min(query_counts)}~{max(query_counts)}, model update0회.",
        f"- 실제 graph nodes{diagnostics['minimum_nodes']}~{diagnostics['maximum_nodes']}, raw features{diagnostics['minimum_raw_features']}~{diagnostics['maximum_raw_features']}. 극소 관측 graph도 제거하지 않았다. 관측량은 `results/finalization.json`에 기록했다.",
        f"- Full-observed에서 같은 view의 모든 arm 확률 최대 차이는 {audits['full_observed_max_arm_probability_delta']:.1f}. 샘플 clip 비율 최대{diagnostics['max_sample_clipping_fraction']:.8f}.",
        '- Preflight: 관측 clamp·label 분리·label-independent completion·moment node permutation·Gaussian covariance sampling·반복 결정성·weights 보존 통과.',
        '- 단일 실행에서 checkpoint별 병렬 실행으로 옮길 때 `--seed` 선택만 추가했다. 변경 전 source/lock과 보존 확률 hash를 남겼고, 불완전 episode 전체를 다시 실행했다. 원래 protocol·수식·모델·bank는 유지했다. 동시 실행 시간을 latency 우위 주장에 사용하지 않는다.',
        '- 원래 source·모델 checkpoint·새 bank hash를 최종 재확인했다. 기존 실험 파일을 수정하지 않았다.', '',
        '## 선행연구와 연구 기록', '',
        '분포형 missing input이나 marginalization 자체는 GCNmf/GSPN과 겹친다. 구조 조건부 feature mixture의 최근 단서는 PEMix다. 현재 구현은 이들 재현이 아니며 novelty·실패 해결을 주장하지 않는다. `README.md`, `PEMix_RECHECK.md`와 research-harness의 structured findings에 해당 범위를 남긴다.', '',
        '[GCNmf](https://arxiv.org/abs/2007.04583) · [GSPN](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6be46e85b1e3ff727c107f2673ce8027-Abstract-Conference.html) · [PEMix](https://arxiv.org/abs/2610.18113)', '']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    E.write(root/'results/completed.json',{'report_sha256':E.C.sha(root/'REPORT.md'),
        'source_lock_verified':True,'bank_and_checkpoint_hashes_verified':True,
        'preserved_pre_restart_predictions_unchanged':True,'primary_passed':decision['primary_passed']})
    print('FINALIZED',flush=True)


if __name__=='__main__':main()
