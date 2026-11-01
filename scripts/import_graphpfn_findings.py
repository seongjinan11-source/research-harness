"""One-time provenance-preserving import of local GraphPFN research reports.

All numeric observations below are transcribed from the archived reports.
Reasons explicitly distinguish diagnostics, hypotheses and unidentified causes.
This script never trains a model, changes an experiment, or adds a paper claim.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
MG = WORK.parent / 'mgpfn'
DEST = ROOT / 'findings/sources/graphpfn-2026-11-01'
MANIFEST = DEST / 'manifest.json'

REPORTS = {
    'history': WORK / 'EXPERIMENT_SUMMARY_MISSING_GRAPHPFN.md',
    'missing_closure': MG / 'docs/closure.md',
    'missing_pilot': MG / 'docs/pilot_results.md',
    'missing_context': MG / 'docs/diag_context_results.md',
    'hetero_gate0': MG / 'heteropfn/docs/gate0.md',
    'hetero_gate1': MG / 'heteropfn/docs/gate1_real.md',
    'hetero_gate1p': MG / 'heteropfn/docs/gate1p.md',
    'imputation': WORK / 'fisf_graphpfn/REPORT.md',
    'views': WORK / 'fisf_graphpfn/context_views/REPORT.md',
    'pubmed': WORK / 'fisf_graphpfn/context_views/PUBMED_REPORT.md',
    'source_error': WORK / 'fisf_graphpfn/source_risk/REPORT.md',
    'native_blocks': WORK / 'hetero_missing/REPORT.md',
    'transfer': WORK / 'schema_transfer/REPORT.md',
    'raw': WORK / 'schema_transfer/RAW_REPORT.md',
    'expanded': WORK / 'schema_transfer/EXPANDED_REPORT.md',
    'accuracy_selector': WORK / 'schema_transfer/ACCURACY_REPORT.md',
    'fusion': WORK / 'schema_transfer/FUSION_REPORT.md',
    'reliability': WORK / 'evidence_reliability/REPORT.md',
    'feature_intervention': WORK / 'feature_intervention/REPORT.md',
    'ratio': WORK / 'counterfactual_residual/REPORT.md',
    'lpc': WORK / 'lp_conditioned_pilot/REPORT.md',
    'lpc_factorial': WORK / 'lp_conditioned_pilot/diagnostics/REPORT.md',
    'evidence': WORK / 'relation_evidence_pfn/REPORT.md',
    'evidence_utility': WORK / 'relation_evidence_pfn/frozen_utility/REPORT.md',
    'role_reversal': WORK / 'plain_role_reversal/REPORT.md',
    'binding': WORK / 'binding_evidence_pilot/REPORT.md',
    'jr': WORK / 'joint_role_pfn/REPORT.md',
    'posterior_shift': WORK / 'joint_role_pfn/posterior_shift/REPORT.md',
    'distillation': WORK / 'joint_role_pfn/distillation/REPORT.md',
    'contrastive': WORK / 'joint_role_pfn/contrastive_decoder/REPORT.md',
    'llbp': WORK / 'learned_likelihood_bp_EAJ_failure_retrospective.md',
    'program': WORK / 'relational_task_program/pilot/REPORT.md',
    'regret': WORK / 'regret_prior_pilot/REPORT.md',
    'regret_scope': WORK / 'regret_prior_pilot/SCOPE_NOTES.md',
    'observation': WORK / 'observation_budget_pilot/REPORT.md',
    'rdb': WORK / 'relational_baselines/native_rdb_summary.csv',
    'rt': WORK / 'relational_baselines/rt_budget_results_summary.csv',
    'rt_coverage': WORK / 'relational_baselines/rt_budget_results_coverage.csv',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive():
    if MANIFEST.exists():
        raise RuntimeError('Import already archived. Preserve the snapshot; create a new dated import for updates.')
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for alias, source in REPORTS.items():
        assert source.is_file(), source
        target = DEST / (alias + source.suffix)
        shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        rows.append({'alias': alias, 'original': str(source), 'snapshot': target.relative_to(ROOT).as_posix(),
            'sha256': digest(source), 'bytes': source.stat().st_size})
    MANIFEST.write_text(json.dumps({'date': '2026-11-01', 'sources': rows,
        'scope': 'Verbatim reports and small summary tables only; raw predictions and checkpoints remain in original experiment directories.'},
        ensure_ascii=False, indent=2), encoding='utf-8')


def refs(*names):
    return [(DEST / (name + REPORTS[name].suffix)).relative_to(ROOT).as_posix() for name in names]


def failure(fid, claim, observation, verdict, reason, level, closed, not_closed,
            source, tags, keywords, reopen, baseline, recheck, excluded=()):
    return {'id': fid, 'kind': 'failure', 'claim': claim, 'observation': observation,
        'verdict': verdict, 'failure_reason': reason, 'reason_status': level,
        'excluded_explanations': list(excluded), 'closed_scope': closed, 'not_closed': not_closed,
        'reopen_when': reopen, 'comparison_floor': baseline, 'recheck': recheck,
        'reopen_tags': tags, 'reopen_keywords': keywords, 'tags': tags, 'keywords': keywords,
        'source_refs': refs(*source), 'evidence': '원 보고서 snapshot 확인',
        'revisit_status': 'open'}


def fact(fid, kind, claim, observation, source, tags, keywords, limit=''):
    item = {'id': fid, 'kind': kind, 'claim': claim, 'observation': observation,
        'source_refs': refs(*source), 'tags': tags, 'keywords': keywords, 'evidence': '원 보고서 snapshot 확인'}
    if kind == 'preserved':
        item['do_not_claim'] = limit
    return item


def project(name, title, setting, entries):
    data = {'project': title, 'date': '2026-11-01', 'source': entries[0]['source_refs'][0],
        'setting': setting, 'closed_scope': '아래 각 항목의 구현·조건·예산에 한정. 미실행 방법과 전체 연구 방향을 실패로 세지 않는다.',
        'related': ['GraphPFN'], 'entries': entries}
    path = ROOT / 'findings' / (name + '.yaml')
    assert not path.exists(), path
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110), encoding='utf-8')


def import_new():
    project('missingness-prior', '초기 Missingness-aware GraphPFN',
        '공식 prior의 소형 binary tasks, missingness mechanism mixture, 단일 학습 seed. NaN 학습과 official imputation 경로를 구분.', [
        failure('MISS-F01', 'Missingness prior 추가가 공식 입력 경로와 미학습 MNAR에서 유리하다.',
            'NaN-input Ours−B4는 label_dep에서 +0.75/+1.43/+0.78%p였지만 degree_dep/self_mask는 약 ±0.1%p. Official imputation의 released GraphPFN이 NaN-trained Ours보다 label_dep에서 0.54/0.63/0.12%p 높았다.',
            '범용 우위 미확보·해당 라인 중단',
            'Raw-NaN 기준선이 공식 imputation보다 약했고, official fill만으로도 mask가 잘 드러났다(AUROC 0.97). Context-query mask 관계 불일치에서 mixed-prior 이득이 유지되지 않았다. 이것만으로 모든 원인을 확정하지는 않는다.',
            'supported', 'NaN 학습의 작은 이득을 official-input robustness나 unseen-MNAR 일반화로 승격',
            'Imputed-input에 맞춘 prior의 효과는 미평가다. 더 넓은 MNAR 식별 문제도 기각하지 않는다.',
            ['missing_closure', 'missing_pilot', 'missing_context'], ['observation-mask', 'prior-mismatch', 'distribution-shift'],
            ['MNAR', 'missingness', 'context-query', 'observation mechanism', 'imputation'],
            '관측 메커니즘의 context-query shift를 다루면서 official preprocessing 대비 추가 가치를 보이는 학습 원리가 나올 때.',
            '공개 GraphPFN + official imputation, 동일 예산 B4',
            '동일 입력 경로·초기화·예산의 새 bank에서 context-query mismatch와 미학습 mask를 paired 평가한다.'),
        fact('MISS-U01', 'untested', 'Official imputed-input에 맞춘 Ours-Imp 최종 비교',
            'B4imp는 255/500 update에서 중단, 저장 checkpoint는 update 200. Ours-Imp와 seed 1 및 imp evaluation은 실행하지 않았다.',
            ['missing_closure'], ['observation-mask', 'eval-protocol'], ['Ours-Imp', 'B4imp', 'stopped before evaluation']),
        fact('MISS-L01', 'lesson', '공식 전처리와 신호 대비 headroom을 먼저 확인한다.',
            'No-missing accuracy 약72.9%, majority 약62%, MCAR 80% 손실 약3.5%p. 약한 입력 경로와의 비교만으로 새 prior의 가치를 주장하지 않는다.',
            ['missing_closure'], ['prior-mismatch', 'eval-protocol'], ['headroom', 'official preprocessing', 'majority'])])

    project('input-views', 'PCFI/FISF 입력 확장', 'Cora/CiteSeer 개발 그래프와 PubMed secondary, frozen GraphPFN, context/mask 일치.', [
        failure('VIEW-F01', 'Support metadata·다중 전파 view·PCFI/FISF 입력 결합이 범용 증분을 준다.',
            '8조건 평균 PCFI 69.65%, PCFI/FISF views 69.73%, multiscale 69.61%. 사전 +1%p 양쪽 parent 초과 기준을 만족한 후보는 없었다.',
            '전체 조건 진입 기준 미통과', '출력에 보인 증분이 작고 조건별 부호가 달랐다. 표현 정보 중복, projection 손실, frozen backbone의 활용 한계 중 무엇이 주원인인지는 분리하지 못했다.',
            'unknown', '현재 frozen 저차원 support/view 추가를 범용 메소드로 확대', 'Native cell-wise provenance encoder를 학습하는 방법과 입력 자체를 바꾸는 학습은 평가하지 않았다.',
            ['views', 'pubmed'], ['observation-mask', 'feature-heterogeneity', 'prior-mismatch'],
            ['provenance', 'support', 'missing', 'feature token', 'multi-view'],
            '관측 provenance를 정보 손실 없이 표현·학습하고 equal-width 값 대조를 넘는 장치가 나올 때.',
            'PCFI/FISF parents, equal-width value controls, 같은 비용의 single predictor',
            '같은 차원·추론 비용으로 새 실제 graph에서 parent 둘과 metadata 없는 값 대조를 함께 비교한다.'),
        fact('VIEW-P01', 'preserved', 'PubMed whole-node secondary에서 input view의 작은 긍정 신호',
            '평균 PCFI 70.05%, FISF 71.50%, input views 73.00%, prediction average 71.97%.', ['pubmed', 'views'],
            ['observation-mask', 'feature-heterogeneity'], ['PubMed', 'PCFI FISF views'],
            '한 graph의 secondary이며 전체 조건 우위·novelty·독립 다중 domain 검증으로 확대하지 않는다.')])

    project('source-error', '출처 중첩·공유 보간 오차 attention', '1 seed, 64 updates×2 episodes, 16 heldout synthetic graphs; stochastic feature 위치를 맞춘 paired 결과.', [
        failure('SRC-F01', 'Source covariance를 보정하면 분류가 개선된다.',
            'Vanilla accuracy 57.01%, covariance 57.04%. Covariance−vanilla +0.032%p, graph CI [−0.092,+0.157]. Source correspondence 파괴 시 working error score는 0.1395 증가했지만 분류 변화는 작았다.',
            '독립적인 task-level 이득 미확인', '오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.',
            'unknown', '현재 source-overlap penalty/log 보정을 메인 방법으로 확대', '출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.',
            ['source_error'], ['uncertainty', 'observation-mask', 'amortized-inference'],
            ['covariance', 'provenance', 'shared error', 'correlated error', 'source dependence'],
            '공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.',
            'Vanilla continuation, matched/log/diagonal, covariance-shuffled controls',
            '분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.'),
        fact('SRC-L01', 'lesson', '추가 사전학습의 효과를 오차 보정 모듈 효과로 치환하지 않는다.',
            'Initial 대비 accuracy는 올랐지만 matched continuation 대비 증분은 약0.03%p였다. Balanced accuracy와 macro-F1은 initial 대비 하락했다.',
            ['source_error'], ['eval-protocol', 'uncertainty'], ['matched continuation', 'proxy vs utility'])])

    project('hetero-baselines', 'Native heterogeneous 입력·기존 relational FM 대조',
        'DBLP native block missing 및 ACM/IMDB/Freebase 개발용 비교. Adapted implementation과 공식 모델 성과를 구분.', [
        failure('HET-F01', 'Adapted HetGFD가 cross-type path 입력보다 큰 증분을 준다.',
            'DBLP k=5 full: cross-type 89.96%, adapted HetGFD 89.92%; drop_A 90.18/90.19%. k=20 full은 둘 다93.55%.',
            '새 메소드 headroom 부족', '관계 경로와 남은 native source가 이미 강한 신호를 전달했다. 격차가 작다는 관찰은 확인했지만 redundancy만이 원인이라는 인과 분해는 하지 않았다.',
            'supported', '현재 DBLP adapted diffusion의 작은 차이를 범용 missing hetero-PFN 기여로 확대',
            '원래 HetGFD/HGT 재현이나 다른 schema에서의 결론이 아니다. 다중 block 소실 조건은 별도 여지가 있다.',
            ['native_blocks'], ['feature-heterogeneity', 'graph-structure-use', 'observation-mask'],
            ['native feature', 'cross-type', 'heterogeneous', 'feature block', 'diffusion'],
            '타입별 feature 의미와 관측 pattern을 보존하며 기존 path redundancy를 넘어서는 장치가 나올 때.',
            'Native cross-type paths + frozen GraphPFN, 강한 attribute-free metapath LP',
            '파생 helper feature 누수를 제거하고, 실제 native block graph에서 missing/full 대비와 최강 구조 기준선을 함께 비교한다.'),
        fact('HET-P01', 'preserved', 'Typed path 정보를 column으로 전달해도 기존 PFN이 이미 활용한다.',
            'Gate1-P의 B5g는 k=3의 세 decision graph에서 사전 val-oracle−1%p 기준을 충족했다. 새 learned path adapter를 그 격차만으로 정당화하지 않았다.',
            ['hetero_gate1p', 'hetero_gate0'], ['graph-structure-use', 'feature-heterogeneity'], ['typed path', 'B5g', 'path columns'],
            'Val oracle은 배포 모델이 아니며, 새로운 학습된 heterogeneous PFN의 성과도 아니다.'),
        fact('HET-L01', 'lesson', '기존 relational FM도 현재 node-classification adapter에서는 자동으로 missing에 강하지 않았다.',
            'RDB-PFN의 ACM k20 mean_count_depth5 full81.53%/missing66.55%, official_depth2 full74.87%/missing35.18%. RT budget 표는 64-query·1 seed 대조로 제한된다.',
            ['rdb', 'rt', 'rt_coverage', 'history'], ['feature-heterogeneity', 'context-scale', 'eval-protocol'], ['RDB-PFN', 'PluRel', 'Relational Transformer', 'coverage']),
        fact('HET-L02', 'lesson', '미리 집계된 helper feature를 독립 native 관측값으로 취급하지 않는다.',
            'ACM/IMDB helper feature는 target feature에서 파생된 집계일 수 있음을 확인했다. DBLP는 서로 다른 native source block으로 별도 처리했다.',
            ['native_blocks'], ['observation-mask', 'eval-protocol'], ['derived helper feature', 'leakage', 'native block'])])

    project('path-selection-fusion', 'Context path 선택과 고정 LP/PFN 혼합', '기존 실제 개발 graph, k5/20/class, 같은 context와 public test. 새 dataset holdout 아님.', [
        failure('XFER-F01', 'DBLP에서 통했던 context-LOO path 선택을 다른 graph로 옮기면 강하다.',
            'Expanded LOO는 ACM k20 75.78% vs uniform LP85.55%, IMDB k5 27.53% vs uniform32.04%. Context accuracy로 목적을 바꿔도 IMDB k20 33.55%였다.',
            '고정 선택 규칙의 전이 실패', '소수 context의 ranking이 query utility와 맞지 않았다. 후보 pool 확대와 accuracy 목적 변경만으로는 해결되지 않았다. 추정 분산·risk mismatch의 상대 기여는 미상이다.',
            'supported', '현재 LOO scalar score로 하나의 metapath를 고르는 규칙', 'Context가 유용한 관계를 알려주는 다른 설정이나 relation 의미 추론 일반은 기각하지 않는다.',
            ['transfer', 'expanded', 'accuracy_selector'], ['uncertainty', 'distribution-shift', 'graph-structure-use'],
            ['leave-one-out', 'relation utility', 'few-shot', 'context selection', 'risk estimation'],
            '극소 context의 추정 불확실성과 실제 query risk를 다루며 task 내 utility를 식별하는 방법이 나올 때.',
            'Uniform/union LP, individual paths, fixed GraphPFN path inputs',
            '독립 graph에서 동일 context·후보 pool·선택 예산으로 사전 규칙을 고정해 검증한다.',
            ['Uniform/union 후보가 없어서만 실패했다는 설명은 확대 pool 대조로 충분하지 않았다.']),
        failure('XFER-F02', 'LP와 PFN 확률의 고정 혼합이 각자의 강점을 보존한다.',
            'k20 FISF/LP 50:50: ACM82.20% vs LP85.55%, IMDB42.54% vs FISF44.55%. 같은 query에서 상보성은 있지만 혼합은 최강 단일 예측기를 못 넘었다.',
            '고정 혼합 메인 방향 중단', '서로 맞히는 query가 다르다는 사실이 고정 확률 평균의 utility를 보장하지 않았다. 중복 증거가 원인일 수 있으나 직접 분리한 결과는 아니다.',
            'unknown', '현재 50:50/equal-thirds 사후 확률 혼합', '모든 ensemble이 불가능한 것도, 각 branch가 feature-only/structure-only인 것도 아니다.',
            ['fusion'], ['uncertainty', 'graph-structure-use'], ['ensemble', 'fusion', 'complementarity', 'double counting'],
            '중복 evidence와 decision risk를 실제 관측량으로 구분하며 최강 단일 predictor를 보존하는 원리가 나올 때.',
            'LP alone, PCFI/FISF+PFN alone, compute-matched fixed mixtures',
            '같은 query·context에서 within-task 오류 보완성과 독립 개발/확인 graph의 평균·worst-case utility를 함께 본다.')])

    project('reliability-selector', 'Predictor reliability Gate A/B', 'ACM/IMDB 99.5% missing, k5/20/class, 세 context/mask draw의 개발용 진단.', [
        failure('SEL-F01', 'Context Brier·local reliability로 PFN/LP의 정답을 선택할 수 있다.',
            'IMDB k20 FISF: PFN44.55%, LP35.41%, Global41.51%, Local41.92%, Oracle59.16%. Local은 최강 predictor보다2.63%p 낮았다.',
            'Gate B 실패, Gate C 보류', 'Brier와 accuracy 목적이 불일치하는 사례를 직접 관측했다. IMDB seed11에서 LP의 test Brier0.6624가 PFN0.6672보다 좋지만 accuracy는35.54% vs44.65%였다. Context noise만으로 설명할 수 없다.',
            'supported', '현재 crossfit Brier·local propagation 기반 predictor 선택 규칙', 'Oracle headroom을 실제로 회수하는 다른 risk estimator는 기각하지 않는다. 학습 selector는 이 실험에서 미실행이다.',
            ['reliability'], ['uncertainty', 'observation-mask', 'context-scale'],
            ['predictor reliability', 'Brier', 'decision risk', 'few-shot selection', 'confidence'],
            '극소 context에서 정확도 효용과 uncertainty를 함께 식별하는 추정법이 나올 때.',
            '최강 단일 PFN/LP, global/local selector, fixed mixture',
            'Query 정답 없는 입력만으로 실제 선택 utility를 독립 graph에서 평가하고 proper score와 accuracy를 분리한다.',
            ['Context Brier 추정 noise만의 문제라는 설명은 실제 test Brier/accuracy 역전으로 충분하지 않다.']),
        fact('SEL-P01', 'preserved', 'Within-query complementarity는 실제로 있다.',
            'ACM k20 FISF oracle90.43%, IMDB59.16%. Oracle은 query 정답으로 만든 상한이다.', ['reliability'],
            ['uncertainty', 'graph-structure-use'], ['oracle', 'complementarity'], '큰 oracle gap만으로 selector 학습의 성공 가능성을 입증하지 않는다.')])

    project('feature-intervention', '실제 관측 feature 기여의 개입 분석', 'ACM/IMDB k20/class, 99.5% missing의 Full/No-observed/Shuffle 및 독립 FISF injection 통제.', [
        failure('FEAT-F01', 'IMDB의 PFN 우위는 남은 실제 관측 feature의 평균 추가 효용이다.',
            'IMDB FISF Full44.55%→No-observed44.51%; controlled Full−Zero0.03%p. 제거 전후 argmax는 평균30.47% 바뀌었으므로 feature가 예측에 무영향이라는 뜻은 아니다.',
            '해당 feature-reliability 동기 약화', '평균상 실제 값 효과가 거의 상쇄됐다. 구조 입력 PFN과 synthetic-input 경로의 차이를 feature-only evidence로 오인할 수 없다. Node별 양·음의 효과가 왜 상쇄되는지는 미상이다.',
            'supported', '현재 IMDB/99.5% 결과의 PFN−LP 격차를 실제 feature 복원 이득으로 설명', '다른 결측률·graph와 query별 feature utility는 기각하지 않는다.',
            ['feature_intervention'], ['observation-mask', 'uncertainty', 'graph-structure-use'],
            ['feature intervention', 'predictive utility', 'alignment', 'synthetic injection'],
            '실제 관측 feature의 양·음 utility를 누수 없이 분리하거나, 유효 feature가 남는 조건을 특정한 방법이 나올 때.',
            'Structure-only PFN, no-observed end-to-end, independent-injection controlled Full/Zero',
            'Mask·context·query·graph를 맞춘 제거/셔플 개입에서 실제 값 경로와 synthetic injection 변화를 분리한다.')])

    project('counterfactual-ratio', 'Frozen counterfactual PFN residual', 'λ=1 고정, LP × PFN-full/PFN-neutral; 실제 개발 graph, 새 학습 없음.', [
        failure('RATIO-F01', 'Full−Neutral log probability가 LP에 더할 새로운 feature evidence다.',
            'FISF ratio ACM82.35%, IMDB34.45%; LP85.55/35.41%, Full PFN74.36/44.55%. Controlled ratio ACM82.22/IMDB34.50%도 개선하지 못했다.',
            'Frozen 비율 보정 중단', 'Neutral이 정확한 feature marginalization이 아니므로 비율을 likelihood ratio로 해석할 근거가 없다. IMDB LP log margin은0.134~0.151로 작고, residual argmax 변경률34.68%였다. 원인별 기여는 분리하지 않았다.',
            'supported', '현재 frozen full/neutral 확률 비율의 λ=1 보정', 'LP-conditioned 학습과 일관된 생성모형의 likelihood ratio 일반은 별개다.',
            ['ratio', 'feature_intervention'], ['uncertainty', 'amortized-inference', 'graph-structure-use'],
            ['likelihood ratio', 'counterfactual', 'neutral', 'evidence correction', 'double counting'],
            '통제 입력의 의미와 feature marginalization을 정의하고 확률 비율이 유효한 증거가 되는 조건을 제시하는 방법이 나올 때.',
            'LP/PFN alone, 50:50, controlled neutral',
            '먼저 feature intervention으로 순효용을 확인하고, 동일 구조·주입 아래 ratio가 정답 방향 보정인지 새 graph에서 평가한다.')])

    project('feature-label-coupling', 'Feature↔Label 양방향 추론', '10K 규모 quadratic prototype, 합성1536 tasks, 2 seeds, 384 updates; 공식 GraphPFN 아님.', [
        failure('COUP-F01', 'Label→feature 반복 피드백이 일방향 추론보다 유용하다.',
            'Matched 일방향72.72%, 양방향72.66%; 차이−0.06%p CI[−0.50,+0.39]. Shape shift 차이−0.01%p CI[−0.52,+0.55].',
            '피드백 증분 미확인', '피드백의 추가 정보를 확보하지 못했다. 관측 feature가 전혀 없으면 latent H가0에 머물고 피드백 제거와 출력이 같았다. 이 구조적 퇴화가 모든 조건의 미개선을 설명하지는 않는다.',
            'supported', '현재 quadratic coupling을 더 반복·확대하는 방향', '비선형 label-conditioned feature inference와 다른 유효 prior는 평가하지 않았다.',
            ['history'], ['graph-structure-use', 'amortized-inference', 'observation-mask'],
            ['bidirectional', 'feature-label', 'feedback', 'latent feature', 'quadratic'],
            '라벨 피드백이 실제로 새로운 feature 상태를 만들고 그 정보가 일방향 모델에 없음을 보이는 추론 연산이 나올 때.',
            '동일 예산 일방향 모델, 피드백 제거, 강한 context ridge baseline',
            'Zero-observation 퇴화와 nonzero 조건을 나눠 같은 초기화·task·예산으로 실제 예측 증분을 비교한다.')])

    project('lp-conditioning', 'LP-conditioned structural residual', '소형 일방향 prototype에 LP offset/내부 conditioning 추가, 2 seeds 및 새 paired factorial; 새 schema 아님.', [
        failure('LPC-F01', 'LP internal conditioning이 output offset과 Plain을 일관되게 넘는다.',
            'Matched conditioned−Plain−0.35%p, heldout_shape+0.90%p; offset 대비−0.03/+0.35%p. 새 factorial 내부 효과+0.36~0.52%p이나 relation 추가효과+0.10 CI[−0.21,+0.42], feature 효과+0.06 CI[−0.11,+0.23]. NLL/Brier는 offset-only가 더 좋았다.',
            'LP residual tuning·대규모 확장 중단', '보인 이득 대부분과 output offset 효과를 구분해야 했고, schema adaptation 특유의 이득이라는 증거가 약했다. 조기 relation aggregation이 원인이라는 설명은 가설이지 확정 원인이 아니다.',
            'unknown', '현재 LP offset/residual conditioning 확대', 'Relation 정보를 합치기 전 보존하는 구조와 별도 semantic schema 전이는 별개다.',
            ['lpc', 'lpc_factorial'], ['graph-structure-use', 'distribution-shift', 'amortized-inference'],
            ['LP conditioning', 'residual', 'relation evidence', 'schema adaptation'],
            '압축된 LP 확률보다 풍부한 context 정보를 활용하며 output offset만의 효과를 넘어서는 연산이 나올 때.',
            'Plain, offset-only, 같은 파라미터·초기화·학습 budget',
            'Relation 수·feature 폭·missing을 분리한 factorial과 새로운 semantic schema에서 내부 conditioning의 독립 증분을 확인한다.')])

    project('relation-evidence', 'Relation-wise Evidence Head', '소형 shared relation backbone, 3 seeds, consistent/noisy/conflict, k1/3/5/10/20/class, 99.5% missing.', [
        failure('EVID-F01', 'Relation 출처를 보존하는 head가 강한 Plain의 class decision을 개선한다.',
            'Conflict Separate86.09%, Mean85.34%, Plain88.96%. Separate−Plain−2.88%p CI[−3.66,−2.09], Separate−Mean+0.74%p CI[+0.49,+1.03].',
            '현재 Evidence Head No-Go', 'Plain 정답을 잃는 mass5.36%가 구제2.48%보다 컸다. 정보 중복과 representation 간섭 중 원인은 식별하지 못했다. Confidence를 낮춰서 NLL이 좋아졌다는 설명은 후속 분석에서 기각됐다.',
            'unknown', '현재 histogram/evidence head의 확대와 calibration/margin loss로 구제', 'Relation source 정보가 불필요하다는 결론이나 다른 내부 추론 연산의 기각은 아니다.',
            ['evidence', 'evidence_utility'], ['graph-structure-use', 'uncertainty'],
            ['relation evidence', 'decision boundary', 'class margin', 'source preservation'],
            'Plain의 강한 결정 능력을 유지하면서 출처별 정보의 독립 가치를 회수하는 장치가 나올 때.',
            'Plain, Mean/Separate 동일 용량 control, fixed context-weighted evidence',
            '동일 task에서 harmed/rescued partition과 NLL/Brier/accuracy를 함께 비교하고 conflict 강도별 증분을 확인한다.',
            ['과신을 줄여 확률 점수가 좋아졌다는 설명: Plain은 confidence63.99% vs accuracy88.96%로 과소확신이었다.']),
        fact('EVID-L01', 'lesson', '확률 점수 개선을 relation 출처의 새 정보나 accuracy 개선으로 치환하지 않는다.',
            'Conflict NLL0.5320→0.3585, Brier0.2905→0.2044; NLL 개선의 산술90.5%는 Mean에서도 나타났다. Confidence63.99→84.31%였다.',
            ['evidence_utility'], ['uncertainty', 'eval-protocol'], ['underconfidence', 'calibration', 'NLL vs accuracy'])])

    project('plain-role-reversal', 'Plain relation adaptation과 CRR 제안 철회', '기존 Plain, exact-degree bipartite 재배선, 역할 교환·mechanism freeze·label shuffle 진단. CRR 추가 학습은 하지 않음.', [
        fact('ROLE-L01', 'lesson', 'Plain이 전역 relation utility 적응을 못한다는 전제로 새 router를 정당화하지 않는다.',
            '공유 relation MLP에 relation-ID embedding이 없고, context 통계를 따라 역할 교환에 적응했다. Conflict end-to-end는 약2.09%p 하락해 완전 robustness는 아니다.',
            ['role_reversal'], ['graph-structure-use', 'distribution-shift'], ['CRR', 'role reversal', 'context routing', 'relation ID']),
        fact('ROLE-L02', 'lesson', 'k=1의 centered label-alignment 통계에는 식별 한계가 있다.',
            '클래스당 context 하나면 서로 다른 class centered one-hot 내적은 모두 −1/C다. 이 통계만 복잡하게 인코딩해도 새 정보가 생기지 않는다. 재배선 경로 품질 차이도 함께 고려해야 한다.',
            ['role_reversal'], ['uncertainty', 'graph-structure-use'], ['k=1', 'identifiability', 'label alignment']),
        fact('ROLE-U01', 'untested', '새 CRR router·query-conditioned relation inference의 독립 학습',
            'CRR 제안은 Plain과 기능이 겹쳐 철회했다. 기존 모델 진단을 새 router를 실제 학습해 실패한 결과로 세지 않는다.',
            ['role_reversal'], ['graph-structure-use', 'adaptation'], ['CRR', 'query-conditioned', 'local relation utility'])])

    project('binding-product', 'Common-context witness 고정 곱셈', '실제 개발 graph, feature-free LP, 새 학습0회, 고정 same-context product.', [
        failure('BIND-F01', 'Class pooling 전 context identity를 보존해 곱하면 실제 graph에서 좋아진다.',
            'k20 pooling-before-product: ACM76.390% vs uniform LP85.555%, IMDB28.311% vs35.413%. IMDB k20 test row 약96.1%는 공통 seed support가0이었다.',
            '고정 common-witness product 중단', '동일 witness를 강제하는 곱셈이 현재 sparse-support graph에서 대부분의 evidence를 제거했다. 표현력 collision toy가 실제 오류 원인이라는 증거는 없었다.',
            'supported', '현재 모든 relation이 같은 context witness를 가져야 하는 고정 곱셈', 'Task별 variable binding·논리 연산을 추론하는 일반 executor는 별개다.',
            ['binding'], ['graph-structure-use', 'compositional-generalization'],
            ['binding', 'witness', 'logical query', 'shared context', 'sparse support'],
            '공유 witness가 필요한 경우와 별개 witness를 허용할 경우를 관측 context로 구분하는 연산이 나올 때.',
            'Uniform LP, class-pooling 후 product, 같은 graph의 강한 PFN',
            '먼저 실제 common-support 빈도와 오류의 관계를 확인하고, 고정 곱셈이 아닌 task-adaptive binding의 independent gain을 비교한다.')])

    project('joint-role-initial', '초기 JR-PFN과 decoder 표현력', 'Feature-free binary SBM, target8+helper4, 총 context1/2/4, Exact Bayes 소형 열거.', [
        fact('JR-P01', 'preserved', 'Exclusive의 joint-role 제약 보존은 강한 Plain/factorized를 넘었다.',
            'Exclusive JR−Large Plain+3.20%p, JR−Factorized+1.76%p. Learned posterior−known prior는+0.14%p에 그쳤다.',
            ['jr'], ['uncertainty', 'amortized-inference'], ['joint role', 'exclusive', 'factorized'],
            '모든 prior에서 우월하거나 context에서 미지 joint dependence를 새로 식별했다는 성과가 아니다.'),
        fact('JR-P02', 'preserved', 'Context posterior에는 실제 predictive value가 있었다.',
            'Exclusive 총 context4에서 역할 정보0.12447 nats, exact role update 기대 NLL 이득0.02278. Decoder/predictive 병목도 함께 있었다.',
            ['posterior_shift'], ['uncertainty', 'amortized-inference'], ['posterior information', 'Bayes', 'conditional decoder'],
            '오라클 posterior의 이득이 곧 학습 모델의 성과나 효율을 보장하지 않는다.'),
        fact('JR-L01', 'lesson', 'Positive-only diffusion의 구조적 불가능성은 loss로 고칠 수 없었다.',
            'Context label이 모두 A일 때 당시 decoder는 모든 query에서 p(A)≥0.5. Exact teacher는 nonedge로 반대 class를 지지할 수 있었다. 이후 contrastive/BP decoder가 이 제한을 넘었다.',
            ['distillation', 'contrastive'], ['graph-structure-use', 'observation-mask'], ['positive diffusion', 'nonedge', 'expressivity']),
        fact('JR-L02', 'lesson', '이미 해결한 중간 병목을 현재 미해결 문제로 다시 제안하지 않는다.',
            '양의 diffusion 표현력 문제는 contrastive decoder로 전환했다. 이후 EAJ/R1의 경쟁력 실패는 별개의 문제이며 기존 LLBP 항목에 보존한다.',
            ['contrastive', 'llbp'], ['eval-protocol', 'amortized-inference'], ['superseded', 'resolved blocker', 'chronology'])])

    project('relational-program', 'Compiler–Executor GraphPFN pilot', 'Fully observed Boolean graph, 문법222개(학습186/미학습36), train/test512/64 graphs, 3 seeds, context총4/8/16.', [
        failure('PROG-F01', '고정 executor 위 compiler가 조합 일반화와 품질·비용에서 강한 대조를 넘는다.',
            '미학습 조합: compiler61.59%/NLL0.68349, Plain56.15%/0.68425, relational features+TabPFN85.18%/0.33352, ProgramGrid87.73%/0.23073. Grid latency도 더 짧았다.',
            '현재 compiler 확장 보류', '관측 정보는 Grid/TabPFN이 회수했지만 compiler는 context 증가에도 약61.6%에 머물렀다. 표현·search guidance·학습 최적화 중 정확한 원인은 식별하지 못했다.',
            'unknown', '현재 작은 compiler를 missing·대형 heterograph로 확대', '프로그램 합성·library learning 일반은 기각하지 않는다. 후보 문법 자체는 학습 중 모두 제공했고 missing은 실험하지 않았다.',
            ['program'], ['program-synthesis', 'compositional-generalization', 'amortized-inference'],
            ['compiler', 'program', 'library', 'search guidance', 'binding'],
            'Context에서 계산 구성을 더 효율적으로 식별하고 bounded enumeration/강한 PFN feature 기준선을 넘어서는 장치가 나올 때.',
            'Cached ProgramGrid grammar prior, relational execution features+TabPFN, matched Plain',
            '같은 grammar와 program prior, 후보 구조 접근, fresh graph와 heldout composition에서 품질과 end-to-end 비용을 맞춰 비교한다.'),
        fact('PROG-U01', 'untested', 'Primitive library를 실제로 학습하거나 missing hidden fact를 공동 주변화하는 확장',
            '실행된 pilot은 고정 grammar/executor의 fully observed 비교다. BDD/weighted model counting, learned library, 무한 grammar 탐색 성과는 없다.',
            ['program'], ['library-learning', 'program-synthesis', 'uncertainty'], ['library learning', 'hidden fact', 'weighted model counting'])])

    project('reference-regret', 'Reference-regret prior curriculum', '공개 GraphPFN continued training, 새64/32 graphs, 3 schedule seeds, 15 runs. Finite bank sampling만 변경.', [
        failure('REG-F01', 'Reference-regret priority가 hard-loss sampling 및 같은 비용의 Random보다 유리하다.',
            'Matched NLL Regret0.76377, Loss0.76467, RandomBudget0.76266. Regret−Loss−0.00090 CI[−0.00543,+0.00404], Regret−RandomBudget+0.00111 CI[−0.00145,+0.00387].',
            '메인 방법 확장 보류', 'Reference의 독립 utility와 채점 비용을 정당화하지 못했다. Stale priority·짧은 학습·input loss 중 원인별 영향은 분리하지 않았다. Loss와 regret의 초기 Spearman0.4264는 다른 task를 강조함을 보일 뿐 효용 증명이 아니다.',
            'unknown', '현재 cached finite-bank reference-regret sampler 확대', '학습된 생성기·library·무한 stream prior learning을 구현하거나 실패시킨 것은 아니다.',
            ['regret', 'regret_scope'], ['pretraining-data', 'prior-mismatch', 'inference-cost'],
            ['regret', 'curriculum', 'task generation', 'hard example', 'reference cost'],
            'Reference의 추가 감독 비용을 회수하면서 실제로 회수 가능한 task difficulty를 식별하는 학습·생성 원리가 나올 때.',
            'Random, Manual, Loss priority, cold-score-cost-matched RandomBudget, frozen FISF',
            '새 graph bank와 고정 비용으로 Loss/RandomBudget을 함께 넘는지 확인하고 task weighting과 generator 학습을 분리한다.'),
        fact('REG-P01', 'preserved', '같은 update 수의 Random 대비 작은 Matched NLL 개선은 있었다.',
            'Regret−Random−0.00185 CI[−0.00353,−0.00034]. Accuracy58.78→58.79%로 정확도 개선은 미확인.',
            ['regret'], ['pretraining-data', 'uncertainty'], ['regret', 'matched updates'],
            'Loss priority·같은 비용 대조·OOD를 넘었다고 확대하지 않는다.'),
        fact('REG-L01', 'lesson', 'Singleton observed feature centering은 입력 정보를 없앨 수 있다.',
            'Test99.5% whole-node에서 PCFI input 전체가0인 episode는 uniform20/32, clustered32/32. 관측 행0~1개에서 centering 손실을 직접 확인했다.',
            ['regret', 'regret_scope'], ['observation-mask', 'feature-heterogeneity'], ['singleton', 'centering', 'all zero', 'normalization']),
        fact('REG-U01', 'untested', '학습된 prior generator·새 mechanism library',
            '실행한 것은 같은 official generator의 finite task bank를 재가중하는 pilot이다.',
            ['regret'], ['pretraining-data', 'library-learning'], ['neural generator', 'library learning'])])

    project('observation-budget', 'Observation-aware subgraph policy', 'Frozen 공개 GraphPFN, 새64/32 graphs, 2 policies×3 seeds, node budget64 학습·32/64/96평가, query8/graph.', [
        failure('OBS-F01', 'Mask-aware 학습 정책이 강한 경로 규칙을 넘는다.',
            'Budget64 Observation/Paths NLL matched1.07193/1.05353, cluster1.08551/1.07685. 차이+0.01839 CI[−0.06529,+0.10213], +0.00866 CI[−0.11473,+0.11820].',
            '주 성공 기준 미통과·확장 보류', 'Mask를 보는 것은 유용했지만 비슷한 효과를 고정 shortest-path closure도 얻었다. Joint subset 표현력·stochastic train/deterministic inference·학습 budget 중 정확한 원인은 분리하지 않았다.',
            'unknown', '현재 mask/geometry shared-node MLP를 더 학습해 메인 방법으로 확대',
            '모든 learned retrieval을 기각하지 않는다. Budget32 cluster 차이−0.19284 CI[−0.39771,+0.00284]는 보조 개발 신호이며 승리가 아니다.',
            ['observation'], ['observation-mask', 'context-scale', 'graph-structure-use'],
            ['subgraph', 'halo', 'retrieval', 'observation policy', 'evidence path'],
            '동일 resident budget에서 mask-aware 공동 부분그래프 구성이 강한 경로 규칙보다 나은 utility를 보이는 장치가 나올 때.',
            'Evidence-paths, PPR, equal-capacity structure policy, random, context prior',
            'Fresh graph에서 사전 budget·mask gate를 고정하고 preprocessing/search를 포함한 비용과 label 예측을 함께 비교한다.'),
        failure('OBS-F02', '부분그래프 선택 개선으로 경쟁력 있는 확률 예측을 확보했다.',
            '사후 Laplace context-prior NLL0.85241/Brier0.55736. Budget64 Observation의 matched NLL은+0.21952 CI[+0.12555,+0.31655], cluster+0.23310 CI[+0.14044,+0.32727] 더 나빴다.',
            '단순 확률 기준선에도 경쟁력 부족', 'Frozen PFN끼리의 상대 개선이 단순 class prior 대비 utility를 보장하지 않았다. Calibration·적은 context·subgraph mismatch·전처리 중 원인은 미상이다.',
            'unknown', 'PPR/구조 정책 대비 개선만으로 좋은 missing-feature predictor라고 주장',
            '확률 점수 비교를 accuracy의 같은 결론으로 자동 확대하지 않는다. Prior 대조는 사후 sanity check다.',
            ['observation'], ['uncertainty', 'prior-mismatch', 'context-scale'],
            ['context prior', 'calibration', 'small context', 'frozen backbone', 'probabilistic prediction'],
            '극소 context·불완전 입력에서 단순 prior보다 좋은 확률 정보를 실제로 회수하는 추론 원리가 나올 때.',
            'Uniform class probability, Laplace context prior, full graph, strongest fixed retrieval',
            '학습 전에 단순 확률·majority 대조를 고정하고 NLL/Brier/accuracy를 독립 graph에서 함께 평가한다.'),
        fact('OBS-P01', 'preserved', '관측 상태를 선택 정책에 주는 독립 NLL 효과',
            'Budget64 Observation−Structure NLL matched−0.16128 CI[−0.27147,−0.05871], cluster−0.24695 CI[−0.38936,−0.11246]. PPR 대비도 두 조건 NLL CI가0 미만이었다.',
            ['observation'], ['observation-mask', 'graph-structure-use'], ['mask aware', 'policy input'],
            'Feature 값의 인과적 효용, strong rule 초과, 확실한 accuracy 개선이나 실제 heterograph 전이로 확대하지 않는다.'),
        fact('OBS-L01', 'lesson', '학습 정책의 선택 단계 속도를 end-to-end 우위로 승격하지 않는다.',
            'Matched budget64 selection1.76ms vs Paths13.73ms지만 PCFI+forward 포함 약146.82/155.90ms. Global statistics와 정책학습 비용은 이 합계 밖이다.',
            ['observation'], ['inference-cost', 'eval-protocol'], ['retrieval latency', 'end-to-end cost'])])


def enrich_legacy():
    """Keep original observations, IDs, comments and evidence labels byte-for-byte."""
    path = ROOT / 'findings/llbp-eaj.yaml'
    before = path.read_text(encoding='utf-8')
    legacy = yaml.safe_load(before)
    definitions = {
        'LLBP-F01': ('Component conditional fit 개선이 최종 predictive 개선으로 이어지지 않았다. Positive diffusion의 표현력 제한도 뒤에 확인했지만 전체 미개선의 유일 원인으로 분해하지 않았다.', 'supported', ['auxiliary-loss', 'graph-structure-use'], ['conditional', 'decoder', 'expressivity'], '표현 가능한 decoder에서 conditional 감독의 독립 predictive 이득을 보이는 방법이 나올 때.', 'Predictive-only JR, Plain predictive distillation', 'Decoder와 teacher를 고정하고 auxiliary의 독립 증분을 같은 task·예산으로 비교한다.'),
        'LLBP-F02': ('Posterior fit 감소와 predictive KL 악화가 함께 관측됐다. 목적의 불일치나 decoder 오차 보상 중 원인을 하나로 확정하지 못했다.', 'unknown', ['auxiliary-loss', 'uncertainty'], ['posterior', 'predictive', 'decoder'], 'Posterior 정확성과 deployment prediction을 동시에 보존하는 감독·추론 원리가 나올 때.', 'Predictive-only, conditional-only, exact posterior substitution', '같은 decoder에서 posterior KL과 deployment predictive KL을 따로 평가한다.'),
        'LLBP-F03': ('추가 likelihood auxiliary는 predictive-only보다 악화했다. Local likelihood fit이 label risk와 정렬되지 않을 가능성은 가설이며 정확한 원인은 미상이다.', 'unknown', ['auxiliary-loss', 'observation-mask'], ['likelihood', 'auxiliary', 'predictive'], 'Local factor 감독이 query risk에 도움이 되는 조건을 증명·검증한 방법이 나올 때.', 'Learned BP predictive-only, fitted constant', '같은 potential·BP·role update에서 loss만 바꿔 independent gain을 본다.'),
        'LLBP-F04': ('Potential과 frozen role posterior의 OOD 불일치 및 conditional 근사 오차가 후보였다. 뒤의 coherent-update 대조가 일부 병목을 지지하지만 모든 shift의 원인을 확정하지 않는다.', 'supported', ['distribution-shift', 'amortized-inference'], ['potential', 'role', 'joint', 'likelihood'], '새 graph 통계로 일관된 role/parameter evidence를 추론하고 단순 EB/grid를 넘는 방법이 나올 때.', 'Constant+update, density estimator, Grid6/81', '같은 conditional branches에서 posterior·potential 교차 대조와 새 density/signal bank를 평가한다.'),
        'LLBP-F05': ('같은 BP branches에서 coherent joint weights가 frozen role q보다 좋았다. 큰 개선을 같은 주변분포의 dependence 효과 전체로 돌릴 수는 없다.', 'confirmed', ['amortized-inference', 'distribution-shift'], ['joint', 'evidence', 'role marginal'], '일관된 evidence 갱신을 미지 family에서도 정확하고 저렴하게 구현하는 장치가 나올 때.', 'Joint grid, same-marginal product, frozen-grid', 'Conditional branches와 marginals를 맞춰 role update와 dependence 증분을 분리한다.'),
        'LLBP-F06': ('학습 evidence의 정확도와 공통 BP 비용이 병목이었다. Posterior-only 속도 증분이 end-to-end에 크게 전달되지 않았다.', 'supported', ['amortized-inference', 'inference-cost'], ['evidence', 'amortized', 'inference', 'shared computation'], '강한 classical update의 품질을 유지하며 전체 branch/evidence 비용을 줄이는 연산이 나올 때.', 'Constant+role-update, exact-product, Joint Grid6/81', 'Posterior 추정만의 시간과 shared BP를 포함한 end-to-end 품질·비용을 분리한다.'),
        'LLBP-F07': ('실제 decoder에 맞춘 predictive weights와 정확한 Bayesian posterior 복원은 다른 목적이었다. Direct predictive 개선은 있었지만 classical 기준선을 넘지 못했다.', 'supported', ['amortized-inference', 'uncertainty'], ['posterior', 'predictive', 'decoder', 'aggregation'], 'Decoder 오차를 다루면서 posterior 해석과 predictive quality를 함께 보장하는 방법이 나올 때.', 'Direct predictive EAJ, Constant+update, exact-grid', '두 KL과 실제 NLL/Brier를 함께 보고 calibrated posterior라는 주장을 별도 검증한다.'),
        'LLBP-F08': ('Low density에서는 exact role update도 실패해 Bethe partition 근사만으로 설명할 수 없었다. Potential·conditional·parameter uncertainty 상호작용은 미해결이다.', 'supported', ['distribution-shift', 'uncertainty'], ['potential', 'sparse', 'density', 'Bayesian'], 'Sparse/OOD potential uncertainty와 conditional inference를 함께 해결하는 연산이 나올 때.', 'Constant+update, R1, Grid6/81', 'Low density를 포함한 새 bank에서 exact/Bethe·conditional 교차를 고정해 본다.'),
        'LLBP-F09': ('Self-update를 미분해 학습하는 경로는 작동했지만 R1 대비 이득이 없었다. Optimization·model bias 중 고유 원인은 미상이다.', 'unknown', ['amortized-inference', 'adaptation'], ['unrolled', 'differentiable', 'belief propagation', 'training'], '동일 계산 budget에서 inference-aware 학습의 독립 이득을 보이는 원리가 나올 때.', 'Frozen R1, R0, Constant+update, Grid6', '추론 시 head 제거 효과와 새 학습 효과를 분리하고 initial state·schedule·예산을 맞춘다.'),
        'LLBP-F10': ('교차 대조에서 learned conditional 손해와 비가법 interaction을 확인했다. Role weights만 고치면 된다는 단일 원인 설명은 지지되지 않는다.', 'supported', ['distribution-shift', 'graph-structure-use'], ['conditional', 'interaction', 'potential', 'BP'], 'Potential·partition·conditional 오차를 일관되게 제어하는 추론 연산이 나올 때.', 'Constant/learned weights×conditional 교차, exact update', '네 교차 조합과 exact-role/BP conditional을 함께 비교해 단일 병목 주장을 검증한다.'),
        'LLBP-F11': ('현재 known-family에서 작은 training-support joint grid가 품질·비용 기준을 더 잘 충족했다. Learned 모델 전체가 불필요하다는 보편 원인은 아니다.', 'confirmed', ['amortized-inference', 'model-based-baseline'], ['grid', 'Bayesian', 'model based', 'unknown family'], '작은 명시적 grid가 처리하지 못하는 실제 family/schema에서 우위를 보이는 범용 추론법이 나올 때.', '사전 prior 고정 Grid6/81, Constant+update, R1', '알려진 family·prior 접근 조건과 end-to-end 비용을 맞춘 새 bank에서 비교한다.'),
        'LLBP-F12': ('병렬 grid와의 실제 latency가 비슷했고 posterior/branch 수 감소가 큰 전체 속도 이점으로 이어지지 않았다. 작업 메모리 절감은 별도로 보존한다.', 'confirmed', ['inference-cost', 'amortized-inference'], ['parallel', 'latency', 'memory', 'scalable', 'shared'], '실제 처리 규모나 memory constraint에서 quality를 유지하는 비용 우위를 입증하는 연산이 나올 때.', '병렬 Grid6/81, R1', '동일 hardware·batch·quality 범위에서 latency 반복 변동과 total/working memory 및 최대 처리 graph를 측정한다.'),
    }
    for entry in legacy['entries']:
        if entry['kind'] != 'failure':
            continue
        reason, level, tags, keywords, when, baseline, check = definitions[entry['id']]
        extra = {'failure_reason': reason, 'reason_status': level, 'reopen_when': when,
            'reopen_tags': tags, 'reopen_keywords': keywords, 'comparison_floor': baseline,
            'recheck': check, 'source_refs': ['findings/sources/llbp-eaj-retrospective-2026-11-01.md'],
            'revisit_status': 'open'}
        pattern = r'(^  - id: ' + re.escape(entry['id']) + r'\n)(.*?)(?=^  - id: |\Z)'
        match = re.search(pattern, before, flags=re.M | re.S)
        assert match and '    failure_reason:' not in match.group(0)
        encoded = yaml.safe_dump(extra, allow_unicode=True, sort_keys=False, width=105)
        encoded = '\n'.join('    ' + line if line else '' for line in encoded.splitlines()) + '\n'
        # Insert before original kind, preserving all original content and evidence labels.
        before = before[:match.end(1)] + encoded + before[match.end(1):]
    path.write_text(before, encoding='utf-8')


def main():
    archive()
    import_new()
    enrich_legacy()
    print('Imported snapshots:', len(REPORTS))


if __name__ == '__main__':
    main()
