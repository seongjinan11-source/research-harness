# FP01 result review — experiments / statistics

2026-11-04. 검토 완료 시점에는 `results/artifact_manifest.json`, 전체 evaluation indices, audit/summary/paired/budget/report가 모두 완성되어 있었다. Source/bank/예측/결과를 수정하지 않았고 새 forward·학습·평가 job·branch 전환을 하지 않았다. 이 reviewer는 bank 및 locked analysis 구현에도 참여했으므로 완전한 외부 심사는 아니다. 이번에는 summary를 그대로 승인하는 데 그치지 않고 primary 저장 NPZ 2,304건의 accuracy/NLL과 stratified bootstrap을 별도 read-only 코드로 재계산했다.

## Verdict

**FP01의 사전 지정 adaptive-method 성공 판정은 False가 맞다.** C−B가 +0.952%p이고97.5%CI가 [−0.585,+2.696]%p여서 CI 하한과1%p point threshold를 모두 충족하지 못했다. C−C_global의 +1.187%p [ +0.213,+2.446 ]%p 및 C의 더 낮은 NLL은 실제 부분 성과로 보존해야 한다. 그러나 full 조건의 절대 feature 추가가치, RP 노출 재학습의 필요성, 반복 사용한 실제 graph에서의 안정적 feature utility는 확인되지 않았다. 이는 의도한 finite conditional model의 완성된 음성/부분양성 결과이며, 이번 검토에서 이를 뒤집을 구현 오류를 발견한 것은 아니다. Full-method success나 실제 missing-feature 문제가 해결됐다는 주장에는 부족하다.

## 1. 실행·감사와 사전 통계 규칙

- `audit.json`:12fits, 새3,072 successful updates/6,144 task presentations, target update0. Synthetic696bank rows×4arms×3seeds×2graph arms=16,704예측, real75×4×3×2=1,800예측. 재사용 public/PCFI/FISF 및2LP의375확률 배열도 별도 검증됐다. 전체 index18,879건은18,504새 예측과375재사용 baseline의 합이지18,879독립 표본이 아니다.
- Artifact manifest에 연결된 summary/paired/audit/budget/report, train/dev/final manifests, full prediction index 해시를 확인했다. Locked `analysis.py` 해시는 `a0202f36...`로 source lock과 일치한다. 원 train512 replay는512개 새 독립 과제가 아니며 inherited graph cost와 구별해야 한다.
- Primary는 full/projected,50/75/90/95%에서 source≥2인 own-feature-missing query다. Rate4개와 fixed training seeds3개를 graph 안에서 평균하고,12개의 N×K층마다4graph를 복원추출해48base graphs를 유지하는5,000회 bootstrap이다. 두 co-primary accuracy 비교의 각각97.5%two-sided percentile CI는 Bonferroni familywise95% 명세와 일치한다.
- 별도 NPZ 재채점2304건 및 동일 층화 재표집으로 두 primary point estimate와 CI를 수치 정밀도 내에서 재현했다. 노드/마스크/rate/seed를 독립 graph로 늘려 세지 않았다. 이 CI는 고정3training seeds와 선택한 shape/scalar 층화 family에 조건부이며 seed 모집단·real graph 모집단 또는 unseen-family 보장은 아니다.
- NLLguard는 **평균 차이≤.01nats**인 사전 실용 기준이다. CI 기반 noninferiority 검정을 통과했다는 뜻은 아니다. Absolute C−none도 사전 gate에서는 point-positive를 요구한다. 이 조건과 더 강한 통계적 feature-utility 입증을 구별해야 한다.

## 2. 긍정적 결과와 실패한 gate를 함께 보존한다

| Full/projected primary | Accuracy | NLL | Brier |
| --- | ---: | ---: | ---: |
| A |52.817%|1.003287|0.525383|
| B |52.800%|1.003349|0.525415|
| C_global |52.565%|0.998006|0.524090|
| C |53.752%|0.974004|0.511943|

| 사전/보조 대조 | Accuracy 차이%p [구간] | NLL 차이 |
| --- | --- | ---: |
| C−B, co-primary |+0.952 [−0.585,+2.696],97.5%|−0.029345|
| C−C_global, co-primary |+1.187 [+0.213,+2.446],97.5%|−0.024002|
| C−none, absolute feature effect |+0.579 [−0.855,+1.929],95%보조|−0.002459|

C−C_global의 accuracy 개선은 사전 핵심 비교 중 하나에서 CI가0을 제외하고1%p point 기준도 넘는다. BAcc/macroF1 및 probability scores도 같은 방향이다. 따라서 이전 feature-law pilot와 똑같이 “adaptive 증분이 전혀 없었다”고 요약하면 이 양성 결과를 지운다. 동시에 두 co-primary를 모두 통과해야 하는 계획을 C−C_global 하나로 교체할 수 없다.

C−B와 C−C_global의 NLL 차이 구간도 각각 [−0.069182,−0.001444], [−0.059758,−0.003516]로 음수다. 이는 저장 결과의 유리한 probability-score 신호이며, 사전 accuracy 실패를 대체하는 새 primary가 아니다. 여러 보조 지표/대조의 multiplicity를 일괄 교정한 결과나 calibration 입증으로 쓰지 않는다. C−B Brier의 구간은0을 포함한다.

C absolute observed−none는 accuracy뿐 아니라 NLL/Brier 구간도0을 포함한다. C_global의 feature effect가 −0.608%p, B가 −0.373%p인 데 비해 C가 +0.579%p라는 사실은 **관측 피처로 인한 손상을 줄이는 것**과 **관측 피처가 graph-only보다 추가 정보를 제공하는 것**을 구별하게 한다. Shared none prediction 때문에 C−B와 difference-in-feature-utility는 같은 대조다. 둘을 독립적인 두 성공으로 세면 안 된다.

## 3. RP/shape/class 및 구조적 경계 해석

**A/B의 interface-training 효과는 현재 거의 없다.** 같은 projected primary에서 B−A는 약−0.017%p다. Raw view에서도 두 head의 점수는 매우 가깝다. 이는 이 checkpoint·512task·256update 구성에서 RP 노출만으로 성능이 복구되지 않았다는 결과다. Projection mismatch 자체가 없었다거나 충분한 학습에서도 절대 중요하지 않다는 증명은 아니다.

Class별 primary 기술 통계에서 C−B는 K2 약+2.366%p, K3 +0.042%p, K5 +0.524%p, K7 +0.876%p다. 최대 효과가 K2에 집중되고 K3의 accuracy 차이는 거의0이다. 모든 class 수에 일관된 adaptive 이득이 확인됐다고 쓰지 않는다. 이 표는 작은 subgroup을 보고 승자를 선택하거나 binary로 새 primary를 바꾸기 위한 근거가 아니다.

Orthogonal 진단은 사전 선택12graphs의75/95%다. 동일한12graphs/rates에서 C−B는 Q view에서 약+1.401%p, RP view에서는 약+0.498%p다. 그러나 Q의75%에서는 C가 좋고95%에서는 C가 B보다 낮다. 작은 descriptive 진단이며 projection covariance, conditioning, RMS/clipping, class mean family의 영향을 인과적으로 분리한 것은 아니다. 특정 view/rate만 뽑아 FP01 gate를 구제하지 않는다. Square16→16은 실제 native D→16의 정확한 matching이 아니라 surrogate라는 제한도 유지한다.

99/99.5%에서 raw/RP와 모든 arm의 accuracy가53.173%로 일치하는 것은 source0/1의 class-exchangeable latent-prior 대칭성에 부합한다. Audit의 singleton−none 최대 확률 오차는3.5763e−7, none의 cross-arm 오차는0이었다. 이는 중요한 구현 sanity이며, 정보가 없는 경계에서 feature 활용에 실패했다는 증거도 새 feature inference 성공의 증거도 아니다. 그 조건을 평균해 유리한/불리한 주장을 만드는 대신 사전 분리한 처리가 타당하다.

## 4. Full/edge와 실제 개발 graph

**Synthetic edge panel은 별도의 중요한 반례다.** C−B accuracy는−0.621%p로 CI가0을 포함하지만, NLL은+0.465185 [ +0.314476,+0.620658 ]로 높다. C−C_global도 accuracy−0.560%p, NLL+0.208659 [ +0.117915,+0.302551 ]다. C의 edge feature effect는 graph-only보다 +2.050%p라는 양성 보조 결과를 갖지만, 더 강한 B/C_global 및 probability metrics와 비교하면 adaptive 우월성으로 읽을 수 없다. Edge는 full prior로 학습한 모델의 사전 지정 추론 개입이므로 이 차이의 유일 원인을 centroid family나 최적화 하나로 단정하지 않는다.

Real은5부모 graph의75기존 episodes다. Full Cora/CiteSeer와2,000-node induced PubMed/Coauthors를 섞어 새로운 독립 holdout으로 부르지 않는다. 아래95% feature effects는 모두 같은 graph/context/query에서 C(full features)−C(none)다.

| Graph | Full graph factors | Edge-only factors |
| --- | ---: | ---: |
| Cora |−0.024%p|+0.245%p|
| CiteSeer |+0.003%p|−0.139%p|
| PubMed |−0.440%p|−1.120%p|
| CoauthorCS |약+0.000%p|+0.015%p|
| CoauthorPhysics |+0.205%p|+0.095%p|

Cora/CiteSeer의 이전 feature harm을 크게 줄인 것은 유효한 현상이다. 예를 들어 full CiteSeer의 B−none−3.613%p가 C에서는+0.003%p가 된다. 그러나 거의 graph-only로 돌아온 결과를 강한 관측값 활용로 승격할 수 없다. Cora edge에서는 C_global의+0.663%p가 C의+0.245%p보다 크다. PubMed는 C의95% feature effect가 모든3반복에서 음수다. CoauthorCS의 극소 양의 변화와 Physics의 mixed-repeat 효과를 독립 통계 확인으로 읽지 않는다. Shuffle도 제거와 다른 개입이며, 일부 alignment 차이만으로 추가 정보량을 증명하지 않는다.

95%에서 Cora/CiteSeer/PubMed/Physics source가 각각 모든7/6/3/5classes를 포함한다. CoauthorCS는12/13/12 of15classes다. 이 post-hoc coverage는 모델 입력이 아니며, source가 단순히 한 개라서 real 효과가 없다는 설명은 해당95%에서 맞지 않는다. 그렇다고 충분한 식별 정보가 보장된다는 뜻도 아니다.

강한 baseline과 minority metrics를 지우지 말아야 한다.

- Full CiteSeer에서 C45.827%는 B44.608%, C_global45.607%, LP38.410%보다 높다. 유리한 개발 accuracy는 보존하되 C−none≈0이라는 사실도 함께 둔다.
- Edge CoauthorCS에서 C72.673%는 FISF→PFN69.601% 및 restart LP68.891%보다 높다. 반면 BAcc58.642%는 restart LP61.689%보다 낮다. 모든 class 구분의 일괄 복구라는 결론은 아니다.
- Edge Physics C86.240%는 harmonic LP85.411%보다 높지만 C BAcc58.111%는 restart LP63.911%와 FISF63.472%보다 낮다. NLL0.9307도 harmonic LP0.5955보다 높다. B의 accuracy86.258%보다 adaptive C가 높지도 않다.
- PubMed edge C66.034%는 restart LP73.690%보다 낮다. 공개 imputation pipeline만 이긴 사실을 최고의 고정 추론 대조를 넘은 것으로 바꾸지 않는다.

기존 Coauthors의 nonedge 삭제 복구와 CiteSeer 손상은 계속 남는다. Dataset마다 full/edge를 골라 높은 점수를 제출하는 predictor는 이번에 정의하지 않았고, 그와 같은 oracle 표를 메인 결과로 만들면 안 된다. 새 FP01의 common graph-only role protocol은 역사적 original Factor의 feature-dependent role update와 다르므로 historical checkpoint의 절대 수치를 A의 학습효과로 직접 빼지 않는다.

## 5. 비용·운영 중단과 구현 실패의 구분

`budget.json`의 새12fits 학습 worker-time 합은305.593초다. 이는 실험 전체 elapsed time이 아니다. A/B721params, C_global3params, C751params와 shared inherited768updates/1,536task presentations를 분리한 것은 적절하다. A에만 각 seed512graph encodes 비용이 붙고 다른 arm은 같은 cache를 재사용하므로, A/B fit seconds를 곧바로 공정한 end-to-end 속도비로 읽으면 안 된다.

Inference는 CPU probability return을 포함하고 common graph encoding/Git/sidecar I/O는 제외한 mixed-shape worker 측정이다. Synthetic median C 약24.44ms 대 B9.37ms라는 비용 증가는 기록할 수 있지만, cold/mixed workload·공유 비용·메타데이터 overhead를 포함한 경쟁 latency benchmark나 end-to-end speedup으로 주장하지 않는다. Real은 약114ms 대104ms 정도이며 graph generation/encoding의 별도 peak가 클 수 있다. Parameter 수가 작다는 것으로 저비용 전체 시스템을 증명하지 않는다.

운영상 root는 concurrency1을 유지했고 descriptive Git snapshot만 cache했다. 모든12fit 이후의 실행 변경이다. Runtime wrapper/runner/lock hashes와 각 seed의 COMPLETE 기록, 동일 start/end Git commit이 남아 있다. Seed11 중단 때 NPZ-only orphan 한 건이 원 해시와 함께 격리되었고, 완성된 pair는 strict resume로 보존됐다. 별도 read-only 비교에서 보존 orphan과 동일 case의 최종 재계산 확률 최대 차이는0, query/truth도 동일했다. 첫 resumed nondegenerate C replay는5.9605e−8 차이였다. 이는 모델·loss·통계 규칙 변경이나 점수에 따른 재선택이 아니라 메타데이터 중단/재계산이다.

`budget.json`의 `discovered_failed_attempt_logs=[]`는 results 아래 training failure.json가 없다는 뜻이다. Runtime의 위 운영 중단/one-orphan/replay 비용까지0이라는 뜻은 아니다. 최종 설명에서는 `EVALUATION_EXECUTION.md`와 runtime manifests를 별도로 연결해야 한다. 기존 fit/prediction timing으로 메타데이터 캐시의 모델 speedup을 주장하면 안 된다.

수치 check와 감사는 finite-grid/tree/cavity/gradient/누출 불변성의 근거이며 continuous Gaussian Bayes 정확성이나 loopy convergence를 증명하지 않는다. 현재 성공 gate 실패를 이 구현 오류 때문이라고 재분류할 직접 근거는 이번 검토에 없다. 수치 근사와 family mismatch는 남아 있는 모델 제한으로 기술하면 된다.

## 결론과 다음 경계

유지할 결론: “C는 같은 finite centroid inference의 전역3parameter 대조보다 synthetic primary accuracy와 probability score를 개선했다. RP-trained preferred-class B 대비 composite primary는 통과하지 못했다. 실제 graph에서는 일부 feature harm 감소와 일부 개발 accuracy 이득이 있으나 안정적인 observed-feature 추가가치는 미확인이다.”

피해야 할 결론: “RP mismatch를 고쳐 real feature failure를 해결했다”, “C-global 대비 유의하므로 FP01 성공”, “99.5%에서 feature gain이0이므로 모델이 실패”, “Coauthors edge accuracy가 높으므로 class-balanced/확률 품질도 우월하다”, “runtime interruption은 새 모델 재학습/seed 재선택이다”.

**FP01은 현재 명세로 닫고 실패한 gate와 부분 성과를 함께 보존하는 것이 맞다.** 같은 final bank에서 iteration/noise/head/loss/temperature/feature view를 골라 구제하지 않는다. 후속 과학적 가설이 있다면 새로운 cycle/사전 specification/final seeds가 필요하다. 이번 검토는 새 실험을 실행하거나 현재 결과를 수정하는 요청이 아니다.
