# Missing-feature GraphPFN 연구 인계서

기준일: 2026-11-03. 최근 업데이트: 누적57편의 문헌과 NSD hidden-token PFN pilot까지 포함한 [실험 결과 요약](RESULTS_SUMMARY_KO.md)과 공유 자료 정리.

## 검토받고 싶은 문제

우리는 **노드 피처가 심하게 결측된 새 그래프에서, target graph 재학습 없이 예측하는 GraphPFN**을 만들고 싶다. 장기적으로 다른 feature 차원과 heterogeneous schema까지 다루되, 현재 실험은 주로 작은 synthetic graph와 일부 실제 그래프에서 수행했다. 여기서 feature missing과 edge observation missing은 다른 문제다.

**불완전한 관측값과 떨어진 context label을 연결하는 추론 구조는 확보했다. 그 구조를 학습하는 것이 강한 통계 모형보다 언제 유리한지는 아직 입증해야 한다.** Factor는 공개 GraphPFN의 개선 모듈이 아니라 별도로 prior 학습한 작은 structured predictor다. 연결 연산·signed evidence·같은 potential의 자기 role-update는 재사용할 구현 자산이다. Distillation의 추가 성과는 현재 teacher 분포 근사 개선이며 분류·분포 전이의 일반적 우위는 아니다.

다음 후보는 현재 함수족으로 부족한 feature–relation 의존성과 그것을 학습·전이하는 추론 연산을 설명해야 한다. 현재 Factor에도 neural potential이 있으므로 모델 전체를 고정 Gaussian factor라고 부르지 않는다. 주요 가정은 binary class/feature-column orientation과 pairwise Ising형 factor이며, 범용 feature semantics를 학습했다고 주장하지 않는다. 이미 구현한 연결 기능에 이름·head·teacher만 추가하는 제안과 구분한다.

이번 검토는 새로운 실험을 즉시 실행하라는 요청이 아니다. 아래 사실과 실패 범위를 검토하고, 설득력 있는 후속 메소드가 있는지 독립적으로 판단해 달라.

## Sheaf-PCFI: synthetic 학습·실제 가중치 고정 비교

2026-11-03에 homophily/heterophily만 남긴 새 PFN pilot을 완료했다. 6arm×3seed,6,912유효updates,64train/32testsyntheticgraphs,실제Cora/CiteSeer/Texas/Wisconsin은가중치고정이다. 주지표는90/99.5% uniform/clustered whole-node missing의결측queryaccuracy다.

| 방법 | Cora | CiteSeer | Texas | Wisconsin |
|---|---:|---:|---:|---:|
| PCFI100+PFN | 50.78 | 32.40 | 28.81 | 22.69 |
| Sheaf-PCFI+PFN | 50.46 | 33.64 | 28.32 | 22.40 |
| 원본피처 PCFI100→RP16+PFN | 52.68 | 33.92 | 32.42 | 24.92 |
| 고정 negative LP | 32.03 | 24.76 | 63.54 | 42.73 |

Synthetic sheaf−PCFI +0.132%p CI[-0.585,+0.968],heterophily실제평균−0.392%p로primarygate를통과하지못했다. **현재2D symmetric-contraction/RP16/384updates구성은확대하지않는다.** Synthetictest의첫좌표직접전달계수a<0비율은0이다. 이통계는2D행렬의다른성분·고유값·반복경로의signed효과를판정하지못하며음의geometry학습의성공/실패로확대하지않는다. Native추론firstrep/seed11에서Texas/Wisconsin일부개선은보존하되primary로교체하지않았다. 원본NSD재현과원본피처공간학습은미검증이다.

초기생성기confound때문에1152updates를폐기하고전부다시학습했다. Checkpoint재실행최대확률차이.007497/선택episodeargmax2.48%변경을공개하며엄격수치재현PASS라고하지않는다. 기존156항목과844paper/source파일은그대로보존했다. [전체보고서](findings/sources/sheaf-pcfi-2026-11-03/REPORT_KO.md),[계산정의](findings/sources/sheaf-pcfi-2026-11-03/METHOD_KO.md),[구조화기록](findings/sheaf-pcfi.yaml).

## NSD hidden-token PFN: 관측에서 시작하는 geometry 갱신

2026-11-03. PCFI warmup/숫자completion대신PFNhiddenstate에서endpointrestriction을학습했다. Mask/PCFI100/identity/static/dynamic을각3seed·1024updates로새로학습했고실제graph학습은0이다. 새64syntheticbasegraph지만family는같고실제4graph는기존developmentbank다.

| 방법 | Cora | CiteSeer | Texas | Wisconsin |
|---|---:|---:|---:|---:|
| pcfi100 | 50.15 | 30.84 | 27.77 | 21.61 |
| identity | 37.16 | 24.24 | 21.43 | 23.28 |
| static | 36.35 | 25.43 | 19.64 | 24.47 |
| dynamic | 38.07 | 22.35 | 19.06 | 23.81 |
| native_pcfi100 | 52.03 | 33.87 | 33.26 | 24.44 |
| negative_lp | 32.03 | 24.76 | 63.54 | 42.73 |

dynamic-mask +1.000059%p CI[0.3503957146611959, 1.6252314285334988]; dynamic-pcfi100 +1.872436%p CI[0.9591497270029059, 2.839915974434959]; dynamic-identity -0.083220%p CI[-0.563776244897672, 0.40735070413055496]; dynamic-static +0.220777%p CI[-0.2799861039889716, 0.7513131305974036]; realhetero {'mask': -1.1361620621828605, 'pcfi100': -3.2524593752169872, 'identity': -0.9208122226674054, 'static': -0.619619386241721, 'native_pcfi100': -7.415763386429836, 'negative_lp': -31.694600458921744}%p; gates={'synthetic': False, 'real_input': False, 'negative_lp_floor': False, 'mask_floor': True, 'expand': False}

현재pilot은확대기준미통과다. 이특정SO2adapter의범위로보존하고모든NSD를기각하지않는다. Numericreplay상태는PASS_WITH_REPLAY_VARIATION이며초기화·taskorder·고정가중치와정보경계를검사했다. [전체보고서](findings/sources/nsd-token-2026-11-03/REPORT_KO.md),[계산정의](findings/sources/nsd-token-2026-11-03/METHOD_KO.md),[구조화기록](findings/nsd-token.yaml).

## 먼저 읽을 자료

이 파일만 읽어도 핵심 판단은 가능하다. 세부 확인은 다음 순서가 좋다.

1. [최신 GP 문헌·결과 검토](findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md)와 [적응 전달 비교](findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md)와 [부호 전달 진단](findings/sources/sign-transport-2026-11-02/REPORT_KO.md)과 [내부 고정 PCFI 재검증](findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md)을 먼저 확인하고, [직전 실제 PCFI 내부화 보고서](findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md)와 [계산 정의](findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md). 실제100-step PCFI를 넣은 네 모델을 학습했으며 이전 generic MLP refiner와 별개다.
2. [Grid6 teacher 판정](findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md), [전체 보고서](findings/sources/grid6-teacher-2026-11-02/REPORT_KO.md), [학습 수식](findings/sources/grid6-teacher-2026-11-02/METHOD.md); 직전 [disjoint-factor 보고서](findings/sources/disjoint-factor-2026-11-02/REPORT_KO.md)와 [모델 수식](findings/sources/disjoint-factor-2026-11-02/METHOD.md)
2. [직전 circuit 보고서](findings/sources/direct-marginal-2026-11-02/REPORT_KO.md), [feature-memory 보고서](findings/sources/feature-memory-2026-11-02/REPORT_KO.md)
3. [누적 실패·보존·미실행 기록](out/research_log.md): 필요한 갈래만 검색한다.
4. `findings/*.yaml`: 각 항목의 `closed_scope`, `not_closed`, `reason_status`, `comparison_floor`, `reopen_when`을 확인한다.
5. `papers/*.yaml`: 확인한 논문의 장치·가정·한계·원문 링크가 있다. 자동 매칭은 해결 근거나 novelty 판정이 아니다.

저장소에는 논문 57편, 연구 기록 164항목, 아이디어 13개가 있다. 39개 프로젝트 묶음이며 실패·중단49, 보존37, 교훈55, 미실행23항목이다. 164번의 독립 실험이라는 뜻은 아니다. 첨부용 ZIP은 버전별 snapshot이다. v16은NSD hidden-token의동일예산학습과static/dynamic비교를추가했다. v15는 Sheaf-PCFI의synthetic-only학습·실제가중치고정pilot과NSD문헌을추가한2026-11-03 snapshot이다. v14는 별도TSFM 공동정보·nuisance frozen 비교까지 포함한다. v13은 diffusion proposal·SMC·ESS의 실제 비교까지 포함한다. v12는 J/V 두 관측 prior의30-model pilot까지 포함한다. v11은56편의 누적 문헌과 관측의 분류 가치에 관한 미실행 재독 메모까지 포함한다. v10은35편과 SSGP/GGP 비교까지 담은 이전 snapshot이다. v9는 직전 참 법칙 없는 적응 전달 snapshot이다. v8은 직전 부호 전달 frozen 진단 snapshot이다. v7는 직전 내부 고정 재검증 snapshot이다. v6는 직전 내부화 학습 snapshot이다. v5는 직전 feature-law Factor snapshot이다. v4는 직전 연구 포지션 snapshot이다. v3는129항목과 Grid6 teacher 후속·실험 요약을 담은 이전 snapshot이고, v2는 disjoint-factor까지다. 대형 raw prediction과 checkpoint는 workspace에 남겨 두었다.

## TSFM 공동 정보·방해 변수 frozen 비교

GraphPFN과 별개의 다음 후보를 실제 실행했다. 새96base task(product64 primary/additive32 control),history512·16개1-step origin,유용2+nuisance8,lag0..3,같은Rademacher주변분포다. Chronos2·이전실제 NI-LoRA·상관screening과모든후보interaction Ridge/Lasso를비교했다. 신규TSFM학습은0이다.

| Product primary | H+uv loss | H+uv+N8 loss |
|---|---:|---:|
| Frozen Chronos2 | 0.52376 | 0.52422 |
| 이전 null LoRA | 0.52378 | 0.52426 |
| 전체 interaction Ridge | 0.08431 | 0.38917 |
| 전체 interaction Lasso | 0.08350 | 0.08552 |

숫자는21quantile평균twice-pinball loss다. Chronos의clean공동정보이득과nuisance손해가함께확인되지않았다. Lasso는참쌍표시없이764개전체후보특징을과거에서fit해history→privilegedoracle gap의99.01%를회수했다. **사전조건에따라2×2학습은보류했으며학습법실패로세지않는다.** Task별fit classical과zero-shotTSFM의차이를명시하고,다른family·실데이터·다른TSFM으로확대하지않는다. [전체보고서](findings/sources/ts-synergy-2026-11-02/REPORT_KO.md),[계산정의](findings/sources/ts-synergy-2026-11-02/METHOD_KO.md),[구조화기록](findings/ts-synergy.yaml).

## Diffusion proposal와 Rényi-2/ESS 실제 비교

기존 Factor3개를 고정하고 conditional denoiser3개를 새로 학습했다(3,072updates). 새96graph/576disjointepisodes,primary matched48graph의6settings와3fixedseed 평균이다.12node·2feature·50/75%missing의 fixed family이며 공개 GraphPFN 본체의 diffusion 개조가 아니다.

| 방법 | Accuracy % | Exact-Factor KL | Grid6 KL |
|---|---:|---:|---:|
| factor_bp | 71.4038 | 0.085031 | 0.091114 |
| factor_exact | 71.6449 | 0.000000 | 0.011446 |
| smc_gibbs_cost | 68.8533 | 0.007171 | 0.017499 |
| diffusion_only | 65.6732 | 0.047351 | 0.060687 |
| diffusion_smc_fixed | 65.1510 | 0.079318 | 0.090732 |
| diffusion_smc_ess | 64.9526 | 0.071562 | 0.085439 |

Diffusion+fixed−비용 대조 SMC accuracy -3.7023 [95% CI -6.3466,-1.2343]%p다. 추론·분류·ESS 증분 gate 모두 실패하여 현재 구현은 확대하지 않는다. 같은 frozen Factor를 정확히 합산하면 Grid6 KL은 크게 낮아지지만 accuracy 증가 +0.2411%p의 CI는0을 포함한다. RNP 학습 목적·DPM-SNC/ReDiSC/RDSMC 원본은 재현하지 않았으므로 해당 방법의 실패로 세지 않는다. [전체 보고서](findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md),[계산 정의](findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md),[구조화 기록](findings/diffusion-factor.yaml).

## J / V 관측 prior 실제 학습 결과

재독 후보의 두 prior를 실제로 실행했다. 30개 새 GraphPFN 기반 checkpoint,6,912updates다. 새64graph·96nodes의 consistent64-state finite family이며, primary는 총 context8개/source4개와 자기 피처 없는 query24개다. 기존192-node Gaussian/probit family와 다르므로 절대 정확도를 전후 개선으로 빼지 않는다.

| 학습 prior | 입력 | Primary accuracy % | NLL |
|---|---|---:|---:|
| random | mask | 55.2951 | 0.690357 |
| geometry | mask | 55.3819 | 0.688894 |
| entropy | mask | 55.7943 | 0.690654 |
| decision | mask | 55.6641 | 0.690199 |
| random_cost | mask | 56.7057 | 0.683048 |
| random | pcfi | 56.9878 | 0.680178 |
| geometry | pcfi | 56.9227 | 0.680114 |
| entropy | pcfi | 57.3134 | 0.680086 |
| decision | pcfi | 56.9444 | 0.680549 |
| random_cost | pcfi | 57.5304 | 0.673814 |

두 Mask primary 모두 통과하지 못했으며 메인 메소드로 확대하지 않는다. Exact-family reference의 feature gain은 +4.0365 [95% CI +1.8880, +6.3802]%p로 별도 보존한다. J와 V는 실제 mask 선택이 약73~88% 겹쳐 일반 우열을 판정하지 않는다. Graph별 paired CI·seed·전체 조건·피처 제거/셔플·cold 생성 비용·random2배 학습 대조를 [전체 보고서](findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md)와 [구조화 기록](findings/decision-value-prior.yaml)에 남겼다.


## Gaussian/probit의 비교 범위와 선행연구 기록

SSGP(IJCAI2007)와 GGP(NeurIPS2018)의 원문·인용·가정·강점·출처별 한계를 추가했다. 현재 Gaussian/probit는 두 논문의 재현이 아니며, 논문 코드로 실행한 대조는 아직 없다. [SSGP](papers/SSGP.yaml), [GGP](papers/GGP.yaml), [문헌과 결과 대조 보고서](findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md).

기존 primary의 보조 paired 비교에서 직접 Gaussian/probit56.154827%는 비특권적 대조11개보다 accuracy·NLL·Brier가 좋았다. PCFI 대비accuracy +4.589585 [95% CI +2.569802, +6.587119]%p,적응 LP 대비 +3.107972 [95% CI +0.741358, +5.489381]%p다. CI는 개별95%이며 multiplicity 미보정이다. 모든 세부 조건·privileged reference·다른 family와의 승리로 넓히지 않는다. 현재 구현은 강한 고정 통계 기준선/추론 자산으로 보존하며 메인 메소드 확정은 보류한다. 기존 frozen 입력 선택기 No-Go와 직접 추론의 긍정 결과를 구분한다. 새학습0.

## 참 법칙 없는 적응 전달과 직접 통계 추론

새64graph·2,048episodes에서0 training으로 비교했다. Primary는 sources8,k4/class,네 label/feature-law 조합 평균이다. 8/192관측=95.83% whole-node missing이며99.5% Bernoulli 조건이 아니다.

| 경로 | Missing-query accuracy % | NLL |
|---|---:|---:|
| 기존 PCFI Plain·일반 복원 | 51.565242 | 0.715365 |
| 같은 PCFI Plain·50:50 두 복원 평균 | 52.354767 | 0.702633 |
| 같은 PCFI Plain·피처만 사용해 적응 | 52.461961 | 0.704598 |
| 같은 PCFI Plain·피처와 context로 적응 | 52.555985 | 0.705381 |
| Context 적응 LP (graph-only) | 53.046855 | 0.694709 |
| 직접 Gaussian/probit 추론 | 56.154827 | 0.672120 |

Context 적응−50:50 accuracy +0.201218 [95% CI -0.127618, +0.537932]%p, context 적응−feature-only +0.094024 [95% CI -0.066909, +0.271023]%p. Primary=False다. 직접 Gaussian/probit의 동일-query feature-removal 차이는 +2.466964 [95% CI +0.884420, +4.166715]%p로 보존한다. 새 neural 학습이나 novelty의 성과가 아니며 frozen 후보 선택기를 확대하지 않는다. [보고서](findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md), [고정 프로토콜](findings/sources/adaptive-transport-2026-11-02/protocol.json), [구조화 판정](findings/adaptive-transport.yaml).

## 부호 전달과 기존 frozen GraphPFN 진단

기존 matched192-update Mask/PCFI Plain checkpoint6개를 새32graph sign-gauged family에서 추가 학습 없이 비교했다. Label heterophily와 feature heterophily를 교차했으며, graph/label/context/mask를 유지하고 feature 부호 패턴을 바꿨다.

| Primary: t1/g1,k4/class,sources8 | Missing-query accuracy % | NLL |
|---|---:|---:|
| 기존 Mask Plain | 52.405028 | 0.709999 |
| 기존 PCFI Plain | 50.190226 | 0.715966 |
| 같은 PCFI Plain + known-law signed 복원 | 56.026584 | 0.707501 |
| 고정 양의 LP | 46.683098 | 0.796353 |
| 고정 signed LP | 56.260834 | 0.687817 |

Known-law signed−PCFI accuracy +5.836358 [95% CI +2.291311, +9.440175]%p. 같은 PCFI→PFN checkpoint에서 부호 법칙을 반영한 복원의 예측적 가치는 확인했다. 그러나 강한 직접 예측·graph-only 대조를 모두 넘는 기준은 통과하지 못해, 새 학습 모델이 필요하다는 근거로 확대하지 않는다. Known-law g는 privileged diagnostic 정보이며 학습된 adaptation이나 새 메소드가 아니다. [보고서](findings/sources/sign-transport-2026-11-02/REPORT_KO.md), [계산 정의](findings/sources/sign-transport-2026-11-02/METHOD_KO.md), [구조화 판정](findings/sign-transport.yaml).

## 내부 고정 PCFI의 새 graph 재검증

기존 checkpoint6개를 새128graph bank에서 추가 학습 없이 확인했다. 원64graph bank를 보고 선택한 후보이며, 생성 family와 학습 seeds11/12/13은 같다.

| 방법 | Missing-query accuracy % | NLL |
|---|---:|---:|
| external | 49.821540 | 0.856885 |
| internal_fixed | 50.041098 | 0.856498 |

Fixed−external accuracy +0.219558, 95% CI [-0.331926, +0.796665]%p. 새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다. NLL/Brier와 비용은 별도로 보고한다. [보고서](findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md), [사전 프로토콜](findings/sources/internal-pcfi-confirmation-2026-11-02/protocol.json), [구조화 판정](findings/internal-pcfi-confirmation.yaml).

## 최신 실제 PCFI 내부화 실험

공개 GraphPFN에서 실제 PCFI를100회 전파+1회 상관 보정으로 구현하고, 외부 처리·내부 고정·context router·strict label-blind router를 각각3seeds/256 updates로 새 학습했다. 이전 generic MLP refiner 재실행이 아니다. Fresh64 final graphs의 uniform/clustered90/99.5%에서 자기 피처 없는 query가 primary다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| external | 49.24327 | 0.867128 |
| internal_fixed | 49.90070 | 0.863062 |
| context | 49.41351 | 0.863255 |
| label_blind | 49.69055 | 0.863001 |

context-external accuracy%p +0.170247 CI[-0.557668,+0.866045], NLL -0.003872 CI[-0.010121,+0.001957]; context-internal_fixed accuracy%p -0.487184 CI[-0.861821,-0.130108], NLL +0.000193 CI[-0.001404,+0.001615]; context-label_blind accuracy%p -0.277040 CI[-0.704964,+0.174764], NLL +0.000254 CI[-0.001276,+0.001713]. 사전 primary=False, full2%p guard=True, 확대 기준=False. 이 구현의 확대는 보류하며, 고유 실패 원인은 미상이다. Context 대조는 labels-only가 아니라 PFN feedback 경로 전체다. [보고서](findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md), [계산 정의](findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md), [구조화 판정](findings/internal-pcfi.yaml).

## 최신 feature-law Factor 실험

새15개 모델을 동일2048 tasks/256 updates/3 seeds/query CE로 학습했다. Gaussian 평균·heavy-tail 평균·분산 차이 law를 모두 학습하고, 새 graph와 미학습 parameter tuple 조합에서 cell만 바꿨다. Graph encoder/factor는 한 역사적 checkpoint로 고정했다. Law 이름 자체의 미관측 전이나 공개 GraphPFN 개선 모듈은 아니다.

| Primary: heldout96 graphs, 자기 피처 없는 query | Accuracy % | NLL | Reference KL |
|---|---:|---:|---:|
| 기존 부호 제약 Factor | 64.6281 | 0.643004 | 0.022515 |
| Flexible-static | 64.4125 | 0.643406 | 0.021705 |
| Flexible-adaptive | 64.4675 | 0.650797 | 0.027830 |
| 고정 law 주변화 exact | 63.4065 | 0.631744 | 0.004696 |
| 고정 law 주변화 BP | 64.0811 | 0.646232 | 0.023510 |
| 새로 학습한 Mask PFN | 53.4883 | 0.693767 | 0.068593 |
| 새로 학습한 PCFI-PFN | 52.9697 | 0.693928 | 0.068545 |

Adaptive−static accuracy +0.054977 CI[-0.976294,+1.037881]%p, adaptive−original -0.160659 CI[-1.202481,+0.821535]%p다. 두 CI에0을 포함해 적응 primary gate를 통과하지 못했다. 이 cell 교체만으로 메인 메소드가 된다는 가설은 닫고 새 loss/head/학습량으로 구제하지 않는다. 원인은 미상이며 부호 제약 제거 자체나 모든 feature-law 학습을 기각하지 않는다. 모든 seed/episode 평균 뒤 **base graph**를 bootstrap했다.

Adaptive 6.759ms, fixed exact 8.861ms, fixed BP 10.077ms. Reference KL은 broad known-family exact reference에 대한 값으로 참 OOD Bayes KL이 아니다. [보고서](findings/sources/feature-law-2026-11-02/REPORT_KO.md), [수식](findings/sources/feature-law-2026-11-02/METHOD.md), [구조화 판정](findings/feature-law-factor.yaml).

## 최신 Grid6 teacher 결과

기존 train2048 tasks·초기화·task order·256 updates를 유지하고 query CE만 exact Grid6 predictive KL로 교체했다. 새6개 모델을 학습하고 기존 CE9개를 새96 graph bank에서 재평가했다. Graph generator 자체를 새로 만든 실험은 아니다.

| 학생 | CE accuracy % / teacher KL | Distill accuracy % / teacher KL | 판정 |
|---|---:|---:|---|
| 공개 Mask PFN, primary | 55.3475 / 0.104788 | 54.6255 / 0.104134 | KL 차이−0.000654 CI[−0.002034,+0.000583], primary 미통과 |
| Factor PFN, secondary | 69.8468 / 0.067257 | 69.5877 / 0.049731 | KL 차이−0.017526 CI[−0.026031,−0.010538], accuracy/NLL 차이 CI는0 포함 |

Matched disjoint48 graphs, total context1/2/4와 observed nodes3/6 및 fixed3 training seeds를 graph 내 평균한 결과다. Mask median59.735ms, Factor8.764ms, Grid6-BP12.425ms이며 품질·2배 latency의 사전 압축 기준은 미통과했다. Factor의 teacher fit 개선은 보존하지만 Mask primary를 대체하거나 새로운 메소드·실제 graph 전이로 승격하지 않는다.

Query 자신도 feature가 없는 사전 secondary에서는 Factor accuracy가 CE64.7164%에서 distill63.5556%로 낮아졌다. 차이−1.1609%p CI[−2.4190,−0.0482]이며 NLL 차이 CI는0을 포함한다. 다중 보조 비교의 CI는 multiplicity 미보정이다. 전체 teacher KL 개선만으로 이 분할에서도 좋아졌다고 주장하지 않는다.

보조 context2/observed6에서 Factor distill의 feature 제거는 accuracy−11.9213%p CI[−19.0972,−4.8611]였다. 기존 CE Factor도 새 bank에서−12.0370%p CI[−19.6759,−4.2824]이므로 실제 disjoint feature 활용의 근거를 보완하지만 그 활용을 distillation이 새로 만들었다고 할 수 없다. 원래 DFACT 기록과 판정은 변경하지 않았다. 자세한 독립 범위와 미해결 원인은 `grid6-teacher.yaml`의 G6T-F01/P01/P02/L01을 확인한다.

## 이전 실험에서 남긴 판단

아래는 서로 다른 과제와 예산의 누적 결과다. 표의 갈래끼리 절대 성능을 직접 비교하면 안 된다. 원인이 분리되지 않은 항목은 원인 미상으로 남겼다.

| 갈래 | 실제로 확인한 것 | 닫은 범위와 남은 범위 | 기록 |
|---|---|---|---|
| PCFI/FISF → GraphPFN 및 입력 결합 | 일부 데이터의 소폭 개선이 전체 조건에서 안정적으로 이어지지 않았다. | 단순 입력 view 결합을 범용 메인 기여로 삼기 어렵다. 모든 imputation 방법을 기각한 것은 아니다. | `input-views.yaml`, `missingness-prior.yaml` |
| LP/PFN 혼합·confidence·Brier selector | Oracle 보완성은 컸지만 실제 context-only selector가 최강 단일 모델을 안정적으로 넘지 못했다. | 이 설정의 사후 선택·평균화는 중단했다. Brier와 accuracy의 최적 predictor가 다를 수 있다. | `reliability-selector.yaml`, `path-selection-fusion.yaml` |
| 관측 feature intervention | 과거 IMDB, 99.5% node missing에서 FISF+PFN은 실제 feature를 제거해도 44.55%→44.51%였다. | 그 조건의 PFN 우위를 실측 feature 때문이라고 설명할 수 없다. 모든 그래프·노드의 feature가 무용하다는 뜻은 아니다. | `feature-intervention.yaml` |
| Frozen counterfactual ratio | LP에 PFN full/neutral 확률 비율을 곱하는 방법은 실패했다. | Neutral PFN은 feature를 정확히 주변화한 분포가 아니다. 비율을 새 feature evidence로 간주하는 근거가 부족했다. | `counterfactual-ratio.yaml` |
| LP conditioning·relation evidence head | 작은 내부 conditioning 이득, Separate>Mean은 있었지만 강한 Plain을 안정적으로 넘지 못했다. | 출처 보존이나 conditioning 일반의 무효가 아니다. 해당 head·offset을 메인 방법으로 확대하지 않는다. | `lp-conditioning.yaml`, `relation-evidence.yaml` |
| 전역 relation router | Plain이 이미 context에 따라 relation role reversal에 적응했다. | 새 전역 router 제안은 중복 때문에 철회했다. 이를 새 모델을 학습해서 실패한 결과로 세지 않는다. | `plain-role-reversal.yaml` |
| Feature↔label feedback·source overlap 보정 | 추가 효과가 없거나 극히 작았다. | 해당 연산의 독립적인 가치가 확인되지 않았다. 정확한 단일 원인은 확정하지 않았다. | `feature-label-coupling.yaml`, `source-error.yaml` |
| Joint relation hypotheses | Exclusive role 제약에서 강한 Plain과 factorized 모델을 이긴 설정이 나왔다. | Joint 제약 효과는 보존한다. 모든 prior에서의 우위나 학습된 posterior 자체의 이득은 아니었다. | `joint-role-initial.yaml` |
| Learned-likelihood BP | Positive diffusion의 표현력 제약을 signed edge/nonedge likelihood로 완화했고 예측이 크게 개선됐다. Unknown pair를 nonedge로 간주하면 안 됐다. | 연산의 효과는 보존한다. Learned potential은 constant potential보다 항상 좋지 않았다. | `llbp-eaj.yaml` |
| EAJ·joint role–parameter evidence | Coherent update는 유효했지만 learned evidence의 오차가 컸다. 별도 posterior head를 제거한 R1도 실제 개선이었다. | 작은 known-family Joint Grid6가 품질·비용에서 강했다. 현재 EAJ/R1을 독립 메인 메소드로 확대하지 않는다. BP 일반을 기각한 것은 아니다. | `llbp-eaj.yaml` |
| 부분그래프 선택 | Mask-aware 학습 정책은 일부 약한 대조보다 좋았지만 관측 source 경로 규칙과 class-prior NLL을 넘지 못했다. | 선택 이후 PCFI·고정 GraphPFN은 그대로였던 실험이다. PCFI 자체를 바꾼 실험과 구분한다. | `observation-budget.yaml` |
| PFN-aware joint imputation | 실제 공개 PFN의 query CE로 imputer를 앞단/중간에서 학습했지만 강한 대조 대비 독립적 성과는 부족했다. | 현재 numeric imputer 구성의 결과다. 완전한 context-conditioned feature posterior는 미실행이다. | `joint-imputation-pfn.yaml` |
| 여러 completion 후 PFN 평균 | Point fill 대비 개선은 있었으나 matched random-fill 대조를 넘는 조건부 completion의 고유 가치는 확인하지 못했다. | 여러 숫자를 넣어 생긴 효과를 posterior inference의 성과로 부를 수 없다. | `feature-marginal-pfn.yaml`, `random-fill-controls.yaml` |
| Mechanism-crossed prior | 원래 NLL gate는 실패했지만 off-diagonal clustered accuracy +3.434%p 신호를 보존했다. | 사후 보조 결과이며 독립 재현·다중 비교 보정이 없다. 이후 memory 실험은 이 효과의 재현이 아니다. | `mechanism-crossed-prior.yaml`, `MCP-P02` |

일반적으로 **추가 학습이 부족해서인지, 정보가 없는지, 표현력·최적화·prior 중 무엇이 원인인지 분리하지 않은 경우가 많다.** 결과가 나쁘다는 이유만으로 그럴듯한 원인을 확정해서는 안 된다.

## 최근 완료한 feature memory 실험

공개 GraphPFN 1.3/LimiX의 기본 sample attention은 label-context 노드를 key/value로 읽는다. Query 자신의 feature residual과 graph adapter 경로는 있으므로 unlabeled node의 feature가 완전히 차단됐다는 뜻은 아니다. Feature memory와 label context를 분리하면 나아지는지 실제 학습으로 비교했다.

- 새 64 train/64 test base graphs, 3 training seeds, 4 arms, 각 512 성공 updates.
- 같은 초기화·과제 순서·파라미터·query CE. 원래 transformer는 고정했다.
- 공개 모델의 두 feature 좌표 묶음을 memory 단위로 사용했다. 완전한 sparse cell-token 모델은 아니다.
- `context`: 기존 방식. `all_nodes`: 모든 노드의 feature memory. `observed`: 관측값이 있는 group만 memory. `pcfi`: 실제 PCFI 입력으로 학습한 기존 방식.
- Primary는 공식/constructed off-diagonal family의 whole-node uniform/clustered 90/99.5% 조건 평균 accuracy다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| Context | 52.3336 | 0.687776 |
| All nodes | 51.9552 | 0.689573 |
| Observed memory | 52.2888 | 0.692156 |
| PCFI | 53.9246 | 0.683446 |
| LP | 53.3203 | 0.777961 |

Observed−Context accuracy는 −0.044759%p, graph 단위 paired 95% CI [−1.501516, +1.460826]이었다. NLL은 오히려 +0.004380 [ +0.000906, +0.007896 ] 높았다. 현재 routing 변경만으로 개선된다는 가설은 지지되지 않았다. Frozen projection, grouping, 학습량과 task 정보량 중 고유 원인은 분리하지 않았다.

## 최근 완료한 직접 주변화 circuit 실험

### 모델과 입력

소형 typed-SBM에서 **normalized graph-conditioned tree mixture**를 새로 구현했다. GSPN/MaM에서 착안했지만 두 논문의 재현은 아니다. MaM을 별도로 학습한 결과도 아니다.

- 12 nodes: target 8/helper 4, relation 2, binary label, 연속 Gaussian feature 2.
- 세 role-prior family인 Independent/Exclusive/Coupled의 descriptor는 모든 학습 모델에 제공했다.
- Graph의 edge/nonedge는 완전히 관측된다. 이 실험에서 결측은 node feature다.
- 구조 encoder는 G와 family만 보고 파라미터를 만든다. 실제 feature와 context label은 leaf evidence로 넣는다.
- 원본 변수 scope가 겹치지 않는 정규화된 tree mixture이고, 결측 Gaussian leaf는 적분값 1로 처리한다.
- 같은 conditional model 안의 정규화·합 규칙·연속 적분을 약 10^-15 오차로 검증했다. 원래 cyclic SBM posterior가 정확하다는 뜻은 아니다.
- Circuit 56,385 parameters. 직접 graph predictor와 공개 Mask GraphPFN을 비교했다. 공개 PFN은 4,012,259 trainable parameters이며 typed context support도 받는다. 파라미터 수는 같지 않다.
- 모든 학습 arm에 같은 2,048 task exposures/256 updates, query CE만 사용했고 3 seeds를 평가했다. Exact predictive distillation이나 새 auxiliary loss를 쓴 실험이 아니다.
- Primary는 새 matched 48 base graphs의 whole-node uniform/clustered 50/90%, 총 context 1/2/4 평균이다. OOD 포함 전체는 192 graphs다.

### 성능과 강한 대조

| 방법 | Accuracy % | NLL | Brier |
|---|---:|---:|---:|
| 공개 Mask PFN | 53.4846 | 0.687080 | 0.493828 |
| Direct graph predictor | 51.1512 | 0.693134 | 0.499986 |
| **Circuit** | **61.3839** | **0.605128** | **0.430740** |
| Grid6-BP | 69.5602 | 0.540755 | 0.375136 |
| Grid6-exact | 69.3680 | 0.541211 | 0.375627 |

Circuit−Mask PFN accuracy는 +7.899306%p [ +3.889009, +11.790761 ], NLL은 −0.081952 [−0.101972, −0.062402]였다. **Neural baseline 대비 개선은 실제 결과로 보존한다.**

반면 Grid6-BP 대비 accuracy는 −8.176257%p [−13.111376, −3.230252], NLL은 +0.064373 [ +0.024247, +0.105302 ]였다. 단일 graph median은 circuit 12.369 ms, Grid6-BP 13.443 ms로 사전 2배 속도 기준도 충족하지 못했다. 단일 GPU 측정이며 외부 GPU 부하는 통제하지 않았다.

Grid6는 알려진 생성 family와 사전에 정한 여섯 parameter support를 이용하는 model-based baseline이다. 실제 test parameter를 제공하는 별도 oracle과 구분한다. 큰 실제 graph의 일반 해법은 아니지만, 현재 실험에서 이겨야 할 강한 비교다.

### 실제 feature 이용과 graph-specific 이득은 구분해야 한다

Context 2/clustered 50%의 보조 intervention에서 관측 feature 제거 시 accuracy −15.45%p, 노드 간 shuffle 시 −13.69%p였다. 이 Gaussian family에서는 실제 feature와 node 대응을 활용한다. Primary와 조건이 다른 보조 결과다.

그런데 설계 고정 뒤 추가한 **known-family feature-only** 대조는 graph를 전혀 쓰지 않고도 NLL 0.605147을 얻었다. Circuit의 NLL 0.605128과의 차이는 −0.000019 [−0.013783, +0.013696]였다. Accuracy 차이 +3.251075%p의 CI [−0.142816, +6.531240]도 0을 포함했다. 동등성 검정은 아니지만 graph-specific 추가 이득을 확인한 결과도 아니다. 이 대조는 Gaussian magnitude 0.8/sd 0.6이라는 정확한 family 상수를 안다.

가장 구체적으로 남은 조건은 **관측 feature가 label context 바깥에만 있는 경우**다. Primary의 사후 분할에서 circuit은 52.69%/NLL 0.69955, Grid6-BP는 64.87%/0.62460이었다. 단순 feature-only는 이 조건에서 class orientation을 알 수 없다. Graph를 이용하면 회수 가능한 정보가 있지만 이 circuit이 충분히 회수하지 못한 조건이다. 사후 기술적 분석이므로 새 primary나 독립 확인으로 쓰지 않는다. Tree 표현력만의 탓으로도 확정하지 않는다.

## 최신 disjoint factor predictor

이 문제를 직접 대상으로 새96 graph bank를 만들고, node class와 feature-column 방향을 latent state로 함께 추론하는 약8천 파라미터 모델을 학습했다. 관측 cell만 factor로 연결하고, graph의 edge/nonedge factor와 finite cavity BP를 통해 context와 feature evidence를 교환한다. 같은 factor의 Bethe score로 role weights를 유도하며 별도 posterior head나 numerical completion은 없다.

모든 arm은 같은2048 tasks/256 updates/3 seeds/query CE를 사용했다. Primary는 새 matched48 graphs의 disjoint 조건이며, context 총1/2/4개와 observed nodes3/6개를 평균한다. Missing은75/50%이며99.5%가 아니다. 이전 bank나 circuit의 절대 수치와 직접 비교하지 않는다.

| 방법 | Accuracy % | NLL |
|---|---:|---:|
| Factor predictor | 65.4004 | 0.628722 |
| Public Mask PFN | 57.0974 | 0.676368 |
| PCFI-trained PFN | 55.2304 | 0.677401 |
| Fixed-mode factor BP | 66.0053 | 0.621022 |
| Grid6-BP | 65.9557 | 0.555143 |

Factor−Mask accuracy+8.3030%p CI[+2.7750,+14.0753], Factor−PCFI+10.1700%p CI[+4.4253,+16.3513]은 보존한다. Neural 대비 NLL 차이 CI는0을 포함한다. Fixed factor 대비 정확도·NLL의 추가 이득은 확인하지 못했고, Grid6-BP 대비 NLL은+0.073580 [+0.019579,+0.150308] 더 높았다. Forward median6.837ms vs11.874ms이며 matched Grid6-exact KL0.056253으로 사전 품질·비용 gate는 실패했다.

Context2/observed6에서 feature shuffle은−11.23%p, graph permutation은−21.76%p로 CI가0을 제외했다. 반면 no-observed−standard는−6.13%p CI[−12.96,+0.93]로 사전 feature-removal gate를 통과하지 못했다. Shuffle 민감도를 feature 추가 순효과의 확정이나 gate 통과로 바꾸지 않는다.

Query 자체도 feature가 없는 사전 secondary에서는 Factor64.60%/NLL0.60005, Mask56.98%/0.67939, PCFI54.48%/0.68409였다. 두 neural 대조 대비 accuracy/NLL 개선 CI가0을 제외했다. 다만 고정 factor 대비 추가 이득은 미확인이고 이 secondary로 primary 판정을 바꾸지 않는다.

**현재 판단은 정확도 성과를 보존하면서 독립 메인 메소드 확대를 보류하는 것이다.** Node class와 latent feature centroid를 함께 추론하는 원리는 이미 [CSBM의 AMP–BP](papers/CSBMInference.yaml)에 있다. 이 pilot이 그 구현을 재현하거나 이겼다는 뜻은 아니다. Unknown feature-law, 새 semantic schema, 실제 heterograph와 큰 graph는 아직 평가하지 않았다.

## 아직 실행하지 않은 범위

다음은 남아 있는 가설이지 검증된 해결책이 아니다.

- 관측 cell을 직접 token으로 만들고 attention까지 학습하는 full sparse architecture. 최근 실행은 frozen backbone의 native group routing이었다.
- 원래 Wander 및 관측 경로를 읽는 observation-aware walk 모델. 이전 부분그래프 선택+PCFI와는 계산 대상이 다르지만, 긴 uniform walk나 기존 label-filtered Wander보다 나은지는 모른다.
- Context labels가 feature 가설의 구성부터 바꾸는 neural feature posterior, 관측 encoding을 공유하는 효율적 다중 가설 처리, 타당한 refinement law를 이용한 marginal consistency.
- PEMix의 장치를 활용한 확률적 첫 feature 변환. Expected token을 통과시킨 Transformer가 전체 posterior predictive와 같지는 않다.
- 실제 MaM 구현, 가변 feature 폭/새 semantic schema/큰 cyclic graph의 circuit 학습.
- Edge observation process가 role/label에 의존하는 MNAR BP. 원래 node feature missing 문제와 구분해야 한다.
- 보존한 mechanism-crossing accuracy 신호의 사전 고정 독립 재현.

## 관련 문헌에서 가져올 수 있는 것

정확한 metadata와 원문 링크는 아래 YAML에 있다. 이 목록이 현재 문헌 전체를 포괄하지는 않는다. 새 후보의 novelty는 추가 원문 검색으로 확인해야 한다.

| 논문 기록 | 이미 알려진 장치와 검토할 차이 |
|---|---|
| [GraphPFN](papers/GraphPFN.yaml) | Synthetic graph prior와 고정 backbone/graph adapter. 기존 모델의 강한 입력·학습 대조를 유지한다. |
| [Ephris](papers/Ephris.yaml) | Context/query feature를 함께 gather하고 압축하는 sparse graph ICL. 단순 feature-memory 분리와 겹침이 크다. |
| [Wander](papers/Wander.yaml) | Walk 기반 graph/task 추론. Label 방문 필터나 긴 walk 자체를 새 기여로 삼을 수 없다. |
| [GSPN](papers/GSPN.yaml) | Missing evidence를 내부에서 주변화하는 probabilistic circuit. 유효한 내부 계산과 원래 graph joint의 정확성은 별개다. |
| [MaM](papers/MaM.yaml) | Conditional/marginal의 합 규칙으로 모델을 구속한다. Discrete 설정을 연속·가변 graph PFN으로 옮기는 문제가 남는다. |
| [PEMix](papers/PEMix.yaml) | 구조 조건부 mixture와 첫 GNN layer에서의 해석적 주변화. PFN 전체에 같은 등식이 성립하지는 않는다. |
| [StructuralMissingPFN](papers/StructuralMissingPFN.yaml) | 불완전 입력에서 직접 예측하며 보조 posterior를 학습하는 PFN. Completion 없이 예측한다는 주장만으로 차별화하기 어렵다. |
| [MCSSL](papers/MCSSL.yaml) | 정보 증가에 따른 martingale consistency. 적절한 조건부 refinement 분포 없이 같은 식만 쓰면 안 된다. |
| [PFNSeparability](papers/PFNSeparability.yaml) | Feature law와 label mechanism의 분리. 우리의 crossing pilot은 원래 gate 실패와 보조 accuracy 신호를 함께 보존했다. |
| [TabImpute](papers/TabImpute.yaml), [GNNmim](papers/GNNmim.yaml) | 각각 query attention 접근과 강한 missing-feature 입력 대조를 검토할 자료다. 두 논문의 task/protocol과 우리 zero-shot 설정을 구분한다. |

## 독립 검토자에게 요청하는 답변

1. **문제 정의를 먼저 비판해 달라.** 현재 관측 정보에서 풀 수 있는 예측 문제인지, known-family oracle gap을 새로운 모델의 필요성으로 잘못 연결하고 있지는 않은지 확인해 달라.
2. **메소드 후보는 최대 두 개만 제안해 달라.** 분리된 feature/label 연결·signed evidence·자기 role-update는 이미 구현했다. 후보마다 현재 factor 함수족으로 부족한 feature–relation 의존성과 이를 학습·전이할 연산을 명시해 달라. 기존 실패 원인이 미상이면 그 가정을 표시해 달라.
3. 후보의 입력, 내부 추론 연산, 학습 목적, 새 graph에서 gradient-free로 하는 일을 수식이나 의사코드로 적어 달라. 새로운 head/loss를 붙이는 것만으로 끝내지 말고 필요한 이유를 설명해 달라.
4. 가장 가까운 선행연구와 이미 실행한 방법을 함께 비교해 달라. 단순 attention, Bayesian averaging, distillation, BP, circuit 또는 PFN 이름 자체를 novelty로 간주하지 말아 달라.
5. 새 모델 학습 전에 점검할 강한 비교와, **한 번의 제한된 실험으로 기각 가능한 기준**을 제시해 달라. 그 뒤 끝없이 진단을 늘리기보다 실제 모델 비교로 이어져야 한다.
6. Accuracy, NLL/Brier, calibration, 계산비용, 실제 graph/schema 전이를 따로 평가해 달라. 개선이 유의하지 않다는 것과 유의하지만 강한 baseline에 진다는 것을 구분해 달라.

설득력 있는 후보가 없으면 없다고 답해도 좋다. 목표를 사후에 바꾸거나, 미실행 범위를 실패로 세거나, 모델 용량·학습량만 늘려 기존 구성을 살리는 제안은 피하고 싶다.

## 바로 전달할 요청문

```text
첨부한 Missing-feature GraphPFN 연구 인계서를 읽고 독립적으로 검토해줘.
최신NSDtokenpilot은warmPCFI없이hiddenrepresentation과restrictionmap을함께학습했어. 결과와닫은범위를확인하고단순tokenNSD추가를미실행새방법으로다시제안하지말아줘.
최신 Sheaf-PCFI pilot의bounded No-Go와negativeLP floor,첫좌표직접전달계수a<0비율0,수치재현한계도함께봐줘. 단순PCFI내부화나signed허용을새아이디어로다시제안하지말아줘.
실제 실행한 결과, 보존한 개선, 미실행 제안을 구분하고, 필요하면 함께 첨부한 보고서와 findings를 확인해줘.

목표는 새 그래프에서 재학습 없이 심한 node-feature missing을 처리하는 GraphPFN이야.
Disjoint Factor PFN은 초기 fresh bank에서 Mask/PCFI PFN보다 정확도가 높았지만, fixed factor 대비 독립 이득은 확인하지 못했고 Grid6보다 전체 NLL이 높았어.
후속 fresh bank에서는 exact Grid6 teacher가 공개 Mask PFN을 확실히 개선하지 못했어.
Factor의 teacher KL은 약26.1% 감소했지만 accuracy 개선은 없었고, query 자신도 feature가 없는 secondary의 accuracy는1.16%p 하락했어.
Physical feature 제거는 CE와 distilled Factor 둘 다 성능을 낮췄으므로 feature 활용 자체는 보존하되, 그것이 distillation으로 새로 생겼다고 해석하지 말아줘.
추가로 feature-law3종을 학습한 cell-only 비교에서도 adaptive64.47%, static64.41%, 기존64.63%로 primary 적응 이득을 확인하지 못했어. 부호 제약을 푸는 것만 다시 제안하지 말아줘.
그 뒤 실제 PCFI100-step 전파를 PFN 내부에 배치해 외부/내부 고정/context/label-blind 네 모델을3seeds로 학습했어. 별도64개 multiclass graph에서 context49.41%, 내부 고정49.90%, 외부49.24%였고 context−고정은−0.487%p CI[−0.862,−0.130]이었어. 현재 bounded weight router의 이득은 미확인이며, 단순히 PCFI를 안에 넣자는 제안을 다시 새 후보로 취급하지 말아줘. 모든 내부화 방법을 기각한 결과는 아니야. 이어 기존 external/fixed checkpoint6개를 새128graph에서 추가 학습 없이 검증했어. Fixed−external은 +0.219558, 95% CI [-0.331926, +0.796665]%p였어. 새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다.
연결 연산·signed evidence·자기 role-update는 이미 구현했어. 현재 학생도 neural potential을 학습하므로 이를 모델 전체가 고정 Gaussian인 것으로 취급하지 말아줘.
다음 후보는 현재 binary column orientation/pairwise factor로 부족한 feature–relation 의존성과, 그것을 학습해 새 graph로 전이하는 구체적인 연산을 설명해줘.

이 실패 범위를 반복하지 않는 메소드 후보를 최대 2개 제안해줘.
후보마다 바꾸는 정보/추론 연산, 학습 목적, 가장 가까운 선행연구와의 차이,
강한 비교 대상, 한 번의 작은 결정적 실험을 적어줘.
특히 unlabeled node의 관측 feature와 떨어진 context label을 graph로 연결하는 문제를 검토하되,
이 방향이 답이라고 가정하지 말고 다른 설명도 비판해줘.

논문 novelty를 단정하지 말고 확인된 사실과 추론을 구분해줘.
새 실험을 바로 실행하거나 저장소를 수정하지 말고, 우선 연구 제안과 비판을 답해줘.
설득력 있는 후보가 없다면 없다고 말해줘.
추가로 label/feature heterophily를 분리한 sign-transport frozen 진단을 했어. Primary known-law signed−PCFI는 +5.836358 [95% CI +2.291311, +9.440175]%p였어. 같은 PCFI→PFN checkpoint에서 부호 법칙을 반영한 복원의 예측적 가치는 확인했다. 그러나 강한 직접 예측·graph-only 대조를 모두 넘는 기준은 통과하지 못해, 새 학습 모델이 필요하다는 근거로 확대하지 않는다. 참 feature-law g를 안 reference라서 새로운 학습/novelty의 성과로 부르지 말아줘.

추가로 참 부호 법칙을 주지 않고 Gaussian/probit evidence로 두 PCFI 입력 후보를 가중 평균했어. Context 적응−50:50는 +0.201218 [95% CI -0.127618, +0.537932]%p로 primary를 통과하지 못했어. 같은 정보를 직접 처리한 고정 Gaussian/probit는56.154827%였고 feature-removal 대비 +2.466964 [95% CI +0.884420, +4.166715]%p였어. 새 학습은0이야. 이미 구현한 통계 추론 기능과 학습의 독립적인 필요성을 구분해 줘.

SSGP/GGP도 원문 기준으로 정리했어. 현재 Gaussian/probit가 기존 primary 평균의 비특권적 대조11개를 개별95CI 기준으로 이겼지만 모든 세부조건을 이긴 것은 아니야. SSGP/GGP 자체의 재현은 아직 하지 않았고, 같은 prior로 새학습한 PFN과도 비교하지 않았어. 기존 graph-GP 원리와 우리 연산의 추가 가치를 구분해 줘.

```
