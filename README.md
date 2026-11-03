# 연구정리 하네스

논문의 장점, 저자가 인정한 한계, **후속 논문·리뷰가 지적한 한계**를 쌓고,
그걸 그래프로 엮어서 아이디어 검증과 접목 후보 탐색을 빠르게 하는 도구.

## 설치
```bash
pip install -r requirements.txt
```

## 쓰는 법
```bash
python src/harness.py build      # out/ 에 엑셀, 그래프, 접목 후보 생성
python src/harness.py validate   # 태그·필드 검사
python src/harness.py tags       # 태그별로 누가 한계이고 누가 해결했는지
python src/harness.py link       # 후속 논문 제목이 하네스 항목과 일치하면 by_id 자동 연결
python src/harness.py revisit --paper 새논문ID  # 실패 병목과 새 논문의 강점 대조

# 아이디어 점검: 차별점 정리 대상, 동기로 쓸 한계, 키워드 매치
python src/harness.py check \
  --tags graph-structure-use,channel-dependency \
  --keywords "PEMS,adjacency" \
  --text "관측 그래프를 MVTS PFN의 prior에 넣는다" \
  --save graph_ts_prior
```

## 내 연구 기록 (`findings/`)
전체 결과를 다시 볼 때는 [실험 결과 요약](RESULTS_SUMMARY_KO.md)을 먼저 읽는다.
실데이터, 최근 여섯 학습 실험과 내부 고정 PCFI 재검증·부호 전달 frozen 진단·참 법칙 없는 적응 전달 비교, 이전 시도·중단 범위를 구분했다.
SSGP·GGP 선행연구와 직접 Gaussian/probit의 추가 비교·메소드 판정도 기록했다.
관측의 기대 분류 가치로 학습 prior를 구성하는 [재독 후보](ideas/decision-value-observation-prior.md)는 실행 전 메모로 보존하고, [J/V 두 prior의 실제 학습 결과](findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md)를 별도로 기록했다.
Diffusion proposal·SMC 보정·Rényi-2/ESS의 [실제 비교](findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md)도 기록했다. 원 논문 재현과 현재 operator pilot의 실패 범위를 구분한다.
별도 시계열 후보의 [공동정보·nuisance frozen 비교](findings/sources/ts-synergy-2026-11-02/REPORT_KO.md)를 기록했다.2×2학습은 사전조건 미충족으로 미실행이며,GraphPFN 갈래의 실패로 합치지 않는다.
다른 연구자나 모델에게 검토를 요청할 때는 [GraphPFN 연구 인계서](RESEARCH_HANDOFF_KO.md)도 함께 전달한다.
현재 목표, 누적 실패의 정확한 범위, 최근 pilot, 강한 대조군, 미실행 후보와 복사할 요청문을 묶었다.
수치의 상세 원문은 인계서에서 연결한 `findings/sources/` 보고서와 구조화 기록을 확인한다.

실패·중단·보존·교훈을 `findings/<project>.yaml`에 쌓는다. `check`는 논문뿐 아니라 이 기록과도 대조해서,
이미 닫은 갈래(닫은 범위와 닫지 않은 것까지), 재사용할 자산, 아직 안 한 것을 함께 보여준다.

각 실패에는 **원인·병목의 근거 수준**, **다시 검토할 조건**, **먼저 이겨야 할 기준선**,
**최소 재검증**, **원문 snapshot**을 남긴다. 원인을 모르면 `unknown`으로 쓰며 추측을
확정 원인처럼 적지 않는다. `findings/_TEMPLATE.yaml`을 복사해 추가한다.

새 논문을 검증해 `papers/`에 추가한 뒤 `validate`와 `build`를 실행하면, 논문의
강점과 실패 항목의 `reopen_tags/reopen_keywords`를 대조해 `Revisit`를 갱신한다.
`revisit --paper ID`로 새 논문만 볼 수도 있다. **검색 단서이며 해결 판정이 아니다.**
같은 tag라도 가정·입력·문제·계산 예산이 다르면 해결책이 아닐 수 있다. Benchmarks와
surveys는 새 추론 장치를 제공한 것으로 매칭하지 않는다. 판정은 자동 변경하지 않는다.
`Bayesian`, `amortized` 같은 일반 단어와 태그 하나만 겹친 경우도 후보에서 제외한다.
기존 논문도 후보로 뜰 수 있으므로 새 결과의 출현으로 오해하지 않는다.

원인 수준은 `confirmed`(적힌 병목·대조 직접 확인), `supported`(진단이 지지),
`hypothesis`(미검증 설명), `unknown`(원인 미상)이다. 직접 확인도 모든 인과 원인을
밝혔다는 뜻은 아니다. 이미 해결된 중간 병목은 현재 미해결 문제와 분리해 남긴다.

2026-11-01 GraphPFN 누적 import의 원문 38개와 SHA-256은
`findings/sources/graphpfn-2026-11-01/manifest.json`에 있다. 원래 실험 보고서와
checkpoint는 수정하지 않았다. 기존 BP/EAJ의 36개 ID와 관찰·판정도 보존했다.

## 결과물 (`out/`)
- `research_harness.xlsx` — Papers / Claims / Candidates / Gaps / My Findings / Backlog / Ideas / Citations / Tags 시트
- `research_log.md` — 시도·실패 이유·보존 효과·미실행을 프로젝트별로 읽는 전체 기록
- `revisit.md`, `revisit_candidates.json` — 실패 병목과 논문 강점의 재검토 단서
- `research_harness.xlsx`의 `Revisit` — 원인 수준·논문 장치·재검토 조건·최소 실험 대조표
- `graph.html` — 브라우저로 여는 인터랙티브 그래프. 논문(박스)과 태그(타원)가 강점(초록)·한계(빨강) 선으로 연결되고, 후속 논문 → 원 논문 비판은 주황 화살표. 오프라인 동작.
- `candidates.md` — 접목 후보 순위와 빈칸(아무도 해결 못 한 축)
- `refs.bib` — 전체 BibTeX. `citation` 블록(확인한 페이지에서 읽은 값)에서 자동 생성

## 논문 추가
`papers/_TEMPLATE.yaml`을 복사해서 `papers/<id>.yaml`로 채운다.
Claude Code에서는 "XXX 논문 추가해줘"라고 하면 `CLAUDE.md`의 절차대로 채운다.

엑셀은 빌드 결과물이다. 수정은 YAML에서 하고 다시 build한다.

그래프에서 `내 연구 기록`과 `재검토 단서`를 켜면 논문→실패의 보라색 점선을 볼 수 있다.
실패 노드를 누르면 원인과 재검토 조건을 읽을 수 있다. 이 저장소의 `out/`이 최신 생성물이며,
저장소 밖에 과거에 복사한 HTML/XLSX는 자동으로 덮어쓰지 않는다.

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

## 노드 직교 frame으로 제한한 PFN

2026-11-03. 기존 edge별 map 대신 Q_i를 노드마다 공유하고 T_ij=Q_i^T Q_j로 제한했다. 입력·전파 위치·초기화·1,024update·train bank를 유지하고 새3개 모델만 학습했다. 비교군도 새64synthetic graph에서 다시 추론했다. 실제4graph는 기존 개발용 평가다.

| 방법 | Cora | CiteSeer | Texas | Wisconsin |
|---|---:|---:|---:|---:|
| pcfi100 | 50.15 | 30.84 | 27.77 | 21.61 |
| identity | 37.16 | 24.24 | 21.43 | 23.28 |
| dynamic | 38.07 | 22.35 | 19.06 | 23.81 |
| node_frame | 36.51 | 24.42 | 19.32 | 22.23 |
| native_pcfi100 | 52.03 | 33.87 | 33.26 | 24.44 |
| negative_lp | 32.03 | 24.76 | 63.54 | 42.73 |

node-identity +0.115674%p CI[-0.43349611351197426, 0.6758997330149052]; node-dynamic -0.020354%p CI[-0.4625205189269549, 0.4064417400329777]; node-static -0.178141%p CI[-0.7422996004065074, 0.35635222011572354]; node-pcfi100 +1.593935%p CI[0.7544764708510517, 2.4803787602259386]; realhetero-nodePCFI -3.915072%p; realhomo-nodePCFI -10.030316%p; gates={'synthetic': False, 'real_input': False, 'negative_lp_floor': False, 'expand': False}

현재 node-frame 구성은 확대 기준을 통과하지 못했다. 같은 bank에서 frame 차원/loss/예산을 조정해 구제하지 않는다. 경로 일관성의 수학적 성립을 실제 전이 성능으로 확대하지 않는다. [전체보고서](findings/sources/node-frame-2026-11-03/REPORT_KO.md), [계산정의](findings/sources/node-frame-2026-11-03/METHOD_KO.md), [구조화기록](findings/node-frame.yaml).

## PCFI의 confidence 가중 공간 확산을 Sheaf로 교체

2026-11-03. 원래 질문으로 돌아와 관측 숫자→confidence 가중 Sheaf 확산→결측 추정→고정 PFN을 비교했다. PCFI confidence·관측값 고정·100회·채널 보정을 유지했고,PFN은학습중에도고정했다. Scalar/Edge/Node를각3seed·1024updates로학습했다.

| 방법 | Cora | CiteSeer | Texas | Wisconsin |
|---|---:|---:|---:|---:|
| pcfi | 50.15 | 30.92 | 27.93 | 21.56 |
| scalar | 49.80 | 30.94 | 27.77 | 21.80 |
| edge | 49.39 | 31.30 | 27.87 | 21.32 |
| node | 49.81 | 31.22 | 28.18 | 22.05 |
| native_pcfi | 52.03 | 33.87 | 33.26 | 24.44 |
| negative_lp | 32.03 | 24.76 | 63.54 | 42.73 |

edge-pcfi -0.243949%p CI[-0.5338936172534267, 0.06024526333356581]; edge-scalar +0.445559%p CI[0.09320573869989934, 0.8435048538160689]; node-pcfi -0.226362%p CI[-0.4600395720623642, 0.027437612457655285]; node-scalar +0.463146%p CI[0.11849629002030168, 0.8351718057452501]; realhetero {'edge': {'pcfi': -0.1501645027210896, 'scalar': -0.1908579766253471, 'native_pcfi': -4.260033840401222, 'negative_lp': -28.53887091289313}, 'node': {'pcfi': 0.3706869668879398, 'scalar': 0.3299934929836823, 'native_pcfi': -3.7391823707921934, 'negative_lp': -28.018019443284103}}%p; gates={'edge': {'synthetic': False, 'real_input': False, 'negative_lp_floor': False, 'expand': False}, 'node': {'synthetic': False, 'real_input': False, 'negative_lp_floor': False, 'expand': False}}

현재 두 후보는 사전 확대 기준을 통과하지 못했다. 일부 대조의 개선이 있으면 분리해 보존하고 같은 bank의 loss/차원/예산 조정으로 구제하지 않는다. 고유원인은미상으로남기고hidden-token실험과구분한다. [전체보고서](findings/sources/pcfi-orthogonal-2026-11-03/REPORT_KO.md), [계산정의](findings/sources/pcfi-orthogonal-2026-11-03/METHOD_KO.md), [구조화기록](findings/pcfi-orthogonal.yaml).

## Confidence Sheaf PFN 구성 요소의 개발 비교

2026-11-03. Context의geometry전용개입,공간confidence×Nodeframe,직접signedscalar를비교했다. 기존과같은초기Phi·512train입력·1024update·고정PFN을유지했고새frontend9개를학습했다. 새32개knownfamily개발graph는label/feature네cell에8개씩배정하고k1/4를함께평가했다. 기존4real은개발재사용이다.

| 방법 | 새 개발 accuracy % | Cora % | CiteSeer % | Texas % | Wisconsin % |
|---|---:|---:|---:|---:|---:|
| pcfi | 44.772 | 50.188 | 30.852 | 28.005 | 21.557 |
| uniform | 44.604 | 48.952 | 32.418 | 29.058 | 22.888 |
| node | 44.852 | 49.800 | 31.275 | 28.178 | 22.114 |
| node_nolabel | 44.583 | 50.009 | 31.046 | 28.315 | 22.760 |
| node_uniform | 44.314 | 48.709 | 32.099 | 26.690 | 21.101 |
| signed | 44.599 | 49.396 | 30.596 | 28.295 | 21.951 |
| node_geometry_shuffle | 44.887 | 49.862 | 31.221 | 28.249 | 22.331 |

node_minus_pcfi=+0.080097%p CI[-0.1365710486064015, 0.3025558766215623]; node_minus_node_nolabel=+0.268850%p CI[0.04220929254982626, 0.5056492014098001]; node_minus_signed=+0.253150%p CI[-0.02066880686233795, 0.5310521697396225]; node_minus_node_geometry_shuffle=-0.035223%p CI[-0.12244736858557316, 0.04753142487188128]; node_uniform_minus_uniform=-0.290566%p CI[-0.5996681003332499, -0.019166541552883186]; node_minus_node_uniform=+0.538148%p CI[0.11773807913328584, 0.9685269136417165]; pcfi_minus_uniform=+0.167485%p CI[-0.28950384160141285, 0.6276919939891128]; signed_minus_pcfi=-0.173053%p CI[-0.4306325276996595, 0.10077196055861028]; node_nolabel_minus_pcfi=-0.188754%p CI[-0.36152359352041336, -0.02118026602979674]; spatial_confidence_by_node_interaction=+0.370663%p CI[0.05541017704068957, 0.6973559827760616]

구성요소의효과를분리해보존하며각비교를전체프로젝트의자동종료기준으로쓰지않는다. Native-PCFI/LPfloor와수치변동을별도보고했다. StaticNode의동치두채널계산과CPUfrontend비용개선도보존한다. [전체보고서](findings/sources/pcfi-reviewer-ablation-2026-11-03/REPORT_KO.md), [계산정의](findings/sources/pcfi-reviewer-ablation-2026-11-03/METHOD_KO.md), [구조화기록](findings/pcfi-reviewer-ablation.yaml).

## 관측 좌표만 고정하는 Confidence Sheaf PFN

2026-11-03. 첫 scalar 좌표만 clamp하고 관측하지 않은 두 번째 좌표를 전파하는 변경을 비교했다. 새 frontend6개/6144update,동일 초기화·512train bank·schedule·고정PFN,새 known-family32base graph,기존4real 개발graph다.

| 방법 | 개발 missing-query accuracy % | Cora % | CiteSeer % | Texas % | Wisconsin % |
|---|---|---|---|---|---|
| pcfi | 50.367 | 50.229 | 30.916 | 27.844 | 21.557 |
| node | 50.350 | 49.835 | 31.223 | 28.178 | 22.048 |
| partial_frozen | 50.362 | 49.800 | 31.514 | 26.424 | 22.388 |
| partial_trained | 50.232 | 49.824 | 31.158 | 27.422 | 22.058 |
| partial_initial | 50.326 | 50.181 | 30.996 | 27.858 | 21.916 |
| partial_random | 47.927 | 38.578 | 25.320 | 26.871 | 24.034 |
| signed_smooth | 49.946 | 49.272 | 31.045 | 28.301 | 23.378 |

partial_trained_minus_node:accuracy=-0.118074%p CI[-0.410523217262834, 0.16035034795655417]; partial_trained_minus_pcfi:accuracy=-0.135065%p CI[-0.3686364602286792, 0.08342222464999093]; partial_frozen_minus_node:accuracy=+0.011438%p CI[-0.19001886361456893, 0.2091231569541227]; partial_trained_minus_partial_frozen:accuracy=-0.129512%p CI[-0.4134588441218917, 0.14871358916660818]; partial_trained_minus_partial_initial:accuracy=-0.093266%p CI[-0.314049007360701, 0.11446042683875222]; partial_trained_minus_partial_random:accuracy=+2.305651%p CI[1.137082186911021, 3.648101343936383]; partial_random_minus_full_random:accuracy=-0.340911%p CI[-0.7137906330390282, 0.017033220980164184]; partial_trained_minus_signed_smooth:accuracy=+0.286140%p CI[-0.08556479459930283, 0.6371842445063916]; signed_smooth_minus_signed:accuracy=-0.313494%p CI[-0.610699974902656, 0.00012718194105000704]; signed_smooth_minus_pcfi:accuracy=-0.421204%p CI[-0.7337652552295654, -0.09699406895156527]

같은 frame에서 source가 하나뿐이면 두 clamp가 같다는 적용 범위도 확인했다. Primary는 그대로 두고 source1/source2+를 secondary로 분리했다. 전체 메소드 novelty의 성공/종료 판정으로 확대하지 않는다. [전체 보고서](findings/sources/pcfi-partial-clamp-2026-11-03/REPORT_KO.md), [계산 정의](findings/sources/pcfi-partial-clamp-2026-11-03/METHOD_KO.md), [구조화 기록](findings/pcfi-partial-clamp.yaml).
