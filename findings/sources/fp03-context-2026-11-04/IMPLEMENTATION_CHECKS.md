# FP03 independent checks

`checks.py`는 real bank·실제 query prediction·학습·optimizer를 사용하지 않는 독립 tiny 검증이다. sklearn을 torch보다 먼저 import한다. 담당 source는 이 파일과 checks.py뿐이며 operator/runner/protocol/historical source는 수정하지 않는다.

## 검증 내용

- NumPy 완전 열거는 모든 class assignments, 모든 unordered graph pair의 Potts score와 literal Normal/GH centroid 적분을 계산한다.
- 별도 generic pairwise NumPy BP는 각 observed cell/class에 K×R gate matrix를 만든다. Production scalar feature gates를 oracle에 재사용하지 않는다. Nonedge MF는 직접 node-pair loop로 계산하며 마지막 readout에 마지막 사용 field를 유지한다.
- b=0 tree는 singleton centroid leaves만 사용해 exact marginal 일치를 요구한다. 여러 observed unknown source가 만드는 feature cycles와 dense nonedge BP의 exact 차이는 진단으로 보존한다.
- Signed b field, theta=d/sqrt(sum d), self/observed-neighbor 제외, complete graph field0, a=0 observed edge, reverse cavity와 마지막 field의 합을 검사한다.
- 세 signed contrast 조합의 float64/float32 class/node/source-row/column/edge 순열, hidden NaN, prior normalization, source0/column-singleton, context clamp, empty graph, graph-off query 및 unanchored symmetry를 검사한다.
- Candidate/fold batch는 35 lanes의 chunk1/16/35로 검사해 partial chunk도 포함한다. Held score/weight parity도 확인한다.
- CS와 같은 class-count metadata만 합성해 m3/14represented/42selected/13extra/1missing-class를 직접 oracle과 비교한다. Stable-ID node/class permutation, equal remaining class counts, extra masking, m0/m1, full-K probabilities를 검사한다.
- Poison 검사는 split을 고정하고 held set 전체와 unused extras의 원래 라벨을 바꾼 뒤, score 전 branch prediction을 비교한다. Held truths는 score의 합법적 target이므로 weights까지 poison 불변을 요구하지 않는다.
- Score weights는 별도 NumPy float64 clip[1e-8,1]·row normalization·represented-class macro log score와 비교한다. Entropy/ESS/KL·tie·empty score, candidate order, probability mixtures와 contrast/TV diagnostics를 검사한다.
- Missing-query loss에서 observed literal likelihood까지의 finite/nonzero gradient를 central difference와 비교한다. Hidden cell와 source0 gradient는0이어야 한다.

## 실행과 PASS 파일

```powershell
& 'C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe' -B -X utf8 -u factor_context_cycle/checks.py
```

Root가 허용한 optional `--gpu`는 tiny float32 invariance smoke만 추가한다. GPU real bank 평가나 학습을 시작하지 않는다. 모든 필수 검사가 실제 PASS하고 source SHA가 실행 전후 같을 때만 `results/checks.json`을 원자 저장한다. 파일에는 source SHA, 실제 오차·근사 진단, optimizer0/real_bank_used=false/real_predictions_used=false를 보존한다. 현 source SHA와 불일치한 과거 PASS는 현 코드의 증거가 아니다.

## 실제 결과와 고정된 source

2026-11-04 최종 source로 CPU 전체 검사 PASS 후, root가 허용한 tiny GPU smoke를 추가한 전체 검사도 PASS했다. CPU 검사 자체3.18초, CPU+GPU3.91초이며 import 시간은 별도다. 실패한 full-check 실행이나 실제 연산 결함은 발견하지 않았다. 매 실행 source SHA를 import 전에 기록하고 검사 뒤 재검증했다. GPU 사전 점검은24GiB 중1639MiB 사용/8% utilization으로 Windows UI가 동작하는 상태였고 root/agent의 다른 계산 job은 없었다. 실제 graph 평가를 실행하지 않았다.

| 검사 | 최대 절대오차/결과 |
|---|---:|
| b0 exact tree marginal | 2.22e-16 |
| 독립 generic pairwise MF·field 직접합 | 8.89e-16 이하 |
| Float64 invariance | 2.22e-16 |
| Float32 CPU 및 tiny GPU invariance | 1.79e-7 |
| Float32 chunk1 대16/35 probability | 2.39e-7 |
| Float32 chunk 간 score/weight | 4.97e-8 / 3.70e-9 |
| Balanced fold/score·entropy/ESS | 5.33e-15 이하 |
| 확률 mixture/contrast/TV 재구성 | 1.45e-8 이하 |
| Literal-likelihood 방향미분 대 central difference | 2.44e-11 이하 |

Hidden cell와 source0 gradient는 정확히0이었다. CS-like split은42selected/13unused/3fold/14represented를 만족했고, held+extra poison은 고정 split의 branch prediction을 bitwise 유지했다. K15 star의 모든 anchor를 제거한 float32 예측은 bitwise uniform이며15개 weight도 prior와 같았다. 테스트는 bank의 실제 CS 배열을 읽지 않고 class-count metadata만 합성했다.

별도 tiny loopy 진단의 MF−exact 최대차이는`.031845168470370555`, MF−dense BP 최대차이는`.011878005154564053`다. 이 값은 근사오차이며 equality assertion으로 숨기지 않았다. 작은 oracle PASS는 큰 graph의 calibration·과학적 gate·독립 real 전이 성능을 증명하지 않는다.

- checks.py SHA256: `f8af6dca9f45daf625fe18963032942c109f55c9c2707a649f2c0c29a2141f4d`
- operators.py SHA256: `7ba1e3f7b7b73751628beb2ac66f2fc4fb263003563b0823ac33a8934d794367`
- run.py SHA256: `8a19b06757b72dd0dfe84c325c25d442a86e5585f602097505786e5b2155c77f`
- results/checks.json SHA256: `024c5c6e6ea2305f4f4cb6c1068e08faa2a01ad2a8dd37805b5b0df3a456d063`

위 source와 본 문서를 이 상태로 고정한다. 이후 source가 바뀌면 기존 PASS를 재사용하지 말고 현 SHA를 대조해야 한다. Historical source SHA도 실행 전후 불변이었다.
