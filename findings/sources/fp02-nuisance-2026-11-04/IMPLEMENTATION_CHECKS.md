# FP02 independent numerical checks

담당 파일은 `checks.py`와 본 문서다. `operators.py`, historical source/bank/checkpoint/prediction, branch는 변경하지 않는다. sklearn을 torch보다 먼저 import한다. 실제 학습·real bank 평가·optimizer step은 수행하지 않는다.

## 검증 원리

- 별도 NumPy 완전 열거는 모든 class assignments에서 literal Normal likelihood와 정규화된 finite-grid prior를 centroid별로 적분한다. Graph score는 모든 unordered pair의 `A ElogB - t theta_i theta_j EB`를 직접 더한다.
- 별도 NumPy dense BP는 generic pairwise factor graph를 사용한다. Production의 scalar feature gates, graph updates, Gamma helpers를 oracle 계산에 재사용하지 않는다. Class-gated cell마다 class/centroid pair matrix를 직접 만든다.
- 완전한 Poisson graph의 N=2 사례와 별도 edge-only tree oracle에는 exact marginal 일치를 요구한다. N>=3의 loopy dense BP 및 sparse nonneighbor MF와 exact 열거의 차이는 진단값이며 0을 요구하지 않는다.
- Gamma 통계는 직접 `i<j` 합으로 만든 edge counts, theta-product exposures, theta interaction과 비교한다. Full symmetric storage의 off-diagonal 셀은 unordered membership 전체이며, 총량 conservation은 upper triangle만 합한다.
- Population은 context를 제외한 node counts, graph exposure/volume은 context를 포함한 theta-weighted counts다. 둘을 다른 수치로 만든 사례에서 구분을 검사한다. Gamma mean과 meanlog, 원래 prior+현재 통계의 갱신, nuisance damping 및 마지막 prediction schedule을 검사한다.
- Nonneighbor field의 self/neighbor 제외와 observed-edge 음수항 중복 방지는 직접 합/독립 graph contraction으로 검사한다.
- 세 arm 모두 class/node/source-row/feature-column/edge-order equivariance, context clamp, 숨은 likelihood의 NaN/큰 값 poison, grid-prior normalization, source0/1, empty edge/one-node 경우를 검사한다. Full outer nuisance update까지 포함한다.
- Missing query의 loss에서 observed literal likelihood까지의 gradient가 유한하고 nonzero인지 확인하고 central difference와 비교한다. Hidden cells의 gradient는 0이어야 한다. Query truth는 public signature/input에 제공하지 않는다.

## 실행과 산출물

기존 research Python으로 실행한다.

```powershell
& 'C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe' -B -X utf8 -u factor_nuisance_cycle/checks.py
```

`--gpu`는 동일 tiny graph의 GPU smoke만 추가한다. 이 선택 검사는 CPU 수학 oracle을 대체하지 않는다. Real graph 입력은 읽지 않는다.

모든 필수 검사가 실제 통과하고 실행 전후 tested source SHA가 동일할 때만 `results/checks.json`을 원자 저장한다. JSON은 `status=PASS`, `optimizer_updates=0`, `real_bank_used=false`, `real_predictions_used=false`, tested SHA와 각 오차/진단값을 포함한다. 실패한 실행은 PASS 파일을 만들지 않으며 기존 파일이 있더라도 root가 현 source SHA를 반드시 확인해야 한다.

## 실제 실행 결과

2026-11-04 `checks.py` CPU 전체 검사 PASS 후 `checks.py --gpu`로 optional tiny GPU 검사까지 PASS했다. Operator 담당자가 source0 zero-gradient 연결과 iterations0 field snapshot의 boundary 처리를 추가한 뒤 그 두 assertion도 suite에 넣어 최종 CPU+GPU 전체 검사를 다시 PASS했다. 매 실행 source hash는 실행 전후 같았으며 실패한 full-check 실행은 없었다. 마지막 JSON은 현재 source의 CPU+GPU 결과이며 검사 자체 약4.00초, import 시간은 별도다.

| 검증 | 실제 최대 절대오차/결과 |
|---|---:|
| Full Poisson tree: 독립 dense BP·production 대 exact enumeration | 2.22e-16 |
| Edge-only tree: 독립 dense BP 대 exact enumeration | 3.33e-16 |
| Gamma count/exposure/interaction 및 보존식 | 7.11e-15 이하 |
| 세 번의 nuisance update·prior 재계산·0.5 damping·field/cavity | 7.11e-15 이하 |
| CPU permutation/hidden poison/clamp/source0·1/empty graph | 5.56e-16 이하 |
| Missing-query loss→observed likelihood 방향미분 대 central difference | 2.52e-11 이하 |
| Tiny GPU invariance smoke | PASS, 5.56e-16 이하 |

세 arm 모두 missing query에서 observed likelihood로 nonzero gradient가 전달됐고 숨은 cell gradient는0이었다. Source0는 autograd로 정확한 zero gradient tensor를 반환했다. Gamma와 population updates도 이 gradient 검사에 포함됐다. Iterations0에서도 반환 node와 저장 field의 readout이 일치했다. Tree 외 세 tiny loopy 사례의 sparse-vs-dense 최대 차이는 `.020677350481450585`, sparse-vs-exact 최대 차이는 `.02835234071089643`였다. 이 차이는 JSON의 진단으로 그대로 남겨 두었으며 exact inference로 주장하지 않는다.

검증한 operator SHA256은 `2baf72cd02f329450bddd89fbf70e5a6df473e90b900aad08685a5d3bbefbf1a`, checks SHA256은 `f7fdee1ba5e6581fbf25d216a80d6b15e256fdbdc3ae2367cd5cedc9469091c7`다. `results/checks.json` SHA256은 `25aae434d3d5e68dc9493c55ce76b3f2879da6cfc2f78bce5eb26efcb3717a4a`다. 이후 source가 바뀌면 해당 PASS를 현 코드의 증거로 재사용할 수 없다. History의 old source hash도 실행 전후 불변을 확인했다. 이것은 작은 수학·구현 검사이며 실제 개발 graph 성능이나 Stage0 go-condition 충족을 뜻하지 않는다.
