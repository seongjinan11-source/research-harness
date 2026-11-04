# FP04 독립 구현 검사

작성일: 2026-11-04. 소유 파일은 `checks.py`와 이 문서뿐이다. 모델·데이터·runner·protocol·기존 결과를 수정하지 않았다. Root가 CLI를 실행하기 전에는 검사 통과나 PASS artifact가 있다고 주장하지 않는다.

## 실행

기본은 작은 CPU 검사이며 실제 bank/정답·학습·optimizer를 사용하지 않는다. 모델 입력의 canonical 값 key는 `x`다. Windows 요구에 따라 sklearn을 torch보다 먼저 import한다.

```powershell
& 'C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe' -B -X utf8 -u observed_cell_cycle/checks.py
```

Root가 자원 상태를 확인한 뒤 같은 명령에 `--gpu`를 붙이면 CPU suite 뒤 작은 CUDA FP32 대칭·bridge·CPU/GPU forward/gradient 비교도 수행한다. Native graph resource smoke는 이 tiny suite의 범위 밖이며 root가 별도로 수행한다. TF32는 끈다.

전체 성공과 source hash 전후 일치 뒤에만 `results/checks.json`에 PASS를 기록한다. `model.py/checks.py/PLAN_v2.md/protocol.json`과 존재하는 `data.py/run.py`의 SHA256을 기록한다. 실패는 기존 PASS를 덮지 않고 timestamp가 있는 별도 `checks_failure_*.json`에 원인·traceback·hash를 보존한다. `optimizer_updates=0`, real/final bank 미사용을 명시한다. 체크 코드 작성 중 AST parse만 실행했으며 suite 자체는 root 실행에 맡겼다.

## 독립성 및 검사 범위

1. **Python-loop full-forward oracle:** 모델 private helper, scatter/einsum reduction을 호출하지 않고 관측-only 통계, ternary seed/class count, directed neighbor 합산, 각 cell/rank의 column·node 합산, 양수 support 분모를 재구성한다. 같은 pointwise MLP/projection/LayerNorm 파라미터를 호출하여 모든 arm의 전체 logits와 파라미터·입력 autograd를 production과 비교한다. 이는 학습 파라미터와 library primitive를 이중 구현한 검사는 아니며, routing·축·합산·분모·round 순서의 독립 검사다.

2. **NumPy contraction oracle:** class/node/column/rank를 literal loop로 순회하여 arbitrary signed weights, 빈 row/column, support holes에서 P/V contraction을 계산한다. 별도 einsum과 oracle 자체를 검증하고 production dense/sparse column read와 비교한다. Column의 비선형 모듈에는 같은 파라미터를 사용한다.

3. **Forward와 gradient 동치:** FP64/FP32에서 sparse/dense, feature chunk1/2/3/전체, class chunk1/2/전체를 비교한다. K3·F5로 마지막 partial chunk를 포함한다. 완전 M1이며 같은 effective input leaf를 공유할 때 observed/mask/PCFI 전체 logits·모든 파라미터·입력 gradient를 비교한다. 일반 입력에서 dense mask는 A=1인 독립 참조식과 비교하므로 공통 M hard gate로 비교를 지우면 실패한다.

4. **누출·대칭·경계조건:** node/class/column permutation, hidden NaN/큰 값 poison, 무관한 target 필드 무시, empty context의 class symmetry와 full-K uniform probability, 동일 seed의 미관측 class equality를 검사한다. Source0/1, 관측값이 모두 0인 경우, 빈 graph, 전부 unknown을 포함한다. 관측된 숫자 0을 missing으로 바꾸면 포함 support/count가 달라지는지 검사한다. Source1 예측의 무효성은 강제하지 않는다.

5. **학습 경로:** 작은 CE에서 value/seed/node/graph/P/column/V/read/head 및 LayerNorm 그룹의 gradient가 유한하고 비퇴화 사례에서 nonzero인지 확인한다. 관측 값의 gradient는 nonzero, 숨긴 원본 값의 gradient는 정확히 0이어야 한다. Source0에서는 unused parameter gradient가 있을 수 있으므로 이를 결함으로 취급하지 않는다.

6. **Balanced bridge:** 5 nodes: context0(class0)—source1, source2—query3, isolated context4(class1). 두 source는 feature0만 관측한다. Static class count는 [1,1]이다. Source1 값에 대한 query class-logit contrast의 derivative가 columns-on에서 nonzero이고 columns-off에서 정확히 0이어야 한다. 모든 graph edge를 제거하면 0이어야 하며, context0—source1 edge만 제거하면 unknown source/query 노드의 class symmetry와 query contrast derivative 0이 복원되어야 한다. 알려진 context 노드 자체는 ternary seed가 다르므로 class symmetry를 요구하지 않는다. 최초 검사의 이 범위 오류와 prelock 수정은 `REPAIR_CHECKS_001.md`에 보존했다. 이 검사는 구현된 경로의 존재를 검증하며 실제 feature utility를 대체하지 않는다.

7. **공통 representation clipping:** Root가 성능 학습·실제 점수 관찰 전에 고정한 `represented_value_clip=10`을 독립 loop의 basis 직전에도 적용한다. Missing PCFI 입력을 ±1234로 둔 경우와 같은 입력을 ±10으로 제한한 경우의 logits·파라미터/관측입력 gradient가 유한하고 일치하는지 FP64/FP32에서 검사한다. FP64 extreme case는 full loop reference와도 비교한다. Basis helper의 7개 입력을 직접 구성해 clamp가 관측 bit나 arm에 따라 달라지지 않는지도 확인한다. Raw PCFI cache, observed-only moments, 원래 입력 배열은 수정하지 않는다.

## 한계

입력 `x`는 외부에서 observed-only RMS scaling/clip을 마친 값이다. 실제 PCFI 알고리즘의 RNG 격리·100-step/alpha 설정·intervention cache provenance와 native preprocessing 자원 사용은 model-only 참조식만으로 검증되지 않는다. 모델은 PCFI의 관측값을 내부에서 다시 clamp한다. 실제 bank 생성/runner audit와 native-shaped smoke를 별도로 통과해야 한다. Tiny oracle 통과는 예측 성능, calibration, Bayesian posterior 정확성이나 novelty를 뜻하지 않는다.
