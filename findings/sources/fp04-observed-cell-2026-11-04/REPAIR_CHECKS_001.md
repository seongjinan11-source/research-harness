# FP04 prelock 검사 수정 001

날짜: 2026-11-04. Root의 실제 CPU+GPU 요청 CLI는 CPU reference/algebra/invariant/boundary 뒤 최초 bridge의 잘못된 assertion에서 멈췄다. 따라서 이 실행은 전체 PASS나 GPU 검사 통과가 아니다. 학습 전·source lock 전 수정이며 모델/설정/성공 기준은 바꾸지 않았다.

- 원본 실패: `results/checks_failure_1793778313974373800.json` (원본 보존).
- 실패 artifact SHA256: `1110eb3636e07b898ac0b0070e52b7784f1626e9b96a90eeaca32b0235a6d8c1`.
- 이전 `checks.py` SHA256: `6a6b5cc68411866cd0ec76df8206e7c21ca114d5e81bf3a01d21fc23ca8c62ea`.
- 수정 `checks.py` SHA256: `a8c1215fde6c4d5128a30d1a726e49668797695c2f133f6bd4cdfa72aa830539`.
- 변경하지 않은 `model.py`: `20aea3daca308d8a049ce6e08991cb39b4d6afacf835adb8aca9c1ff74bf0792`.
- 변경하지 않은 `protocol.json`: `9e2f12448b88899f0071ef7fc5e8564b2ebe734121aa9ec012a5689e28a83bc2`.

## 원인과 독립 확인

Balanced bridge에서 context0—source1 edge를 제거한 뒤 모든 노드의 class0/1 logits가 같아야 한다고 검사한 것이 오류다. Context0와 context4는 각각 서로 다른 positive/other ternary seed를 그대로 받는다. 이 노드들의 class contrast는 edge cut 뒤에도 허용된다. 대칭이 복원되는 대상은 unknown인 source1,source2,query3이다.

별도 작은 CPU 진단으로 원래 모델·동일 seed·edge cut 입력을 재구성했다. 각 노드의 class0−class1 logit은 `[-0.7201807753889475, 0, 0, 0, +0.7201807753889475]`였다. Unknown 노드의 최대 contrast는 정확히 0, source1 값에 대한 query contrast derivative는 `-9.69861447723382e-22`였다. 원 실패의 0.72018은 context 노드 자체의 정상 seed 차이와 일치한다. 이 증거는 모델 결함보다 assertion 범위 오류를 직접 지지한다.

## 최소 수정과 재실행

`context<0`인 노드에만 동일한 tolerance의 class-symmetry assertion을 적용하고 그 오차를 결과에 기록한다. Query derivative, columns-off/graph-off, 원래 bridge-on nonzero 요구와 나머지 검사는 그대로다. `IMPLEMENTATION_CHECKS.md`도 이 범위를 명시하도록 고쳤다. 실제 전체 suite 재실행과 PASS 생성은 root CLI가 수행한다. 이전 실패 artifact는 수정·삭제하지 않는다.
