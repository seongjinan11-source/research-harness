# Label / feature memory 분리 실험

기존 공개 GraphPFN의 sample attention에서 feature token이 읽는 key/value 범위를 바꾼 작은 matched pilot이다.

- `context`: 기존 label-context 노드만 읽는다.
- `all_nodes`: feature token은 전체 노드를, label token은 context만 읽는다.
- `observed`: feature token은 해당 native feature group에 실측값이 있는 노드만 읽는다. Label token은 context만 읽는다.
- `pcfi`: 기존 context-only GraphPFN을 PCFI 입력으로 실제 학습한다.

공개 backbone은 두 feature를 묶으므로 이번 구현의 memory 단위는 cell이 아니라 **두 좌표의 group**이다. 두 좌표의 원래 mask는 유지하고, padding은 관측 evidence로 세지 않는다. 빈 memory에는 학습 파라미터가 없는 zero key/value만 허용한다. 완전한 sparse cell-token 아키텍처의 실험이라고 부르지 않는다.

네 arm은 같은 초기화, 4,012,259 trainable parameters, 같은 512 logical updates와 과제 순서를 사용한다. 기존 transformer는 고정한다. 새 64 train / 64 test base graphs를 사용하며 공식 prior와 controlled crossed family를 함께 학습한다. 새 생성기나 loss를 추가한 효과가 아니다.

Primary는 공식 및 controlled off-diagonal의 whole-node uniform/clustered 90/99.5% 조건을 평균한 **accuracy**다. NLL/Brier는 따로 보고한다. Observed가 context와 all_nodes를 모두 넘어야 관측 memory 구성의 독립 이득으로 판정한다. PCFI 및 LP 우위는 더 강한 별도 gate다. 기존 MCP의 NLL gate와 secondary accuracy 신호는 바꾸지 않는다.

실행 조건은 `protocol.json`, 사전 검사는 `results/preflight.json`, source/bank 고정은 `results/lock.json`에 남긴다. 완료 후 `REPORT.md`와 `results/paired.json`에서 최종 결과를 볼 수 있다. CI의 표집 단위는 query가 아닌 base graph이며 세 training seed에 조건부다.

## 재현

같은 Python/DGL/공개 checkpoint 환경에서 `experiment.py prepare`, `preflight`, `run --seed 11/12/13`, `analyze.py` 순서다. 기존 bank와 partial training을 조용히 덮어쓰거나 다시 시작하지 않는다. 원래 GraphPFN 및 이전 실험 파일은 수정하지 않는다.
