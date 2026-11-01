# Feature memory 분리 — 완료 결과

**판정: 현재 공개 GraphPFN의 feature-group memory routing에서는 accuracy 개선을 확인하지 못했다.**
기존 sample attention의 context-only key/value 제한은 확인됐지만, 이를 바꾸는 것만으로 이전 실패가 해결된다는 가설은 지지되지 않았다. 고유한 실패 원인은 여전히 미상이다.

- 새 64 train / 64 test base graphs, 3 fixed training seeds, 4 arms, arm/seed당 512 성공 updates.
- 같은 public initialization·파라미터·과제 순서·query CE. 기존 transformer 고정. PCFI arm은 실제 PCFI 입력으로 학습했다.
- Primary는 공식 및 controlled off-diagonal의 whole-node uniform/clustered90/99.5% accuracy. CI는 base graph64개를 resample한다.
- 공개 모델의 native 두 좌표 group을 memory 단위로 썼다. 완전한 sparse cell-token 모델의 결과가 아니다.

| 모델 | Accuracy (%) | NLL | Brier |
|---|---:|---:|---:|
| context | 52.3336 | 0.687776 | 0.494768 |
| all_nodes | 51.9552 | 0.689573 | 0.496561 |
| observed | 52.2888 | 0.692156 | 0.498759 |
| pcfi | 53.9246 | 0.683446 | 0.490617 |
| lp | 53.3203 | 0.777961 | 0.519445 |
| class_prior | 51.0254 | 0.689640 | 0.496611 |

| Accuracy 차이 | %p와 paired95% CI |
|---|---|
| observed − context | -0.044759 [-1.501516, +1.460826] |
| observed − all_nodes | +0.333659 [-0.948130, +1.613515] |
| observed − pcfi | -1.635742 [-3.420207, +0.063324] |
| observed − lp | -1.031494 [-4.421438, +2.356160] |

Observed의 NLL은 context 대비 +0.004380 [+0.000906, +0.007896], PCFI 대비 +0.008710 [+0.003985, +0.013666] 높았다. Accuracy의 CI가0을 포함하는 것과 확률 품질 악화는 구분한다.
Clustered90/off-diagonal의 observed arm은 feature를 섞었을 때 오히려 NLL이 낮아졌다. 이 실행에서는 실제 feature–node 정렬을 유익하게 활용했다고 주장하지 않는다. 이것을 모든 실제 feature의 무용함으로 확대하지 않는다.

저장된 예측 16,896개를 독립 재채점했고 총 6,144 성공 updates와 frozen parameters 보존을 확인했다.
이번 bank와 mixed-prior512-update 비교는 이전 MCP의 +3.43%p를 독립 재현한 실험이 아니다. 이전 NLL gate와 긍정 신호를 수정하지 않는다.
전체 결과·사전 고정 조건: REPORT.md, protocol.json, results/paired.json, interventions.json, final_integrity.json.
