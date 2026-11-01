# 실패 병목과 논문 강점: 재검토 단서

자동 매칭은 논문 metadata의 강점·태그·키워드 검색이다. 해결·novelty·재개 승인이 아니다. 새 논문 원문에서 필요한 장치와 가정을 확인하고, 아래 최소 재검증을 통과해야 한다. 기존 논문도 포함하므로 목록에 나타났다는 사실이 새 해결책의 출현을 뜻하지 않는다.

원인 수준: 직접 확인 / 진단이 지지 / 미검증 가설 / 원인 미상. 직접 확인은 적힌 병목·대조를 확인했다는 뜻이며 모든 인과 원인을 밝혔다는 뜻은 아니다.

## REG-F01 — Reference-regret priority가 hard-loss sampling 및 같은 비용의 Random보다 유리하다.

- 판정: 메인 방법 확장 보류
- 원인·병목 [원인 미상]: Reference의 독립 utility와 채점 비용을 정당화하지 못했다. Stale priority·짧은 학습·input loss 중 원인별 영향은 분리하지 않았다. Loss와 regret의 초기 Spearman0.4264는 다른 task를 강조함을 보일 뿐 효용 증명이 아니다.
- 반증·배제한 설명: 확정된 배제 없음
- 재검토 조건: Reference의 추가 감독 비용을 회수하면서 실제로 회수 가능한 task difficulty를 식별하는 학습·생성 원리가 나올 때.
- 먼저 이겨야 할 기준선: Random, Manual, Loss priority, cold-score-cost-matched RandomBudget, frozen FISF
- 최소 재검증: 새 graph bank와 고정 비용으로 Loss/RandomBudget을 함께 넘는지 확인하고 task weighting과 generator 학습을 분리한다.
- 닫지 않은 것: 학습된 생성기·library·무한 stream prior learning을 구현하거나 실패시킨 것은 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md); [findings/sources/graphpfn-2026-11-01/regret_scope.md](../findings/sources/graphpfn-2026-11-01/regret_scope.md)

- [PFNSeparability](https://arxiv.org/abs/2610.14989): 복수 태그 일치 (약한 단서); tags=pretraining-data, prior-mismatch; keywords=-
  - 기록된 강점: Feature law와 conditional label law의 의존성을 prior 설계의 명시적 선택으로 다룬다.
