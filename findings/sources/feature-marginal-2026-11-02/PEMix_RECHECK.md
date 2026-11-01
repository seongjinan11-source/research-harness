# PEMix를 보고 무엇을 다시 확인하는가

원문: [When Missingness Follows the Graph](https://arxiv.org/abs/2610.18113), 2026-10-13 v1. 서지와 본문 §4, Appendix E를 확인해 research-harness에 추가했다.

PEMix의 구조 조건부 feature mixture는 관측값 전파가 아닌 입력 분포를 만드는 단서다. 기존 JIMP-F01/F02는 bounded numeric point-refiner의 학습·배치 위치에 관한 결과라서, 분포를 통한 frozen completion 연산까지 닫지는 않는다. 이것이 두 실패를 해결했다는 뜻은 아니다. 이번 pilot은 PEMix 재현도 아니다.

검토한 선행연구 때문에 다음 주장은 보류한다.

- 분포로 missing feature를 다루는 것 자체가 새롭다.
- 여러 대치를 평균하면 PFN다운 새로운 학습 원리가 된다.
- Gaussian conditional이 공식 SCM의 참 conditional이다.
- 최신 논문이 태그로 연결됐으니 기존 실패가 해결됐다.

실제 신규 가설은 더 좁다. **완성된 feature graph를 PFN에 통과시킬 때, missing node들의 조건부 상관을 유지해야 하는가?** 같은 조건부 평균·각 variance·regime draws를 유지하고 rho 내 cross-node covariance만 제거한 대조가 이 질문을 겨냥한다. 관측된 피처의 실제 정보가 충분히 남아 있다는 가정도 하지 않는다. Extreme whole-node missing에서 관측 0~1개 graph를 그대로 포함한다.

이 대조가 이겨도 새로운 학습법이나 unseen schema 전이가 입증되지는 않는다. 특히 zero-fill에 맞춰 추가 학습한 PFN을 frozen으로 쓰므로, distributional input을 학습 중 사용한 PFN과의 결론은 별개다. PCFI도 같은 checkpoint의 입력 대조이고 별도의 PCFI-trained checkpoint 재비교는 아니다. Positive result라면 그 비교와 PEMix/GSPN 등 확률적 입력 방법을 다음 기준선으로 삼아야 한다.

현재 추가 auxiliary loss·density head·posterior head를 학습하지 않는다. 먼저 고정한 연산 자체의 예측 가치를 새 graph seed에서 평가한다.
