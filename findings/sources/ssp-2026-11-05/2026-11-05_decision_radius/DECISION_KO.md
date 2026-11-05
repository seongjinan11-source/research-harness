# Decision radius: 직접 선행 확인 후 신규성 주축으로 개발하지 않음

2026-11-05. 사용자 요청에 따라 size residual→순위 전환의 sharp 조건을 계획하고 세 독립 reviewer의 수학·문헌·실험설계 검토를 받았다. 구현·새 bank 실행 이전에 직접 선행을 확인했다.

R=min margin/dual sensitivity는 affine multiclass classifier의 exact adversarial radius와 동일한 구조다. [DeepFool, CVPR2016 §3.1](https://arxiv.org/pdf/1511.04599), [Hein & Andriushchenko, NeurIPS2017](https://papers.neurips.cc/paper/6821-formal-guarantees-on-the-robustness-of-a-classifier-against-adversarial-manipulation)의 선행과 대응한다. Binary linear selection의 stability radii도 기존 연구다. Source 범위와 추가 출처는 PLAN_REVIEW_NOVELTY.md 및 PROOF_REVIEW_THEORY.md에 있다.

Count perturbation으로 제한하고 full-K crossing-pair만 검사하는 특수화는 유용한 진단일 수 있다. 그러나 이것만으로 이론 novelty를 크게 강화했다고 주장할 수 없다. Constructed witness의 .5R/1.5R refit 실험은 correctness 검사이며 미지 shape에 대한 독립 예측 증거가 아니다.

따라서 이 cycle의192-case 수치 실험은 실행하지 않았다. Plan·리뷰·수학적 검토는 보존한다. 새로운 메소드 이름으로 바꾸거나 locked 결과에 적용해 확인 성공을 만들지 않는다. 다음에는 count nuisance의 유연성과 composition의 실제 식별 정보·추천 손실 사이의 조건을 직접 분석한다.
