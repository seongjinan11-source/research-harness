Factor에서는 다음 경로가 실제 계산으로 구현됐어.

**Context label → graph 관계 → 피처 관측 노드 → 공유 feature-column 잠재변수 → 다른 노드의 class**

완성된 feature 행렬을 먼저 만드는 대신, 피처의 의미와 노드 class를 함께 추론해. 우리가 처음 이야기했던 “PCFI를 앞에 붙이는 것을 넘어, 예측하면서 결측 정보를 다루자”에 가까운 구현이 나온 셈이야. 다만 공개 GraphPFN을 개선한 모듈이 아니라, 별도로 학습한 작은 structured predictor라는 구분은 필요해. [Factor 계산 정의](C:/Users/seongjin/Desktop/code/gts_pfn/disjoint_factor_pilot/METHOD.md)

반대로 **teacher 결과는 이 계산의 효과와 감독 신호의 효과를 잘 분리해 줘.** 피처 제거에 따른 하락은 CE Factor에도 이미 있었어. 그러니 연결 능력을 distillation의 공으로 돌리면 안 돼. Distillation이 추가한 것은 현재로서는 **teacher 분포에 더 가까워지는 효과**야. 피처 없는 query의 정확도는 오히려 보조 비교에서 하락했으므로, 전체 KL 감소만 보고 성공으로 묶기도 어렵고. [Teacher 판정](C:/Users/seongjin/Desktop/code/gts_pfn/grid6_teacher_pilot/REVIEW_KO.md)

그래서 현재 연구 상태를 나는 이렇게 표현하겠어.

> **불완전한 관측값과 떨어진 context label을 연결하는 추론 구조는 확보했다. 그 구조를 학습하는 것이 고정된 통계 모형보다 언제 유리한지는 아직 입증해야 한다.**

이건 “아무것도 안 남았다”와는 달라. **작동하는 연산은 남았고, 학습의 독립적인 필요성이 아직 약한 상태**야. Factor의 teacher KL이 fixed-mode factor보다 유의하게 낮아진 결과도 있으니 학습의 가치가 전혀 없다는 결론도 맞지 않아. 다만 이를 실제 분류 성능과 새로운 그래프 분포에서의 이점으로 연결하지 못했어.

앞으로 후보를 찾을 때도 기준이 바뀌어야 해. **분리된 피처·라벨 연결, signed evidence, 자기 role-update는 이미 구현한 기능**이야. 이를 다시 새 이름의 메소드로 제안하면 반복이 돼. 다음 후보는 **현재 고정 Gaussian factor로는 표현하기 어려운 피처–관계 의존성을 어떻게 학습하고 전이할 것인지**까지 구체적으로 설명해야 해. 단순히 teacher나 head를 더 추가하는 것으로는 현재 남은 질문에 답하기 어려워.\\
