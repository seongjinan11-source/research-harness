# 성능표 확인 전 추가한 보수적인 기준선

수정bank의 학습을 시작한 시점이며 어떤 arm의 비교 성능표도 확인하지 않았다. 기존 primary와 학습 schedule은 바꾸지 않는다.

학습 후 실제graph에서 `공개초기화 PFN+mask`, `공개초기화 PFN+RP16후PCFI100`, `공개초기화 PFN+native PCFI100후RP16`도 평가한다. 모든 경우 같은 입력과context를사용하고 target학습은0이다. 이는 **현재 wrapper·RP16·단일forward 인터페이스**에서의 초기화 기준선이며,공식GraphPFN의full전처리/10ensemble pipeline을재현한것은아니다.

이 기준선은 synthetic adaptation 자체가공개초기화보다나빠질가능성을놓치지않기위해추가한다. 다른학습조건이나방법을이결과로선택하지않는다. PCFI native-before-RP의사전대조도그대로유지한다.

추가로 동일context의 양의LP와context-frequency prior를 단순 구조/빈도 대조로 보고한다. 이것을 CPGNN/CLP 원 논문 재현이라고 부르지 않는다. Synthetic gate와 real development gate를 사후 교체하지 않으며, 이 보조 기준선을 넘지 못하면 범용 개선 주장을 제한한다.
