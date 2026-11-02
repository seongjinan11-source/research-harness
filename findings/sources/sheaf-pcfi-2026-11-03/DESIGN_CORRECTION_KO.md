# 결과 판정 전 생성기 교정

2026-11-03. 첫 실행에서 class 수를 `classes[index % 6]`, feature law를 `index % 2`로 배정하여 class 수와 feature law가 결합되는 구현 문제가 발견됐다. Feature law와 label mode는 교차했지만 class 수가 feature law의 지름길이 될 수 있었다.

숫자 성능표를 열어 판단하기 전에 이 문제를 발견했다. Mask arm3개(각384updates,총1152updates)는 완료됐으며 synthetic 예측과 real 예측 일부도 계산된 상태였다. PCFI/weighted/signed/sheaf arm은 아직 학습하지 않았다. 세 실행을 중단하고 전체 폴더를 `../sheaf_pcfi_pilot_design_confound_v0`에 보존했다. 이 checkpoint·예측·학습 update는 이후 비교에서 전부 제외한다.

수정은 class 수를 graph별 RNG에서 독립적으로 추출하는 것이다. 구조·피처법칙의2×2 배정은 유지한다. 새 synthetic bank는 수정한 함수로 다시 만들고 모든 arm/seed를 원 public initialization에서 다시 학습한다. RNG 순서가 달라져 그래프·피처·class 배정도 달라지며, 이전 synthetic checkpoint를 이어 학습하지 않는다.

Real raw graph와 관측 mask/context 입력 bank는 동일하게 보존·재사용한다. 따라서 real 그래프를 완전히 새로운 독립 holdout으로 부르지 않으며, 원래대로4dataset development pilot이다. Real 입력 생성에는 위 synthetic class-count 문제가 관여하지 않는다. 초기 성능을 보고 real 입력이나 학습 budget을 선택한 것이 아니다.

이 교정은 학습 실패 판정이나 개선 메소드가 아니라 설계 통제를 복구한 조치다. 총 계산 비용에서는 폐기된1152updates도 따로 보고한다.
