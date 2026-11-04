# FP03 종료 기록

FP03의 고정 15-candidate/mean-score context calibration 개발 gate는 10개 중 5개 통과로 전체 FAIL이다. 감사와 세 독립 결과 검토는 완료됐다. 그래프당 rep0, frozen seed11 한 개이므로 일반적인 missing PFN의 실패나 독립 실데이터 검증으로 확대하지 않는다.

None 평균 accuracy는 adaptive60.910%, uniform-prior35.269%, fixed-edge62.523%, inherited-edge63.750%다. Adaptive는 uniform보다 +25.641%p 개선했지만 fixed-edge보다 다섯 그래프 모두 낮다. 95% 관측 피처 추가 효과는 평균 −0.107%p다. 확률 점수와 accuracy의 판정을 바꾸지 않는다.

새 scalar/rank-one potential 및 nonedge MF family는 inherited pair-specific potential/dense BP의 실제 full endpoint를 포함하지 않는다. 따라서 calibration, family 변경과 근사의 영향을 독립적으로 식별하지 못했다. Best-single candidate는 mixture의 상한이 아니다.

본 실험은 보통 클래스당 context4개이며 singleton-label은 수치 경계 검사다. 클래스 비율은 해석 요인이지만 연구 목표는 missing-feature PFN이다. 두-anchor 후속은 미실행 제안으로만 보존한다. 원 코드, lock, bank, 결과 및 세 RESULT 문서를 변경하지 않는다.

사용자 요청에 따라 다음 갈래는 관측 cell부터 end-to-end로 학습하는 PFN이다. FP03 결과를 새 모델의 결과로 바꾸거나 같은 bank에서 temperature/grid를 구제하지 않는다. Knowledge archive 통합은 별도 bookkeeping이며 본 문서는 이미 완료된 결과의 간결한 판정이다.
