# 내부 고정 PCFI의 새 그래프 재현성 검증

기존 개발 결과를 보고 선택한 `internal_fixed` 후보가 외부 PCFI보다 나은지, 기존 checkpoint 여섯 개를 그대로 사용해 새 그래프 128개에서 확인한다. 추가 학습과 모델 선택은 없다. 같은 생성 family와 같은 학습 seed를 사용하므로 새로운 학습 seed 집단, 실제 그래프, 미지 schema에 대한 검증은 아니다.

원 실험의 고정된 `pcfi_shared.py`, `pcfi_model.py`, checkpoint loader와 generator 설정을 import하며 수정하지 않는다. 원 bank 세 episode를 모델별로 replay하고, 원 코드·lock·checkpoint 및 이 실험 코드·프로토콜의 해시를 잠근 뒤 최종 bank를 생성한다. 새 graph seed와 실제 graph fingerprint가 원 train/dev/test와 겹치지 않는지 확인한다.

Primary는 클래스당 최대 context 4개, uniform/clustered whole-node 90/99.5% 결측 네 조건에서 자기 피처가 전혀 없는 query의 accuracy다. Episode 내 query, base graph 내 네 조건과 고정된 세 학습 seed를 차례로 평균하고, base graph 128개를 단위로 paired bootstrap 5,000회를 수행한다. 내부 고정−외부의 95% CI 하한이 0을 넘고, full-feature accuracy의 CI 하한이 −2%p보다 높아야 재현 성공으로 판정한다. 기준은 새 bank 생성 전에 잠근다.

NLL/Brier, 전체 query, 개별 결측 조건, cell missing과 k=1, 학습 seed별 결과는 별도 보조 분석이다. 하위집단 CI는 다중비교 보정을 하지 않는다. 기존 결과와 새 bank를 합쳐 primary 성공을 주장하지 않는다. 0-source episode도 포함하며 원래 관측값과 query 정답을 숨긴 처리 원칙을 유지한다.

외부는 PCFI 100회+correlation 1회를 PFN 이전에 수행한다. 내부 고정은 PFN layer 0/3/6/9 이전에 PCFI 25회씩 수행하고 마지막에 correlation을 적용한다. 둘 다 PFN 12층 한 번을 사용한다. 내부 PCFI 자체는 고정이지만 해당 checkpoint의 PFN adapter/encoder/decoder는 원 실험에서 학습됐다.

성공하면 FP/FISF 내부화 비교를 별도 프로토콜로 설계한다. 실패하거나 불확실하면 그 후속 학습을 자동으로 시작하지 않는다. 원 개발 bank의 양의 관찰은 보존하며, 이를 사후적으로 동등성이나 모든 내부화의 불가능성으로 해석하지 않는다.
