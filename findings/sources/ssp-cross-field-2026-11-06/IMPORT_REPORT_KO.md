# 타분야 해결 선행의 논문 정리 import

2026-11-06. 사용자의 “가져와서 정리 리스트에 넣자” 요청에 따라 기존 79편에 8편을 추가한다. 기존 RegConfounding은 중복 생성하지 않고 9편의 연결에 포함한다. 새 성능 실험이나 메소드 변경은 없다.

| ID | 해결 원리 | SSP 접점·적용 경계 |
|---|---|---|
| RegConfounding (기존) | Nuisance shrinkage가 관심 효과에 전달되는 RIC | 메커니즘 선행. Causal estimand와 BO ranking의 차이 유지. |
| BayesianCausalForests | Propensity를 반영하고 prognostic/treatment prior 분리 | 관측 수집 구조를 고려할 동기. Adaptive BO에 대한 보장 아님. |
| SpatialPlus | Spatial nuisance와 correlated covariate의 residualization | Count-dependent composition residualization의 인접 선행. 완전 collinearity의 정보는 복원하지 못함. |
| FunctionalHorseshoe | 계수 zero 대신 함수 subspace deviation 수축 | Count 구조 prior의 직접 인접 선행. Composition contrast prior 보존과 같지 않음. |
| SmoothFunctionalShrinkage | 구조적 수축에 spline smoothness 제어 결합 | 유연한 count curve 후보. 현재 SSPd3의 불충분성은 전제하지 않음. |
| NonlinearFunctionalShrinkage | Taylor/Jacobian으로 nonlinear shape family 활용 | 포화 count curve의 soft prior 동기. Exact global invariance 아님. |
| PooledHorseshoeQF | Gaussian common mean + horseshoe deviations | Shared count+HS residual의 강한 단순 대조. 원문은 국가 pooling. |
| LAVA | Dense+sparse latent sum에 별도 ridge/lasso penalty | 일반 분해 자체의 신규성 경계. 예측 총합과 성분 식별을 구분. |
| DoubleMachineLearning | Orthogonal score + cross-fitting | Nuisance 오차의 1차 영향 제어. Finite-sample BO posterior·regret 보장 아님. |

각 YAML의 citation.verified_from에 실제 확인한 primary URL을 넣고, 본문 확인 위치를 notes에 기록했다. 공식 코드 URL을 확인하지 못한 항목은 추측해서 채우지 않는다. Analyst 한계는 편당 2개 이하로 구분한다. 각 항목에서 문제·해결 원리·가정·강점·한계·비교·SSP hook을 제공한다.

Spatial+는 Wiley 최종 출판판(2022)의 저자·권호·페이지·DOI를 사용한다. 다른 항목은 직접 확인한 arXiv판을 보수적으로 인용한다. 특히 DML은 읽은 v4(2016)의 제목과 저자 6명을 사용한다. 2018 발표판 및 최신 arXiv판의 제목·저자 목록과 섞지 않는다. BCF citation 저자는 PDF 첫 페이지의 Carlos M. Carvalho 표기를 사용한다.

원래 cross-field REVIEW_KO.md는 당시의 조사 기록으로 그대로 snapshot한다. 그 문서의 “아직 canonical에 추가하지 않았다”는 말은 조사 시점의 상태다. 이번 import와 생성물 검증 완료 여부는 이 cycle의 COMPLETION.json에 별도로 기록한다. 기존 실험의 signed mechanism 실패, GP projection 음성 결과, corrected linear regret 양성 결과와 SSPd3 선택을 모두 유지한다.

원문 링크는 각 YAML과 엑셀 Citations에 있다. 새 finding SSP-XF-U01의 9개 명시적 문헌 연결은 기존 가설을 해결 완료로 바꾸지 않으며 검색·재검토의 단서만 제공한다.
