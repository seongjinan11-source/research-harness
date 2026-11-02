# Gaussian/probit 결과와 SSGP·GGP 문헌 검토

이번 기록은 기존 adaptive-transport bank의 추가 비교와 논문 정리다. 새 graph·모델·학습 update는0이며, 새 독립 검증이나 SSGP/GGP 재현 실험이 아니다. 기존 ADTR-F01/P01/L01 판정과 source snapshots는 유지한다.

## 문헌에서 확인한 계산과 우리 구현의 차이

| 항목 | SSGP, IJCAI 2007 | GGP, NeurIPS 2018 | 현재 직접 Gaussian/probit |
|---|---|---|---|
| Covariance | Graph regularizer로 수정한 kernel | P K(X,X) Pᵀ | Graph low-pass Gaussian covariance의 sign-gauged 가설 |
| Label 처리 | Probit와 EP | Robust-max와 inducing-point variational inference | Context의 sign 제약과 고정 Sobol 적분 |
| Feature 처리 | 입력 좌표에서 kernel 계산 | 이웃의 feature kernel 결합 | 관측행만으로 singular score를 만들고 Gaussian 관측으로 조건화 |
| 과제 적응 | Evidence 기반 모델 선택 | ELBO로 target 데이터의 파라미터 최적화 | 고정24가설을 관측 evidence로 주변화; 새 optimizer 학습0 |
| 이번에 재현했나? | 아니오 | 아니오 | 기존 adaptive-transport 실험의 대조군 |

[SSGP 공식 논문](https://www.ijcai.org/Proceedings/07/Papers/171.pdf)과 [GGP 공식 논문](https://proceedings.neurips.cc/paper/2018/hash/1fc214004c9481e4c8073e85323bfd4b-Abstract.html)의 구조·가정·강점·한계·인용은 papers/SSGP.yaml과 papers/GGP.yaml에 정리했다. GGP의 [공식 코드](https://github.com/yincheng/GGP) README도 확인했다. 한계의 self와 analyst를 구분했고, 읽지 않은 후속 논문이나 리뷰의 비판은 만들지 않았다.

우리 구현은 두 논문의 재현도, 새로운 GP 원리의 입증도 아니다. 겹치는 선행 원리는 graph covariance를 통한 확률 분류다. 관측 score와 분리된 context를 함께 조건화하고 g/t를 주변화하는 현재 구성의 차별성은 별도로 평가해야 하며, 기존 구성요소의 조합만으로 novelty가 확정되지는 않는다. 이 판단은 조사자 해석이다.

## 기존 primary 조건에서의 추가 paired 비교

새64 base graphs,192nodes,8correlated columns,source8/192=95.83% whole-node missing,k4 labels/class다. Label/feature 네 법칙과 기존 세 고정 training seeds를 graph 안에서 평균했다. 현재 Gaussian/probit 정확도는56.154827%,NLL0.672120,Brier0.479871이다. Original statistics와 모든 입력 SHA가 일치하는지 확인한 뒤 저장된 metric/probability를 재집계했다.

다음은 Gaussian−대조의 차이다. Bootstrap은 기존과 같은 graph 단위5,000회다. 이 비교들은 같은 bank를 본 뒤 추가한 보조 분석이며 pointwise95CI를 사용한다. 11개 비교 전체에 대한 family-wise 유의성이나 새로운 사전 primary 성공을 주장하지 않는다.

| 비특권적 대조 | Accuracy 차이%p [95% CI] | NLL 차이 [95% CI] | Brier 차이 [95% CI] |
|---|---:|---:|---:|
| Mask PFN | +3.797403 [95% CI +1.555741, +6.019568] | -0.033324 [95% CI -0.042134, -0.024630] | -0.029931 [95% CI -0.037851, -0.022093] |
| 일반 PCFI→PFN | +4.589585 [95% CI +2.569802, +6.587119] | -0.043245 [95% CI -0.052592, -0.033770] | -0.038474 [95% CI -0.046914, -0.029902] |
| 고정 parity PCFI→PFN | +3.949441 [95% CI +1.813129, +6.118670] | -0.047007 [95% CI -0.057068, -0.037140] | -0.040338 [95% CI -0.048919, -0.032079] |
| 50:50 두 복원 평균 | +3.800061 [95% CI +1.681170, +5.912160] | -0.030513 [95% CI -0.037902, -0.022871] | -0.028457 [95% CI -0.035463, -0.021262] |
| Feature-only 적응 PCFI→PFN | +3.692866 [95% CI +1.652477, +5.729643] | -0.032478 [95% CI -0.040299, -0.024485] | -0.029856 [95% CI -0.037082, -0.022471] |
| Context 적응 PCFI→PFN | +3.598842 [95% CI +1.536745, +5.635105] | -0.033261 [95% CI -0.041242, -0.025097] | -0.030450 [95% CI -0.037792, -0.022878] |
| 양의 LP | +3.358392 [95% CI +1.028041, +5.675842] | -0.061318 [95% CI -0.076301, -0.047654] | -0.043910 [95% CI -0.052533, -0.035120] |
| Signed LP | +3.041067 [95% CI +0.602935, +5.374370] | -0.093141 [95% CI -0.120197, -0.070608] | -0.065830 [95% CI -0.079375, -0.051862] |
| Context 적응 LP | +3.107972 [95% CI +0.741358, +5.489381] | -0.022589 [95% CI -0.029979, -0.015259] | -0.020881 [95% CI -0.027945, -0.013921] |
| 같은 Gaussian/probit·피처 제거 | +2.466964 [95% CI +0.884420, +4.166715] | -0.009081 [95% CI -0.013619, -0.004556] | -0.008591 [95% CI -0.012854, -0.004317] |
| Uniform | +6.852143 [95% CI +4.297486, +9.340412] | -0.021027 [95% CI -0.029090, -0.012736] | -0.020129 [95% CI -0.027830, -0.012320] |

이 primary 조건의 평균에서는 모든 비특권적 대조에 대해 accuracy CI 하한>0,NLL/Brier CI 상한<0였다. 따라서 Gaussian 대조의 우위를 유의하지 않은 점추정으로 축소하지 않는다. 정확히 같은 primary missing-query 집합의 피처 제거 비교는 기존 ADTR-P01의+2.466964 [95% CI +0.884420,+4.166715]%p를 유지한다.

## 모든 조건과 이전 방법을 이긴 것은 아니다

참 label-law t를 받은 LP reference는57.135862%로 Gaussian보다 점추정이 높았다. Gaussian−reference 차이는 -0.981035 [95% CI -3.304919, +1.346377]%p이며 CI는0을 포함한다. 이 reference는 privileged information을 받으므로 비특권적 대조와 분리한다.

동일 source8/k4에서도 t0/g0의 양의 LP57.254319%가 Gaussian55.815288%보다 높았고, t1/g1의 signed LP57.017405%가 Gaussian56.494366%보다 높았다. 이는 보조 cell의 점추정 반례이며 해당 차이의 유의성을 주장하지 않는다. 전체32개 세부 cell 중 비특권적 방법의 점추정이 더 높은 cell은22개였다. 전체 목록은 PRIMARY_COMPARISONS.json에 있다.

현재 Gaussian covariance는 생성 family의 저주파 구조를 활용하며, feature는 한 잠재 신호를 공유한다. 비교한 PFN은 다른 prior로 학습된 frozen checkpoint다. 그러므로 같은 family로 재학습한 PFN, 기존 Grid6, 실제 graph, arbitrary heterophily나 새로운 schema까지 모두 이겼다는 뜻은 아니다. 숨겨진 feature 값·query 정답과 참 g/t는 실행 가능한 Gaussian 추론에 추가 공개하지 않았다.

## 메소드로서의 판정

현재 Gaussian/probit는 성능이 확인된 고정 통계 기준선이자 후속 연구에서 사용할 추론 자산으로 보존한다. 메인 메소드의 novelty와 범용성이 확정됐다는 판정은 보류한다. 신경망이 없다는 이유로 배제하는 것이 아니라, 선행 GP 원리와 구별되는 연산 및 현재 family 밖의 근거가 아직 부족하다는 판단이다. 이는 Gaussian 추론 전체에 대한 No-Go가 아니다.

SSGP/GGP 원문은 우리 missing-feature PFN 문제를 이미 해결했다는 증거가 아니다. 반대로 우리 구현이 두 원 논문보다 우월하다는 실험도 아직 없다. 두 논문 자체의 재현은 미실행으로 명시하고 학습 실패로 세지 않는다. 이번 기록 요청으로 새 모델이나 GPU 학습을 시작하지 않았다.

## 보존과 재검토

Papers의 verified_from에는 공식 원문/학회 metadata를 넣었다. SSGP 페이지1059–1064는 공식 PDF footer에서 확인했고, GGP 페이지1683–1694는 공식 metadata를 따랐다. 검색의 상충한 페이지 수는 채택하지 않았다. GGP 코드 링크는 저자 repository README로 확인했으며, SSGP의 전용 공식 코드는 확인하지 못해 링크를 만들지 않았다.

각 논문에 대해 revisit --paper를 실행한 결과도 보존했다. 자동 매칭은 재검토용 검색 단서이며 실패 해결이나 학습 재개의 판정이 아니다. 원142개 finding,699개 보존 source 파일,33개 기존 paper YAML의 hash를 비교한다.
