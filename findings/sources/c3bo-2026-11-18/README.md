# C3BO blue revision — 2026-11-18 archive

사용자가 지정한 최근 파란색 수정 제출본에 대응하는 로컬 자료의 보존본이다. 기존 SSP 자료와 최초 제출본을 대체하지 않는다.

| 자료 | 위치 |
|---|---|
| 최신 로컬 파란색 PDF | [manuscript_blue.pdf](manuscript_blue.pdf) |
| 편집 가능한 Overleaf 프로젝트 | [overleaf/current_blue](overleaf/current_blue), main document: `main_v2.tex` |
| Overleaf 업로드용 ZIP | [current_blue_source.zip](overleaf/current_blue_source.zip) |
| 앞선 파란색 개정 소스 ZIP 5개 | [earlier_blue_sources](overleaf/earlier_blue_sources) |
| C3BO 및 기존 runner의 로컬 소스 | [workspace/AnalogCoder/official_genbo](workspace/AnalogCoder/official_genbo) |
| 11월 14–17일 실행·분석 코드, 프로토콜, 요약 | [workspace/ssp-share/research_cycles](workspace/ssp-share/research_cycles) |
| 원시 관측·trajectory·입력 데이터 | [data](data), lossless `tar.zst` shards와 파일별 SHA-256 |
| 13개 baseline adapter와 외부 구현 | 저장소 루트의 [BASELINES.md](../../BASELINES.md) 및 [baseline 출처 보존본](baseline_provenance/BASELINES.md) |

현재 프로젝트 디렉터리는 11월 17일 수정과 scaling 그림을 포함한다. 11월 15일 ZIP과 동일한 버전으로 취급하지 않는다. 원격 Overleaf/OpenReview에서 현재 제공 중인 PDF와의 byte 동일성은 이번 패키징에서 검증하지 않았다. PDF와 원고 내용을 이번 업로드를 위해 수정하지 않았다.

주요 실행 기록: `2026-11-15_c3bo_no_gp`는 4개 linear 정책의 policy-only main/held-out 비교, `2026-11-17_prior_no_gp`는 prior 대조, `2026-11-15_c3bo_ablation`은 초기 count 설계 대조다. `2026-11-14_c3bo_weak_size`는 약한 size 효과, `2026-11-14_online_gp_size`와 `2026-11-17_online_gp_size_extension`은 GP pilot이다. OR-Library 자료는 `2026-11-15_or_library_*`, 2,000-item scaling은 `2026-11-15_c3bo_online_2000`와 `2026-11-16_c3bo_*`, PyOED 보강은 `2026-11-14_pyoed_meta_baselines`와 `2026-11-17_pyoed_no_gp`에 있다. 각 PROTOCOL과 RESULTS가 실험 범위를 정의하며, 서로 다른 설정의 수치를 합쳐 새 성능 주장으로 해석하지 않는다.

## 데이터 복원

이 디렉터리에서 Python 3.14 이상으로 `python extract_data.py --destination workspace`를 실행한다. `workspace/ssp-share`와 `workspace/AnalogCoder/official_genbo`의 원래 상대 배치를 복원한다. 모든 shard를 받고 실행한다. 동일한 파일을 여러 실험에서 재사용한 경우 한 번만 저장하고 SHA-256으로 확인한 복사본을 복원한다. 원본 JSON을 축약하거나 평가 행을 제거하지 않았다.

기존 외부 baseline 소스는 저장소 루트의 `code/tmp/external_baselines`, `code/third_party`, `external_baselines`, `baseline_support`에도 있다. runner의 원래 외부 경로, pretrained weights와 패키지 의존성은 해당 manifest 및 BASELINES.md를 따른다. 이 보존본은 새 머신에서의 전체 실험 실행을 인증한 배포판이 아니다. 코드·데이터를 import하거나 objective를 호출하지 않고 파일·압축·소스 입력을 검증했다.

## 보존과 출처

소스는 원래 byte를 보존했다. 데이터 archive는 원시 파일을 lossless 압축하고 원래 상대 경로와 hash를 기록한다. 런타임 로그, PID, 가상환경, 중복 manuscript 사본과 대형 pretrained checkpoint는 제외했다. 제3자 baseline에 새 라이선스를 부여하지 않으며 원래 출처·라이선스를 따른다. 기록된 실패·음성 결과는 해당 실험의 요약과 데이터에 함께 남긴다.
