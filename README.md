# 연구정리 하네스

논문의 장점, 저자가 인정한 한계, **후속 논문·리뷰가 지적한 한계**를 쌓고,
그걸 그래프로 엮어서 아이디어 검증과 접목 후보 탐색을 빠르게 하는 도구.

## 설치
```bash
pip install -r requirements.txt
```

## 쓰는 법
```bash
python src/harness.py build      # out/ 에 엑셀, 그래프, 접목 후보 생성
python src/harness.py validate   # 태그·필드 검사
python src/harness.py tags       # 태그별로 누가 한계이고 누가 해결했는지
python src/harness.py link       # 후속 논문 제목이 하네스 항목과 일치하면 by_id 자동 연결
python src/harness.py revisit --paper 새논문ID  # 실패 병목과 새 논문의 강점 대조

# 아이디어 점검: 차별점 정리 대상, 동기로 쓸 한계, 키워드 매치
python src/harness.py check \
  --tags graph-structure-use,channel-dependency \
  --keywords "PEMS,adjacency" \
  --text "관측 그래프를 MVTS PFN의 prior에 넣는다" \
  --save graph_ts_prior
```

## 내 연구 기록 (`findings/`)
전체 결과를 다시 볼 때는 [실험 결과 요약](RESULTS_SUMMARY_KO.md)을 먼저 읽는다.
실데이터, 최근 여섯 학습 실험과 내부 고정 PCFI 재검증·부호 전달 frozen 진단·참 법칙 없는 적응 전달 비교, 이전 시도·중단 범위를 구분했다.
SSGP·GGP 선행연구와 직접 Gaussian/probit의 추가 비교·메소드 판정도 기록했다.
관측의 기대 분류 가치로 학습 prior를 구성하는 [재독 후보](ideas/decision-value-observation-prior.md)는 실행 전 메모로 보존하고, [J/V 두 prior의 실제 학습 결과](findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md)를 별도로 기록했다.
Diffusion proposal·SMC 보정·Rényi-2/ESS의 [실제 비교](findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md)도 기록했다. 원 논문 재현과 현재 operator pilot의 실패 범위를 구분한다.
다른 연구자나 모델에게 검토를 요청할 때는 [GraphPFN 연구 인계서](RESEARCH_HANDOFF_KO.md)도 함께 전달한다.
현재 목표, 누적 실패의 정확한 범위, 최근 pilot, 강한 대조군, 미실행 후보와 복사할 요청문을 묶었다.
수치의 상세 원문은 인계서에서 연결한 `findings/sources/` 보고서와 구조화 기록을 확인한다.

실패·중단·보존·교훈을 `findings/<project>.yaml`에 쌓는다. `check`는 논문뿐 아니라 이 기록과도 대조해서,
이미 닫은 갈래(닫은 범위와 닫지 않은 것까지), 재사용할 자산, 아직 안 한 것을 함께 보여준다.

각 실패에는 **원인·병목의 근거 수준**, **다시 검토할 조건**, **먼저 이겨야 할 기준선**,
**최소 재검증**, **원문 snapshot**을 남긴다. 원인을 모르면 `unknown`으로 쓰며 추측을
확정 원인처럼 적지 않는다. `findings/_TEMPLATE.yaml`을 복사해 추가한다.

새 논문을 검증해 `papers/`에 추가한 뒤 `validate`와 `build`를 실행하면, 논문의
강점과 실패 항목의 `reopen_tags/reopen_keywords`를 대조해 `Revisit`를 갱신한다.
`revisit --paper ID`로 새 논문만 볼 수도 있다. **검색 단서이며 해결 판정이 아니다.**
같은 tag라도 가정·입력·문제·계산 예산이 다르면 해결책이 아닐 수 있다. Benchmarks와
surveys는 새 추론 장치를 제공한 것으로 매칭하지 않는다. 판정은 자동 변경하지 않는다.
`Bayesian`, `amortized` 같은 일반 단어와 태그 하나만 겹친 경우도 후보에서 제외한다.
기존 논문도 후보로 뜰 수 있으므로 새 결과의 출현으로 오해하지 않는다.

원인 수준은 `confirmed`(적힌 병목·대조 직접 확인), `supported`(진단이 지지),
`hypothesis`(미검증 설명), `unknown`(원인 미상)이다. 직접 확인도 모든 인과 원인을
밝혔다는 뜻은 아니다. 이미 해결된 중간 병목은 현재 미해결 문제와 분리해 남긴다.

2026-11-01 GraphPFN 누적 import의 원문 38개와 SHA-256은
`findings/sources/graphpfn-2026-11-01/manifest.json`에 있다. 원래 실험 보고서와
checkpoint는 수정하지 않았다. 기존 BP/EAJ의 36개 ID와 관찰·판정도 보존했다.

## 결과물 (`out/`)
- `research_harness.xlsx` — Papers / Claims / Candidates / Gaps / My Findings / Backlog / Ideas / Citations / Tags 시트
- `research_log.md` — 시도·실패 이유·보존 효과·미실행을 프로젝트별로 읽는 전체 기록
- `revisit.md`, `revisit_candidates.json` — 실패 병목과 논문 강점의 재검토 단서
- `research_harness.xlsx`의 `Revisit` — 원인 수준·논문 장치·재검토 조건·최소 실험 대조표
- `graph.html` — 브라우저로 여는 인터랙티브 그래프. 논문(박스)과 태그(타원)가 강점(초록)·한계(빨강) 선으로 연결되고, 후속 논문 → 원 논문 비판은 주황 화살표. 오프라인 동작.
- `candidates.md` — 접목 후보 순위와 빈칸(아무도 해결 못 한 축)
- `refs.bib` — 전체 BibTeX. `citation` 블록(확인한 페이지에서 읽은 값)에서 자동 생성

## 논문 추가
`papers/_TEMPLATE.yaml`을 복사해서 `papers/<id>.yaml`로 채운다.
Claude Code에서는 "XXX 논문 추가해줘"라고 하면 `CLAUDE.md`의 절차대로 채운다.

엑셀은 빌드 결과물이다. 수정은 YAML에서 하고 다시 build한다.

그래프에서 `내 연구 기록`과 `재검토 단서`를 켜면 논문→실패의 보라색 점선을 볼 수 있다.
실패 노드를 누르면 원인과 재검토 조건을 읽을 수 있다. 이 저장소의 `out/`이 최신 생성물이며,
저장소 밖에 과거에 복사한 HTML/XLSX는 자동으로 덮어쓰지 않는다.
