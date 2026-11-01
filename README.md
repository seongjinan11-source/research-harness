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

# 아이디어 점검: 차별점 정리 대상, 동기로 쓸 한계, 키워드 매치
python src/harness.py check \
  --tags graph-structure-use,channel-dependency \
  --keywords "PEMS,adjacency" \
  --text "관측 그래프를 MVTS PFN의 prior에 넣는다" \
  --save graph_ts_prior
```

## 내 연구 기록 (`findings/`)
실패·중단·보존·교훈을 `findings/<project>.yaml`에 쌓는다. `check`는 논문뿐 아니라 이 기록과도 대조해서,
이미 닫은 갈래(닫은 범위와 닫지 않은 것까지), 재사용할 자산, 아직 안 한 것을 함께 보여준다.

## 결과물 (`out/`)
- `research_harness.xlsx` — Papers / Claims / Candidates / Gaps / My Findings / Backlog / Ideas / Citations / Tags 시트
- `graph.html` — 브라우저로 여는 인터랙티브 그래프. 논문(박스)과 태그(타원)가 강점(초록)·한계(빨강) 선으로 연결되고, 후속 논문 → 원 논문 비판은 주황 화살표. 오프라인 동작.
- `candidates.md` — 접목 후보 순위와 빈칸(아무도 해결 못 한 축)
- `refs.bib` — 전체 BibTeX. `citation` 블록(확인한 페이지에서 읽은 값)에서 자동 생성

## 논문 추가
`papers/_TEMPLATE.yaml`을 복사해서 `papers/<id>.yaml`로 채운다.
Claude Code에서는 "XXX 논문 추가해줘"라고 하면 `CLAUDE.md`의 절차대로 채운다.

엑셀은 빌드 결과물이다. 수정은 YAML에서 하고 다시 build한다.
