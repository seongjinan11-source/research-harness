# 연구정리 하네스 — Claude Code 작업 규칙

최신 논문(ML 근본론, graph FM, PFN, TSFM, BO, EDA)의 장점과 한계를 누적하고,
그래프로 엮어 아이디어 검증과 접목 후보 탐색에 쓰는 저장소다.

## 구조
- `papers/<id>.yaml` — 논문 한 편 = 파일 하나. **유일한 원본**. 형식은 `papers/_TEMPLATE.yaml`.
- `taxonomy.yaml` — 강점과 한계에 공통으로 붙이는 태그. 같은 태그가 A의 한계이자 B의 강점이면 접목 후보가 된다.
- `ideas/<이름>.yaml` — `check --save`로 저장한 아이디어.
- `out/` — 빌드 결과물 (엑셀, 그래프, 후보). 직접 고치지 않는다.
- `src/harness.py` — build / validate / tags / check.

## "논문 추가해줘" 요청을 받으면
1. 논문 정체를 확정한다: 정확한 제목, 저자, venue, arXiv/OpenReview 링크, 코드 링크.
2. 본문을 읽고 problem, core_idea, key_assumptions, strengths를 채운다.
3. limitations는 출처를 구분해서 채운다. 이게 이 하네스의 핵심이다.
   - `self`: 저자가 limitation/conclusion에서 인정한 것
   - `cited`: **이 논문을 인용한 후속 논문**이 지적한 것. 후속 논문 제목·연도(`by`)와 링크(`by_link`) 필수. 실제 본문이나 초록에서 확인한 것만 넣는다.
   - `review`: OpenReview 리뷰어가 지적한 weakness. `by_link`는 forum URL.
   - `analyst`: 조사자 판단. 최대 2개.
4. 태그는 `taxonomy.yaml`에 있는 것만 쓴다. 맞는 게 없으면 taxonomy에 먼저 추가하고 쓴다.
5. builds_on, compared_against, open_questions, hooks(사용자 연구와의 접점)를 채운다.
6. `python src/harness.py validate` → 경고 0개 확인 → `python src/harness.py build`.

## 아이디어 점검 요청을 받으면
1. 아이디어를 taxonomy 태그 2~4개와 키워드로 바꾼다.
2. `python src/harness.py check --tags ... --keywords ... --text "..." --save <이름>`
3. 결과를 이렇게 해석해서 보고한다.
   - 같은 축을 강점으로 가진 논문은 **기각 사유가 아니라 차별점을 정리할 대상**이다.
   - 같은 축의 한계(특히 cited/review)는 동기로 인용할 근거다.
   - 하네스에 관련 논문이 없으면, 외부 검색으로 선행연구를 찾아 먼저 추가할지 제안한다.

## 작성 규칙
- 설명은 한국어, ML 용어는 영어 그대로. 키워드 나열이 아니라 바로 읽히는 문장으로.
- 논문 문장을 복사하지 않는다. 요약·의역만.
- 확인 못 한 인용·한계는 넣지 않는다. 개수보다 정확성.
- 같은 대상을 여러 이름으로 부르지 않는다. 분야 표준 용어를 쓴다.
- 작업이 끝나면 git commit 한다 (이 하네스는 한 번 날아간 적이 있다).

## 사용자 연구 (hooks 작성 시 참고)
- 불균형 라벨 context에서 graph PFN 노드 분류 (ImbNodePFN; class-balanced vs instance-balanced context sampling, prior에 label skew 주입)
- feature 공간이 다른 그래프로 graph PFN 확장 (tSVD 대신 per-graph codebook 입력 인터페이스)
- 관측 그래프를 다변량 시계열 PFN에 넣기 (PEMS 등 교통 데이터)
- MVTSFM이 irrelevant/nuisance context를 무시하지 못하는 문제, regime shift 후 obsolete history
- 조합 BO (count-constrained, amortized proposal), TabPFN surrogate MOBO
