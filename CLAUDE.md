# 연구정리 하네스 — Claude Code 작업 규칙

최신 논문(ML 근본론, graph FM, PFN, TSFM, BO, EDA)의 장점과 한계를 누적하고,
그래프로 엮어 아이디어 검증과 접목 후보 탐색에 쓰는 저장소다.

## 구조
- `papers/<id>.yaml` — 논문 한 편 = 파일 하나. **유일한 원본**. 형식은 `papers/_TEMPLATE.yaml`.
- `taxonomy.yaml` — 강점과 한계에 공통으로 붙이는 태그. 같은 태그가 A의 한계이자 B의 강점이면 접목 후보가 된다.
- `ideas/<이름>.yaml` — `check --save`로 저장한 아이디어.
- `findings/<project>.yaml` — **내 연구의 실패·중단·보존·교훈·미실행 기록**. 원문은 `findings/sources/`. check가 이 기록과도 대조한다.
- `out/` — 빌드 결과물 (엑셀, 그래프, 후보). 직접 고치지 않는다.
- `src/harness.py` — build / validate / tags / link / check.

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
6. `citation` 블록을 채운다. 아래 "인용 규칙"을 반드시 지킨다.
7. 한계를 지적한 후속 논문(`cited`)이 이 하네스에 이미 있으면 `by_id`에 그 id를 적는다. 없으면 그 후속 논문도 새 항목으로 추가할지 사용자에게 묻는다.
8. `python src/harness.py link` (by 제목이 하네스 논문과 일치하면 by_id 자동 기입) → `python src/harness.py validate` → 경고 0개 확인 → `python src/harness.py build`.

## 인용 규칙 (할루시네이션 금지)
- `citation`의 모든 값은 `verified_from`에 적은 페이지에서 **직접 읽은 것**만 쓴다. 기억이나 추정으로 채우지 않는다.
- 저자는 전원, 원문 순서, 페이지에 표기된 철자 그대로. "et al."로 줄이지 않는다.
- 학회 게재는 proceedings·OpenReview·학회 사이트에서 확인됐을 때만 `inproceedings`로 쓴다. 확인 못 하면 `misc` + `arxiv`로 둔다.
- pages, volume, publisher처럼 확인 못 한 필드는 줄을 지운다. 빈칸이 틀린 값보다 낫다.
- BibTeX는 직접 쓰지 않는다. build가 `citation` 블록에서 `out/refs.bib`를 생성한다.
- `cited` 한계의 `by`(제목·연도)도 같은 기준이다. 지적한 문장을 실제로 본 논문만 넣는다.

## 실패 기록·연구 회고를 받으면
1. 원문을 `findings/sources/`에 그대로 저장한다.
2. `findings/<project>.yaml`에 항목을 나눠 적는다. kind는 failure(실패·중단) / preserved(보존한 결과) / lesson(교훈) / untested(미실행, 실패로 세지 않음).
3. failure는 claim, observation(수치는 원문 그대로), verdict, closed_scope(정확히 무엇을 닫았는지), not_closed(닫지 않은 것)를 반드시 쓴다. preserved는 do_not_claim(확대 금지 주장)을 쓴다.
4. 원문에 없는 수치·원인·판정을 만들지 않는다. 태그는 taxonomy.yaml에서 고르고, 검색에 걸릴 keywords를 넉넉히 단다.
5. failure_reason과 reason_status(confirmed/supported/hypothesis/unknown)를 기록한다. 원인 미상인 결과에 설명을 지어내지 않는다. 배제한 설명은 excluded_explanations에 별도로 쓴다.
6. reopen_when, reopen_tags, reopen_keywords, comparison_floor, recheck, source_refs를 채운다. 새 논문이 무엇을 바꾸어야 재검토하는지와 최소 독립 실험을 명시한다.
7. 미실행과 제안 철회는 실제 학습 실패로 세지 않는다. 보존할 긍정 결과·후속으로 해결된 중간 병목을 함께 남긴다. 기존 관찰·ID·판정을 덮어쓰지 않는다.
8. validate → build를 실행하고, source snapshot hash 및 기존 ID 보존을 검사한다.

## 새 논문과 내 실패의 접점을 볼 때
1. 검증한 논문 metadata를 추가한 뒤 `python src/harness.py revisit --paper ID`를 실행한다.
2. 자동 매칭은 강점·태그·키워드의 검색 단서다. 원문에서 장치·가정·입력·비용이 재검토 조건을 충족하는지 확인하기 전에는 해결·재개·novelty라고 말하지 않는다.
3. Benchmark/survey를 해결 메소드로 삼지 않는다. 기존에 이미 쓰거나 이긴/진 방법이 다시 뜬 경우도 구분한다.
4. `recheck`의 강한 기준선과 정보·예산 대조를 유지한다. 필요하면 독립 graph/schema에서 먼저 작은 재검증을 한다. 자동 매칭만으로 GPU 실험을 재개하지 않는다.

## 아이디어 점검 요청을 받으면
1. 아이디어를 taxonomy 태그 2~4개와 키워드로 바꾼다.
2. `python src/harness.py check --tags ... --keywords ... --text "..." --save <이름>`
3. 결과를 이렇게 해석해서 보고한다.
   - 같은 축을 강점으로 가진 논문은 **기각 사유가 아니라 차별점을 정리할 대상**이다.
   - 같은 축의 한계(특히 cited/review)는 동기로 인용할 근거다.
   - 하네스에 관련 논문이 없으면, 외부 검색으로 선행연구를 찾아 먼저 추가할지 제안한다.
   - "내 연구에서 이미 확인한 것"에 failure가 걸리면, 새 아이디어가 그 항목의 **닫은 범위** 안에 있는지 밖에 있는지를 먼저 판단해서 말한다. 이름만 바꾼 반복이면 분명히 지적하고, 닫지 않은 것에 해당하면 실패 기록을 근거로 기각하지 않는다.
   - 강한 기준선 비교를 재학습·확장 실험보다 먼저 하라는 교훈(LLBP-L6)을 실험 계획에 반영한다.

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
- 조합최적화(CO): RL(GRPO)로 학습하는 LLM 휴리스틱 자동설계(CALM 계열)에서 작은 고정 instance 집합 보상의 잡음과 크기·용량 층이 섞인 보상(TSP, online bin packing), neural CO solver와의 비교
- 편향 context 선택(community 집중) 아래 frozen GraphPFN 노드 분류 (findings BSEL)
