#!/usr/bin/env python3
"""연구정리 하네스.

papers/*.yaml (논문별 장점·한계) 를 누적하고, 이를 엑셀·그래프·접목 후보로 만든다.

  python src/harness.py build                       # 엑셀 + 그래프 + 접목 후보 생성
  python src/harness.py validate                    # 태그·필드 검사만
  python src/harness.py tags                        # 태그별 사용 현황
  python src/harness.py check --tags a,b --keywords "x,y" --text "아이디어 설명" [--save 이름]
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"
IDEAS_DIR = ROOT / "ideas"
FINDINGS_DIR = ROOT / "findings"
OUT_DIR = ROOT / "out"
TAXONOMY = ROOT / "taxonomy.yaml"
VIS_JS = ROOT / "src" / "vendor" / "vis-network.min.js"

# 한계의 출처별 신뢰 가중치: 외부에서 확인된 한계일수록 접목 근거로 강하다.
SOURCE_WEIGHT = {"cited": 1.0, "review": 0.8, "self": 0.6, "analyst": 0.4}
SOURCE_KO = {"cited": "후속 논문 지적", "review": "리뷰 지적", "self": "저자 인정", "analyst": "조사자 판단"}
KIND_KO = {"failure": "실패·중단", "preserved": "보존한 결과", "lesson": "교훈", "untested": "미실행 (실패 아님)"}


# ---------------------------------------------------------------- loading
def load_taxonomy():
    t = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8")) or {}
    return t.get("areas", {}) or {}, t.get("tags", {}) or {}


def load_dir(d: Path):
    items = []
    if not d.exists():
        return items
    for p in sorted(d.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        data["_file"] = p.name
        items.append(data)
    return items


def load_findings():
    """findings/*.yaml: 내 연구의 실패·보존·교훈·미실행 기록. 항목마다 project 정보를 붙여 평탄화."""
    out = []
    for f in load_dir(FINDINGS_DIR):
        for e in f.get("entries") or []:
            e = dict(e)
            e["project"] = f.get("project", f["_file"])
            e["project_scope"] = f.get("closed_scope", "")
            e["_file"] = f["_file"]
            out.append(e)
    return out


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def claims_of(p):
    rows = []
    for c in as_list(p.get("strengths")):
        rows.append({"paper": p["id"], "kind": "strength", "text": c.get("text", ""),
                     "tags": as_list(c.get("tags")), "source": "", "by": "", "by_link": ""})
    for c in as_list(p.get("limitations")):
        rows.append({"paper": p["id"], "kind": "limitation", "text": c.get("text", ""),
                     "tags": as_list(c.get("tags")), "source": c.get("source", ""),
                     "by": c.get("by", ""), "by_link": c.get("by_link", "")})
    return rows


# ---------------------------------------------------------------- citations
CITE_TYPES = {"inproceedings", "article", "misc"}
_STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "via", "with", "from", "towards", "toward", "is", "are", "when"}


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def bib_keys(papers):
    keys, used = {}, defaultdict(int)
    for p in papers:
        c = p.get("citation") or {}
        authors = as_list(c.get("authors"))
        last = authors[0].split()[-1] if authors else p["id"]
        last = re.sub(r"[^A-Za-z]", "", unicodedata.normalize("NFKD", last)).lower() or re.sub(r"\W", "", p["id"]).lower()
        words = [w for w in re.findall(r"[A-Za-z0-9]+", str(c.get("title", p.get("title", "")))) if w.lower() not in _STOP]
        key = f"{last}{c.get('year', p.get('year', ''))}{(words[0] if words else 'paper').lower()}"
        used[key] += 1
        keys[p["id"]] = key if used[key] == 1 else f"{key}{chr(ord('a') + used[key] - 1)}"
    return keys


def _tex(v):
    return str(v).replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def bibtex(p, key):
    c = p.get("citation") or {}
    if not c:
        return ""
    typ = c.get("type", "misc")
    f = [("title", "{" + _tex(c.get("title", "")) + "}"), ("author", " and ".join(as_list(c.get("authors"))))]
    if typ == "inproceedings" and c.get("booktitle"):
        f.append(("booktitle", _tex(c["booktitle"])))
    if typ == "article" and c.get("journal"):
        f.append(("journal", _tex(c["journal"])))
    for k in ("volume", "pages", "publisher"):
        if c.get(k):
            f.append((k, _tex(c[k])))
    f.append(("year", c.get("year", "")))
    if c.get("arxiv"):
        if typ == "misc":
            f.append(("howpublished", f"arXiv preprint arXiv:{c['arxiv']}"))
        f += [("eprint", c["arxiv"]), ("archivePrefix", "arXiv")]
    if c.get("url"):
        f.append(("url", c["url"]))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f if str(v) != "")
    return f"@{typ}{{{key},\n{body}\n}}"


def title_index(papers):
    idx = {}
    for p in papers:
        for t in (p.get("title"), (p.get("citation") or {}).get("title")):
            if t:
                idx[norm_title(t)] = p["id"]
    return idx


def resolve_by_id(lim, tidx):
    """by_id가 없으면 by 제목으로 하네스 안의 논문을 찾는다."""
    if lim.get("by_id"):
        return lim["by_id"]
    by = norm_title(re.sub(r"\([^)]*\)\s*$", "", str(lim.get("by", ""))))
    if not by:
        return None
    for t, pid in tidx.items():
        if by == t or (len(t) > 25 and (by.startswith(t) or t.startswith(by))):
            return pid
    return None


def critique_links(papers):
    """(비판한 논문 id, 비판받은 논문 id, 한계) 목록."""
    tidx, ids, out = title_index(papers), {p["id"] for p in papers}, []
    for p in papers:
        for lim in as_list(p.get("limitations")):
            if lim.get("source") != "cited":
                continue
            bid = resolve_by_id(lim, tidx)
            if bid in ids and bid != p["id"]:
                out.append((bid, p["id"], lim))
    return out


def backlog(papers):
    """한계를 지적했지만 아직 하네스에 없는 후속 논문 목록 (다음에 추가할 후보)."""
    tidx, ids = title_index(papers), {p["id"] for p in papers}
    items = {}
    for p in papers:
        for lim in as_list(p.get("limitations")):
            if lim.get("source") != "cited":
                continue
            bid = resolve_by_id(lim, tidx)
            if bid in ids:
                continue
            title = re.sub(r"\s*\([^)]*\)\s*$", "", str(lim.get("by", ""))).strip()
            k = norm_title(title)
            it = items.setdefault(k, {"title": lim.get("by", ""), "link": lim.get("by_link", ""), "targets": set()})
            it["targets"].add(p["id"])
    return sorted(items.values(), key=lambda x: (-len(x["targets"]), x["title"]))


# ---------------------------------------------------------------- validate
def validate(papers, ideas, areas, tags, findings=()):
    warns, ids = [], set()
    for p in papers:
        pid = p.get("id")
        if not pid:
            warns.append(f"{p['_file']}: id 없음")
            continue
        if pid in ids:
            warns.append(f"중복 id: {pid}")
        ids.add(pid)
        if p["_file"] != f"{pid}.yaml":
            warns.append(f"{p['_file']}: 파일명과 id({pid})가 다름")
        for k in ("title", "problem", "core_idea"):
            if not p.get(k):
                warns.append(f"{pid}: '{k}' 비어 있음")
        for a in as_list(p.get("areas")):
            if a not in areas:
                warns.append(f"{pid}: 미등록 area '{a}'")
        if p.get("proposed_tags"):
            warns.append(f"{pid}: proposed_tags 있음 → taxonomy.yaml에 반영 후 지울 것: {list(p['proposed_tags'])}")
        for c in claims_of(p):
            for t in c["tags"]:
                if t not in tags:
                    warns.append(f"{pid}: 미등록 tag '{t}'")
            if c["kind"] == "limitation":
                if c["source"] not in SOURCE_WEIGHT:
                    warns.append(f"{pid}: limitation source '{c['source']}' 잘못됨")
                if c["source"] in ("cited", "review") and not c["by_link"]:
                    warns.append(f"{pid}: {c['source']} 한계에 by_link 없음")
    tidx = title_index(papers)
    for p in papers:
        pid, c = p.get("id"), p.get("citation")
        if not pid:
            continue
        if not c:
            warns.append(f"{pid}: citation 블록 없음")
        else:
            if c.get("type") not in CITE_TYPES:
                warns.append(f"{pid}: citation.type '{c.get('type')}' 잘못됨")
            for k in ("title", "authors", "year", "verified_from"):
                if not c.get(k):
                    warns.append(f"{pid}: citation.{k} 비어 있음")
            if any("et al" in str(a).lower() for a in as_list(c.get("authors"))):
                warns.append(f"{pid}: citation.authors에 et al. — 저자 전원을 적을 것")
            if c.get("type") == "inproceedings" and not c.get("booktitle"):
                warns.append(f"{pid}: inproceedings인데 booktitle 없음")
            if c.get("type") == "article" and not c.get("journal"):
                warns.append(f"{pid}: article인데 journal 없음")
            if c.get("type") == "misc" and not (c.get("arxiv") or c.get("url")):
                warns.append(f"{pid}: misc인데 arxiv/url 없음")
            if c.get("title") and norm_title(c["title"]) != norm_title(p.get("title", "")):
                warns.append(f"{pid}: title과 citation.title이 다름")
        for lim in as_list(p.get("limitations")):
            if lim.get("by_id") and lim["by_id"] not in ids:
                warns.append(f"{pid}: by_id '{lim['by_id']}'가 하네스에 없음")
            if lim.get("source") == "cited" and not lim.get("by_id"):
                guess = resolve_by_id(lim, tidx)
                if guess:
                    warns.append(f"{pid}: by '{str(lim.get('by'))[:40]}…' → by_id: {guess} 로 연결 가능 (link 명령으로 자동 기입)")
    for i in ideas:
        for t in as_list(i.get("tags")):
            if t not in tags:
                warns.append(f"idea {i.get('id')}: 미등록 tag '{t}'")
    fids = set()
    for e in findings:
        if e.get("id") in fids:
            warns.append(f"finding 중복 id: {e.get('id')}")
        fids.add(e.get("id"))
        if e.get("kind") not in KIND_KO:
            warns.append(f"finding {e.get('id')}: kind '{e.get('kind')}' 잘못됨")
        for t in as_list(e.get("tags")):
            if t not in tags:
                warns.append(f"finding {e.get('id')}: 미등록 tag '{t}'")
    return warns


# ---------------------------------------------------------------- analysis
def tag_weights(papers):
    """흔한 태그일수록 매칭 근거가 약하다: weight = 1 / (1 + ln df), df = 태그를 쓴 논문 수."""
    import math
    df = defaultdict(set)
    for p in papers:
        for c in claims_of(p):
            for t in c["tags"]:
                df[t].add(p["id"])
    return {t: 1.0 / (1.0 + math.log(len(ps))) for t, ps in df.items()}


def combination_candidates(papers, top=40):
    """한계 하나(A) ← 강점(B) 매칭을 한 건씩 점수화한다.

    점수 = 한계 출처 신뢰도 × Σ(겹친 태그의 희소성 가중치) + 분야 교차 보너스(0.3).
    태그 매칭은 후보를 '제안'할 뿐이고, 실제로 메워지는지는 사람이 판단한다.
    """
    by_id = {p["id"]: p for p in papers}
    tw = tag_weights(papers)
    out = []
    for a in papers:
        for lim in as_list(a.get("limitations")):
            lt = set(as_list(lim.get("tags")))
            if not lt:
                continue
            for b in papers:
                if b["id"] == a["id"]:
                    continue
                scored = []
                for st in as_list(b.get("strengths")):
                    ov = lt & set(as_list(st.get("tags")))
                    if ov:
                        scored.append((sum(tw.get(t, 1.0) for t in ov), sorted(ov), st.get("text", "")))
                if not scored:
                    continue
                scored.sort(key=lambda x: -x[0])
                best = scored[0][0]
                keep = [x for x in scored if x[0] >= best - 1e-9][:2]
                tags_hit = sorted({t for x in keep for t in x[1]})
                cross = not (set(as_list(by_id[a["id"]].get("areas"))) & set(as_list(b.get("areas"))))
                score = SOURCE_WEIGHT.get(lim.get("source"), 0.5) * best + (0.3 if cross else 0.0)
                out.append({
                    "target": a["id"], "donor": b["id"], "cross_area": cross, "score": round(score, 2),
                    "tags": tags_hit, "limitation": lim.get("text", ""), "source": lim.get("source", ""),
                    "by": lim.get("by", ""), "strengths": [x[2] for x in keep],
                })
    out.sort(key=lambda x: (-x["score"], x["target"], x["donor"]))
    return out[:top]


def tag_table(papers, tags):
    lim, st = defaultdict(set), defaultdict(set)
    for p in papers:
        for c in claims_of(p):
            for t in c["tags"]:
                (lim if c["kind"] == "limitation" else st)[t].add(p["id"])
    rows = []
    for t in sorted(set(tags) | set(lim) | set(st)):
        L, S = sorted(lim[t]), sorted(st[t])
        if not L and not S:
            status = "미사용"
        elif L and not S:
            status = "빈칸 (해결한 논문 없음)"
        elif L:
            status = "부분 해결"
        else:
            status = "강점만 있음"
        rows.append({"tag": t, "desc": tags.get(t, "(미등록)"), "limited": L, "addressed": S, "status": status})
    order = {"빈칸 (해결한 논문 없음)": 0, "부분 해결": 1, "강점만 있음": 2, "미사용": 3}
    rows.sort(key=lambda r: (order[r["status"]], -len(r["limited"]), r["tag"]))
    return rows


def check_idea(papers, tags_q, keywords, text="", findings=()):
    tags_q, kws = set(tags_q), [k.lower() for k in keywords if k]
    prior, motive, kw_hits = [], [], []
    for p in papers:
        for c in claims_of(p):
            ov = tags_q & set(c["tags"])
            if ov and c["kind"] == "strength":
                prior.append((p, c, ov))
            if ov and c["kind"] == "limitation":
                motive.append((p, c, ov))
        blob = " ".join(str(p.get(k, "")) for k in ("title", "problem", "core_idea", "key_assumptions",
                                                    "open_questions", "hooks", "notes"))
        blob += " " + " ".join(c["text"] for c in claims_of(p))
        hits = [k for k in kws if k in blob.lower()]
        if hits:
            kw_hits.append((p, hits))
    motive.sort(key=lambda x: -SOURCE_WEIGHT.get(x[1]["source"], 0))

    L = []
    L.append(f"# 아이디어 점검 ({dt.date.today()})\n")
    if text:
        L.append(f"> {text}\n")
    L.append(f"태그: {', '.join(sorted(tags_q)) or '-'} / 키워드: {', '.join(keywords) or '-'}\n")

    is_eval = lambda p: bool({"benchmark", "survey"} & set(as_list(p.get("areas"))))
    methods = [x for x in prior if not is_eval(x[0])]
    evals = [x for x in prior if is_eval(x[0])]
    L.append("## 1. 차별점을 정리해야 할 선행연구")
    L.append("같은 축을 이미 강점으로 내세운 방법 논문이에요. 기각 사유가 아니라, 무엇이 다른지 한 줄로 써둘 대상입니다.\n")
    if methods:
        for p, c, ov in methods:
            L.append(f"- **{p['id']}** [{', '.join(sorted(ov))}] {c['text']}")
    else:
        L.append("- 하네스 안에는 아직 없음. 외부 검색으로 한 번 더 확인할 것.")
    if evals:
        L.append("\n평가·분석 연구 중 이 축을 다룬 것 (실험 설계·평가 protocol 참고용):")
        for p, c, ov in evals:
            L.append(f"- **{p['id']}** [{', '.join(sorted(ov))}] {c['text']}")
    L.append("")

    L.append("## 2. 동기로 인용할 수 있는 한계")
    L.append("이 아이디어가 겨냥하는 축에서 기존 논문이 약하다고 확인된 지점이에요. 후속 논문·리뷰 지적이 가장 강한 근거입니다.\n")
    if motive:
        for p, c, ov in motive:
            src = SOURCE_KO.get(c["source"], c["source"])
            by = f" — {c['by']}" if c["by"] else ""
            L.append(f"- **{p['id']}** ({src}{by}) [{', '.join(sorted(ov))}] {c['text']}")
    else:
        L.append("- 없음. 동기를 받쳐줄 한계가 하네스에 아직 없다는 뜻이니, 관련 논문을 먼저 추가하면 좋아요.")
    L.append("")

    L.append("## 3. 키워드가 등장하는 논문")
    if kw_hits:
        for p, hits in kw_hits:
            L.append(f"- **{p['id']}**: {', '.join(hits)}")
    else:
        L.append("- 없음")
    L.append("")

    # 4. 내 연구 기록
    hits = defaultdict(list)
    for e in findings:
        ov = tags_q & set(as_list(e.get("tags")))
        blob = " ".join(str(e.get(k, "")) for k in ("claim", "observation", "closed_scope", "not_closed", "do_not_claim"))
        blob += " " + " ".join(as_list(e.get("keywords")))
        kh = [k for k in kws if k in blob.lower()]
        if ov or kh:
            hits[e.get("kind")].append((e, sorted(ov), kh))
    L.append("## 4. 내 연구에서 이미 확인한 것")
    if not hits:
        L.append("- 걸리는 기록 없음")
    head = {"failure": "### 이미 닫은 갈래 — 같은 가설을 이름만 바꿔 반복하는 건 아닌지 확인",
            "lesson": "### 반복된 교훈",
            "preserved": "### 보존한 결과 — 재사용할 수 있는 자산",
            "untested": "### 아직 안 한 것 — 실패로 세지 않음"}
    for kind in ("failure", "lesson", "preserved", "untested"):
        if not hits.get(kind):
            continue
        L.append(head[kind])
        for e, ov, kh in hits[kind]:
            why = ", ".join(ov + kh)
            L.append(f"- **{e['id']}** ({e['project']}) [{why}] {e.get('claim', '')}")
            if kind == "failure":
                L.append(f"  - 판정: {e.get('verdict', '')}. {e.get('observation', '')}")
                if e.get("closed_scope"):
                    L.append(f"  - 닫은 범위: {e['closed_scope']}")
                if e.get("not_closed"):
                    L.append(f"  - 닫지 않은 것: {e['not_closed']}")
            elif kind == "preserved" and e.get("do_not_claim"):
                L.append(f"  - {e.get('observation', '')} (확대 금지: {e['do_not_claim']})")
            elif e.get("observation"):
                L.append(f"  - {e['observation']}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- excel
def write_excel(papers, ideas, areas, tags, cands, tagrows, path: Path, findings=()):
    keys = bib_keys(papers)
    crit = critique_links(papers)
    critiqued_by, critiques = defaultdict(list), defaultdict(list)
    for a_, b_, _ in crit:
        critiqued_by[b_].append(a_)
        critiques[a_].append(b_)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    HEAD_FILL = PatternFill("solid", fgColor="1F3A5F")
    HEAD_FONT = Font(bold=True, color="FFFFFF")
    WRAP = Alignment(wrap_text=True, vertical="top")
    FILL = {"strength": PatternFill("solid", fgColor="E8F5E9"),
            "limitation": PatternFill("solid", fgColor="FDECEA")}
    LINK = Font(color="1155CC", underline="single")

    wb = Workbook()

    def sheet(title, headers, widths, rows, first=False, fills=None, links=None):
        ws = wb.active if first else wb.create_sheet()
        ws.title = title
        ws.append(headers)
        for c in ws[1]:
            c.fill, c.font, c.alignment = HEAD_FILL, HEAD_FONT, Alignment(vertical="center", wrap_text=True)
        for r_i, row in enumerate(rows):
            ws.append(row)
            for c in ws[ws.max_row]:
                c.alignment = WRAP
                if fills and fills[r_i]:
                    c.fill = fills[r_i]
            for col in (links or []):
                cell = ws.cell(row=ws.max_row, column=col)
                if isinstance(cell.value, str) and cell.value.startswith("http"):
                    cell.hyperlink, cell.font = cell.value, LINK
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "B2"
        if rows:
            ws.auto_filter.ref = ws.dimensions
        return ws

    j = lambda xs: "\n".join(f"• {x}" for x in as_list(xs))
    c = lambda xs: ", ".join(as_list(xs))

    n_lim = sum(len(as_list(p.get("limitations"))) for p in papers)
    n_ext = sum(1 for p in papers for x in as_list(p.get("limitations")) if x.get("source") in ("cited", "review"))
    guide = [
        ["생성 시각", dt.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["논문 수", len(papers)],
        ["한계 수 (그중 후속 논문·리뷰 지적)", f"{n_lim} ({n_ext})"],
        ["", ""],
        ["Papers", "논문별 요약. 문제, 핵심 아이디어, 가정, 열린 질문, 내 연구와의 접점."],
        ["Claims", "장점·한계 한 줄씩. 한계는 출처(저자 인정 / 후속 논문 지적 / 리뷰 지적 / 조사자 판단)를 구분해요."],
        ["Candidates", "A의 한계 하나와 B의 강점이 같은 태그로 걸린 건. 'A의 약점을 B의 장치로 메운다'는 접목 후보예요. 점수 = 한계 출처 신뢰도 × 태그 희소성 (+분야 교차 0.3). 후보 제안용이고, 실제로 메워지는지는 직접 판단."],
        ["Gaps", "태그별로 '한계로 지적한 논문'과 '해결했다는 논문'을 모은 표. '빈칸'은 아무도 해결 못 한 축이라 아이디어 후보지예요."],
        ["My Findings", "내 연구의 실패·중단·보존·교훈 기록 (findings/*.yaml). check가 이 기록과도 대조해요. '닫지 않은 것'까지 같이 읽어야 실패를 과하게 일반화하지 않아요."],
        ["Backlog", "한계를 지적했지만 아직 하네스에 항목이 없는 후속 논문. 다음에 추가할 후보예요."],
        ["Ideas", "check --save 로 저장한 아이디어 기록."],
        ["Citations", "논문별 인용 정보와 BibTeX. 모든 필드는 '확인한 페이지'에서 직접 읽은 값이에요. 전체 BibTeX는 out/refs.bib."],
        ["Tags", "태그 정의. 새 태그는 taxonomy.yaml에 먼저 추가."],
        ["", ""],
        ["편집 원칙", "이 엑셀은 빌드 결과물이에요. 내용 수정은 papers/*.yaml에서 하고 build를 다시 돌리세요."],
    ]
    ws = sheet("안내", ["항목", "내용"], [34, 110], guide, first=True)
    for row in ws.iter_rows(min_row=2, max_col=1):
        row[0].font = Font(bold=True)

    rows, link_cols = [], [18, 19]
    for p in papers:
        lims = as_list(p.get("limitations"))
        ext = sum(1 for x in lims if x.get("source") in ("cited", "review"))
        rows.append([
            p["id"], p.get("title", ""), p.get("authors", ""), p.get("venue", ""), p.get("year", ""),
            c(p.get("areas")), p.get("problem", ""), p.get("core_idea", ""), j(p.get("key_assumptions")),
            len(as_list(p.get("strengths"))), f"{len(lims)} ({ext})", c(p.get("builds_on")),
            c(sorted(set(critiqued_by[p["id"]]))), c(sorted(set(critiques[p["id"]]))),
            j(p.get("open_questions")), j(p.get("hooks")),
            keys.get(p["id"], ""),
            (p.get("links") or {}).get("paper", ""), (p.get("links") or {}).get("code", ""),
            p.get("notes", ""), str(p.get("added", "")),
        ])
    sheet("Papers",
          ["id", "title", "authors", "venue", "year", "areas", "문제", "핵심 아이디어", "가정",
           "강점 수", "한계 수 (외부 지적)", "builds_on", "비판한 후속 논문", "이 논문이 비판한 논문",
           "열린 질문", "내 연구 접점", "인용 키", "paper", "code", "notes", "added"],
          [14, 40, 18, 14, 7, 14, 40, 60, 40, 8, 12, 18, 22, 22, 45, 45, 22, 30, 30, 30, 11], rows, links=link_cols)

    rows, fills = [], []
    tidx = title_index(papers)
    for p in papers:
        for cl in claims_of(p):
            by_id = ""
            if cl["kind"] == "limitation" and cl["source"] == "cited":
                by_id = resolve_by_id(cl, tidx) or ""
            rows.append([cl["paper"], "강점" if cl["kind"] == "strength" else "한계",
                         SOURCE_KO.get(cl["source"], ""), c(cl["tags"]), cl["text"], cl["by"], by_id, cl["by_link"]])
            fills.append(FILL[cl["kind"]])
    sheet("Claims", ["paper", "구분", "출처", "tags", "내용", "지적한 논문", "지적한 논문 id", "링크"],
          [14, 7, 14, 26, 90, 40, 16, 36], rows, fills=fills, links=[8])

    rows = []
    for i, cd in enumerate(cands, 1):
        src = SOURCE_KO.get(cd["source"], cd["source"]) + (f" — {cd['by']}" if cd["by"] else "")
        rows.append([i, cd["target"], cd["limitation"], src, cd["donor"], "\n".join(f"• {x}" for x in cd["strengths"]),
                     c(cd["tags"]), cd["score"], "예" if cd["cross_area"] else ""])
    sheet("Candidates", ["순위", "한계가 있는 논문", "한계", "한계 출처", "메워줄 논문", "메워줄 강점", "겹친 태그", "점수", "분야 교차"],
          [6, 14, 60, 30, 14, 60, 22, 7, 9], rows)

    rows = [[r["tag"], r["desc"], r["status"], c(r["limited"]), c(r["addressed"])] for r in tagrows]
    gap_fill = PatternFill("solid", fgColor="FFF4CE")
    sheet("Gaps", ["tag", "설명", "상태", "한계로 지적한 논문", "해결했다는 논문"],
          [22, 60, 20, 36, 36], rows, fills=[gap_fill if r["status"].startswith("빈칸") else None for r in tagrows])

    kfill = {"failure": PatternFill("solid", fgColor="FDECEA"), "preserved": PatternFill("solid", fgColor="E8F5E9"),
             "lesson": PatternFill("solid", fgColor="E8F0FE"), "untested": None}
    rows = [[e.get("project", ""), e.get("id", ""), KIND_KO.get(e.get("kind"), e.get("kind", "")), e.get("claim", ""),
             e.get("observation", ""), e.get("verdict", ""), e.get("closed_scope", ""),
             e.get("not_closed", "") or e.get("do_not_claim", ""), c(e.get("tags")), c(e.get("keywords")), e.get("evidence", "")]
            for e in findings]
    sheet("My Findings", ["project", "id", "구분", "가설·주장", "관찰", "판정", "닫은 범위", "닫지 않은 것 / 확대 금지", "tags", "keywords", "근거"],
          [22, 11, 12, 45, 70, 22, 40, 50, 26, 30, 10], rows, fills=[kfill.get(e.get("kind")) for e in findings])

    rows = [[b["title"], ", ".join(sorted(b["targets"])), len(b["targets"]), b["link"]] for b in backlog(papers)]
    sheet("Backlog", ["후속 논문 (아직 하네스에 없음)", "이 논문이 비판한 항목", "비판 수", "링크"], [70, 30, 9, 40], rows, links=[4])

    rows = [[i.get("id", ""), str(i.get("date", "")), i.get("text", ""), c(i.get("tags")), c(i.get("keywords"))]
            for i in ideas]
    sheet("Ideas", ["id", "date", "아이디어", "tags", "keywords"], [18, 11, 90, 30, 30], rows)

    rows = []
    for p in papers:
        ct = p.get("citation") or {}
        venue = ct.get("booktitle") or ct.get("journal") or (f"arXiv:{ct['arxiv']}" if ct.get("arxiv") else "")
        rows.append([p["id"], keys.get(p["id"], ""), ct.get("type", ""), ct.get("title", ""),
                     "; ".join(as_list(ct.get("authors"))), venue, ct.get("year", ""), ct.get("url", ""),
                     "\n".join(as_list(ct.get("verified_from"))), bibtex(p, keys.get(p["id"], ""))])
    sheet("Citations", ["id", "인용 키", "type", "title", "authors", "venue", "year", "url", "확인한 페이지", "BibTeX"],
          [14, 24, 12, 45, 50, 40, 7, 34, 40, 70], rows, links=[8])

    rows = [[k, "tag", v] for k, v in tags.items()] + [[k, "area", v] for k, v in areas.items()]
    sheet("Tags", ["이름", "종류", "설명"], [24, 8, 90], rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------- graph
AREA_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#EECA3B"]


def graph_data(papers, ideas, areas, tags, findings=()):
    ids = {p["id"] for p in papers}
    used_areas = [a for a in areas if any(a in as_list(p.get("areas")) for p in papers)]
    area_color = {a: AREA_COLORS[i % len(AREA_COLORS)] for i, a in enumerate(used_areas)}
    nodes, edges, seen_ext = [], [], set()
    used_tags = set()

    for p in papers:
        pa = as_list(p.get("areas"))
        color = area_color.get(pa[0], "#888") if pa else "#888"
        detail = {
            "kind": "paper", "id": p["id"], "title": p.get("title", ""), "venue": p.get("venue", ""),
            "areas": pa, "problem": p.get("problem", ""), "core_idea": p.get("core_idea", ""),
            "strengths": [{"text": x.get("text", ""), "tags": as_list(x.get("tags"))} for x in as_list(p.get("strengths"))],
            "limitations": [{"text": x.get("text", ""), "tags": as_list(x.get("tags")),
                             "source": SOURCE_KO.get(x.get("source"), x.get("source", "")),
                             "by": x.get("by", ""), "by_link": x.get("by_link", "")} for x in as_list(p.get("limitations"))],
            "open_questions": as_list(p.get("open_questions")), "hooks": as_list(p.get("hooks")),
            "links": p.get("links") or {},
        }
        nodes.append({"id": f"p:{p['id']}", "label": p["id"], "group": "paper", "shape": "box",
                      "color": {"background": color, "border": color}, "font": {"color": "#fff"},
                      "area": pa, "detail": detail})
        for b in as_list(p.get("builds_on")):
            if b in ids:
                edges.append({"from": f"p:{p['id']}", "to": f"p:{b}", "type": "builds_on"})
            else:
                if b not in seen_ext:
                    seen_ext.add(b)
                    nodes.append({"id": f"x:{b}", "label": b, "group": "external", "shape": "dot", "size": 6,
                                  "color": "#bbb", "font": {"color": "#999", "size": 11},
                                  "detail": {"kind": "external", "id": b}})
                edges.append({"from": f"p:{p['id']}", "to": f"x:{b}", "type": "builds_on_ext"})
        for c in claims_of(p):
            for t in c["tags"]:
                used_tags.add(t)
                if c["kind"] == "strength":
                    edges.append({"from": f"p:{p['id']}", "to": f"t:{t}", "type": "strength", "text": c["text"]})
                else:
                    edges.append({"from": f"p:{p['id']}", "to": f"t:{t}", "type": "limitation",
                                  "source": c["source"], "source_ko": SOURCE_KO.get(c["source"], c["source"]),
                                  "text": c["text"], "by": c["by"]})

    for a_, b_, lim in critique_links(papers):
        edges.append({"from": f"p:{a_}", "to": f"p:{b_}", "type": "critiques", "text": lim.get("text", "")})

    for i in ideas:
        nodes.append({"id": f"i:{i['id']}", "label": i["id"], "group": "idea", "shape": "star", "size": 14,
                      "color": "#EECA3B", "detail": {"kind": "idea", **{k: i.get(k) for k in ("id", "text", "tags")}}})
        for t in as_list(i.get("tags")):
            used_tags.add(t)
            edges.append({"from": f"i:{i['id']}", "to": f"t:{t}", "type": "idea"})

    fcolor = {"failure": "#9E3B3B", "preserved": "#3B7D4A", "lesson": "#3B5B9E", "untested": "#999999"}
    for e in findings:
        nodes.append({"id": f"f:{e['id']}", "label": e["id"], "group": "finding", "shape": "diamond", "size": 9,
                      "color": fcolor.get(e.get("kind"), "#777"), "font": {"color": "#555", "size": 10},
                      "detail": {"kind": "finding", "fkind": KIND_KO.get(e.get("kind"), ""), **{k: e.get(k, "") for k in
                                 ("id", "project", "claim", "observation", "verdict", "closed_scope", "not_closed", "do_not_claim")},
                                 "tags": as_list(e.get("tags"))}})
        for t in as_list(e.get("tags")):
            used_tags.add(t)
            edges.append({"from": f"f:{e['id']}", "to": f"t:{t}", "type": "finding"})

    for t in sorted(used_tags):
        nodes.append({"id": f"t:{t}", "label": t, "group": "tag", "shape": "ellipse",
                      "color": {"background": "#f3f3f3", "border": "#999"}, "font": {"color": "#333", "size": 12},
                      "detail": {"kind": "tag", "id": t, "desc": tags.get(t, "(미등록)")}})
    return {"nodes": nodes, "edges": edges, "areas": area_color}


GRAPH_HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>연구정리 하네스 그래프</title>
<style>
 body{margin:0;font-family:-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;display:flex;height:100vh;color:#222}
 #side{width:380px;overflow:auto;border-right:1px solid #ddd;padding:14px 16px;box-sizing:border-box;font-size:13px;line-height:1.5}
 #net{flex:1}
 h1{font-size:16px;margin:0 0 8px} h2{font-size:14px;margin:14px 0 4px} h3{font-size:13px;margin:10px 0 2px}
 .ctl label{display:block;margin:2px 0} input[type=text]{width:100%;padding:5px;box-sizing:border-box;margin:6px 0}
 .lg span{display:inline-block;margin-right:10px} .sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:4px}
 ul{padding-left:18px;margin:4px 0} li{margin:3px 0} .tag{display:inline-block;background:#eee;border-radius:3px;padding:0 4px;margin:0 2px;font-size:11px}
 .src{color:#888;font-size:11px} a{color:#1155cc}
</style></head><body>
<div id="side">
 <h1>연구정리 하네스</h1>
 <div class="src">__STAMP__</div>
 <input id="q" type="text" placeholder="논문·태그 검색 후 Enter">
 <div class="ctl">
  <label><input type="checkbox" id="e_strength" checked> 강점 (논문 → 태그)</label>
  <label><input type="checkbox" id="e_limitation" checked> 한계 (논문 → 태그)</label>
  <label><input type="checkbox" id="e_crit" checked> 비판 (후속 논문 → 원 논문)</label>
  <label><input type="checkbox" id="e_builds" checked> builds_on</label>
  <label><input type="checkbox" id="e_ext"> 외부 모델 노드</label>
  <label><input type="checkbox" id="e_weak" checked> 조사자 판단 한계 포함</label>
  <label><input type="checkbox" id="e_find"> 내 연구 기록 (실패·보존·교훈)</label>
 </div>
 <div class="lg" style="margin-top:6px">
  <span><i class="sw" style="background:#2e7d32"></i>강점</span>
  <span><i class="sw" style="background:#c62828"></i>한계(외부 지적=굵게)</span>
  <span><i class="sw" style="background:#E67E22"></i>비판</span>
  <span><i class="sw" style="background:#4C78A8"></i>builds_on</span>
 </div>
 <div id="areas" class="lg" style="margin-top:4px"></div>
 <div id="info"><h2>사용법</h2><p>노드를 클릭하면 내용이 여기 나와요. 태그 노드를 누르면 그 축을 강점으로 가진 논문과 한계로 가진 논문이 갈라져 보입니다. 같은 태그에 초록선과 빨간선이 함께 걸려 있으면 접목 후보예요.</p></div>
</div>
<div id="net"></div>
<script>__VIS__</script>
<script>
const DATA = __DATA__;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tg = ts => (ts||[]).map(t => `<span class="tag">${esc(t)}</span>`).join("");
document.getElementById("areas").innerHTML = Object.entries(DATA.areas).map(([a,c]) => `<span><i class="sw" style="background:${c};height:10px"></i>${esc(a)}</span>`).join("");

function styleEdge(e){
  const s = {id: e.from+"|"+e.to+"|"+e.type+"|"+Math.random(), from:e.from, to:e.to, type:e.type, source:e.source};
  if(e.type==="strength"){ s.color={color:"#2e7d32",opacity:.7}; s.width=1.5; s.title=e.text; }
  else if(e.type==="limitation"){ const strong = e.source==="cited"||e.source==="review";
    s.color={color:"#c62828",opacity:.75}; s.width= strong?3:1.2; s.dashes = e.source==="analyst";
    s.title = (e.by? "["+e.by+"] ":"") + e.text; }
  else if(e.type==="idea"){ s.color={color:"#d4a400"}; s.dashes=true; }
  else if(e.type==="finding"){ s.color={color:"#888",opacity:.5}; s.dashes=[2,4]; s.width=1; }
  else if(e.type==="critiques"){ s.color={color:"#E67E22"}; s.arrows="to"; s.width=2.2; s.title="비판: "+e.text; }
  else { s.color={color:"#4C78A8"}; s.arrows="to"; s.width=1.5; }
  return s;
}
const nodes = new vis.DataSet(DATA.nodes.map(n => ({...n, title: n.detail && n.detail.title ? n.detail.title : undefined})));
const edges = new vis.DataSet(DATA.edges.map(styleEdge));
const nodeView = new vis.DataView(nodes, {filter: n => (n.group!=="external" || ck("e_ext")) && (n.group!=="finding" || ck("e_find"))});
const edgeView = new vis.DataView(edges, {filter: e => {
  if(e.type==="strength") return ck("e_strength");
  if(e.type==="limitation") return ck("e_limitation") && (e.source!=="analyst" || ck("e_weak"));
  if(e.type==="builds_on") return ck("e_builds");
  if(e.type==="critiques") return ck("e_crit");
  if(e.type==="finding") return ck("e_find");
  if(e.type==="builds_on_ext") return ck("e_builds") && ck("e_ext");
  return true; }});
function ck(id){ return document.getElementById(id).checked; }
const net = new vis.Network(document.getElementById("net"), {nodes:nodeView, edges:edgeView}, {
  physics:{solver:"forceAtlas2Based", forceAtlas2Based:{gravitationalConstant:-60, springLength:120}, stabilization:{iterations:300}},
  interaction:{hover:true, tooltipDelay:120}, nodes:{margin:8}, edges:{smooth:{type:"continuous"}}});
["e_strength","e_limitation","e_crit","e_builds","e_ext","e_weak","e_find"].forEach(id => document.getElementById(id).onchange = () => {nodeView.refresh(); edgeView.refresh();});

function paperHTML(d){
  const L = d.links||{};
  return `<h2>${esc(d.id)}</h2><div>${esc(d.title)}</div><div class="src">${esc(d.venue)} · ${esc((d.areas||[]).join(", "))}
   ${L.paper?` · <a href="${esc(L.paper)}" target="_blank">paper</a>`:""}${L.code?` · <a href="${esc(L.code)}" target="_blank">code</a>`:""}</div>
   <h3>문제</h3><div>${esc(d.problem)}</div><h3>핵심 아이디어</h3><div>${esc(d.core_idea)}</div>
   <h3>강점</h3><ul>${d.strengths.map(s=>`<li>${esc(s.text)} ${tg(s.tags)}</li>`).join("")}</ul>
   <h3>한계</h3><ul>${d.limitations.map(s=>`<li>${esc(s.text)} ${tg(s.tags)}<br><span class="src">${esc(s.source)}${s.by?" — "+(s.by_link?`<a href="${esc(s.by_link)}" target="_blank">${esc(s.by)}</a>`:esc(s.by)):""}</span></li>`).join("")}</ul>
   ${(()=>{const cb=DATA.edges.filter(e=>e.type==="critiques"&&e.to==="p:"+d.id).map(e=>e.from.slice(2)); const ct=DATA.edges.filter(e=>e.type==="critiques"&&e.from==="p:"+d.id).map(e=>e.to.slice(2));
      return (cb.length?`<h3>이 논문을 비판한 후속 논문</h3><div>${[...new Set(cb)].map(esc).join(", ")}</div>`:"")+(ct.length?`<h3>이 논문이 비판한 논문</h3><div>${[...new Set(ct)].map(esc).join(", ")}</div>`:"");})()}
   ${d.open_questions.length?`<h3>열린 질문</h3><ul>${d.open_questions.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`:""}
   ${d.hooks.length?`<h3>내 연구 접점</h3><ul>${d.hooks.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`:""}`;
}
function tagHTML(d){
  const ins = DATA.edges.filter(e => e.to==="t:"+d.id);
  const S = ins.filter(e=>e.type==="strength"), Lm = ins.filter(e=>e.type==="limitation");
  const nm = id => id.split(":")[1];
  return `<h2>${esc(d.id)}</h2><div>${esc(d.desc)}</div>
   <h3>이 축을 강점으로 가진 논문 (${S.length})</h3><ul>${S.map(e=>`<li><b>${esc(nm(e.from))}</b> ${esc(e.text)}</li>`).join("")||"<li>없음</li>"}</ul>
   <h3>이 축이 한계인 논문 (${Lm.length})</h3><ul>${Lm.map(e=>`<li><b>${esc(nm(e.from))}</b> ${esc(e.text)}<br><span class="src">${esc(e.source_ko)}${e.by?" — "+esc(e.by):""}</span></li>`).join("")||"<li>없음</li>"}</ul>
   ${(()=>{const F=ins.filter(e=>e.type==="finding").map(e=>nodes.get(e.from)).filter(Boolean); return F.length?`<h3>내 연구 기록 (${F.length})</h3><ul>${F.map(n=>`<li><b>${esc(n.label)}</b> [${esc(n.detail.fkind)}] ${esc(n.detail.claim)}</li>`).join("")}</ul>`:"";})()}
   ${S.length&&Lm.length?"<p><b>접목 후보:</b> 위 강점 논문의 장치로 아래 한계를 메울 수 있는지 볼 만해요.</p>":(Lm.length&&!S.length?"<p><b>빈칸:</b> 하네스 안에서 이 한계를 해결한 논문이 아직 없어요.</p>":"")}`;
}
net.on("click", ev => {
  if(!ev.nodes.length) return;
  const n = nodes.get(ev.nodes[0]); const d = n.detail||{}; let h="";
  if(d.kind==="paper") h = paperHTML(d);
  else if(d.kind==="tag") h = tagHTML(d);
  else if(d.kind==="finding") h = `<h2>${esc(d.id)}</h2><div class="src">${esc(d.project)} · ${esc(d.fkind)}</div><h3>가설·주장</h3><div>${esc(d.claim)}</div>
     ${d.observation?`<h3>관찰</h3><div>${esc(d.observation)}</div>`:""}${d.verdict?`<h3>판정</h3><div>${esc(d.verdict)}</div>`:""}
     ${d.closed_scope?`<h3>닫은 범위</h3><div>${esc(d.closed_scope)}</div>`:""}${d.not_closed?`<h3>닫지 않은 것</h3><div>${esc(d.not_closed)}</div>`:""}
     ${d.do_not_claim?`<h3>확대 금지</h3><div>${esc(d.do_not_claim)}</div>`:""}<div>${tg(d.tags)}</div>`;
  else if(d.kind==="idea") h = `<h2>💡 ${esc(d.id)}</h2><div>${esc(d.text)}</div><div>${tg(d.tags)}</div>`;
  else h = `<h2>${esc(d.id)}</h2><div class="src">하네스 밖의 모델 (builds_on 대상)</div>`;
  document.getElementById("info").innerHTML = h;
});
document.getElementById("q").addEventListener("keydown", e => {
  if(e.key!=="Enter") return; const q = e.target.value.trim().toLowerCase(); if(!q) return;
  const hit = nodes.get().find(n => n.label.toLowerCase().includes(q));
  if(hit){ net.selectNodes([hit.id]); net.focus(hit.id,{scale:1.3,animation:true}); net.body.emitter.emit("click",{nodes:[hit.id]}); }
});
</script></body></html>"""


def write_graph(gdata, path: Path):
    js = VIS_JS.read_text(encoding="utf-8")
    page = (GRAPH_HTML
            .replace("__VIS__", js.replace("</script>", "<\\/script>"))
            .replace("__DATA__", json.dumps(gdata, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__STAMP__", html.escape(f"생성 {dt.datetime.now():%Y-%m-%d %H:%M}")))
    path.write_text(page, encoding="utf-8")


def write_candidates_md(cands, tagrows, path: Path):
    L = [f"# 접목 후보 ({dt.date.today()})\n",
         "A의 한계 하나와 B의 강점이 같은 태그로 걸린 경우예요. 점수는 한계 출처 신뢰도(후속 논문 1.0, 리뷰 0.8, "
         "저자 0.6, 조사자 0.4) × 겹친 태그의 희소성이고, 분야가 다르면 0.3을 더해요. 여러 논문에 흔한 태그일수록 가중치가 낮아요.",
         "태그 매칭은 후보를 제안할 뿐이에요. B의 장치가 A의 한계를 실제로 메우는지는 직접 판단해야 합니다.\n"]
    for i, cd in enumerate(cands[:25], 1):
        flag = " · 분야 교차" if cd["cross_area"] else ""
        src = SOURCE_KO.get(cd["source"], cd["source"]) + (f" — {cd['by']}" if cd["by"] else "")
        L.append(f"## {i}. {cd['target']} ← {cd['donor']} [{', '.join(cd['tags'])}] (점수 {cd['score']}{flag})")
        L.append(f"- **{cd['target']} 한계** ({src}): {cd['limitation']}")
        for s_ in cd["strengths"]:
            L.append(f"- **{cd['donor']} 강점**: {s_}")
        L.append("")
    blanks = [r for r in tagrows if r["status"].startswith("빈칸")]
    L.append("# 빈칸: 한계로만 지적되고, 하네스 안에 해결한 논문이 없는 축\n")
    for r in blanks:
        L.append(f"- **{r['tag']}** ({r['desc']}): {', '.join(r['limited'])}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description="연구정리 하네스")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("validate")
    sub.add_parser("tags")
    sub.add_parser("link", help="cited 한계의 by 제목이 하네스 논문과 일치하면 by_id를 YAML에 기입")
    ck = sub.add_parser("check")
    ck.add_argument("--tags", default="")
    ck.add_argument("--keywords", default="")
    ck.add_argument("--text", default="")
    ck.add_argument("--save", default="", help="ideas/<이름>.yaml 로 저장하고 out/checks/<이름>.md 작성")
    a = ap.parse_args(argv)

    areas, tags = load_taxonomy()
    papers, ideas, findings = load_dir(PAPERS_DIR), load_dir(IDEAS_DIR), load_findings()
    warns = validate(papers, ideas, areas, tags, findings)

    if a.cmd == "validate":
        print("\n".join(warns) or "문제 없음")
        return 1 if warns else 0

    if a.cmd == "link":
        tidx, n = title_index(papers), 0
        for p in papers:
            path = PAPERS_DIR / p["_file"]
            lines = path.read_text(encoding="utf-8").split("\n")
            out, i = [], 0
            lims = [l for l in as_list(p.get("limitations")) if l.get("source") == "cited" and not l.get("by_id")]
            want = {str(l.get("by")): resolve_by_id(l, tidx) for l in lims}
            for ln in lines:
                out.append(ln)
                m = re.match(r"^(\s+)by:\s*(.+)$", ln)
                if m:
                    val = m.group(2).strip().strip('"').strip("'")
                    pid = want.get(val)
                    if pid and pid != p["id"]:
                        out.append(f"{m.group(1)}by_id: {pid}")
                        n += 1
            if out != lines:
                path.write_text("\n".join(out), encoding="utf-8")
        print(f"by_id {n}개 기입")
        return 0

    if a.cmd == "tags":
        for r in tag_table(papers, tags):
            print(f"{r['tag']:<22} {r['status']:<16} 한계:{','.join(r['limited']) or '-'}  강점:{','.join(r['addressed']) or '-'}")
        return 0

    if a.cmd == "check":
        tq = [t.strip() for t in a.tags.split(",") if t.strip()]
        kw = [k.strip() for k in a.keywords.split(",") if k.strip()]
        unknown = [t for t in tq if t not in tags]
        if unknown:
            print(f"경고: 미등록 태그 {unknown}", file=sys.stderr)
        report = check_idea(papers, tq, kw, a.text, findings)
        print(report)
        if a.save:
            name = re.sub(r"[^\w\-]+", "_", a.save)
            IDEAS_DIR.mkdir(exist_ok=True)
            (IDEAS_DIR / f"{name}.yaml").write_text(yaml.safe_dump(
                {"id": name, "date": str(dt.date.today()), "text": a.text, "tags": tq, "keywords": kw},
                allow_unicode=True, sort_keys=False), encoding="utf-8")
            (OUT_DIR / "checks").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "checks" / f"{name}.md").write_text(report, encoding="utf-8")
            print(f"\n저장: ideas/{name}.yaml, out/checks/{name}.md (다음 build 때 그래프·엑셀에 반영)")
        return 0

    # build
    if warns:
        print("경고:\n  " + "\n  ".join(warns))
    cands = combination_candidates(papers)
    tagrows = tag_table(papers, tags)
    OUT_DIR.mkdir(exist_ok=True)
    write_excel(papers, ideas, areas, tags, cands, tagrows, OUT_DIR / "research_harness.xlsx", findings)
    write_graph(graph_data(papers, ideas, areas, tags, findings), OUT_DIR / "graph.html")
    write_candidates_md(cands, tagrows, OUT_DIR / "candidates.md")
    keys = bib_keys(papers)
    (OUT_DIR / "refs.bib").write_text("\n\n".join(b for b in (bibtex(p, keys[p["id"]]) for p in papers) if b) + "\n", encoding="utf-8")
    print(f"논문 {len(papers)}편, 내 연구 기록 {len(findings)}건, 아이디어 {len(ideas)}개, 접목 후보 {len(cands)}건")
    print("→ out/research_harness.xlsx, out/graph.html, out/candidates.md, out/refs.bib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
