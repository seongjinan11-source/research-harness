"""Retrieve paper strengths worth checking against recorded blockers.

Lexical retrieval only: a match never changes a failure verdict or proves that
a paper solves a blocker. Benchmarks and surveys are not treatment methods.
"""
from __future__ import annotations

REASON_KO = {"confirmed": "직접 확인", "supported": "진단이 지지", "hypothesis": "미검증 가설", "unknown": "원인 미상"}
GENERIC_TERMS = {"amortized", "inference", "bayesian", "posterior", "predictive",
    "conditional", "decoder", "role", "joint", "evidence", "auxiliary", "potential",
    "likelihood", "adaptation", "context", "graph", "feature", "calibration"}


def seq(value):
    return value if isinstance(value, list) else ([value] if value else [])


def source_links(entry):
    return "; ".join(f"[{ref}](../{ref})" for ref in seq(entry.get("source_refs")))


def revisit_candidates(papers, findings, paper_id=None):
    candidates = []
    for entry in findings:
        if (entry.get("kind") != "failure" or not entry.get("reopen_when")
                or entry.get("revisit_status") == "superseded"):
            continue
        wanted = set(seq(entry.get("reopen_tags")))
        keywords = seq(entry.get("reopen_keywords"))
        for paper in papers:
            if paper_id and paper.get("id") != paper_id:
                continue
            if {"benchmark", "survey"} & set(seq(paper.get("areas"))):
                continue
            matches = []
            for strength in seq(paper.get("strengths")):
                overlap = sorted(wanted & set(seq(strength.get("tags"))))
                # Only positive method descriptions, never limitations or a cited criticism.
                text = " ".join(str(paper.get(k, "")) for k in ("title", "core_idea"))
                text += " " + strength.get("text", "")
                hits = sorted({word for word in keywords if str(word).casefold() in text.casefold()})
                specific = [word for word in hits if str(word).casefold() not in GENERIC_TERMS]
                if overlap and (specific or len(overlap) >= 2):
                    matches.append({"strength": strength.get("text", ""), "tags": overlap,
                        "keywords": hits, "specific_keywords": specific})
            if not matches:
                continue
            overlap = sorted({t for match in matches for t in match["tags"]})
            hits = sorted({k for match in matches for k in match["keywords"]})
            specific = sorted({k for match in matches for k in match["specific_keywords"]})
            candidates.append({
                "finding_id": entry["id"], "project": entry.get("project", ""),
                "claim": entry.get("claim", ""), "verdict": entry.get("verdict", ""),
                "reason_status": entry.get("reason_status", "unknown"),
                "failure_reason": entry.get("failure_reason", ""),
                "reopen_when": entry["reopen_when"], "recheck": entry.get("recheck", ""),
                "comparison_floor": entry.get("comparison_floor", ""),
                "source_refs": seq(entry.get("source_refs")),
                "paper_id": paper["id"], "paper_title": paper.get("title", ""),
                "paper_link": (paper.get("links") or {}).get("paper", ""),
                "matches": matches, "tags": overlap, "keywords": hits,
                "retrieval_score": 2 * len(overlap) + min(3, len(specific)),
                "signal": "구체 키워드+태그 일치 (미검증)" if specific else "복수 태그 일치 (약한 단서)",
                "status": "원문·가정·범위 대조 필요; 기존 판정 유지",
            })
    return sorted(candidates, key=lambda r: (-r["retrieval_score"], r["finding_id"], r["paper_id"]))


def revisit_markdown(findings, candidates):
    lines = ["# 실패 병목과 논문 강점: 재검토 단서", "",
        "자동 매칭은 논문 metadata의 강점·태그·키워드 검색이다. 해결·novelty·재개 승인이 아니다. "
        "새 논문 원문에서 필요한 장치와 가정을 확인하고, 아래 최소 재검증을 통과해야 한다. "
        "기존 논문도 포함하므로 목록에 나타났다는 사실이 새 해결책의 출현을 뜻하지 않는다.", "",
        "원인 수준: 직접 확인 / 진단이 지지 / 미검증 가설 / 원인 미상. "
        "직접 확인은 적힌 병목·대조를 확인했다는 뜻이며 모든 인과 원인을 밝혔다는 뜻은 아니다.", ""]
    by_finding = {}
    for item in candidates:
        by_finding.setdefault(item["finding_id"], []).append(item)
    for entry in findings:
        if entry.get("kind") != "failure":
            continue
        lines += [f"## {entry['id']} — {entry.get('claim', '')}", "",
            f"- 판정: {entry.get('verdict', '')}",
            f"- 원인·병목 [{REASON_KO.get(entry.get('reason_status'), '미기록')}]: {entry.get('failure_reason', '미기록')}",
            f"- 반증·배제한 설명: {'; '.join(seq(entry.get('excluded_explanations'))) or '확정된 배제 없음'}",
            f"- 재검토 조건: {entry.get('reopen_when', '추가 명세 필요')}",
            f"- 먼저 이겨야 할 기준선: {entry.get('comparison_floor', '추가 명세 필요')}",
            f"- 최소 재검증: {entry.get('recheck', '추가 명세 필요')}",
            f"- 닫지 않은 것: {entry.get('not_closed', '')}",
            f"- 원문: {source_links(entry) or entry.get('source', '')}", ""]
        matches = by_finding.get(entry["id"], [])
        if not matches:
            lines += ["현재 catalog에서 장치 단서가 잡히지 않았다. 관련 문헌의 부재나 해결 불가능을 뜻하지 않는다.", ""]
        for item in matches:
            label = f"[{item['paper_id']}]({item['paper_link']})" if item["paper_link"] else item["paper_id"]
            lines += [f"- {label}: {item['signal']}; tags={', '.join(item['tags'])}; "
                f"keywords={', '.join(item['keywords']) or '-'}"]
            for match in item["matches"]:
                lines.append("  - 기록된 강점: " + match["strength"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def research_log_markdown(findings):
    lines = ["# GraphPFN 연구 기록", "",
        "실패·중단, 보존한 효과, 교훈, 미실행을 구분한 누적 기록이다. "
        "숫자는 보존한 원문 보고서에서 옮겼고, 원인 미상인 결과에 사후 인과 설명을 붙이지 않는다. "
        "새 논문과의 연결은 revisit.md의 재검토 단서에서 확인한다.", ""]
    projects = {}
    for entry in findings:
        projects.setdefault(entry.get("project", ""), []).append(entry)
    kinds = {"failure": "실패·중단", "preserved": "보존", "lesson": "교훈", "untested": "미실행"}
    lines += [f"총 {len(projects)}개 프로젝트 · {len(findings)}개 기록 (" +
        ", ".join(f"{label} {sum(e['kind'] == kind for e in findings)}개" for kind, label in kinds.items()) + ")", ""]
    for project, entries in projects.items():
        lines += ["## " + project, ""]
        if entries[0].get("setting"):
            lines += ["실험 조건: " + entries[0]["setting"], ""]
        for entry in entries:
            lines += [f"### {entry['id']} [{kinds[entry['kind']]}] {entry.get('claim', '')}", "",
                entry.get("observation", ""), ""]
            if entry["kind"] == "failure":
                lines += [f"- 판정: {entry.get('verdict', '')}",
                    f"- 원인·병목 [{REASON_KO.get(entry.get('reason_status'), '미기록')}]: {entry.get('failure_reason', '')}",
                    f"- 닫은 범위: {entry.get('closed_scope', '')}",
                    f"- 닫지 않은 것: {entry.get('not_closed', '')}",
                    f"- 재검토 조건: {entry.get('reopen_when', '')}",
                    f"- 기준선: {entry.get('comparison_floor', '')}",
                    f"- 최소 재검증: {entry.get('recheck', '')}"]
            elif entry.get("do_not_claim"):
                lines.append("- 확대 금지: " + entry["do_not_claim"])
            lines += ["- 원문: " + source_links(entry), ""]
    return "\n".join(lines).rstrip() + "\n"
