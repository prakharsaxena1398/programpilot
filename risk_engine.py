"""ProgramPilot deterministic risk engine.

Consumes a list of validated findings and produces:
  - a scored, ordered RAID log (pydantic-free plain records),
  - a severity x likelihood heatmap matrix with counts,
  - dissent notes where severity disagreement exists,
  - an executive summary.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from agents import Finding

SEVERITY_POINTS = {"low": 1, "medium": 2, "high": 3}
LIKELIHOOD_POINTS = {"low": 1, "medium": 2, "high": 3}

OWNER_BY_CATEGORY = {
    "Scope & Outcomes": "Product",
    "Data & Integration": "Data Engineering",
    "Model & Evaluation": "ML Lead",
    "Compliance & Safety": "Legal & Safety",
    "Delivery & Schedule": "PMO",
}

SEVERITY_ORDER = ("high", "medium", "low")
LIKELIHOOD_ORDER = ("high", "medium", "low")

# Cell background color by product score (severity x likelihood).
HEAT_COLORS = {
    1: "#d8f3dc", 2: "#e2f0cb", 3: "#fff3bf",
    4: "#ffe8a3", 6: "#ffd6a5", 9: "#ffadad",
}


def risk_score(finding: Finding) -> int:
    """score = severity_points x likelihood_points (low=1, medium=2, high=3)."""
    return SEVERITY_POINTS[finding.severity] * LIKELIHOOD_POINTS[finding.likelihood]


def build_raid(findings: List[Finding]) -> List[dict]:
    """Assemble the RAID log, highest score first, IDs R-01, R-02, ..."""
    ordered = sorted(
        findings,
        key=lambda f: (-risk_score(f), f.category, f.title),
    )
    raid: List[dict] = []
    for i, f in enumerate(ordered, start=1):
        raid.append(
            {
                "ID": f"R-{i:02d}",
                "Category": f.category,
                "Risk": f.title,
                "Evidence quote": f.evidence_quote,
                "Severity": f.severity,
                "Likelihood": f.likelihood,
                "Score": risk_score(f),
                "Owner suggestion": OWNER_BY_CATEGORY.get(f.category, "Program Lead"),
                "Recommendation": f.recommendation,
            }
        )
    return raid


def build_heatmap(findings: List[Finding]) -> Dict[tuple, List[str]]:
    """Matrix counts: {(severity, likelihood): [RAID ids]}."""
    raid = build_raid(findings)
    id_lookup = {(r["Category"], r["Risk"]): r["ID"] for r in raid}
    matrix: Dict[tuple, List[str]] = defaultdict(list)
    for f in findings:
        key = (f.severity, f.likelihood)
        matrix[key].append(id_lookup.get((f.category, f.title), "?"))
    return dict(matrix)


def heatmap_counts(matrix: Dict[tuple, List[str]]) -> Dict[tuple, int]:
    return {k: len(v) for k, v in matrix.items()}


def heatmap_html(matrix: Dict[tuple, List[str]]) -> str:
    """Render a color-coded severity x likelihood HTML table."""
    counts = heatmap_counts(matrix)
    total = sum(counts.values())

    def cell_style(score: int) -> str:
        color = HEAT_COLORS.get(score, "#e9ecef")
        return f"background:{color};padding:10px;text-align:center;border:1px solid #ced4da;"

    rows = []
    header_cells = "".join(
        f'<th style="padding:8px;border:1px solid #ced4da;background:#e9ecef;">'
        f"Likelihood: {lik.capitalize()}</th>"
        for lik in LIKELIHOOD_ORDER
    )
    rows.append(
        f'<tr><th style="padding:8px;border:1px solid #ced4da;background:#e9ecef;">'
        f"Severity</th>{header_cells}</tr>"
    )
    for sev in SEVERITY_ORDER:
        cells = []
        for lik in LIKELIHOOD_ORDER:
            score = SEVERITY_POINTS[sev] * LIKELIHOOD_POINTS[lik]
            count = counts.get((sev, lik), 0)
            ids = matrix.get((sev, lik), [])
            label = f"{count}" + (f"<br><small>{', '.join(ids[:4])}</small>" if ids else "")
            cells.append(f'<td style="{cell_style(score)}">{label}</td>')
        rows.append(
            f'<tr><th style="padding:8px;border:1px solid #ced4da;background:#e9ecef;">'
            f"{sev.capitalize()} ({SEVERITY_POINTS[sev]})</th>{''.join(cells)}</tr>"
        )
    return (
        '<div style="font-family:sans-serif;">'
        f'<p style="margin:4px 0 8px 0;color:#495057;">Risks plotted: {total} '
        "(severity x likelihood; score = severity points x likelihood points)</p>"
        f'<table style="border-collapse:collapse;">{"".join(rows)}</table></div>'
    )


def detect_dissent(findings: List[Finding], mode: str = "demo") -> List[str]:
    """Surface disagreement notes.

    A dissent is recorded when two or more findings in the same category
    disagree on severity. LIVE mode: this is genuine reviewer disagreement
    between agents. DEMO mode: the dissent is synthesized only when the
    document genuinely triggers two detectors in the same category whose
    severities differ.
    """
    by_category: Dict[str, dict] = defaultdict(dict)
    for f in findings:
        by_category[f.category][(f.category, f.title, f.severity)] = f

    notes: List[str] = []
    for category, members in by_category.items():
        severities = {f.severity for f in members.values()}
        if len(members) < 2 or len(severities) < 2:
            continue
        descriptions = "; ".join(
            f"'{f.title}' is {f.severity}" for f in members.values()
        )
        origin = (
            "reviewer agents disagree"
            if mode == "live"
            else "two independent detectors in this lens disagree"
        )
        notes.append(
            f"DISSENT [{category}]: severity disagreement ({origin}). "
            f"{descriptions}. The war-room should reconcile whether the weaker "
            "severity is justified before the risk is accepted."
        )
        if len(notes) >= 2:
            break
    return notes


def executive_summary(findings: List[Finding], raid: List[dict], dissent: List[str]) -> str:
    """Deterministic plain-text executive summary."""
    if not findings:
        return (
            "No findings were produced. Either the document is unusually clean or "
            "it was too short to review. Paste a fuller program charter and re-run."
        )
    high = [f for f in findings if f.severity == "high"]
    top = raid[:3]
    category_counts: Dict[str, int] = defaultdict(int)
    for f in findings:
        category_counts[f.category] += 1
    dominant = max(category_counts.items(), key=lambda kv: kv[1])

    overall = "HIGH" if len(high) >= 3 else ("ELEVATED" if high else "MODERATE")
    lines = [
        f"ProgramPilot review complete: {len(findings)} risks logged "
        f"({len(high)} high severity). Overall risk posture: {overall}.",
        "",
        f"Most exposed area: {dominant[0]} ({dominant[1]} finding(s)).",
        "",
        "Top risks by score:",
    ]
    for r in top:
        lines.append(
            f"  {r['ID']} [{r['Score']}] {r['Risk']} "
            f"(owner: {r['Owner suggestion']})"
        )
    if dissent:
        lines.append("")
        lines.append(
            f"{len(dissent)} dissent note(s) recorded - reviewers disagree on "
            "severity for at least one area; see the Dissent notes panel."
        )
    lines.append("")
    lines.append(
        "Recommended next step: walk the RAID log with the suggested owners and "
        "agree mitigations before the next steering meeting."
    )
    return "\n".join(lines)
