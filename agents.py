"""ProgramPilot reviewer agents.

Five specialist reviewer agents scan an AI program charter / PRD and return
structured findings (pydantic-validated).

Modes:
  - LIVE: OpenAI-compatible chat completions via `requests` (env vars
    OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL).
  - DEMO (default, fully offline): deterministic rule-based detectors that
    scan the document with regex/keywords. Evidence quotes are always verbatim
    sentences from the document.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import List, Literal, Optional

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

CATEGORIES = (
    "Scope & Outcomes",
    "Data & Integration",
    "Model & Evaluation",
    "Compliance & Safety",
    "Delivery & Schedule",
)

SEVERITIES = ("low", "medium", "high")


class Finding(BaseModel):
    """A single structured risk finding produced by a reviewer agent."""

    title: str = Field(min_length=3)
    category: str
    severity: Literal["low", "medium", "high"]
    likelihood: Literal["low", "medium", "high"]
    evidence_quote: str = Field(min_length=5)
    recommendation: str = Field(min_length=3)

    @field_validator("category")
    @classmethod
    def category_must_be_known(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(
                f"category '{v}' not one of {CATEGORIES}"
            )
        return v


class AgentReport(BaseModel):
    """Top-level envelope returned by every reviewer agent."""

    agent: str
    findings: List[Finding] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Agent definitions
# --------------------------------------------------------------------------

_JSON_CONTRACT = (
    "Return STRICT JSON only (no prose, no markdown fences) with this exact shape: "
    '{"agent": "<your agent name>", "findings": [{"title": str, "category": one of '
    '["Scope & Outcomes", "Data & Integration", "Model & Evaluation", '
    '"Compliance & Safety", "Delivery & Schedule"], '
    '"severity": "low|medium|high", "likelihood": "low|medium|high", '
    '"evidence_quote": "<verbatim span copied from the document>", '
    '"recommendation": str}]}. '
    "The evidence_quote MUST be a verbatim substring of the document. "
    "If you find no risks, return an empty findings list."
)

AGENTS = [
    {
        "name": "Scope & Outcomes Reviewer",
        "category": "Scope & Outcomes",
        "system_prompt": (
            "You are the Scope & Outcomes reviewer on an AI program risk war-room. "
            "Your lens: business outcomes, success metrics, scope creep, stakeholder "
            "alignment, and whether the program defines measurable, testable goals. "
            "Flag vague metrics, undefined success criteria, unbounded scope, and "
            "missing user acceptance paths. " + _JSON_CONTRACT
        ),
    },
    {
        "name": "Data & Integration Reviewer",
        "category": "Data & Integration",
        "system_prompt": (
            "You are the Data & Integration reviewer on an AI program risk war-room. "
            "Your lens: data availability and quality, retrieval sources, integration "
            "with enterprise systems, latency, vendor dependence and lock-in, and "
            "operational fallback behavior. Flag single-vendor dependence without an "
            "exit plan, fragile integrations, and missing data readiness work. "
            + _JSON_CONTRACT
        ),
    },
    {
        "name": "Model & Evaluation Reviewer",
        "category": "Model & Evaluation",
        "system_prompt": (
            "You are the Model & Evaluation reviewer on an AI program risk war-room. "
            "Your lens: model choice, fine-tuning decisions, evaluation plans, "
            "benchmarks, golden datasets, regression testing, and monitoring. Flag "
            "custom model work with no evaluation plan, no baseline comparison, and "
            "no post-launch quality monitoring. " + _JSON_CONTRACT
        ),
    },
    {
        "name": "Compliance & Safety Reviewer",
        "category": "Compliance & Safety",
        "system_prompt": (
            "You are the Compliance & Safety reviewer on an AI program risk war-room. "
            "Your lens: PII and sensitive data handling, privacy safeguards, regulatory "
            "exposure, autonomous-agent behavior, and human oversight. Flag personal "
            "data flowing into prompts without masking or anonymization, and agentic "
            "behavior without human-in-the-loop controls. " + _JSON_CONTRACT
        ),
    },
    {
        "name": "Delivery & Schedule Reviewer",
        "category": "Delivery & Schedule",
        "system_prompt": (
            "You are the Delivery & Schedule reviewer on an AI program risk war-room. "
            "Your lens: timelines, phasing, team capacity, dependencies, and delivery "
            "realism. Flag compressed schedules for AI work, missing contingency, "
            "pilot-to-production gaps, and undefined exit criteria. " + _JSON_CONTRACT
        ),
    },
]


# --------------------------------------------------------------------------
# LIVE mode (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------

def _llm_config() -> dict:
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    }


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _call_agent(agent: dict, doc: str) -> str:
    cfg = _llm_config()
    payload = {
        "model": cfg["model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": agent["system_prompt"]},
            {
                "role": "user",
                "content": (
                    "Review the following AI program document and report risks.\n\n"
                    "DOCUMENT START\n" + doc + "\nDOCUMENT END"
                ),
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def quote_in_doc(quote: str, doc: str) -> bool:
    """Whitespace-tolerant verbatim containment check."""
    norm_doc = re.sub(r"\s+", " ", doc).strip()
    norm_quote = re.sub(r"\s+", " ", quote).strip()
    return bool(norm_quote) and norm_quote in norm_doc


def _run_live_agent(agent: dict, doc: str) -> Optional[AgentReport]:
    """Call one agent, validate with pydantic, retry once, else drop."""
    last_error = ""
    for attempt in range(2):
        try:
            raw = _call_agent(agent, doc)
            report = AgentReport.model_validate_json(_strip_json_fences(raw))
            # Enforce verbatim evidence quotes; drop findings that fail.
            verified = [f for f in report.findings if quote_in_doc(f.evidence_quote, doc)]
            dropped = len(report.findings) - len(verified)
            if dropped:
                warnings.warn(
                    f"{agent['name']}: dropped {dropped} finding(s) whose "
                    "evidence_quote was not a verbatim span of the document."
                )
            report.findings = verified
            return report
        except (requests.RequestException, ValidationError, KeyError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == 0:
                warnings.warn(f"{agent['name']}: first attempt failed, retrying once.")
    warnings.warn(f"{agent['name']}: dropped after validation failure. Reason: {last_error}")
    return None


def run_live_review(doc: str) -> List[Finding]:
    findings: List[Finding] = []
    for agent in AGENTS:
        report = _run_live_agent(agent, doc)
        if report is not None:
            findings.extend(report.findings)
    return findings


# --------------------------------------------------------------------------
# DEMO mode (deterministic, offline detectors)
# --------------------------------------------------------------------------

class Detector:
    """A rule-based detector standing in for one agent's review lens."""

    def __init__(
        self,
        agent: str,
        category: str,
        title: str,
        severity: str,
        likelihood: str,
        recommendation: str,
        trigger: str,
        exceptions: Optional[str] = None,
        context: Optional[str] = None,
        sentence_exclusions: Optional[str] = None,
    ):
        self.agent = agent
        self.category = category
        self.title = title
        self.severity = severity
        self.likelihood = likelihood
        self.recommendation = recommendation
        self.trigger = re.compile(trigger, re.IGNORECASE)
        self.exceptions = re.compile(exceptions, re.IGNORECASE) if exceptions else None
        self.context = re.compile(context, re.IGNORECASE) if context else None
        # Per-sentence exclusion: the detector skips sentences matching it.
        self.sentence_exclusions = (
            re.compile(sentence_exclusions, re.IGNORECASE) if sentence_exclusions else None
        )


DETECTORS = [
    Detector(
        agent="Model & Evaluation Reviewer",
        category="Model & Evaluation",
        title="Custom model / fine-tuning without an evaluation plan",
        severity="high",
        likelihood="medium",
        recommendation=(
            "Define an evaluation plan before any fine-tuning work: build a golden "
            "test set of at least 100 real queries, set target metrics, and compare "
            "against a strong-prompt baseline on a frontier model."
        ),
        trigger=r"\b(fine[- ]?tun\w*|custom model|proprietary model|train(?:ing|ed)? (?:a |our |the )?(?:custom |proprietary )?model)\b",
        exceptions=r"(evaluat\w*|eval set|benchmark|golden set|test set|regression suite)",
    ),
    Detector(
        agent="Compliance & Safety Reviewer",
        category="Compliance & Safety",
        title="Personal / payment data flowing into prompts without masking",
        severity="high",
        likelihood="high",
        recommendation=(
            "Add a preprocessing layer that masks or pseudonymizes PII before it "
            "reaches the model, restrict retention of prompt logs, and get a privacy "
            "review (DPIA) sign-off before the pilot."
        ),
        trigger=r"\b(PII|personally identifiable|personal data|payment data|credit card|customer e[- ]?mails?|dealer e[- ]?mails?|names (?:and|&) (?:phone|email|address))\b",
        exceptions=r"(anonymi\w*|differential privacy|masking|redact\w*|pseudonymi\w*|tokeni[sz]ation)",
    ),
    Detector(
        agent="Delivery & Schedule Reviewer",
        category="Delivery & Schedule",
        title="Compressed timeline: pilot with a model component in weeks",
        severity="high",
        likelihood="medium",
        recommendation=(
            "Re-baseline the schedule: split the pilot into a 2-week data and "
            "prompt-pipeline phase followed by a measured pilot window, and add "
            "explicit contingency for retrieval-quality rework."
        ),
        trigger=r"\b\d+\s*[- ]?\s*weeks?\b",
        exceptions=r"(buffer|realistic|contingen\w*|phase ?d|stretch goal)",
        context=r"\b(chatbot|assistant|LLM|RAG|model|fine[- ]?tun\w*|agent)\b",
    ),
    Detector(
        agent="Data & Integration Reviewer",
        category="Data & Integration",
        title="Single LLM vendor without fallback or exit plan",
        severity="high",
        likelihood="medium",
        recommendation=(
            "Isolate the vendor behind an internal gateway with a provider-abstraction "
            "layer, keep prompts and evals portable, and negotiate an exit clause plus "
            "a second-source provider for failover."
        ),
        trigger=r"\b(vendor|API provider|LLM provider|single provider|OpenAI|Anthropic|Azure OpenAI|subscription plan)\b",
        exceptions=r"(fallback|exit clause|multi[- ]vendor|portable|abstraction|second source|switch provider)",
    ),
    Detector(
        agent="Compliance & Safety Reviewer",
        category="Compliance & Safety",
        title="Autonomous agent behavior without human-in-the-loop controls",
        severity="high",
        likelihood="medium",
        recommendation=(
            "Constrain the assistant to suggest-only mode for consequential actions "
            "(orders, quotes, warranty claims) and require explicit human approval "
            "before any tool call that changes external state."
        ),
        trigger=r"\b(agents?|agentic|autonomous\w*)\b",
        exceptions=r"(human in the loop|human[- ]in[- ]the[- ]loop|approval|approver|supervision|supervised|escalat\w*|review before)",
    ),
    Detector(
        agent="Scope & Outcomes Reviewer",
        category="Scope & Outcomes",
        title="Vague success metrics without measurable targets",
        severity="medium",
        likelihood="high",
        recommendation=(
            "Replace qualitative goals with numeric targets, e.g. CSAT >= 4.2/5, "
            "fitment-answer accuracy >= 90% on the golden set, and >= 60% weekly "
            "borrower adoption by week 8."
        ),
        trigger=r"\b(accuracy|CSAT|adoption|satisfaction|user satisfaction|success)\b",
        exceptions=None,
        context=None,
        # A sentence containing a number already has a measurable figure.
        sentence_exclusions=r"\d",
    ),
    Detector(
        agent="Scope & Outcomes Reviewer",
        category="Scope & Outcomes",
        title="Pilot / MVP without defined success or exit criteria",
        severity="high",
        likelihood="medium",
        recommendation=(
            "Write down pilot exit criteria before kickoff: minimum accuracy, "
            "minimum weekly active borrowers, escalation rate ceiling, and an explicit "
            "go / no-go decision meeting."
        ),
        trigger=r"\b(pilot|MVP|proof of concept|POC|phase 1)\b",
        exceptions=r"(success criteria|exit criteria|definition of done|acceptance criteria|KPIs?\b|go[ /-]?no[ -]?go)",
    ),
    Detector(
        agent="Data & Integration Reviewer",
        category="Data & Integration",
        title="Enterprise integration without fallback or degraded mode",
        severity="medium",
        likelihood="medium",
        recommendation=(
            "Specify integration failure behavior: request timeouts, retry with "
            "backoff, a cached last-known-good catalog, and a degraded-mode answer "
            "path when the ERP is unreachable."
        ),
        trigger=r"\b(integrat\w* with|ERP|CRM|real[- ]time|inventory system|catalog service)\b",
        exceptions=r"(fallback|timeout|retry|circuit breaker|degraded mode|backup|cache\b)",
    ),
]


def split_sentences(text: str) -> List[str]:
    """Split a document into sentence-like segments.

    Splitting happens only at sentence-terminal punctuation so that hard
    line wraps inside a paragraph never break a sentence in half. Each
    returned segment is an exact verbatim substring of the input.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def run_demo_review(doc: str) -> List[Finding]:
    """Offline deterministic review. Each detector reports at most one finding
    (its first matching sentence), keeping output predictable."""
    findings: List[Finding] = []
    sentences = split_sentences(doc)
    for det in DETECTORS:
        if det.context is not None and not det.context.search(doc):
            continue
        # Exception keywords suppress the detector at document level, e.g. a
        # fine-tuning mention does not fire when an evaluation plan exists
        # anywhere in the document.
        if det.exceptions is not None and det.exceptions.search(doc):
            continue
        for sentence in sentences:
            if not det.trigger.search(sentence):
                continue
            if det.sentence_exclusions is not None and det.sentence_exclusions.search(sentence):
                continue
            findings.append(
                Finding(
                    title=det.title,
                    category=det.category,
                    severity=det.severity,
                    likelihood=det.likelihood,
                    evidence_quote=sentence,
                    recommendation=det.recommendation,
                )
            )
            break  # one finding per detector
    return findings


def run_review(doc: str, mode: str = "demo"):
    """Run all five reviewers in the requested mode.

    Returns (findings, warnings_list).
    """
    captured = []

    def _catch(message, category, *args, **kwargs):
        captured.append(str(message))

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        if mode == "live":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("LIVE mode requires OPENAI_API_KEY to be set.")
            findings = run_live_review(doc)
        elif mode == "demo":
            findings = run_demo_review(doc)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        captured.extend(str(w.message) for w in wlist)
    return findings, captured
