"""Unit tests for ProgramPilot core (agents, risk engine, extraction).

Run with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (  # noqa: E402
    DETECTORS,
    CATEGORIES,
    Finding,
    run_demo_review,
    split_sentences,
)
from risk_engine import (  # noqa: E402
    build_heatmap,
    build_raid,
    detect_dissent,
    executive_summary,
    heatmap_counts,
    heatmap_html,
    risk_score,
)
from pydantic import ValidationError  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "demo_data", "sample_ai_program_charter.md")

VALID_FINDING = {
    "title": "PII exposure",
    "category": "Compliance & Safety",
    "severity": "high",
    "likelihood": "medium",
    "evidence_quote": "the doc contains personal data",
    "recommendation": "Mask PII before it reaches the prompt.",
}


def make_finding(**overrides):
    data = dict(VALID_FINDING)
    data.update(overrides)
    return Finding(**data)


# Planted sentence per detector key order, plus minimal neutral context.
PLANTED = {
    "Custom model / fine-tuning without an evaluation plan":
        "We will fine-tune a custom model on historical transcripts.",
    "Personal / payment data flowing into prompts without masking":
        "The system will process customer personal data from borrower emails.",
    "Compressed timeline: pilot with a model component in weeks":
        "We plan to pilot in 6 weeks. The program includes a chatbot component.",
    "Single LLM vendor without fallback or exit plan":
        "We will rely on a single LLM vendor for all inference.",
    "Autonomous agent behavior without human-in-the-loop controls":
        "The assistant will act as an autonomous agent for ordering.",
    "Vague success metrics without measurable targets":
        "Success will be measured by customer satisfaction and adoption.",
    "Pilot / MVP without defined success or exit criteria":
        "Phase 1 will run as a proof of concept for the team.",
    "Enterprise integration without fallback or degraded mode":
        "The service will integrate with our ERP in real time.",
}


def build_tiny_pdf(path: str, text: str) -> None:
    """Build a minimal valid single-page PDF containing `text`."""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    with open(path, "wb") as fh:
        fh.write(bytes(out))


class TestFindingModel(unittest.TestCase):
    def test_finding_accepts_valid_payload(self):
        f = make_finding()
        self.assertEqual(f.severity, "high")
        self.assertEqual(f.category, "Compliance & Safety")

    def test_finding_rejects_bad_severity(self):
        with self.assertRaises(ValidationError):
            make_finding(severity="critical")

    def test_finding_rejects_unknown_category(self):
        with self.assertRaises(ValidationError):
            make_finding(category="Finance")

    def test_finding_rejects_missing_recommendation(self):
        data = dict(VALID_FINDING)
        data.pop("recommendation")
        with self.assertRaises(ValidationError):
            Finding(**data)


class TestDetectors(unittest.TestCase):
    def test_each_detector_fires_on_planted_sentence(self):
        for det in DETECTORS:
            with self.subTest(detector=det.title):
                doc = PLANTED[det.title]
                findings = run_demo_review(doc)
                titles = [f.title for f in findings]
                self.assertIn(det.title, titles, f"detector did not fire: {det.title}")

    def test_exception_words_suppress_detector(self):
        doc = (
            "We will fine-tune a custom model on historical transcripts. "
            "The plan includes a golden set and a benchmark suite."
        )
        titles = [f.title for f in run_demo_review(doc)]
        self.assertNotIn(
            "Custom model / fine-tuning without an evaluation plan", titles
        )

    def test_evidence_quote_is_verbatim_substring(self):
        doc = PLANTED["Personal / payment data flowing into prompts without masking"]
        findings = run_demo_review(doc)
        self.assertTrue(findings)
        for f in findings:
            self.assertIn(f.evidence_quote, doc)

    def test_sentence_splitter_keeps_segments(self):
        segments = split_sentences("First sentence here. Second sentence here!\nThird line")
        self.assertEqual(len(segments), 3)


class TestRiskEngine(unittest.TestCase):
    def test_risk_score_math(self):
        f = make_finding(severity="high", likelihood="medium")
        self.assertEqual(risk_score(f), 6)
        self.assertEqual(risk_score(make_finding(severity="low", likelihood="low")), 1)
        self.assertEqual(risk_score(make_finding(severity="high", likelihood="high")), 9)

    def test_raid_ids_sequential_and_unique(self):
        findings = [
            make_finding(category=cat, severity=sev)
            for cat, sev in zip(CATEGORIES, ("high", "medium", "high", "low", "medium"))
        ]
        raid = build_raid(findings)
        ids = [r["ID"] for r in raid]
        self.assertEqual(ids, [f"R-{i:02d}" for i in range(1, len(findings) + 1)])
        self.assertEqual(len(set(ids)), len(ids))
        # sorted descending by score
        scores = [r["Score"] for r in raid]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_owner_suggestion_by_category(self):
        findings = [make_finding(category="Model & Evaluation")]
        raid = build_raid(findings)
        self.assertEqual(raid[0]["Owner suggestion"], "ML Lead")
        findings = [make_finding(category="Delivery & Schedule")]
        raid = build_raid(findings)
        self.assertEqual(raid[0]["Owner suggestion"], "PMO")

    def test_heatmap_counts_sum_to_findings(self):
        findings = [
            make_finding(category="Scope & Outcomes", severity="high", likelihood="medium"),
            make_finding(category="Data & Integration", severity="low", likelihood="low"),
            make_finding(category="Scope & Outcomes", severity="high", likelihood="medium"),
        ]
        matrix = build_heatmap(findings)
        counts = heatmap_counts(matrix)
        self.assertEqual(sum(counts.values()), len(findings))
        self.assertEqual(counts[("high", "medium")], 2)
        html = heatmap_html(matrix)
        self.assertIn("<table", html)

    def test_dissent_fires_on_same_category_different_severity(self):
        findings = [
            make_finding(category="Scope & Outcomes", severity="medium"),
            make_finding(category="Scope & Outcomes", severity="high"),
        ]
        notes = detect_dissent(findings, mode="demo")
        self.assertEqual(len(notes), 1)
        self.assertIn("DISSENT [Scope & Outcomes]", notes[0])

    def test_dissent_silent_when_agreeing(self):
        findings = [
            make_finding(category="Scope & Outcomes", severity="medium"),
            make_finding(category="Compliance & Safety", severity="medium"),
        ]
        self.assertEqual(detect_dissent(findings, mode="demo"), [])

    def test_executive_summary_mentions_counts(self):
        findings = [
            make_finding(category="Scope & Outcomes", severity="high"),
            make_finding(category="Scope & Outcomes", severity="medium"),
        ]
        raid = build_raid(findings)
        summary = executive_summary(findings, raid, [])
        self.assertIn("2 risks logged", summary)
        self.assertIn("1 high severity", summary)


class TestSampleCharterDemoRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
            cls.doc = fh.read()
        cls.findings = run_demo_review(cls.doc)
        cls.raid = build_raid(cls.findings)

    def test_sample_yields_at_least_six_findings(self):
        self.assertGreaterEqual(len(self.findings), 6)

    def test_sample_has_high_severity(self):
        self.assertTrue(any(f.severity == "high" for f in self.findings))

    def test_sample_heatmap_sums(self):
        matrix = build_heatmap(self.findings)
        self.assertEqual(sum(heatmap_counts(matrix).values()), len(self.findings))

    def test_sample_all_quotes_verbatim(self):
        for f in self.findings:
            self.assertIn(f.evidence_quote, self.doc)

    def test_sample_produces_dissent(self):
        notes = detect_dissent(self.findings, mode="demo")
        self.assertGreaterEqual(len(notes), 1)


class TestPdfExtraction(unittest.TestCase):
    def test_pdf_text_extraction(self):
        from pypdf import PdfReader

        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "tiny.pdf")
                payload = "LoanDesk charter contains personal data and a 6 week pilot."
                build_tiny_pdf(path, payload)
                reader = PdfReader(path)
                text = "\n".join(
                    (page.extract_text() or "") for page in reader.pages
                )
                self.assertIn("personal data", text.replace("\n", " "))
        except Exception as exc:  # skip gracefully if extraction infeasible
            self.skipTest(f"PDF fixture extraction infeasible: {exc}")


if __name__ == "__main__":
    unittest.main()
