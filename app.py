"""ProgramPilot - Multi-Agent Risk War-Room for AI Program Delivery.

Gradio Blocks UI. Paste or upload an AI program charter / PRD, run the five
specialist reviewer agents, and inspect the scored RAID log, risk heatmap,
executive summary, and dissent notes.
"""
from __future__ import annotations

import os

import gradio as gr
import pandas as pd
from pypdf import PdfReader

from agents import run_review
from risk_engine import (
    build_heatmap,
    build_raid,
    detect_dissent,
    executive_summary,
    heatmap_html,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "demo_data", "sample_ai_program_charter.md")

MODE_BANNERS = {
    "demo": (
        "**MODE: DEMO** - deterministic rule-based detectors, fully offline. "
        "No network calls are made; findings come from regex/keyword scanners."
    ),
    "live": (
        "**MODE: LIVE** - LLM reviewer agents via OpenAI-compatible chat "
        "completions (env: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL). "
        "Findings are pydantic-validated; invalid JSON is retried once, then dropped."
    ),
}


def extract_text_from_file(path: str) -> str:
    """Extract text from .md/.txt/.pdf uploads using pypdf for PDFs."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def run_pipeline(doc_text: str, mode: str):
    """Full review pipeline -> (exec summary, heatmap HTML, RAID df, dissent)."""
    if not doc_text or not doc_text.strip():
        empty = pd.DataFrame(
            columns=[
                "ID", "Category", "Risk", "Evidence quote", "Severity",
                "Likelihood", "Score", "Owner suggestion", "Recommendation",
            ]
        )
        return (
            "No document provided. Paste text or upload a .md/.txt/.pdf file.",
            "<p style='color:#6c757d;'>No document provided.</p>",
            empty,
            "",
        )
    try:
        findings, warnings_list = run_review(doc_text, mode)
    except RuntimeError as exc:
        return (
            f"Review failed: {exc}",
            "<p style='color:#dc3545;'>Review failed.</p>",
            pd.DataFrame(),
            "",
        )
    raid = build_raid(findings)
    matrix = build_heatmap(findings)
    dissent = detect_dissent(findings, mode)
    summary = executive_summary(findings, raid, dissent)
    if warnings_list:
        summary += "\n\nAgent warnings:\n" + "\n".join(f"- {w}" for w in warnings_list)
    raid_df = pd.DataFrame(raid)
    if raid_df.empty:
        raid_df = pd.DataFrame(
            columns=[
                "ID", "Category", "Risk", "Evidence quote", "Severity",
                "Likelihood", "Score", "Owner suggestion", "Recommendation",
            ]
        )
    return (
        summary,
        heatmap_html(matrix),
        raid_df,
        "\n\n".join(dissent) if dissent else "No dissent notes: reviewers agree on severities.",
    )


def on_run(doc_text: str, mode: str):
    banner = MODE_BANNERS[mode]
    summary, heat, df, dissent = run_pipeline(doc_text, mode)
    return summary, heat, df, dissent, banner


def on_run_sample():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
        doc = fh.read()
    mode = "demo"
    banner = MODE_BANNERS[mode]
    summary, heat, df, dissent = run_pipeline(doc, mode)
    return doc, summary, heat, df, dissent, banner


def on_upload(file) -> str:
    if file is None:
        return ""
    return extract_text_from_file(file)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ProgramPilot Risk War-Room") as demo:
        banner = gr.Markdown(MODE_BANNERS["demo"])
        gr.Markdown(
            "# ProgramPilot - Multi-Agent Risk War-Room\n"
            "Five specialist reviewers (Scope & Outcomes, Data & Integration, "
            "Model & Evaluation, Compliance & Safety, Delivery & Schedule) scan "
            "your AI program charter, a deterministic risk engine scores "
            "severity x likelihood into a RAID log and heatmap, and dissent "
            "notes surface where reviewers disagree."
        )

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=["demo", "live"],
                    value="demo",
                    label="Review mode",
                    info="demo = offline detectors; live = LLM agents (needs OPENAI_API_KEY)",
                )
                file_input = gr.File(
                    label="Upload charter (.md / .txt / .pdf)",
                    file_types=[".md", ".txt", ".pdf"],
                    type="filepath",
                )
                doc_input = gr.Textbox(
                    label="Program charter / PRD text",
                    lines=16,
                    placeholder="Paste your AI program charter here...",
                )
                with gr.Row():
                    run_btn = gr.Button("Run review", variant="primary")
                    sample_btn = gr.Button("Run full demo on sample charter")

            with gr.Column(scale=2):
                exec_summary = gr.Textbox(
                    label="Executive summary", lines=10, interactive=False
                )
                heatmap = gr.HTML(label="Risk heatmap (severity x likelihood)")
                gr.Markdown("### RAID log")
                raid_table = gr.DataFrame(
                    headers=[
                        "ID", "Category", "Risk", "Evidence quote", "Severity",
                        "Likelihood", "Score", "Owner suggestion", "Recommendation",
                    ],
                    wrap=True,
                    interactive=False,
                )
                dissent_notes = gr.Textbox(
                    label="Dissent notes", lines=6, interactive=False
                )

        file_input.change(
            fn=on_upload, inputs=file_input, outputs=doc_input
        )
        mode.change(
            fn=lambda m: MODE_BANNERS[m], inputs=mode, outputs=banner
        )
        run_btn.click(
            fn=on_run,
            inputs=[doc_input, mode],
            outputs=[exec_summary, heatmap, raid_table, dissent_notes, banner],
        )
        sample_btn.click(
            fn=on_run_sample,
            inputs=[],
            outputs=[doc_input, exec_summary, heatmap, raid_table, dissent_notes, banner],
        )
    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7863")),
    )
