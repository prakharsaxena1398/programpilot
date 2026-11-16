# ProgramPilot - Multi-Agent Risk War-Room for AI Program Delivery

ProgramPilot reviews AI program charters and PRDs with five specialist
reviewer agents, scores the findings with a deterministic risk engine, and
produces a RAID log, a severity x likelihood risk heatmap, and dissent notes
where reviewers disagree.

## The problem it solves

AI programs fail on risks nobody flagged in the review meeting: no evaluation
plan, PII leaking into prompts, vendor lock-in, unrealistic timelines.
ProgramPilot surfaces those risks before the steering meeting, with evidence
quoted verbatim from the document itself.

## Agent architecture

Five specialist reviewers scan the charter, each through a distinct lens:

| Agent | Lens | Category |
| --- | --- | --- |
| Scope & Outcomes Reviewer | measurable goals, success criteria, scope creep | Scope & Outcomes |
| Data & Integration Reviewer | data readiness, integrations, vendor dependence | Data & Integration |
| Model & Evaluation Reviewer | fine-tuning decisions, evaluation plans, monitoring | Model & Evaluation |
| Compliance & Safety Reviewer | PII handling, privacy, autonomous behavior | Compliance & Safety |
| Delivery & Schedule Reviewer | timelines, phasing, delivery realism | Delivery & Schedule |

Each agent returns strict JSON validated by pydantic:
`{agent, findings: [{title, category, severity, likelihood, evidence_quote,
recommendation}]}`. `evidence_quote` must be a verbatim span of the document;
findings whose quotes do not verify are dropped with a warning.

The deterministic risk engine then:

- computes `score = severity_points x likelihood_points` (low=1, medium=2, high=3),
- assembles the RAID log with sequential IDs (R-01, R-02, ...) ordered by score,
  with owner suggestions by category (Scope -> Product, Data -> Data Engineering,
  Model -> ML Lead, Compliance -> Legal & Safety, Delivery -> PMO),
- builds the severity x likelihood heatmap with counts,
- surfaces dissent notes where the same category carries different severities.

## Demo mode vs live mode

- **DEMO (default, fully offline)**: five rule-based detectors mirror the agent
  lenses with regex/keyword scanners (fine-tuning without an evaluation plan,
  PII without masking, compressed week-based timelines, single vendor without
  an exit clause, autonomous agents without human oversight, vague success
  metrics, pilots without exit criteria, integrations without fallback).
  Evidence quotes are the actual matched sentences. No network access needed.
- **LIVE**: each agent is called via OpenAI-compatible chat completions
  (`POST {OPENAI_BASE_URL}/chat/completions`, temperature 0.2). Set
  `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL` (default
  `https://api.openai.com/v1`) and `LLM_MODEL` (default `gpt-4o-mini`).
  Invalid JSON is retried once, then dropped with a warning.

The mode banner in the UI always shows which mode produced the current output.

## Quickstart

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:7863
```

Click "Run full demo on sample charter" to run the offline pipeline on the
bundled borrower-assistant charter (it triggers 8 findings, several high
severity, plus dissent notes). Or paste/upload your own charter (.md, .txt,
.pdf) and click "Run review".

For live mode:

```bash
export OPENAI_API_KEY=sk-...            # required
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
export LLM_MODEL=gpt-4o-mini            # optional
python app.py   # switch the radio to "live" in the UI
```

## Deploy to Hugging Face Spaces

1. Create a new Space with SDK "Gradio".
2. Upload `app.py`, `agents.py`, `risk_engine.py`, `requirements.txt`,
   `demo_data/sample_ai_program_charter.md` (keep the folder layout).
3. Add `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `LLM_MODEL`) as
   Space secrets if you want live mode; without a key, demo mode works as-is.
4. The app binds to `0.0.0.0` and honors the `GRADIO_SERVER_PORT` env var, so
   it works unmodified on Spaces.

## Tests

```bash
python -m unittest discover -s tests -v
```

21 tests cover: pydantic Finding accept/reject, every detector firing on a
planted sentence, exception-word suppression, verbatim evidence quotes,
risk score math (high x medium = 6), RAID ID sequencing and uniqueness,
owner mapping, heatmap count invariants, dissent detection, the sample
charter demo run (>= 6 findings, >= 1 high severity), and PDF text
extraction through a generated tiny PDF fixture.

## Honest limitations

- Demo detectors are keyword/regex based: they can miss risks phrased
  unusually and can be fooled by negation that is not in their exception
  lists. They approximate, not replace, LLM judgment.
- Live mode output quality depends entirely on the configured model; strict
  JSON adherence and verbatim quotes are enforced but cannot be guaranteed.
- Dissent detection compares severities within a category; it does not model
  nuanced arguments between reviewers.
- Severity and likelihood defaults in demo mode are fixed per rule, not
  contextual.
- The risk engine is deliberately simple (multiplicative score, fixed owner
  map); it is a war-room aid, not a calibrated risk model.
- PDF extraction relies on pypdf; scanned or image-only PDFs yield no text.
- No persistence: results live only in the current UI session.
