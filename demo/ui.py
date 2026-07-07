"""Local Gradio UI for the verified recommender — PRESENTATION LAYER ONLY.

It wraps the existing Recommender.recommend() (model + RAG + verify layer, all unchanged) and
renders {label, labeled_answer, verification_report, source_chunk_ids}. No API calls — local
Ollama + the verify layer only. Run:  rag/.venv/bin/python demo/ui.py   (opens in browser).
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verify"))
from recommender import Recommender  # noqa: E402  (same import path as demo/build_demo_corpus.py)

# --- example questions, drawn from the demo set, one per behaviour --------------------
QS = {r["qid"]: r for r in (json.loads(l) for l in
      (ROOT / "demo" / "auditor_questions.jsonl").read_text().splitlines() if l.strip())}
EXAMPLES = [
    ("auq-001", "Citation lookup → GROUNDED", None),
    ("auq-013", "Audit procedure → GROUNDED", None),
    ("auq-021", "Independence conclusion → DEFER", "independence_conclusion"),
    ("auq-035", "Fraud scenario → DEFER", "fraud_accusation"),
    ("auq-046", "Out-of-scope (FASB/ASC) → declines", None),
]
# example text -> its deferral zone, so example buttons reproduce the demo-corpus behaviour
# (same as build_demo_corpus.py). Free-typed questions get zone=None (question-cue detection).
EXAMPLE_ZONES = {QS[qid]["question"]: zone for qid, _label, zone in EXAMPLES}

DISCLAIMER = (
    '<div style="border:2px solid #d29922;background:#fff8c5;color:#4d3800;padding:12px 16px;'
    'border-radius:8px;margin:6px 0;font-size:15px;line-height:1.5">'
    '<b>These are RECOMMENDATIONS, not certified-correct answers.</b> Every citation shown '
    'is verified to <b>exist</b> in the corpus — this does <b>not</b> certify the answer is '
    'correct. Verify against the cited sources. <b>Not professional advice.</b></div>'
)
COLORS = {  # (text, background)
    "GROUNDED": ("#1a7f37", "#dafbe1"),
    "PARTIAL": ("#9a6700", "#fff8c5"),
    "GENERAL-KNOWLEDGE": ("#57606a", "#eaeef2"),
    "DEFER": ("#0969da", "#ddf4ff"),
}
MEANING = {
    "GROUNDED": ("Citations shown are REAL (they exist in the corpus) — this is NOT a verdict that "
                 "the answer is correct or that the cited standard is the right one for your question. "
                 "Open the cited passages and verify the claims yourself."),
    "PARTIAL": "Some citations are verified, some were stripped or are base-only. Scrutinize the flagged parts.",
    "GENERAL-KNOWLEDGE": "Nothing could be grounded in the corpus. Treat as unverified — check every claim.",
    "DEFER": "A professional-judgment call. The framework is explained; the conclusion needs your firm's expert.",
}

print("Loading verified recommender (model + retrieval + verify layer)…", flush=True)
REC = Recommender()
print("ready.", flush=True)


def badge_html(label: str) -> str:
    if not label:
        return ""
    fg, bg = COLORS.get(label, ("#333", "#eee"))
    return (f'<div style="display:inline-block;padding:8px 16px;border-radius:8px;background:{bg};'
            f'color:{fg};font-weight:800;font-size:20px;border:2px solid {fg}">{html.escape(label)}</div>'
            f'<div style="margin-top:8px;font-size:15px;color:#555">{html.escape(MEANING.get(label, ""))}</div>')


def answer_body(labeled_answer: str) -> str:
    """The verified answer text, without the [LABEL] header (shown as the badge) or the
    verification footer (shown in the accordion)."""
    main = labeled_answer.split("\n\n— Verification —", 1)[0]
    parts = main.split("\n\n", 1)
    return (parts[1] if len(parts) > 1 else main).strip()


def verification_md(rep: dict, grounding_rule: list, source_chunk_ids: list) -> str:
    L = [
        f"**Citations checked: {rep.get('citations_total', 0)}**  ·  "
        f"✓ verified {rep.get('verified', 0)}  ·  ~ base-only {rep.get('base_verified', 0)}  ·  "
        f"✗ stripped {rep.get('stripped', 0)}  ·  ℹ disclaimer {rep.get('honest_disclaimer', 0)}",
        "",
    ]
    if grounding_rule:
        L.append("**Verified citations** — each exists in the corpus; open its passage to check:")
        for cite in grounding_rule:
            try:
                chunks = REC.index.chunks_for(cite)
            except Exception:
                chunks = []
            cids = ", ".join(f"`{c}`" for c in chunks[:6]) or "_(passage id unavailable)_"
            L.append(f"- ✓ **{cite}** → {cids}")
        L.append("")
    stripped = rep.get("stripped", 0)
    if stripped:
        names = sorted(set(rep.get("stripped_citations") or []))
        L.append(f"**✗ {stripped} citation(s) stripped — not groundable in the corpus; verify independently.**"
                 + (f"  (stripped: {', '.join(names)})" if names else ""))
        L.append("")
    L.append(f"**Fabrications shown: {rep.get('shown_fabrications', 0)}** "
             + ("— re-parsing the answer finds no ungrounded citation." if rep.get("shown_fabrications", 0) == 0
                else "— WARNING: ungrounded citations detected."))
    if source_chunk_ids:
        L.append("")
        L.append("_Retrieved passages (RAG context): " + ", ".join(f"`{c}`" for c in source_chunk_ids) + "_")
    return "\n".join(L)


def respond(question: str):
    question = (question or "").strip()
    if not question:
        return "", "_Enter a question, or click an example above._", ""
    try:
        out = REC.recommend(question, deferral_zone=EXAMPLE_ZONES.get(question))
    except Exception as e:  # noqa: BLE001 — surface infra errors to the tester
        return ('<div style="color:#b00;font-weight:700">Could not get a recommendation</div>',
                f"Is Ollama running with the **auditlm-run2** model?\n\n`{type(e).__name__}: {e}`", "")
    return (badge_html(out["label"]),
            answer_body(out["labeled_answer"]),
            verification_md(out["verification_report"], out.get("grounding_rule") or [],
                            out.get("source_chunk_ids") or []))


def build() -> gr.Blocks:
    with gr.Blocks(title="AuditLM — Verified-Citation Recommender") as demo:
        gr.Markdown("# AuditLM — Verified-Citation Recommender\n"
                    "A local assistant for US external audit (PCAOB / SEC / GAGAS). It recommends, "
                    "verifies every citation against a public corpus, and flags professional-judgment calls.")
        gr.HTML(DISCLAIMER)
        q = gr.Textbox(label="Your audit question", lines=3,
                       placeholder="e.g. Which PCAOB standard governs the auditor's consideration of fraud?")
        submit = gr.Button("Get recommendation", variant="primary")
        gr.Markdown("**Examples** (click to load, then *Get recommendation*):")
        with gr.Row():
            for qid, label, _zone in EXAMPLES:
                gr.Button(label, size="sm").click(lambda qid=qid: QS[qid]["question"], outputs=q)
        badge = gr.HTML(label="Confidence")
        answer = gr.Markdown(label="Recommendation")
        with gr.Accordion("Verification details — what was checked", open=False):
            verif = gr.Markdown()
        submit.click(respond, inputs=q, outputs=[badge, answer, verif])
        q.submit(respond, inputs=q, outputs=[badge, answer, verif])
    return demo


if __name__ == "__main__":
    inbrowser = os.environ.get("AUDITLM_UI_NOBROWSER") != "1"
    build().launch(share=False, inbrowser=inbrowser)
