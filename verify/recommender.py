"""Recommender orchestration — the end of the verified-citation pipeline.

End-to-end wrapper; the model and RAG layer are UNTOUCHED:

  question -> RAG retrieve (existing rag/, tiered_guardrail policy)
           -> model generate (auditlm-run2 via ollama, matching eval serving)   [unchanged]
           -> parser -> verifier -> confidence label                            [verify layer]
           -> {labeled_answer, verification_report, source_chunk_ids}

The display layer DEDUPS the verifier's (deliberately verbose) annotations: stripped citations
collapse to a short inline "[unverified]" marker plus one summary line ("ASC 842 — cited 19×,
not groundable"), and the repeated base_verified note collapses to one summary line. The
verifier stays verbose internally (verification_report is exact); the recommender presents
cleanly.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# select the run-2 grounding policy before importing the grounder (it reads the env at import)
os.environ.setdefault("AUDITLM_GROUNDING", "tiered_guardrail")

import httpx  # noqa: E402
from ground import Grounder, TIERED_GUARDRAIL_POLICY  # noqa: E402
from citation_index import CitationIndex  # noqa: E402
from verifier import verify, shown_ungrounded  # noqa: E402
from confidence import label_answer, has_gk_disclaimer  # noqa: E402

# Same neutral system framing as the eval harness (assurancebench/src/models.py) — apples to
# apples with how auditlm-run2 was scored.
SYSTEM = ("You are an assistant for US external audit (PCAOB standards) and US "
          "GAAP accounting. Answer precisely and cite standards by their identifiers "
          "(e.g. AS 2301.05, ASC 606, 17 CFR 210.2-01) where relevant.")

MODEL = "auditlm-run2"
HOST = "http://localhost:11434"
INLINE_UNVERIFIED = "[unverified]"


class Recommender:
    def __init__(self, k: int = 5, model: str = MODEL, host: str = HOST):
        self.grounder = Grounder(k=k)
        self.index = CitationIndex.load_or_build()
        self.model = model
        self.host = host

    def _generate(self, grounded_prompt: str) -> str:
        r = httpx.post(f"{self.host}/api/chat", json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": grounded_prompt}],
            "options": {"temperature": 0.0, "num_ctx": 8192}}, timeout=300.0)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _render(self, output: str, result, label) -> str:
        """Clean, deduped display: header + body (short inline markers) + verification footer."""
        body = output
        for a in sorted(result.annotations, key=lambda a: a.start, reverse=True):
            if a.tag == "stripped":
                body = body[:a.start] + INLINE_UNVERIFIED + body[a.end:]
            # verified / base_verified / honest_disclaimer: keep the citation inline as-is
        # GROUNDED lines NAME the citation (verified/base re-parse GROUNDED — safe to show).
        # UNGROUNDED lines show COUNTS ONLY — never a bare citation token — so re-parsing the
        # shown answer yields zero ungrounded cites (criterion 1 is machine-checkable). The
        # named stripped list stays in verification_report.stripped_citations (the log).
        groups: "OrderedDict[tuple, int]" = OrderedDict()
        for a in result.annotations:
            if a.tag in ("verified", "base_verified"):
                groups[(a.citation, a.tag)] = groups.get((a.citation, a.tag), 0) + 1
        lines = []
        for (cite, tag), n in groups.items():
            times = f" [cited {n}×]" if n > 1 else ""
            if tag == "verified":
                lines.append(f"  ✓ {cite} — verified: real corpus passage available{times}")
            else:
                lines.append(f"  ~ {cite} — base standard confirmed; exact paragraph NOT verified — check it{times}")
        stripped_n = sum(1 for a in result.annotations if a.tag == "stripped")
        disc_n = sum(1 for a in result.annotations if a.tag == "honest_disclaimer")
        if stripped_n:
            lines.append(f"  ✗ {stripped_n} citation(s) not groundable in the corpus — stripped; "
                         f"verify independently (see stripped_citations)")
        if disc_n:
            lines.append(f"  ℹ {disc_n} citation(s) the model flagged as general knowledge "
                         f"(not corpus-grounded)")
        footer = ("— Verification —\n" + "\n".join(lines)) if lines else "— Verification — (no citations to check)"
        return f"[{label.label}] {label.headline}\n{label.detail}\n\n{body.strip()}\n\n{footer}"

    def recommend(self, question: str, deferral_zone: str | None = None) -> dict:
        passages = self.grounder.retrieve(question)
        grounded = self.grounder.build_prompt(question, passages, policy=TIERED_GUARDRAIL_POLICY)
        output = self._generate(grounded)
        result = verify(output, self.index)
        grounding_rule = sorted({a.citation for a in result.annotations
                                 if a.tag in ("verified", "base_verified")})
        label = label_answer(result.report, question=question, deferral_zone=deferral_zone,
                             grounding_rule=grounding_rule, gk_disclaimer=has_gk_disclaimer(output))
        labeled = self._render(output, result, label)
        # AUTHORITATIVE criterion-1 metric: measured on the EXACT string returned/shown.
        report = dict(result.report)
        report["shown_fabrications"] = shown_ungrounded(labeled, self.index)
        return {
            "label": label.label,
            "grounding_rule": label.grounding_rule,
            "labeled_answer": labeled,
            "verification_report": report,
            "source_chunk_ids": [p.get("id") for p in passages],
            "raw_output": output,
        }


if __name__ == "__main__":
    import json
    SMOKE = [
        ("Which PCAOB standard governs audit documentation requirements?", None),
        ("What are the lessor disclosure requirements under ASC 842?", None),
        ("Our firm's 401(k) plan holds a mutual fund that owns shares in our audit client. "
         "Are we still independent?", "independence_conclusion"),
        ("What procedures address the existence assertion for accounts receivable?", None),
        ("Which standard sets the auditor's responsibilities regarding fraud?", None),
    ]
    rec = Recommender()
    for q, zone in SMOKE:
        out = rec.recommend(q, deferral_zone=zone)
        print("=" * 100)
        print("Q:", q)
        print("LABEL:", out["label"], "| report:", json.dumps(out["verification_report"]))
        print("source_chunk_ids:", out["source_chunk_ids"])
        print(out["labeled_answer"])
        print()
