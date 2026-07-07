"""Grounding wrapper — retrieve hybrid passages and build the grounding prompt.

Model-agnostic by design: it produces a prompt string from a question + retrieved
passages, so the SAME wrapper serves both Base+RAG and Base+SFT+RAG.

Two grounding policies (select via AUDITLM_GROUNDING=tiered|strict; default tiered):
  - TIERED (default; the SFT run uses it): strict on citations, soft on substance.
  - STRICT (ablation): strict everywhere. Both preserved — see PHASE3A_RAG_RESULTS.md.
"""

from __future__ import annotations

import os

from corpus import DEFAULT_CHUNKS
from hybrid import HybridRetriever

# Two grounding policies, both preserved for the RAG-baseline ablation. Select with the
# AUDITLM_GROUNDING env var: "tiered" (default) or "strict". SFT uses TIERED
# (the model learns the strict-citation part as a skill while keeping soft-reasoning
# fallback); STRICT remains reproducible for the paper's ablation.

# STRICT (v1) — strict everywhere: cite only retrieved passages, say "not found" if the
# source is absent. Produced the headline citation win (0.48 -> 0.80) AND the disclosure
# drop (0.45 -> 0.22): with only a topic stub retrieved, the model suppressed its own
# knowledge and answered "not explicitly stated".
STRICT_POLICY = (
    "Use ONLY the retrieved source passages below to ground your answer.\n\n"
    "CITATION RULES (strict): When you cite a standard, paragraph, or a specific "
    "disclosure/factual requirement, cite ONLY from the retrieved passages, using the "
    "exact identifier shown in brackets (e.g., AS 2301.05, ASC 606, 17 CFR 210.2-01). "
    "Do NOT state a standard number, paragraph, or specific requirement that does not "
    "appear in the retrieved passages. If the passages do not contain the source needed "
    "to answer, say the relevant source was not found in the retrieved material rather "
    "than guessing a citation.\n"
    "REASONING (synthesize): For audit procedures, concepts, comparisons, and "
    "explanations, use the passages as grounding and apply audit reasoning to give a "
    "clear, structured answer in your own words, anchored to the cited sources."
)

# TIERED (v2) — STRICT on citations (protects the citation win + blocks fabricated
# standard numbers, incl. confabulating sub-citations from a stub's 'Subtopics'), SOFT
# on substance (when retrieval is thin, answer from own knowledge instead of suppressing
# it). Recovers most of the disclosure/filing drop at a small citation/comparison cost.
TIERED_POLICY = (
    "Answer the question using the retrieved source passages below, under a TWO-TIER "
    "policy: be STRICT about citations, but answer FULLY on substance.\n\n"
    "CITATIONS (strict): When you state a specific standard NUMBER or paragraph "
    "(e.g., AS 2301.05, ASC 606, 17 CFR 210.2-01), cite ONLY identifiers that actually "
    "appear in the retrieved passages, using the exact identifier shown in brackets. Do "
    "NOT invent, guess, or infer a standard number or paragraph that is not in the "
    "passages, and do NOT fabricate sub-citations from a passage's metadata (a topic "
    "stub listing 'Subtopics: 10, 20' is NOT a set of citations). If you are unsure of "
    "the exact identifier, state the requirement without attaching a specific number "
    "rather than guessing one.\n"
    "SUBSTANCE (answer fully — including from your own knowledge when retrieval is thin): "
    "The passages may be incomplete — sometimes only a topic-level stub or tangential "
    "notes that do not actually contain the requirement asked about. In that case, "
    "ANSWER FROM YOUR OWN KNOWLEDGE and give a complete, substantive answer; do NOT say "
    "the information is 'not stated' or refuse just because the passages don't spell it "
    "out. Where useful, note which parts are grounded in the passages versus drawn from "
    "general knowledge.\n"
    "For procedures, concepts, comparisons, and disclosure requirements: synthesize a "
    "clear, structured answer in your own words — cite precisely where a passage supports "
    "a specific standard, and reason from your own knowledge where the passages fall short."
)

# Run-2 guardrail (counters the over-grounding-on-given-inputs failure, e.g. calc
# substituting a retrieved SAB-99 rule for an explicit 70%). Appended to the tiered policy
# for BOTH run-2 training (grounded demos) and run-2 serving (AUDITLM_GROUNDING=tiered_guardrail),
# so train and inference match. The RAG baseline's plain TIERED_POLICY is left unchanged
# for reproducibility.
GUARDRAIL = ("\nUSE GIVEN INPUTS: Use the specific inputs, numbers, and facts given in the "
             "question. Do NOT substitute a general rule or threshold from the passages for an "
             "explicit value the question provides.")
TIERED_GUARDRAIL_POLICY = TIERED_POLICY + GUARDRAIL

POLICIES = {"strict": STRICT_POLICY, "tiered": TIERED_POLICY,
            "tiered_guardrail": TIERED_GUARDRAIL_POLICY}
DEFAULT_POLICY = os.environ.get("AUDITLM_GROUNDING", "tiered").lower()
GROUNDING_POLICY = POLICIES.get(DEFAULT_POLICY, TIERED_POLICY)   # default tiered


def _passage_label(chunk: dict) -> str:
    """The bracket identifier shown to the model for a passage."""
    cite = chunk.get("citation")
    if cite:
        return cite
    md = chunk.get("metadata") or {}
    extra = md.get("company") or md.get("section_title") or chunk.get("title") or ""
    return f"{chunk['source']} {chunk['doc_type']}{(' — ' + extra) if extra else ''}".strip()


class Grounder:
    def __init__(self, k: int = 5, chunks_path=DEFAULT_CHUNKS, passage_chars: int = 1100):
        self.retriever = HybridRetriever(chunks_path)
        self.k = k
        self.passage_chars = passage_chars

    def retrieve(self, question: str) -> list[dict]:
        return [self.retriever.chunks[cid] for cid, _ in self.retriever.search(question, self.k)]

    def build_prompt(self, question: str, passages: list[dict], policy: str | None = None) -> str:
        blocks = []
        for i, c in enumerate(passages, 1):
            body = " ".join((c.get("text") or "").split())[: self.passage_chars]
            blocks.append(f"[{i}] ({_passage_label(c)}) {body}")
        joined = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
        return (f"{policy or GROUNDING_POLICY}\n\nRETRIEVED PASSAGES:\n{joined}\n\n"
                f"QUESTION:\n{question}\n\nAnswer:")

    def ground(self, question: str) -> tuple[str, list[dict]]:
        passages = self.retrieve(question)
        return self.build_prompt(question, passages), passages
