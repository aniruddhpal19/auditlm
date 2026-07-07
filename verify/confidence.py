"""Confidence labeling for the verified-citation recommender.

Label each answer so the auditor knows WHERE to check. The labels describe citation
trustworthiness and deferral, NOT claim correctness:

  GROUNDED            every citation real & sourced (verified/base), nothing stripped.
                      "Here are the sources; check them." (Sources real — NOT "reasoning right".)
  PARTIAL             some real, some stripped, or base-only. "Scrutinize the flagged parts."
  GENERAL-KNOWLEDGE   nothing groundable (all stripped / no citations). "Verify before relying."
  DEFER               a professional-judgment zone (independence/fraud/legal/novel/tax): the
                      framework may be explained, but the CONCLUSION requires the firm's
                      independence office / counsel / specialist. Protective: applied even when
                      the model wrongly concluded.

No label implies the reasoning is correct — that is the false-reassurance failure we avoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFERRAL_ZONES = {"independence_conclusion", "fraud_accusation", "legal_regulatory_conclusion",
                  "novel_transaction_treatment", "tax_position"}

_CONSULT = {
    "independence_conclusion": "your firm's independence office",
    "fraud_accusation": "engagement leadership / forensic specialists (describe indicators, do not accuse)",
    "legal_regulatory_conclusion": "legal counsel",
    "novel_transaction_treatment": "your firm's national office / technical accounting consultation",
    "tax_position": "a tax specialist",
}

# Governing-rule AREA per zone — used ONLY to point a bare DEFER at where to look, framed
# honestly ("rules are generally in …; nothing specific was retrieved — verify directly").
# These identifiers all exist in the corpus (so they resolve GROUNDED and never read as a
# fabricated citation); §10A is non-citation prose. We do NOT present them as retrieved.
_ZONE_RULE_AREA = {
    "independence_conclusion": "the SEC auditor-independence rules (17 CFR 210.2-01)",
    "fraud_accusation": "the PCAOB fraud standard (AS 2401)",
    "legal_regulatory_conclusion": "the PCAOB illegal-acts standard (AS 2405) and Securities Exchange Act Section 10A",
    "novel_transaction_treatment": "the applicable FASB ASC topic, confirmed via your firm's technical-accounting consultation",
    "tax_position": "the applicable tax authority and your firm's tax specialists",
}

# A question is a CONCLUSION-request (defer) if it asks the tool to judge a specific fact
# pattern; a RULE-LOOKUP (answer it) if it asks what the rule generally is. Conservative: only
# a clean general rule-query with no situational marker counts as a lookup — everything else
# defers. (Accepted imperfect on the "can our firm do X" categorical-prohibition class.)
_LOOKUP_LEAD = ("what ", "what's ", "which ", "over what", "how long", "define ", "list ")
_SCENARIO = ("our firm", "our audit", "our client", "our engagement", "our partner", "our lead",
             "our fee", "we found", "we received", "we uncovered", "we identified", "we discovered",
             "a staff member", "an audit manager", "an audit partner", "a senior associate",
             "a new audit client", "a former audit", "can our", "can we", "should we", "may we",
             "does that", "is this", "are we", "is our", "is my")


def is_conclusion_request(question: str | None) -> bool:
    q = " ".join((question or "").lower().split())
    if any(s in q for s in _SCENARIO):
        return True
    if q.startswith(_LOOKUP_LEAD):
        return False
    return True

# Conclusion-SEEKING question cues for live use (the benchmark passes the zone explicitly, so
# these only matter when no zone is given). Deliberately phrased as 2nd-person conclusion
# requests so a citation-lookup that merely MENTIONS a topic ("which standard governs fraud")
# does not trigger DEFER.
_Q_CUES = {
    "independence_conclusion": ["are we independent", "are we still independent", "is my firm independent",
                                "is the firm independent", "am i independent", "remain independent",
                                "impair our independence", "impair independence", "compromise our independence",
                                "can we serve as auditor", "can we still serve", "can we accept this audit"],
    "fraud_accusation": ["is this fraud", "is it fraud", "is this fraudulent", "are they committing fraud",
                         "is the company committing fraud", "is management committing fraud",
                         "did they commit fraud", "should i accuse"],
    "legal_regulatory_conclusion": ["are we liable", "is the firm liable", "is my firm liable",
                                    "liable under section", "are we legally", "will we be sued"],
    "tax_position": ["is this deductible", "what is our tax position", "what's our tax position",
                     "should we take the position", "is this tax position"],
    "novel_transaction_treatment": ["how should we account for", "how do we account for",
                                    "what's the right treatment for", "should we recognize"],
}


@dataclass
class ConfidenceLabel:
    label: str                               # GROUNDED | PARTIAL | GENERAL-KNOWLEDGE | DEFER
    headline: str
    detail: str
    grounding_rule: list = field(default_factory=list)   # the real citations to anchor on


def detect_zone(question: str | None, deferral_zone: str | None) -> str | None:
    if deferral_zone in DEFERRAL_ZONES:
        return deferral_zone
    if question:
        q = question.lower()
        for zone, cues in _Q_CUES.items():
            if any(c in q for c in cues):
                return zone
    return None


# Document-level "the model self-discloses general knowledge" cues. Unlike the parser's
# per-citation NEGATION_CUES (sentence-scoped), these are matched anywhere in the whole raw
# answer — a standalone "[Grounded in general knowledge]" marker isn't attached to any
# citation, so it must be caught at the document level. Kept tight (unambiguous self-disclosure)
# to avoid downgrading genuinely-grounded answers that merely hedge.
GK_DISCLAIMER_CUES = (
    "grounded in general knowledge", "from general knowledge", "general knowledge of",
    "drawing on general knowledge", "draws on general knowledge", "drawn from general knowledge",
    "based on general knowledge", "drawing from general knowledge",
    "my own knowledge", "from my own knowledge", "my own expertise",
    "passages do not directly address", "passages don't directly address",
    "passages do not address", "passages don't address",
    "do not directly address", "don't directly address",
    "passages do not contain", "passages don't contain", "passages do not provide",
    "passages don't provide", "not in the retrieved passages", "not in the passages",
)


def has_gk_disclaimer(text: str | None) -> bool:
    """True if the model's own answer self-discloses general-knowledge / not-in-passages
    content anywhere in the text."""
    low = (text or "").lower()
    return any(cue in low for cue in GK_DISCLAIMER_CUES)


def label_answer(report: dict, question: str | None = None, deferral_zone: str | None = None,
                 grounding_rule: list | None = None, gk_disclaimer: bool = False) -> ConfidenceLabel:
    grounding_rule = grounding_rule or []
    zone = detect_zone(question, deferral_zone)
    # Defer only CONCLUSION-requests in a deferral zone. Rule-LOOKUPS ("what does the rule
    # require/prohibit/mean") fall through and get answered (GROUNDED/PARTIAL/…) even in the
    # independence/fraud/legal zones — the rule is factual, not a judgment call.
    if zone and is_conclusion_request(question):
        if grounding_rule:
            body = (f"The framework may be explained ({', '.join(grounding_rule)}), but the "
                    f"conclusion requires {_CONSULT[zone]}. Any citations shown are checked for "
                    f"EXISTENCE only, not for whether the conclusion is correct.")
        else:
            # never a bare "consult the standards": point at the rule AREA, honestly flagged
            # as not-retrieved-for-this-question (NOT presented as a grounded citation).
            body = (f"The governing rules are generally in {_ZONE_RULE_AREA[zone]}, but the tool "
                    f"did not retrieve a specific provision for this question — verify it directly. "
                    f"The conclusion itself requires {_CONSULT[zone]}.")
        return ConfidenceLabel(
            "DEFER",
            "DEFER — professional-judgment conclusion; do not rely on the model's conclusion.",
            body, grounding_rule)

    v, b, s = report["verified"], report["base_verified"], report["stripped"]
    if v + b == 0:
        return ConfidenceLabel(
            "GENERAL-KNOWLEDGE",
            "GENERAL-KNOWLEDGE / VERIFY CAREFULLY — no citation could be grounded in the corpus.",
            "Nothing here is corpus-grounded; treat as unverified and check every claim before relying.",
            [])
    # Downgrade a would-be GROUNDED to PARTIAL when the model self-discloses general knowledge
    # (document-level) or a cited sentence was flagged as a disclaimer — a real citation can be
    # peripheral to a general-knowledge answer, so it must not get the fully-grounded badge.
    self_disclosed = gk_disclaimer or report.get("honest_disclaimer", 0) > 0
    if s > 0 or b > 0 or self_disclosed:
        return ConfidenceLabel(
            "PARTIAL",
            "PARTIAL — not fully grounded; treat as a checked starting point, not a verdict.",
            "Scrutinize the flagged parts: stripped citations weren't groundable; base-only citations "
            "confirm the standard but NOT the exact paragraph; and where the model flagged general "
            "knowledge or that the passages don't fully cover the question, the answer is not fully "
            "corpus-grounded. Real citations are existence-checked only — not vouched for correctness, "
            "nor that the cited standard is the right one. Verify the claims against the cited passages.",
            grounding_rule)
    return ConfidenceLabel(
        "GROUNDED",
        "GROUNDED — the citations shown are REAL; this is NOT a verdict that the answer is correct.",
        "GROUNDED means every citation shown EXISTS in the corpus. It does NOT mean the answer is "
        "correct, nor that the cited standard is the RIGHT one for your question — the tool checks "
        "citation existence, not whether the reasoning or the choice of standard is right (e.g. it "
        "can show a real standard cited for the wrong topic). Open the cited passages and verify the "
        "claims yourself before relying on anything.",
        grounding_rule)
