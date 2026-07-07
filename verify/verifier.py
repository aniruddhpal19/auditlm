"""The verifier — the core gate of the verified-citation recommender.

Deterministic gate: raw model output -> annotated output + verification_report, where NO
fabricated citation survives in what is SHOWN.

resolve() -> action mapping:
  GROUNDED_EXACT      keep, tag="verified", attach chunk_id(s)
  GROUNDED_BASE       keep base, tag="base_verified", append unverified-paragraph note
  OUT_OF_CORPUS_STUB  STRIP, replace with placeholder, tag="stripped"   (e.g. ASC)
  FAMILY_ABSENT       STRIP + flag                                       (AU-C/IFRS/IAS/ISA)
  UNRECOGNIZED        STRIP + flag                                       (fabrications, AS 9999.01)
  honest_disclaimer   keep untouched, tag="honest_disclaimer"           (from the parser)

The honesty boundary (critical): the verifier confirms citation EXISTENCE, never citation
CORRECTNESS. tag="verified" means the cited paragraph is real and retrievable — NOT that the
surrounding claim is right (e.g. it cannot vouch that AS 2310.24 is the correct paragraph for
an A/R-existence procedure). GROUNDED_BASE goes further and explicitly flags the exact
paragraph/subsection as unverified — it must never vouch for the paragraph-level claim, since
that is exactly where the run-2 dangerous output lived (210.2-01(c)(3) cited for a (c)(1)
issue). Claim correctness is the auditor's check — that's "recommend, and they check".
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_index import CitationIndex, Resolution  # noqa: E402
from citation_parser import parse  # noqa: E402

UNVERIFIED_PARA_NOTE = " (specific paragraph not verified against the corpus; base standard confirmed)"
STRIP_PLACEHOLDER = "[citation could not be verified against the corpus — verify independently]"

# resolve() -> tag (action). honest_disclaimer is handled before resolution.
_TAG = {
    Resolution.GROUNDED_EXACT: "verified",
    Resolution.GROUNDED_BASE: "base_verified",
    Resolution.OUT_OF_CORPUS_STUB: "stripped",
    Resolution.FAMILY_ABSENT: "stripped",
    Resolution.UNRECOGNIZED: "stripped",
}
# the strip states that are genuine fabrications (for the fabrications_caught breakdown),
# distinct from out-of-scope ASC stubs which are also stripped but are real-elsewhere.
FABRICATION_RES = {Resolution.UNRECOGNIZED.value, Resolution.FAMILY_ABSENT.value}
# ALL ungrounded resolutions — what criterion 1 forbids in SHOWN text (stubs included).
UNGROUNDED_RES = {Resolution.OUT_OF_CORPUS_STUB.value, Resolution.FAMILY_ABSENT.value,
                  Resolution.UNRECOGNIZED.value}


@dataclass
class Annotation:
    citation: str            # canonical identifier
    surface: str             # raw matched text in the output
    start: int
    end: int
    resolution: str          # resolve() value, or "HONEST_DISCLAIMER"
    tag: str                 # verified | base_verified | stripped | honest_disclaimer
    chunk_ids: list = field(default_factory=list)


@dataclass
class VerificationResult:
    annotated_text: str
    report: dict
    annotations: list        # list[Annotation], in document order


def shown_ungrounded(text: str, index: CitationIndex) -> int:
    """Count ungrounded citations PRESENT in `text` — the exact string shown to the
    auditor/judge — excluding honest-disclaimer spans. Ungrounded = OUT_OF_CORPUS_STUB +
    FAMILY_ABSENT + UNRECOGNIZED. This is THE criterion-1 metric and it must be measured on
    the actual shown answer (not an intermediate string); it must be 0."""
    n = 0
    for sp in parse(text):
        if sp.honest_disclaimer:
            continue
        if index.resolve(sp.citation_normalized).value in UNGROUNDED_RES:
            n += 1
    return n


def verify(output: str, index: CitationIndex | None = None) -> VerificationResult:
    index = index or CitationIndex.load_or_build()
    anns: list[Annotation] = []
    for sp in parse(output):
        if sp.honest_disclaimer:
            anns.append(Annotation(sp.citation_normalized, sp.text, sp.start, sp.end,
                                   "HONEST_DISCLAIMER", "honest_disclaimer"))
            continue
        r = index.resolve(sp.citation_normalized)
        tag = _TAG[r]
        chunks = index.chunks_for(sp.citation_normalized) if tag in ("verified", "base_verified") else []
        anns.append(Annotation(sp.citation_normalized, sp.text, sp.start, sp.end, r.value, tag, chunks))

    # Edit the text right-to-left so earlier offsets stay valid; replace only the citation
    # token (keep the surrounding sentence intact).
    text = output
    for a in sorted(anns, key=lambda a: a.start, reverse=True):
        if a.tag == "base_verified":
            text = text[:a.start] + a.surface + UNVERIFIED_PARA_NOTE + text[a.end:]
        elif a.tag == "stripped":
            text = text[:a.start] + STRIP_PLACEHOLDER + text[a.end:]
        # verified / honest_disclaimer: leave the text untouched

    tags = Counter(a.tag for a in anns)
    stripped = [a for a in anns if a.tag == "stripped"]
    total = len(anns)
    report = {
        "citations_total": total,
        "verified": tags["verified"],
        "base_verified": tags["base_verified"],
        "stripped": tags["stripped"],
        "honest_disclaimer": tags["honest_disclaimer"],
        "stripped_citations": [a.citation for a in stripped],
        "fabrication_rate": (tags["stripped"] / total) if total else 0.0,
        # finer, honest breakdown of the strips:
        "fabrications_caught": sum(1 for a in stripped if a.resolution in FABRICATION_RES),
        "out_of_corpus_stub_stripped": sum(1 for a in stripped if a.resolution == Resolution.OUT_OF_CORPUS_STUB.value),
        # criterion 1, measured on THIS function's annotated_text. NOTE: the deployed answer
        # is the recommender's labeled_answer (header + body + footer), so the authoritative
        # criterion-1 number is recomputed there via shown_ungrounded(labeled_answer). This
        # field reflects the verifier's own strip output (kept for the unit tests).
        "shown_fabrications": shown_ungrounded(text, index),
    }
    return VerificationResult(text, report, anns)


if __name__ == "__main__":
    import json
    demo = ("Per AS 9999.01 and Section 210.2-01(c)(3) the firm is not independent. "
            "See AS 2310.24. Under ASC 842 lessors disclose lease income. "
            "The rest is drawn from general knowledge of ASC 718.")
    res = verify(demo)
    print("--- annotated ---\n" + res.annotated_text)
    print("\n--- report ---\n" + json.dumps(res.report, indent=2))
