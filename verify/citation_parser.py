"""Citation parser for the verified-citation recommender.

Extract citation spans from free-text model output, each with its enclosing sentence (so the
verifier can strip precisely without mangling prose) and a canonical identifier (so the index
lookup is reliable across surface variants).

Single source: the finder regex and canonicalize() live in citation_index.py and are imported
here, not duplicated. The parser adds two things on top:
  1. enclosing-sentence segmentation + char spans;
  2. negation / honest-disclosure detection: a citation inside a "not in the
     retrieved passages / from general knowledge / topic stub" sentence is the model being
     transparent, not asserting. It is passed through tagged honest_disclaimer, NOT stripped.
     Detection is scoped to the ENCLOSING SENTENCE — a disclaimer elsewhere in the answer must
     not silence a citation that was asserted confidently in its own sentence.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from citation_index import CITATION_FINDER, canonicalize  # noqa: E402

# Honest-disclosure cues: the three core cues plus the surface variants seen in real
# run-2 output ("drawn from general knowledge of ASC 718"). Matched case-insensitively
# anywhere in the citation's enclosing sentence.
NEGATION_CUES = (
    "not in the retrieved passage", "not in the passage", "not in the retrieved material",
    "not found in", "not contained in", "not explicitly stated", "not stated in",
    "do not provide", "does not provide", "not in the corpus", "could not be verified",
    "from general knowledge", "general knowledge of", "drawn from general knowledge",
    "based on general knowledge", "my own knowledge", "from my own", "own expertise",
    "topic stub", "no specific citation",
)


@dataclass
class Span:
    text: str                  # the raw matched surface form, e.g. "Section 210.2-01(c)(3)"
    start: int                 # char offset into the output (for precise stripping)
    end: int
    citation_normalized: str   # canonical identifier, e.g. "17 CFR 210.2-01(c)(3)"
    family: str                # AS | 17 CFR | GAGAS | ASC | AU-C | IFRS | IAS | ISA
    enclosing_sentence: str
    honest_disclaimer: bool    # True => honest-disclosure transparency; pass through, never strip


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence (start, end) char spans. Newlines end a sentence; '.'/'!'/'?' end one only
    when not preceded by a digit (so citation periods like 'AS 2310.24' don't split)."""
    spans, start, i, n = [], 0, 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            if text[start:i + 1].strip():
                spans.append((start, i + 1))
            start = i + 1
        elif c in ".!?":
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < n else " "
            if not prev.isdigit() and (nxt.isspace() or i + 1 == n):
                spans.append((start, i + 1))
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                start = i = j
                continue
        i += 1
    if text[start:n].strip():
        spans.append((start, n))
    return spans


def _enclosing(spans: list[tuple[int, int]], pos: int, text: str) -> tuple[int, int]:
    for s, e in spans:
        if s <= pos < e:
            return s, e
    return (0, len(text))   # fallback: whole text


def is_negation_context(sentence: str) -> bool:
    low = sentence.lower()
    return any(cue in low for cue in NEGATION_CUES)


def parse(output: str) -> list[Span]:
    """Return citation spans (one per occurrence, in document order)."""
    sents = _sentence_spans(output)
    spans: list[Span] = []
    for m in CITATION_FINDER.finditer(output):
        raw = m.group()
        parsed = canonicalize(raw)
        if parsed is None:
            continue
        family, canonical, _base, _has_para = parsed
        s, e = _enclosing(sents, m.start(), output)
        sentence = output[s:e].strip()
        spans.append(Span(
            text=raw, start=m.start(), end=m.end(),
            citation_normalized=canonical, family=family,
            enclosing_sentence=sentence,
            honest_disclaimer=is_negation_context(sentence),
        ))
    return spans


if __name__ == "__main__":
    demo = ('Under Section 210.2-01(c)(3) the firm is not independent (17 CFR 210.2-01[c][3]). '
            'The disclosures are drawn from general knowledge of ASC 718 requirements.')
    for sp in parse(demo):
        tag = "honest_disclaimer" if sp.honest_disclaimer else "asserted"
        print(f"  {sp.text!r:32} -> {sp.citation_normalized!r:26} [{sp.family}] {tag}")
