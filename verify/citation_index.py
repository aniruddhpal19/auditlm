"""Citation index for the verified-citation recommender.

The single source of truth for "does this citation exist in our corpus." Build once from
data/corpus/chunks.jsonl, cache to verify/data/identifier_index.json, query in O(1).

The grounding scope is locked: PCAOB auditing standards (AS), SEC independence /
Reg S-X (17 CFR 210.x), and GAGAS. FASB ASC is present only as topic stubs -> resolves
OUT_OF_CORPUS_STUB (out of scope, must be flagged). AU-C / AU / IFRS / IAS / ISA are
recognised-but-not-carried -> FAMILY_ABSENT. The 8,668 SEC 10-K filing chunks are applied
prose (no `citation` field) and are deliberately NOT a grounding source -- a filing merely
*mentioning* "ASC 606" must not make ASC 606 look grounded.

The identifier regexes here are the single source; the citation parser imports them.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # auditlm/
CHUNKS = ROOT / "data" / "corpus" / "chunks.jsonl"
CACHE = Path(__file__).resolve().parent / "data" / "identifier_index.json"


class Resolution(str, Enum):
    GROUNDED_EXACT = "GROUNDED_EXACT"          # the cited unit is a real corpus passage
    GROUNDED_BASE = "GROUNDED_BASE"            # base standard present, exact paragraph is not
    OUT_OF_CORPUS_STUB = "OUT_OF_CORPUS_STUB"  # family in corpus only as a topic stub (ASC)
    FAMILY_ABSENT = "FAMILY_ABSENT"            # recognised family we deliberately don't carry
    UNRECOGNIZED = "UNRECOGNIZED"              # unparseable, or parseable but base absent (fabricated)


# ---- identifier families (single source — the parser reuses canonicalize()) ----------
# Ordered so the more specific prefixes (ASC, 17 CFR, GAGAS) are tried before bare AS.
GROUNDED_FAMILIES = ("AS", "17 CFR", "GAGAS")
FAMILIES_ABSENT = ("AU-C", "IFRS", "IAS", "ISA")

# canonicalize() also absorbs the surface variants the run-2 outputs showed, so it is the
# single normalization authority: "Section 210.2-01" -> "17 CFR 210.2-01"; bracket
# subsections "[c][3]" -> "(c)(3)"; "ASC Topic 842" -> "ASC 842".
_RE_ASC = re.compile(r"(?i)^ASC\s+(?:Topic\s+)?(\d{3})((?:-\d+)*)\b")
_RE_CFR = re.compile(r"(?i)^(?:17\s*CFR|Section)\s*(\d{3})\.(\d+(?:-\d+)?)((?:[\(\[][A-Za-z0-9]+[\)\]])*)")
_RE_GAGAS = re.compile(r"(?i)^GAGAS\s+(\d+)(?:\.(\d+))?")
_RE_AS = re.compile(r"(?i)^AS\s+(\d{3,4})(?:\.(\d+[A-Z]?))?")
_RE_AUC = re.compile(r"(?i)^AU-?C?\s+\d{3}")
_RE_IFRS = re.compile(r"(?i)^IFRS\s+\d+")
_RE_IAS = re.compile(r"(?i)^IAS\s+\d+")
_RE_ISA = re.compile(r"(?i)^ISA\s+\d+")

# Unanchored finder over free model text — SAME identifier shapes as the anchored regexes
# above (the parser imports this; single source, no duplication). Includes the surface
# aliases (Section / [..] / ASC Topic) so spans are caught, then canonicalize() normalizes.
CITATION_FINDER = re.compile(r"(?i)\b(?:"
    r"AS\s+\d{3,4}(?:\.\d+[A-Z]?)?"
    r"|(?:17\s*CFR|Section)\s+\d{3}\.\d+(?:-\d+)?(?:[\(\[][A-Za-z0-9]+[\)\]])*"
    r"|GAGAS\s+\d+(?:\.\d+)?"
    r"|ASC\s+(?:Topic\s+)?\d{3}(?:-\d+)*"
    r"|AU-?C?\s+\d{3}(?:\.\d+)?"
    r"|IFRS\s+\d+|IAS\s+\d+|ISA\s+\d+"
    r")")


def canonicalize(cite: str):
    """Parse one citation string -> (family, canonical, base, has_paragraph) or None.

    canonical = normalised full identifier (e.g. "AS 2110.07", "17 CFR 210.2-01(c)(1)").
    base      = the standard/section without the paragraph/subsection (e.g. "AS 2110").
    has_paragraph = True when the cite specifies a unit below the base (.07, (c)(1), -10-25).
    """
    s = " ".join((cite or "").split())
    m = _RE_ASC.match(s)
    if m:
        topic, sub = m.group(1), m.group(2) or ""
        return ("ASC", f"ASC {topic}{sub}", f"ASC {topic}", bool(sub))
    m = _RE_CFR.match(s)
    if m:
        base = f"17 CFR {m.group(1)}.{m.group(2)}"
        subs = (m.group(3) or "").replace("[", "(").replace("]", ")")   # [c][3] -> (c)(3)
        return ("17 CFR", base + subs, base, bool(subs))
    m = _RE_GAGAS.match(s)
    if m:
        ch, para = m.group(1), m.group(2)
        canon = f"GAGAS {ch}" + (f".{para}" if para else "")
        return ("GAGAS", canon, f"GAGAS {ch}", bool(para))
    m = _RE_AS.match(s)
    if m:
        num, para = m.group(1), m.group(2)
        canon = f"AS {num}" + (f".{para}" if para else "")
        return ("AS", canon, f"AS {num}", bool(para))
    if _RE_AUC.match(s):
        return ("AU-C", s, s, False)
    if _RE_IFRS.match(s):
        return ("IFRS", s, s, False)
    if _RE_IAS.match(s):
        return ("IAS", s, s, False)
    if _RE_ISA.match(s):
        return ("ISA", s, s, False)
    return None


class CitationIndex:
    def __init__(self, exact, base, asc_stubs, families_in_corpus, families_absent, stats=None):
        self.exact = exact                      # canonical -> [chunk_id, ...]
        self.base = base                        # base standard -> [chunk_id, ...]
        self.asc_stubs = set(asc_stubs)         # "ASC 606" topic stubs we carry
        self.families_in_corpus = families_in_corpus
        self.families_known_but_absent = families_absent
        self.stats = stats or {}

    # ---- build ----------------------------------------------------------------------
    @classmethod
    def build(cls, chunks_path: Path = CHUNKS) -> "CitationIndex":
        exact: dict[str, list[str]] = defaultdict(list)
        base: dict[str, list[str]] = defaultdict(list)
        asc_stubs: set[str] = set()
        skipped_other = 0
        family_counts: dict[str, int] = defaultdict(int)

        with open(chunks_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                md = r.get("metadata") or {}
                cid = r["id"]
                # structured metadata: standard_id is an authoritative base (PCAOB).
                std = md.get("standard_id")
                if std:
                    parsed = canonicalize(std)
                    if parsed and parsed[0] in GROUNDED_FAMILIES:
                        base[parsed[2]].append(cid)
                # the `citation` field is the clean per-passage identifier (only the 5
                # authoritative doc-types carry it; 10-K filings have none).
                cite = r.get("citation")
                if not cite:
                    continue
                parsed = canonicalize(cite)
                if parsed is None:
                    skipped_other += 1           # e.g. SAB topics — present but out of scope
                    continue
                fam, canon, b, has_para = parsed
                if fam == "ASC":
                    asc_stubs.add(b)
                    family_counts["ASC"] += 1
                    continue
                if fam not in GROUNDED_FAMILIES:
                    skipped_other += 1
                    continue
                exact[canon].append(cid)
                base[b].append(cid)
                family_counts[fam] += 1

        stats = {
            "total_identifiers": len(exact) + len(base) + len(asc_stubs),
            "exact_keys": len(exact),
            "base_keys": len(base),
            "asc_stubs": len(asc_stubs),
            "by_family": dict(sorted(family_counts.items())),
            "skipped_other_citation_chunks": skipped_other,
        }
        families_in_corpus = list(GROUNDED_FAMILIES) + (["ASC"] if asc_stubs else [])
        return cls(dict(exact), dict(base), asc_stubs, families_in_corpus,
                   list(FAMILIES_ABSENT), stats)

    # ---- cache ----------------------------------------------------------------------
    def to_json(self, path: Path = CACHE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "exact": self.exact, "base": self.base,
            "asc_stubs": sorted(self.asc_stubs),
            "families_in_corpus": self.families_in_corpus,
            "families_known_but_absent": self.families_known_but_absent,
            "stats": self.stats,
        }, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path = CACHE) -> "CitationIndex":
        d = json.loads(Path(path).read_text())
        return cls(d["exact"], d["base"], d["asc_stubs"], d["families_in_corpus"],
                   d["families_known_but_absent"], d.get("stats"))

    @classmethod
    def load_or_build(cls, chunks_path: Path = CHUNKS, cache: Path = CACHE,
                      rebuild: bool = False) -> "CitationIndex":
        if cache.exists() and not rebuild:
            return cls.from_json(cache)
        idx = cls.build(chunks_path)
        idx.to_json(cache)
        return idx

    # ---- query ----------------------------------------------------------------------
    def resolve(self, cite: str) -> Resolution:
        parsed = canonicalize(cite)
        if parsed is None:
            return Resolution.UNRECOGNIZED
        family, canonical, b, has_para = parsed
        if family in FAMILIES_ABSENT:
            return Resolution.FAMILY_ABSENT
        if family == "ASC":
            # we carry ASC only as topic stubs -> never a confident citation
            return Resolution.OUT_OF_CORPUS_STUB
        if canonical in self.exact:
            return Resolution.GROUNDED_EXACT
        if b in self.base:
            # base present: a bare standard reference is fully grounded; a specific
            # paragraph we couldn't match is base-verified (paragraph unconfirmed).
            return Resolution.GROUNDED_BASE if has_para else Resolution.GROUNDED_EXACT
        # parseable, in-corpus family, but this base is absent (e.g. 17 CFR 229.303,
        # or a fabricated AS 4101.03) -> ungrounded, must be stripped.
        return Resolution.UNRECOGNIZED

    def chunks_for(self, cite: str) -> list[str]:
        parsed = canonicalize(cite)
        if parsed is None:
            return []
        _, canonical, b, _ = parsed
        return self.exact.get(canonical) or self.base.get(b) or []


if __name__ == "__main__":
    idx = CitationIndex.load_or_build(rebuild=True)
    print("built + cached ->", CACHE)
    print(json.dumps(idx.stats, indent=2))
