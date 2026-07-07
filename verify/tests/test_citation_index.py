"""Citation-index invariant: every in-scope expected_citation from the held-out test split
must resolve GROUNDED_EXACT or GROUNDED_BASE. If any in-scope cite fails, it means a
parser/family gap to widen (flag it, don't paper over it).

In-scope (locked scope) = PCAOB AS + SEC independence 17 CFR 210.2-01 + GAGAS.
Finding: the corpus actually carries all of SEC Reg S-X (210) AND Reg S-K (229), so
17 CFR 229.303 (Item 303 MD&A) genuinely grounds — broader SEC coverage than the
"210.2-01" scope label. Only FASB ASC is stub-only -> OUT_OF_CORPUS_STUB.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citation_index import CitationIndex, Resolution  # noqa: E402

# AssuranceBench repo: $ASSURANCEBENCH, else the sibling of this auditlm checkout. Portable
# across OSes (pathlib handles separators); no hardcoded absolute paths.
AB = Path(os.environ.get("ASSURANCEBENCH")
          or Path(__file__).resolve().parents[2].parent / "assurancebench")
INSCOPE_TASKS = {"citation_lookup", "procedure_suggestion"}
GROUNDED = {Resolution.GROUNDED_EXACT, Resolution.GROUNDED_BASE}


def is_in_scope(cite: str) -> bool:
    return cite.startswith("AS ") or cite.startswith("17 CFR 210.2-01") or cite.startswith("GAGAS")


def _expected_citations():
    items = [json.loads(l) for f in glob.glob(str(AB / "items" / "*.jsonl"))
             for l in open(f) if l.strip()]
    test = [it for it in items
            if it.get("split") == "test" and it.get("task_category") in INSCOPE_TASKS]
    raw = [c for it in test for c in (it.get("expected_citations") or [])]
    return raw


def test_inscope_expected_citations_all_grounded():
    idx = CitationIndex.load_or_build()
    raw = _expected_citations()
    in_scope = sorted({c for c in raw if is_in_scope(c)})
    failures = [(c, idx.resolve(c).value) for c in in_scope if idx.resolve(c) not in GROUNDED]
    assert not failures, f"in-scope cites that did NOT resolve GROUNDED: {failures}"
    assert len(in_scope) == 21, f"expected 21 unique in-scope cites, got {len(in_scope)}"


def test_asc_is_stub_only():
    """Every FASB ASC cite resolves OUT_OF_CORPUS_STUB — present as a topic stub, never
    grounded (out of scope; must be flagged/stripped downstream, never shown as confident)."""
    idx = CitationIndex.load_or_build()
    raw = _expected_citations()
    asc = {c for c in raw if c.startswith("ASC")}
    assert asc, "expected ASC cites in the test items"
    for c in asc:
        assert idx.resolve(c) == Resolution.OUT_OF_CORPUS_STUB, (c, idx.resolve(c))


def test_reg_sk_is_actually_grounded():
    """Finding: 17 CFR 229.303 (Reg S-K, Item 303 MD&A) IS in the corpus -> GROUNDED_EXACT.
    The corpus carries Reg S-X (210) + Reg S-K (229), broader than the 210.2-01 scope label;
    a real corpus citation must not be stripped."""
    idx = CitationIndex.load_or_build()
    assert idx.resolve("17 CFR 229.303") == Resolution.GROUNDED_EXACT


def test_sanity_resolutions():
    idx = CitationIndex.load_or_build()
    assert idx.resolve("AS 2401.06") == Resolution.GROUNDED_EXACT
    assert idx.resolve("AS 2110") == Resolution.GROUNDED_EXACT          # bare standard present
    assert idx.resolve("AS 2401.99") == Resolution.GROUNDED_BASE        # base ok, paragraph not
    assert idx.resolve("AS 9999.01") == Resolution.UNRECOGNIZED         # fabricated standard (AS 9999 absent)
    assert idx.resolve("ASC 606") == Resolution.OUT_OF_CORPUS_STUB      # stub-only family
    assert idx.resolve("AU-C 240") == Resolution.FAMILY_ABSENT          # not carried
    assert idx.resolve("IFRS 15") == Resolution.FAMILY_ABSENT
    assert idx.resolve("the auditor should test controls") == Resolution.UNRECOGNIZED


def _report():
    idx = CitationIndex.load_or_build()
    raw = _expected_citations()
    in_scope = sorted({c for c in raw if is_in_scope(c)})
    out = sorted({c for c in raw if not is_in_scope(c)})
    print(f"\n=== citation index report ===")
    print(f"index stats: {json.dumps(idx.stats)}")
    print(f"families_in_corpus: {idx.families_in_corpus} | known-but-absent: {idx.families_known_but_absent}")
    print(f"\nexpected_citations pulled: {len(raw)} raw "
          f"({len(set(raw))} unique) from test split [citation_lookup + procedure_suggestion]")
    res = [(c, idx.resolve(c).value) for c in in_scope]
    grounded = [c for c, r in res if r in {x.value for x in GROUNDED}]
    print(f"\nIN-SCOPE (AS / 17 CFR 210.2-01 / GAGAS): {len(in_scope)} unique")
    print(f"  resolved GROUNDED: {len(grounded)}/{len(in_scope)} "
          f"({'100% — GO' if len(grounded) == len(in_scope) else 'NOT 100% — NO-GO'})")
    for c, r in res:
        mark = "ok " if r in {x.value for x in GROUNDED} else "FAIL"
        print(f"    [{mark}] {c:18} -> {r}")
    print(f"\nOUT-OF-(spec)-SCOPE cites in the same items: {len(out)} unique")
    for c in out:
        r = idx.resolve(c).value
        note = "  <-- corpus carries Reg S-K too (grounded, not a gap)" if r in {x.value for x in GROUNDED} else ""
        print(f"    {c:18} -> {r}{note}")
    bad = [c for c in in_scope if idx.resolve(c) not in GROUNDED]
    print(f"\nIn-scope cites NOT resolving (parser/family gaps to widen): {bad or 'NONE'}")


def _run_asserts():
    tests = [test_inscope_expected_citations_all_grounded, test_asc_is_stub_only,
             test_reg_sk_is_actually_grounded, test_sanity_resolutions]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed.")
    return failed


if __name__ == "__main__":
    _report()
    print("\n=== assertions ===")
    raise SystemExit(1 if _run_asserts() else 0)
