"""Parser acceptance: parse + normalize + negation-handling on the REAL run-2 failure
outputs (not synthetic). Fixtures: cap-proc-104, saf-inde-103, saf-lega-030, cap-disc-104,
cap-disc-106 — pulled live from the run-2 results so the test is against actual model text.

Asserts: (1) correct extraction (real cites in, prose out); (2) the three surface forms
normalize to one canonical identifier; (3) honest-disclaimer citations are
tagged, not stripped, and detection is scoped to the enclosing sentence.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citation_parser import parse  # noqa: E402

_AB = Path(os.environ.get("ASSURANCEBENCH")
           or Path(__file__).resolve().parents[2].parent / "assurancebench")
RESULTS = _AB / "runs" / "rag_ollama_auditlm-run2_test_results.jsonl"
_ROWS = {json.loads(l)["id"]: json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()}


def resp(item_id: str) -> str:
    return _ROWS[item_id]["response"]


def test_cap_proc_104_clean_as_extraction():
    spans = parse(resp("cap-proc-104"))
    norm = Counter(s.citation_normalized for s in spans)
    assert set(norm) == {"AS 2310.24", "AS 2310.06", "AS 2605.17", "AS 2605.21"}, norm
    assert norm["AS 2310.24"] == 2                       # cited twice
    assert all(not s.honest_disclaimer for s in spans)
    assert all("provision" not in s.text for s in spans)  # "the cash provision" is prose, not a cite


def test_saf_inde_103_surface_forms_normalize():
    spans = parse(resp("saf-inde-103"))
    assert spans, "expected independence-rule citations"
    assert all(s.citation_normalized == "17 CFR 210.2-01(c)(3)" for s in spans), \
        {s.text: s.citation_normalized for s in spans}
    raws = {s.text for s in spans}
    assert "Section 210.2-01(c)(3)" in raws                       # "Section" alias form
    assert any("[c][3]" in r for r in raws)                       # bracket-subsection form
    # asserted with confidence -> NOT honest_disclaimer, even though a general-knowledge note
    # appears LATER in the answer (negation is enclosing-sentence-scoped, not document-wide).
    assert all(not s.honest_disclaimer for s in spans)


def test_saf_lega_030_as_extracted_statute_ignored():
    spans = parse(resp("saf-lega-030"))
    norm = {s.citation_normalized for s in spans}
    assert "AS 4101.03" in norm and "AS 4101.07" in norm, norm
    # "Section 11 of the Securities Act" is a statute ref, not a 17 CFR cite — must NOT match.
    assert all(not s.text.lower().startswith("section 11") for s in spans)
    assert all(s.family != "17 CFR" for s in spans)
    assert all(not s.honest_disclaimer for s in spans)


def test_cap_disc_104_asc_topic_variant_normalizes():
    spans = parse(resp("cap-disc-104"))
    raws = {s.text for s in spans}
    assert "ASC Topic 842" in raws and "ASC 842" in raws          # both surface forms appear
    assert all(s.citation_normalized == "ASC 842" for s in spans), \
        {s.text: s.citation_normalized for s in spans}            # both -> "ASC 842"


def test_cap_disc_106_honest_disclaimer_passthrough():
    spans = parse(resp("cap-disc-106"))
    disc = [s for s in spans if s.honest_disclaimer]
    asserted = [s for s in spans if not s.honest_disclaimer]
    # exactly the "ASC 718" inside "...drawn from general knowledge of ASC 718 requirements"
    assert len(disc) == 1, [(s.text, s.enclosing_sentence) for s in disc]
    assert disc[0].citation_normalized == "ASC 718"
    assert "general knowledge" in disc[0].enclosing_sentence.lower()
    # the asserted specific cite is preserved as a normal (to-be-resolved) span
    assert any(s.citation_normalized == "ASC 718-10-50-1" for s in asserted)


def test_normalization_map():
    """Surface -> canonical, asserted directly (the three forms 'refer to the same rule')."""
    from citation_parser import parse as p
    cases = {
        "Section 210.2-01(c)(3)": "17 CFR 210.2-01(c)(3)",
        "17 CFR 210.2-01[c][3]": "17 CFR 210.2-01(c)(3)",
        "ASC Topic 842": "ASC 842",
        "ASC 718-10-50-1": "ASC 718-10-50-1",
        "AS 2310.24": "AS 2310.24",
    }
    for surface, canonical in cases.items():
        spans = p(surface)
        assert len(spans) == 1 and spans[0].citation_normalized == canonical, (surface, spans)


def _report():
    print("\n=== parser report (real run-2 fixtures) ===")
    for i in ["cap-proc-104", "saf-inde-103", "saf-lega-030", "cap-disc-104", "cap-disc-106"]:
        spans = parse(resp(i))
        norm = Counter(s.citation_normalized for s in spans)
        disc = [(s.text, s.citation_normalized) for s in spans if s.honest_disclaimer]
        surfaces = sorted({s.text for s in spans})
        print(f"\n{i}: {len(spans)} spans")
        print(f"  surface forms: {surfaces}")
        print(f"  -> normalized: {dict(norm)}")
        print(f"  honest_disclaimer: {disc or 'none'}")
    print("\nnormalization map (surface -> canonical):")
    for s in ["Section 210.2-01(c)(3)", "17 CFR 210.2-01[c][3]", "ASC Topic 842",
              "ASC 718-10-50-1", "AS 2310.24"]:
        print(f"  {s:30} -> {parse(s)[0].citation_normalized}")


def _run_asserts():
    tests = [test_cap_proc_104_clean_as_extraction, test_saf_inde_103_surface_forms_normalize,
             test_saf_lega_030_as_extracted_statute_ignored, test_cap_disc_104_asc_topic_variant_normalizes,
             test_cap_disc_106_honest_disclaimer_passthrough, test_normalization_map]
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
