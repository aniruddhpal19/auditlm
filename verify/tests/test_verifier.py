"""Verifier acceptance: the verifier on the REAL run-2 fixtures + a pure fabrication.
The non-negotiable (criterion 1): zero fabrications survive in SHOWN text. Plus: GROUNDED_BASE
cites are kept-but-flagged (never vouched), strips replace cleanly without mangling the
sentence, and honest-disclaimer cites are untouched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citation_index import CitationIndex  # noqa: E402
from verifier import verify, UNVERIFIED_PARA_NOTE, STRIP_PLACEHOLDER  # noqa: E402

_AB = Path(os.environ.get("ASSURANCEBENCH")
           or Path(__file__).resolve().parents[2].parent / "assurancebench")
RESULTS = _AB / "runs" / "rag_ollama_auditlm-run2_test_results.jsonl"
_ROWS = {json.loads(l)["id"]: json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()}
_IDX = CitationIndex.load_or_build()


def resp(i):
    return _ROWS[i]["response"]


def test_cap_proc_104_real_exact_paragraphs_kept_not_vouched():
    """AS 2310.24 etc. are REAL exact paragraphs -> verified (kept), not
    base_verified. The verifier confirms existence only; it does NOT vouch the 'cash provision'
    claim — annotated text is unchanged (no endorsement added), the auditor checks the claim."""
    res = verify(resp("cap-proc-104"), _IDX)
    assert res.report["verified"] == 5
    assert res.report["base_verified"] == 0
    assert res.report["stripped"] == 0
    assert res.report["shown_fabrications"] == 0
    assert res.annotated_text == resp("cap-proc-104")          # nothing stripped/flagged
    assert UNVERIFIED_PARA_NOTE not in res.annotated_text       # not falsely downgraded
    assert "AS 2310.24" in res.annotated_text and "provision" in res.annotated_text  # claim left for the auditor to check


def test_saf_inde_103_subsection_base_verified_flagged_not_vouched():
    """The dangerous-output fixture. 17 CFR 210.2-01(c)(3) -> base_verified: rule kept, exact
    subsection flagged unverified. It must NOT be vouched (the model used (c)(3) for a (c)(1)
    financial-interest issue)."""
    res = verify(resp("saf-inde-103"), _IDX)
    assert res.report["base_verified"] == 6
    assert res.report["stripped"] == 0
    assert res.report["shown_fabrications"] == 0
    assert UNVERIFIED_PARA_NOTE in res.annotated_text           # flagged, not silently passed
    assert "210.2-01" in res.annotated_text                     # base rule kept (it's real)
    base = [a for a in res.annotations if a.citation == "17 CFR 210.2-01(c)(3)"]
    assert base and all(a.tag == "base_verified" for a in base)
    assert all(a.tag != "verified" for a in base)               # never vouched as exact


def test_saf_lega_030_real_as_kept():
    res = verify(resp("saf-lega-030"), _IDX)
    assert res.report["verified"] == 3                          # AS 4101.03 x2, AS 4101.07
    assert res.report["stripped"] == 0
    assert res.report["shown_fabrications"] == 0
    assert "AS 4101.03" in res.annotated_text                   # real + apt -> kept


def test_cap_disc_104_asc_stripped_cleanly():
    res = verify(resp("cap-disc-104"), _IDX)
    assert res.report["stripped"] == 19 and res.report["verified"] == 0
    assert res.report["out_of_corpus_stub_stripped"] == 19
    assert res.report["shown_fabrications"] == 0
    assert "ASC 842" not in res.annotated_text                  # every ungrounded ASC removed
    assert STRIP_PLACEHOLDER in res.annotated_text


def test_cap_disc_106_disclaimer_kept_asserted_stripped():
    res = verify(resp("cap-disc-106"), _IDX)
    assert res.report["honest_disclaimer"] == 1
    assert res.report["stripped"] == 2                          # asserted ASC 718 + ASC 718-10-50-1
    assert res.report["shown_fabrications"] == 0
    # the self-disclaimed ASC 718 ("from general knowledge") is preserved verbatim
    assert "general knowledge of ASC 718" in res.annotated_text
    # but the confidently-asserted specific cite is stripped
    assert "ASC 718-10-50-1" not in res.annotated_text


def test_pure_fabrication_stripped():
    res = verify("Per AS 9999.01 the auditor must rotate every year.", _IDX)
    assert res.report["fabrications_caught"] == 1
    assert res.report["shown_fabrications"] == 0
    assert "AS 9999.01" not in res.annotated_text
    assert res.annotated_text.startswith("Per ") and res.annotated_text.endswith("every year.")


def test_strip_does_not_mangle_sentence():
    res = verify("The rule is ASC 842 here.", _IDX)
    assert res.annotated_text == f"The rule is {STRIP_PLACEHOLDER} here."


def test_criterion_1_zero_shown_fabrications_all_fixtures():
    """The non-negotiable across every fixture + a fabrication: nothing fabricated is shown."""
    for i in ["cap-proc-104", "saf-inde-103", "saf-lega-030", "cap-disc-104", "cap-disc-106"]:
        assert verify(resp(i), _IDX).report["shown_fabrications"] == 0, i
    assert verify("AS 9999.01 and IFRS 15 and AU-C 240 all apply.", _IDX).report["shown_fabrications"] == 0


def _report():
    print("\n=== verifier report (real run-2 fixtures) ===")
    for i in ["cap-proc-104", "saf-inde-103", "saf-lega-030", "cap-disc-104", "cap-disc-106"]:
        res = verify(resp(i), _IDX)
        acts = ", ".join(sorted({f"{a.citation}->{a.tag}" for a in res.annotations}))
        print(f"\n{i}:")
        print(f"  actions: {acts}")
        print(f"  report: {json.dumps(res.report)}")
    res = verify("Per AS 9999.01 the auditor must rotate.", _IDX)
    print(f"\nfabrication AS 9999.01:")
    print(f"  annotated: {res.annotated_text}")
    print(f"  report: {json.dumps(res.report)}")


def _run_asserts():
    tests = [test_cap_proc_104_real_exact_paragraphs_kept_not_vouched,
             test_saf_inde_103_subsection_base_verified_flagged_not_vouched,
             test_saf_lega_030_real_as_kept, test_cap_disc_104_asc_stripped_cleanly,
             test_cap_disc_106_disclaimer_kept_asserted_stripped, test_pure_fabrication_stripped,
             test_strip_does_not_mangle_sentence, test_criterion_1_zero_shown_fabrications_all_fixtures]
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
