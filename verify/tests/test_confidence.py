"""Confidence-label logic (no network): the four labels + zone detection, incl. the
conclusion-seeking guard (a lookup that merely mentions 'fraud' must NOT become DEFER)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confidence import label_answer  # noqa: E402


def _rep(verified=0, base=0, stripped=0, honest=0):
    total = verified + base + stripped + honest
    return {"verified": verified, "base_verified": base, "stripped": stripped,
            "honest_disclaimer": honest, "citations_total": total}


def test_grounded():
    lab = label_answer(_rep(verified=3), question="Which standard governs documentation?")
    assert lab.label == "GROUNDED"
    assert "EXIST" in lab.detail and "NOT mean" in lab.detail   # existence != correctness


def test_partial_mixed_and_base_only():
    assert label_answer(_rep(verified=2, stripped=1)).label == "PARTIAL"
    assert label_answer(_rep(base=4)).label == "PARTIAL"        # base-only -> scrutinize paragraph


def test_general_knowledge():
    assert label_answer(_rep(stripped=19)).label == "GENERAL-KNOWLEDGE"
    assert label_answer(_rep()).label == "GENERAL-KNOWLEDGE"    # no citations at all


def test_defer_explicit_zone_and_question_cue():
    lab = label_answer(_rep(verified=1), deferral_zone="independence_conclusion",
                       grounding_rule=["17 CFR 210.2-01"])
    assert lab.label == "DEFER" and "independence office" in lab.detail
    assert "17 CFR 210.2-01" in lab.detail
    lab2 = label_answer(_rep(verified=1), question="Our 401k owns the client — are we still independent?")
    assert lab2.label == "DEFER"


def test_defer_not_triggered_by_topic_mention():
    """'which standard governs fraud' is a lookup, not a fraud accusation -> not DEFER."""
    lab = label_answer(_rep(verified=2), question="Which standard sets responsibilities regarding fraud?")
    assert lab.label == "GROUNDED"


def _run():
    tests = [test_grounded, test_partial_mixed_and_base_only, test_general_knowledge,
             test_defer_explicit_zone_and_question_cue, test_defer_not_triggered_by_topic_mention]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed.")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run() else 0)
