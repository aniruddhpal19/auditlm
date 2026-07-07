"""Corpus citation index + verification — every standard citation in a generated
demonstration is checked against the public corpus so we never teach the student to
hallucinate a standard number. Demos with any unverifiable citation are dropped.

Index (from data/corpus/*.jsonl):
  - PCAOB AS standards (standard_id "AS NNNN") + exact paragraph citations ("AS NNNN.NN")
  - FASB ASC topics ("ASC NNN")
  - SEC regs ("17 CFR 210.x / 229.x") and SABs ("SAB No. N")
  - GAGAS ("GAGAS N.NN")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"

# IN-CORPUS families — verifiable against the public corpus.
PATTERNS = {
    "AS": re.compile(r"\bAS\s?(\d{3,4})(?:\.(\w+))?\b"),
    "ASC": re.compile(r"\bASC\s?(\d{3,4})(?:-[\d-]+)?\b"),
    "CFR": re.compile(r"\b17\s?CFR\s?(\d{3})\.([\w.-]+)\b"),
    "SAB": re.compile(r"\bSAB\s?(?:No\.?\s?)?(\d{2,3})\b"),
    "GAGAS": re.compile(r"\bGAGAS\s?(\d+)\.(\d+)\b"),
}

# OUT-OF-CORPUS families — frameworks the corpus does NOT contain. A specific citation
# to one of these (a NUMBER is required, so a bare framework-name mention in a comparison
# is not flagged) is UNVERIFIABLE: the verifier cannot confirm it, so it must never pass
# as "clean". Opus citing these means it answered from memory (the behavior to train out).
OUT_OF_CORPUS = {
    "AU-C": re.compile(r"\bAU-?C\s?(?:[Ss]ection\s?)?(\d{3})(?:\.\w+)?\b"),
    "AU":   re.compile(r"\bAU(?!-?C)\s?(?:[Ss]ection\s?)?(\d{3})(?:\.\w+)?\b"),
    "SAS":  re.compile(r"\bSAS\s?(?:No\.?\s?)?(\d{1,3})\b"),
    "IFRS": re.compile(r"\bIFRS\s?(\d{1,2})\b"),
    "IAS":  re.compile(r"\bIAS\s?(\d{1,2})\b"),
    "ISA":  re.compile(r"\bISA\s?(\d{3})\b"),
}


def _load(name):
    p = CORPUS / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def build_index() -> dict:
    pcaob = _load("pcaob_standards.jsonl")          # all fields are TOP-LEVEL in the source files
    idx = {
        "AS_std": {r["standard_id"] for r in pcaob if r.get("standard_id")},
        "AS_para": {r["citation"] for r in pcaob if r.get("citation")},
        "ASC": {str(r["asc_topic"]) for r in _load("fasb_asc_topics.jsonl") if r.get("asc_topic")},
        "CFR": {r["citation"] for r in _load("sec_regulations.jsonl") if r.get("citation")},
        "SAB": {str(r["sab_number"]) for r in _load("sec_sab.jsonl") if r.get("sab_number")},
        "GAGAS": {r["citation"] for r in _load("gao_yellowbook.jsonl") if r.get("citation")},
    }
    return idx


def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def extract(text: str) -> list[tuple[str, str]]:
    """Return distinct (citation, family) — in-corpus families AND out-of-corpus ones."""
    out = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            if kind == "AS":
                out.append((f"AS {m.group(1)}" + (f".{m.group(2)}" if m.group(2) else ""), "AS"))
            elif kind == "ASC":
                out.append((f"ASC {m.group(1)}", "ASC"))
            elif kind == "CFR":
                out.append((f"17 CFR {m.group(1)}.{m.group(2)}", "CFR"))
            elif kind == "SAB":
                out.append((f"SAB No. {m.group(1)}", "SAB"))
            elif kind == "GAGAS":
                out.append((f"GAGAS {m.group(1)}.{m.group(2)}", "GAGAS"))
    for fam, pat in OUT_OF_CORPUS.items():
        for m in pat.finditer(text):
            out.append((f"{fam} {m.group(1)}", fam))
    return _dedup(out)


def verify_one(cite: str, family: str, idx: dict) -> bool:
    if family == "AS":
        return ("AS " + cite.split()[1].split(".")[0]) in idx["AS_std"]   # standard-level
    if family == "ASC":
        return cite.split()[1] in idx["ASC"]                              # topic-level (no FASB prose)
    if family == "CFR":
        return cite in idx["CFR"]
    if family == "SAB":
        return cite.split("No. ")[-1] in idx["SAB"]
    if family == "GAGAS":
        return cite in idx["GAGAS"]
    return False                                                          # out-of-corpus -> unverifiable


def verify_text(text: str, idx: dict) -> dict:
    """Honest citation audit. A demo is `clean` only if EVERY citation is corpus-verified;
    any out-of-corpus citation (AU-C/IFRS/ISA/...) or not-in-corpus AS/ASC is unverifiable."""
    verified, unverifiable = [], []
    for cite, fam in extract(text):
        if verify_one(cite, fam, idx):
            verified.append(cite)
        else:
            reason = f"out-of-corpus:{fam}" if fam in OUT_OF_CORPUS else "not-in-corpus"
            unverifiable.append((cite, reason))
    return {"verified": verified, "unverifiable": unverifiable, "clean": not unverifiable}


if __name__ == "__main__":
    idx = build_index()
    print("index sizes:", {k: len(v) for k, v in idx.items()})
    for t in ["Per AS 2301.05 and ASC 606, see also 17 CFR 210.2-01 and SAB No. 99.",
              "Inventory observation is governed by AU-C Section 501.11 and IFRS 15.",
              "PCAOB AS apply to issuers while AICPA AU-C apply to non-issuers (no number).",
              "This is governed by AS 9999 and ASC 999."]:
        print(repr(t), "->", verify_text(t, idx))
