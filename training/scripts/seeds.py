"""Benchmark-weighted, in-corpus, deduplicated seed pools for full capability generation.

Each seed is (task_category, topic). Topics are IN-CORPUS by construction: citation seeds
derive from the 51 PCAOB AS titles + curated ASC topics; the rest are curated audit subjects
that the corpus (PCAOB AS / SEC / FASB ASC topics / GAGAS) actually covers. The SCOPE
constraint in generation + the corpus verifier are the safety nets behind this.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"

# benchmark weighting (capability tasks) -> demo share
WEIGHTS = {
    "citation_lookup": 0.25, "procedure_suggestion": 0.20, "concept_explanation": 0.15,
    "disclosure_review": 0.08, "filing_summarization": 0.08, "comparison_differentiation": 0.07,
    "document_drafting": 0.07, "analytical_flagging": 0.03, "calculation_support": 0.03,
}

# Major ASC topics (number -> name) the corpus indexes at topic level; used for ASC-side
# citation + disclosure seeds (answered from parametric knowledge, with disclosure).
ASC = {
    "606": "revenue from contracts with customers", "842": "leases", "820": "fair value measurement",
    "740": "income taxes", "326": "credit losses (CECL)", "350": "goodwill and other intangibles",
    "805": "business combinations", "718": "stock-based compensation", "850": "related party disclosures",
    "260": "earnings per share", "280": "segment reporting", "450": "loss contingencies",
    "715": "pension and OPEB", "330": "inventory", "360": "property, plant and equipment",
    "230": "statement of cash flows", "250": "accounting changes and error corrections",
    "220": "comprehensive income", "235": "significant accounting policies", "210": "balance sheet classification",
    "323": "equity method investments", "321": "investments in equity securities", "470": "debt",
    "606-10-50": "revenue disaggregation and contract balances", "275": "risks and uncertainties",
    "932": "extractive activities disclosures", "944": "insurance contract disclosures",
    "270": "interim reporting", "310": "receivables", "320": "investments in debt securities",
    "405": "liabilities", "410": "asset retirement and environmental obligations",
    "420": "exit or disposal cost obligations", "460": "guarantees",
    "480": "distinguishing liabilities from equity", "505": "equity",
    "610": "other income (gains and losses)", "720": "other expenses",
    "730": "research and development", "815": "derivatives and hedging",
    "825": "financial instruments", "830": "foreign currency matters",
    "835": "interest (capitalization and imputation)", "855": "subsequent events",
    "860": "transfers and servicing of financial assets", "958": "not-for-profit entities",
    "980": "regulated operations",
}


def _pcaob_titles() -> list[tuple[str, str]]:
    recs = [json.loads(l) for l in (CORPUS / "pcaob_standards.jsonl").read_text().splitlines() if l.strip()]
    seen, out = set(), []
    for r in recs:
        sid, t = r.get("standard_id"), (r.get("title") or "").strip()
        if sid and t and sid not in seen:
            seen.add(sid); out.append((sid, t))
    return out


PROC_ACCOUNTS = [
    "cash and bank balances", "accounts receivable", "inventory", "property, plant and equipment",
    "investment securities at fair value", "goodwill and intangible assets", "accounts payable",
    "accrued liabilities", "long-term debt and borrowings", "revenue", "payroll and compensation expense",
    "income tax provision and deferred taxes", "stockholders' equity", "lease assets and liabilities",
    "derivative and hedging instruments", "pension and OPEB obligations",
    "warranty and product-return reserves", "the allowance for credit losses",
]
PROC_ASSERTIONS = ["existence/occurrence", "completeness", "valuation/allocation",
                   "rights and obligations", "cutoff", "presentation and disclosure"]
PROC_RISK = [
    "management override of controls", "the presumed fraud risk in revenue recognition",
    "an entity's ability to continue as a going concern", "related-party transactions",
    "significant accounting estimates with measurement uncertainty", "subsequent events",
    "journal entries and other adjustments for fraud", "using the work of a company's specialist",
    "IT general controls supporting automated controls", "first-year audit opening balances",
]

CONCEPTS = [
    "audit risk model (inherent, control, detection risk)", "reasonable assurance vs absolute assurance",
    "sufficiency vs appropriateness of audit evidence", "professional skepticism",
    "materiality and performance materiality", "tolerable misstatement", "control risk assessment",
    "the auditor's risk assessment procedures", "significant risk", "fraud triangle",
    "management override of controls", "the expectation gap", "due professional care",
    "the engagement quality review", "the auditor's report opinion types",
    "critical audit matters", "key audit matters vs CAMs", "going concern substantial doubt",
    "the auditor's responsibility for fraud vs error", "internal control over financial reporting",
    "entity-level vs process-level controls", "the control environment (COSO components)",
    "walkthroughs", "tests of design vs operating effectiveness", "deficiency severity (material weakness)",
    "the rollforward of interim control testing", "sampling risk vs nonsampling risk",
    "substantive analytical procedures", "the search for unrecorded liabilities",
    "the management representation letter", "subsequent events (type I vs type II)",
    "the predecessor-successor auditor communication", "component vs group auditor responsibility",
    "the use of a specialist (auditor's vs management's)", "audit documentation completion and retention",
    "independence in fact vs appearance", "the audit committee's oversight role",
    "the cycle approach to the audit", "assertions about classes of transactions vs balances",
    "the difference between a review and an audit", "negative vs positive confirmations",
    "the integrated audit (financial statements + ICFR)", "PCAOB inspection vs peer review",
    "auditor's responsibilities for other information in the annual report",
    "the dual-dating of the auditor's report", "emphasis-of-matter vs explanatory paragraphs",
    "the going-concern assessment period", "evaluating the qualitative aspects of accounting practices",
    "the auditor's consideration of laws and regulations",
    "the timing of audit procedures (interim vs year-end)", "dual-purpose tests",
    "the reliability of audit evidence by source", "reperformance vs observation vs inspection",
    "analytical procedures in planning vs as substantive tests vs final review",
    "engagement partner rotation requirements", "an issuer vs a non-issuer",
    "the use of internal auditors to provide direct assistance",
    "the effect of a service organization (SOC 1) report on the audit",
    "the link between assessed risk and the nature, timing, and extent of procedures",
    "the auditor's responsibility for required supplementary information",
    "the auditor's report addressee and signature requirements",
    "evaluating the appropriateness of the going-concern basis of accounting",
]

COMPARISONS = [
    "qualified opinion vs adverse opinion vs disclaimer of opinion",
    "tests of controls vs substantive procedures", "a material weakness vs a significant deficiency in ICFR",
    "inherent risk vs control risk", "type I vs type II subsequent events",
    "the auditor's specialist vs a company's specialist", "vouching vs tracing (direction of testing)",
    "positive vs negative confirmations", "an audit vs a review engagement",
    "fraudulent financial reporting vs misappropriation of assets",
    "a critical audit matter vs an emphasis-of-matter paragraph",
    "entity-level vs transaction-level controls", "performance materiality vs tolerable misstatement",
    "design effectiveness vs operating effectiveness of a control",
    "a financial-statement-only audit vs an integrated audit",
    "Regulation S-X vs Regulation S-K", "the existence assertion vs the completeness assertion",
    "principal auditor vs component auditor responsibilities",
    "a scope limitation vs a GAAP departure (effect on the opinion)",
    "reasonable assurance vs limited assurance",
    "an unmodified opinion with an emphasis-of-matter paragraph vs a qualified opinion",
    "substantive analytical procedures vs tests of details",
    "the predecessor auditor vs successor auditor responsibilities",
    "a recognized vs nonrecognized subsequent event",
    "an audit of a single financial statement vs a complete set",
    "a financial statement assertion vs an audit objective",
    "control deficiency severity by likelihood and magnitude",
    "the engagement quality reviewer vs the engagement partner role",
]

FILINGS = [
    "the auditor's report 'Basis for Opinion' section", "the opinion paragraph of an unqualified report",
    "Item 7 MD&A of a 10-K", "the MD&A liquidity and capital resources discussion",
    "Item 9A Controls and Procedures of a 10-K", "the critical audit matters section of the auditor's report",
    "Item 1A Risk Factors of a 10-K", "Item 3 Legal Proceedings of a 10-K",
    "the management's report on internal control over financial reporting",
    "the auditor's report on internal control over financial reporting",
    "the significant accounting policies footnote", "the subsequent events footnote",
    "the related-party transactions footnote", "the commitments and contingencies footnote",
    "the segment reporting footnote", "the income tax footnote rate reconciliation",
    "the revenue recognition policy disclosure", "the fair value measurements footnote",
    "the going-concern explanatory paragraph", "the report of independent registered public accounting firm",
    "the auditor's report tenure statement (year first engaged)",
    "Item 8 Financial Statements and Supplementary Data of a 10-K",
    "the contractual obligations table in MD&A", "the off-balance-sheet arrangements disclosure",
    "the schedule of valuation and qualifying accounts", "the disclosure controls and procedures conclusion",
    "the cybersecurity risk management disclosure (Item 1C)", "the debt and credit-facility footnote",
    "the stock-based compensation footnote", "the pension and OPEB footnote",
    "the leases footnote (lessee)", "the business combination footnote",
]

DRAFTING = [
    "the opinion paragraph of an unqualified auditor's report",
    "an audit inquiry to management about a significant unusual related-party transaction",
    "a management representation letter paragraph on subsequent events",
    "an attorney letter of audit inquiry request", "a bank confirmation request",
    "a memo documenting the going-concern assessment conclusion",
    "an audit committee communication on a significant deficiency",
    "a memo evaluating an uncorrected misstatement against materiality",
    "the emphasis-of-matter paragraph for a going-concern uncertainty",
    "a workpaper documenting a substantive analytical procedure over interest expense",
    "a journal-entry testing approach memo for the management-override risk",
    "an internal control deficiency evaluation memo",
    "a fraud brainstorming (engagement team discussion) summary memo",
    "the auditor's report explanatory paragraph for a change in accounting principle",
    "a memo supporting the materiality determination for the engagement",
    "a planning memo summarizing the assessed risks of material misstatement",
    "an audit program step for accounts receivable confirmation",
    "a memo concluding on the appropriateness of the going-concern basis",
    "a positive accounts receivable confirmation request",
    "a summary of audit differences narrative", "a representation letter paragraph on fraud",
    "an inquiry of internal audit about identified control deficiencies",
    "a memo evaluating a specialist's competence and objectivity",
    "a communication to those charged with governance on significant findings",
    "a workpaper documenting the search for unrecorded liabilities",
    "a memo on the sufficiency of audit evidence for revenue",
    "a disclosure checklist conclusion memo for the financial statements",
    "an emphasis-of-matter paragraph for a justified change in accounting principle",
]

ANALYTICAL = [
    "gross margin rising sharply with flat selling prices and input costs",
    "accounts receivable growing far faster than revenue with rising DSO",
    "operating cash flow negative and declining while net income grows",
    "inventory turnover slowing while management asserts full realizability",
    "an effective tax rate that swings sharply with no explained driver",
    "a sudden spike in round-dollar manual journal entries near period-end",
    "revenue concentrated in the final days of each quarter",
    "the allowance for doubtful accounts shrinking as receivables age",
    "capitalized costs rising while repairs-and-maintenance expense falls",
    "a large unexplained 'other' line in operating expenses",
    "days payable outstanding lengthening sharply near year-end",
]

CALCULATIONS = [
    "overall materiality at 5% of pre-tax income of $40 million",
    "performance materiality at 75% of overall materiality of $2 million",
    "the clearly-trivial threshold at 5% of performance materiality",
    "projecting a sample misstatement to the population (ratio projection)",
    "overall materiality benchmarked at 1% of total assets of $800 million",
    "the sample size effect of increasing desired confidence",
    "a known plus likely misstatement total vs performance materiality",
    "materiality for a not-for-profit benchmarked on total expenses",
    "the effect of tolerable misstatement on attribute sample size",
    "evaluating whether aggregated misstatements exceed materiality",
]


def build_seeds(target_total: int = 400, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []

    # citation: PCAOB AS titles (51) + ASC topics + a few reg/SAB
    cit = [("citation_lookup", t.lower()) for _, t in _pcaob_titles()]
    cit += [("citation_lookup", f"{name} (the governing ASC topic)") for name in ASC.values()]
    cit += [("citation_lookup", x) for x in [
        "the qualifications and independence requirements for accountants under Regulation S-X",
        "the management's discussion and analysis requirements under Regulation S-K",
        "the SEC staff guidance on the assessment of materiality of misstatements",
        "the SEC staff guidance on revenue recognition", "government auditing standards independence (GAGAS)"]]

    proc = [("procedure_suggestion", f"the {a} assertion for {acct}")
            for acct in PROC_ACCOUNTS for a in PROC_ASSERTIONS]
    proc += [("procedure_suggestion", f"the audit response to {r}") for r in PROC_RISK]

    pools = {
        "citation_lookup": cit,
        "procedure_suggestion": proc,
        "concept_explanation": [("concept_explanation", c) for c in CONCEPTS],
        "disclosure_review": [("disclosure_review", f"{name} (ASC {num})") for num, name in ASC.items()],
        "filing_summarization": [("filing_summarization", f) for f in FILINGS],
        "comparison_differentiation": [("comparison_differentiation", c) for c in COMPARISONS],
        "document_drafting": [("document_drafting", d) for d in DRAFTING],
        "analytical_flagging": [("analytical_flagging", a) for a in ANALYTICAL],
        "calculation_support": [("calculation_support", c) for c in CALCULATIONS],
    }

    for cat, w in WEIGHTS.items():
        want = round(w * target_total)
        pool = list(dict.fromkeys(pools[cat]))   # dedup, preserve order
        rng.shuffle(pool)
        if want > len(pool):
            want = len(pool)                       # never duplicate a seed to pad
        out.extend(pool[:want])

    # final dedup across everything, stable
    return list(dict.fromkeys(out))


if __name__ == "__main__":
    s = build_seeds()
    from collections import Counter
    c = Counter(cat for cat, _ in s)
    print(f"total seeds: {len(s)}  (deduped)")
    for cat in WEIGHTS:
        print(f"  {cat:28} {c[cat]}")
    print("\nsamples:")
    for cat, topic in s[:3] + s[-3:]:
        print(f"  [{cat}] {topic}")
