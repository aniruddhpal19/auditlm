# AuditLM — Corpus Statistics

Unified corpus of **4,280 documents** and **6,345,710 tokens** (Qwen2.5 tokenizer), segmented into **12,048 passages**.

## Pipeline

- Loaded **4,288** raw records from 6 source collectors.
- Exact duplicates removed: **5**.
- Near-duplicates removed within EDGAR 10-Ks (MinHash-LSH, Jaccard ≥ 0.92): **1**.
- Dropped by quality filter (tiny/garbled, < 12 tokens or non-English): **2**.
- Final: **4,280** documents.

## Per-source (documents, tokens)

| source | docs | tokens | median tok/doc | max tok/doc |
|---|---|---|---|---|
| SEC | 2,086 | 6,017,223 | 1296 | 146,528 |
| PCAOB | 1,421 | 242,467 | 113 | 3,818 |
| GAO | 688 | 78,626 | 81 | 5,547 |
| FASB | 85 | 7,394 | 86 | 148 |

## Per-source (passages after segmentation)

| source | passages |
|---|---|
| SEC | 9,790 |
| PCAOB | 1,473 |
| GAO | 700 |
| FASB | 85 |

## Passage size (tokens)

- count: 12,048 | median: 800 | mean: 603 | max: 800
- passages at the 800-token cap (i.e. from split records): 7,604

## Per doc_type

| doc_type | docs | tokens |
|---|---|---|
| 10-K | 1,809 | 5,344,500 |
| regulation | 246 | 338,092 |
| staff_guidance | 31 | 334,631 |
| auditing_standard | 1,421 | 242,467 |
| government_auditing_standard | 688 | 78,626 |
| asc_topic_structure | 85 | 7,394 |

## Sample records (one per source)

**PCAOB — AS 1000.01**  
> .01 The auditor has a fundamental obligation to protect investors through the preparation and issuance of informative, accurate, and independent auditor’s reports. This responsibility transcends an au…

**SEC — 17 CFR 210.1-01**  
> (a) This part (together with the Financial Reporting Releases (part 211 of this chapter)) sets forth the form and content of and requirements for financial statements required to be filed as a part of…

**FASB — ASC 105**  
> ASC Topic 105 — accounting-side citation key (FASB Accounting Standards Codification). Subtopics: 10. Referenced by 27 US-GAAP taxonomy elements, e.g. GrandfatheredESOPExpenseRecognitionDividendsPaidT…

**GAO — GAGAS 1.01**  
> 1.01 This chapter provides guidance for engagements conducted in accordance with generally accepted government auditing standards (GAGAS). This chapter also a. explains the types of auditors and audit…

## Quality flags / known limitations

- **GAGAS currency:** GAO content is the **2018 revision** (GAO-21-368G); the 2024 revision is the current effective version — a v1.1 refresh candidate (clean machine-readable 2024 PDF not yet available).
- **EDGAR notes (6 of 91 filings):** name-only / index-only note headings (BofA, Goldman, IBM, McDonald's, BlackRock, Disney) yield no footnotes (reports + CAMs still captured) — deferred to LLM-assisted extraction.
- **Flattened tables:** financial-statement tables lose column structure in text extraction; note prose is preserved.
- **SAB 114:** a broad multi-topic codification update, so its single extracted SAB-Codification topic is approximate.
- **FASB ASC:** topic/subtopic **number** scaffolding only (from the public XBRL taxonomy); the licensed/bot-protected ASC prose is not reproduced.
- **Wells Fargo** 10-K skipped (cover-page primary document; financials incorporated by reference).

