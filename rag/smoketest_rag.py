"""RAG smoke test for AuditLM.

Tests:
1. Grounder loads successfully.
2. Hybrid retrieval returns passages.
3. Retrieved passages are inserted into the grounding prompt.
4. Local Ollama llama3.1:8b generates an answer.

Usage:
    python .\rag\smoketest_rag.py
    python .\rag\smoketest_rag.py cap-cita-101
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Make rag modules importable.
ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "rag"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAG))

from ground import Grounder


DEFAULT_IDS = ["cap-cita-101", "cap-proc-102", "saf-inde-101"]

# AssuranceBench location can be overridden:
#   $env:ASSURANCEBENCH="C:\path\to\assurancebench"
AB = Path(
    os.environ.get(
        "ASSURANCEBENCH",
        str(ROOT / "assurancebench"),
    )
)


def load_items(ids: list[str]) -> list[dict]:
    """Load requested benchmark items if AssuranceBench is available."""
    if not AB.exists():
        print(f"[smoke] AssuranceBench not found at: {AB}")
        return []

    found: dict[str, dict] = {}

    for f in (AB / "items").glob("*.jsonl"):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")

        for line in text.splitlines():
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if item.get("id") in ids:
                found[item["id"]] = item

    return [found[i] for i in ids if i in found]


def ollama_available() -> bool:
    """Check whether Ollama is installed and responding."""
    try:
        p = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return p.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ollama_generate(prompt: str, model: str = "llama3.1:8b") -> str:
    """Generate using local Ollama."""
    p = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if p.returncode != 0:
        raise RuntimeError(
            "Ollama failed:\n"
            + (p.stderr or p.stdout or "unknown error")
        )

    return p.stdout.strip()


def test_grounder() -> Grounder:
    print("=" * 90)
    print("AUDITLM RAG SMOKE TEST")
    print("=" * 90)

    print("\n[1/3] Loading Grounder...")
    g = Grounder(k=5)
    print("      Grounder: READY")

    return g


def test_query(g: Grounder, question: str) -> tuple[str, list[dict]]:
    print("\n[2/3] Testing retrieval...")
    print(f"      Q: {question}")

    prompt, passages = g.ground(question)

    print(f"      Retrieved passages: {len(passages)}")
    print(f"      Prompt length: {len(prompt):,} chars")

    if not passages:
        raise RuntimeError("Grounder returned zero passages.")

    print("\n      Retrieved citations:")

    for i, c in enumerate(passages, 1):
        cite = c.get("citation") or (
            f"{c.get('source', '?')}/{c.get('doc_type', '?')}"
        )

        print(f"        [{i}] {cite}")

    # Verify passage markers exist in the actual prompt.
    missing = []

    for i in range(1, len(passages) + 1):
        if f"[{i}]" not in prompt:
            missing.append(i)

    if missing:
        raise RuntimeError(
            f"Retrieved passages missing from prompt: {missing}"
        )

    print(
        f"      Prompt contains all {len(passages)} retrieved passage markers: YES"
    )

    return prompt, passages


def main() -> int:
    ids = sys.argv[1:] or DEFAULT_IDS

    # ------------------------------------------------------------------
    # 1. Grounder / retrieval
    # ------------------------------------------------------------------
    g = test_grounder()

    # ------------------------------------------------------------------
    # 2. Benchmark questions, if available
    # ------------------------------------------------------------------
    items = load_items(ids)

    if items:
        print(f"\nLoaded {len(items)} benchmark item(s) from:")
        print(f"  {AB}")

        questions = [
            (item["id"], item["question"])
            for item in items
        ]
    else:
        print("\nAssuranceBench items unavailable.")
        print("Running a direct RAG query instead.")

        questions = [
            ("direct-rag-test", "What is the auditor's responsibility regarding going concern?")
        ]

    # ------------------------------------------------------------------
    # 3. Retrieval + optional Ollama generation
    # ------------------------------------------------------------------
    for item_id, question in questions:
        print("\n" + "=" * 90)
        print(f"ITEM: {item_id}")
        print(f"QUESTION: {question}")

        prompt, passages = test_query(g, question)

        # --------------------------------------------------------------
        # Ollama
        # --------------------------------------------------------------
        print("\n[3/3] Testing local Ollama generation...")

        if not ollama_available():
            print("      Ollama: NOT AVAILABLE")
            print("\n      RAG retrieval itself is working.")
            print("      Install/start Ollama and ensure llama3.1:8b exists")
            print("      if you want the generation portion tested.")
            continue

        print("      Ollama: AVAILABLE")
        print("      Model: llama3.1:8b")
        print("      Generating answer...")

        try:
            answer = ollama_generate(prompt)

            if not answer:
                raise RuntimeError("Ollama returned an empty answer.")

            print("\n      GROUNDED ANSWER:")
            print("      " + " ".join(answer.split())[:1000])

        except Exception as e:
            print(f"\n      Ollama generation FAILED: {e}")
            return 1

    print("\n" + "=" * 90)
    print("RAG SMOKE TEST COMPLETE")
    print("=" * 90)
    print("FAISS index:      OK")
    print("Sentence model:   OK")
    print("Grounder:         OK")
    print("Hybrid retrieval: OK")
    print("Citation index:   OK")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
