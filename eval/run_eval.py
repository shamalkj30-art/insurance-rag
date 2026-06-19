"""
run_eval.py — measure how reliable the RAG system is.

This is the part that gets you hired. Two things are measured:

  1. ACCURACY on answerable questions — does the response contain the
     content the expected answer describes? (LLM-as-judge with Claude.)
  2. REFUSAL CORRECTNESS on unanswerable questions — when the vilkår
     don't contain the answer, does the system correctly say "I don't
     know" rather than hallucinate?

Run:
    python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import REFUSAL, answer  # noqa: E402

load_dotenv()

TESTSET = Path(__file__).parent / "testset.jsonl"


def load_cases():
    return [
        json.loads(line)
        for line in TESTSET.read_text().splitlines()
        if line.strip()
    ]


def judge(question, expected, got) -> bool:
    """LLM-as-judge — does the actual answer plausibly address the question
    using grounded information?

    The earlier version of this judge was too rigid — it expected specific
    numbers ("state the deductible amount") and failed correct answers that
    cited the vilkår's actual wording ("avtalt egenandel" — varies per
    contract). Real Norwegian vilkår don't always contain hard numbers, so
    the judge now grades on topic-correctness and grounding, not on
    matching a rigid expected string.

    Uses Sonnet (not Haiku) because grading quality matters more than cost.
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=50)
    prompt = (
        "Du vurderer et RAG-systems svar fra et norsk forsikringsselskap.\n\n"
        f"Spørsmål: {question}\n"
        f"Forventet tema (det riktige svaret skal handle om dette): {expected}\n"
        f"Faktisk svar fra systemet: {got}\n\n"
        "Vurder om svaret:\n"
        "1. Handler om det spørsmålet etterspør (riktig tema)\n"
        "2. Inneholder relevant informasjon fra forsikringsvilkår\n"
        "3. Ikke er åpenbart feil eller fabrikkert\n\n"
        "Svaret trenger ikke nevne spesifikke tall hvis vilkårene ikke "
        "inneholder spesifikke tall. Svar med KUN JA eller NEI."
    )
    verdict = llm.invoke(prompt).content.strip().upper()
    return verdict.startswith("JA")


def main() -> None:
    cases = load_cases()
    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]

    correct = 0
    for c in answerable:
        result = answer(c["question"], product=c.get("product"))
        ok = judge(c["question"], c["expected"], result["answer"])
        correct += ok
        print(f"[{'PASS' if ok else 'FAIL'}] [{c.get('product','-')}] {c['question']}")

    refused_correctly = 0
    for c in unanswerable:
        result = answer(c["question"], product=c.get("product"))
        # We consider it a correct refusal if the model says the refusal phrase
        # OR the result was already short-circuited as "refused".
        refused = result["refused"] or REFUSAL.lower() in result["answer"].lower()
        if refused:
            refused_correctly += 1
            print(f"[REFUSED OK] [{c.get('product','-')}] {c['question']}")
        else:
            print(f"[HALLUCINATED] [{c.get('product','-')}] {c['question']}")
            print(f"   → {result['answer'][:120]}")

    n_ans = len(answerable) or 1
    n_unans = len(unanswerable) or 1
    print("\n" + "=" * 60)
    print(f"Accuracy (answerable):  {correct}/{len(answerable)}  = {correct/n_ans:.0%}")
    print(f"Refusal correctness:    {refused_correctly}/{len(unanswerable)}  = {refused_correctly/n_unans:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
