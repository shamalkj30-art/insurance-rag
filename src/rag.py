"""
rag.py — retrieval + grounded answer generation.

What makes this more than a demo:
  1. **Product-filtered retrieval** — the customer's product (Bil, Innbo,
     etc.) gates which chunks are even considered. This prevents the agent
     from answering a bil question with an innbo clause.
  2. **Strict grounding** — the model is instructed to answer ONLY from the
     retrieved passages and to refuse honestly when the passages don't
     contain the answer. The eval harness measures this refusal behaviour.
  3. **Citations on every answer** — source filename + page so the customer
     (or a saksbehandler reviewing) can trace any claim back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS

from ingest import make_embeddings
from products import Product

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "vectorstore"
# TOP_K — chunks to retrieve. We tuned this through the eval:
#   k=5  → 4/10 accurate, several genuine retrieval misses
#   k=10 → 6/10
#   k=15 → 7/10  (current)
# Bigger isn't always better (more context = more noise) but legal-style
# vilkår are repetitive enough that wider recall pays off here. The
# remaining failures are recall-bound — a cross-encoder reranker would
# be the next lever (see README "what I'd do differently").
TOP_K = 15

REFUSAL = "Jeg fant ikke svar på dette i vilkårene jeg har tilgang til."

SYSTEM_PROMPT = f"""Du er en hjelpsom og presis kundeassistent for et norsk forsikringsselskap.

REGLER (følg disse strengt — det er viktigere enn å gi et fullstendig svar):
1. Svar BARE basert på de oppgitte vilkårsutdragene under «Vilkår».
2. KRITISK: Hvis vilkårene ikke inneholder svaret, eller du er usikker:
   - START svaret ditt med eksakt denne setningen, ordrett:
     "{REFUSAL}"
   - Deretter kan du anbefale at kunden kontakter en saksbehandler i
     Gjensidige, eller foreslå hvor de ellers kan finne svaret.
   - Ikke omformuler "jeg vet ikke" — bruk den eksakte setningen.
3. Etter svaret ditt, list opp kildene du brukte (filnavn og side) —
   med mindre du brukte regel 2.
4. Bruk klart, vennlig norsk. Hold svaret kort (maks 4-6 setninger).
5. Hvis kunden spør om noe som ikke gjelder deres valgte produkt, si det
   høflig og tilby å hjelpe med riktig produkt.
"""


# Phrases that indicate the model is refusing — used by the eval to count
# "honest refusals" even when the model paraphrases slightly. Keeps the
# system robust without weakening the strict prompt above.
_REFUSAL_SIGNALS = (
    REFUSAL.lower(),
    "ikke i vilkårene",
    "finnes ikke i vilkårene",
    "vilkårene jeg har tilgang til",
    "ikke noe vilkårene omhandler",
    "kan dessverre ikke hjelpe",
)


def looks_like_refusal(text: str) -> bool:
    t = text.lower()
    return any(sig in t for sig in _REFUSAL_SIGNALS)


_store: Optional[FAISS] = None


def _load_store() -> FAISS:
    global _store
    if _store is None:
        _store = FAISS.load_local(
            str(INDEX_DIR),
            make_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _store


def _format_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "ukjent")
        page = d.metadata.get("page", "?")
        # E5 stored chunks with a "passage:" prefix; strip it before showing
        # the text to the LLM — it's only meaningful to the embedding model.
        text = d.page_content.removeprefix("passage: ").strip()
        blocks.append(f"[{i}] ({src}, side {page})\n{text}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    product: Optional[Product] = None,
    k: int = TOP_K,
) -> dict:
    """
    Answer a question, optionally restricted to one product's vilkår.

    Returns:
        {
            "answer": str,
            "sources": [{"source": ..., "page": ...}, ...],
            "refused": bool,
        }
    """
    store = _load_store()

    # E5 query format — must match the prefix used at ingest time.
    query = f"query: {question}"

    # Product filter: keeps the agent honest. If product is None we fall
    # back to global retrieval (e.g. for first-turn product detection).
    filter_ = {"product": product} if product else None
    docs = store.similarity_search(query, k=k, filter=filter_)

    if not docs:
        return {
            "answer": (
                f"{REFUSAL} Jeg anbefaler at du kontakter en saksbehandler "
                "i Gjensidige for å være helt sikker."
            ),
            "sources": [],
            "refused": True,
        }

    context = _format_context(docs)
    llm = ChatAnthropic(
        model="claude-sonnet-4-6", temperature=0, max_tokens=1024,
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Vilkår:\n{context}\n\nSpørsmål: {question}"),
    ]
    response = llm.invoke(messages).content

    refused = looks_like_refusal(response)
    sources = (
        [] if refused else [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
            for d in docs
        ]
    )
    return {"answer": response, "sources": sources, "refused": refused}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Hva er egenandelen ved kollisjon?"
    result = answer(q, product="Bil")
    print(result["answer"])
    print("\nKilder:")
    for s in result["sources"]:
        print(f"  - {s['source']} (side {s['page']})")
