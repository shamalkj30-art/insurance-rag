"""
rag.py — retrieval + grounded answer generation.

The three things that make this more than a RAG demo:

  1. **Per-product retrieval.** The user picks the exact product they
     have ("Bil Pluss", "Reise Pluss", "Alvorlig sykdom", ...) and the
     retriever is restricted to that one PDF. No cross-contamination
     between products with different rules.

  2. **Strict grounding + honest refusal.** The model is told to answer
     only from the retrieved passages and to refuse — using a specific
     refusal phrase — when the passages don't contain the answer. The
     eval harness measures both accuracy and refusal correctness.

  3. **Conversational tone.** The system prompt asks the model to talk
     like a knowledgeable colleague reading the vilkår alongside you,
     not like a customer-service bot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS

from ingest import make_embeddings
from products import LABEL_TO_FILE

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "vectorstore"

# Top-K — tuned through the eval:
#   k=5  → 4/10 answerable correct (lots of recall misses)
#   k=10 → 6/10
#   k=15 → 7/10 (current).
# Diminishing returns past 15; remaining failures are reranker problems
# (right chunk exists but isn't in the top-K), not generation problems.
TOP_K = 15

REFUSAL = "Jeg fant ikke svar på dette i vilkårene jeg har tilgang til."

SYSTEM_PROMPT = f"""Du er en kunnskapsrik kollega som hjelper kunden å lese forsikringsvilkårene sine. Du sitter med dokumentet foran deg og forklarer hva som faktisk står der — som en samtale, ikke som en kundeservice-mal.

Snakk naturlig norsk. Bruk «du» og «jeg». Ingen markedsføringsspråk. Ingen «vi har mottatt» eller «takk for at du tok kontakt» — det er ikke en sak, det er et spørsmål.

REGLER (følg disse — de er viktigere enn å gi et komplett svar):
1. Bruk BARE informasjon fra vilkårsutdragene under «Vilkår».
2. Hvis vilkårene ikke gir svar, eller du er usikker: START svaret med eksakt denne setningen, ordrett: "{REFUSAL}"
   Etterpå kan du foreslå hva kunden kan gjøre videre (sjekke forsikringsbeviset, ringe Gjensidige, osv.). Ikke omformuler — bruk akkurat den setningen.
3. List kildene du brukte til slutt (filnavn + side), med mindre du brukte regel 2.
4. Hold svaret kort. To til fire setninger holder som regel.
"""

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
        text = d.page_content.removeprefix("passage: ").strip()
        blocks.append(f"[{i}] ({src}, side {page})\n{text}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    product_label: Optional[str] = None,
    k: int = TOP_K,
) -> dict:
    """Answer a question, restricted to one product's vilkår.

    Returns: {"answer": str, "sources": [...], "refused": bool}
    """
    store = _load_store()
    query = f"query: {question}"

    # Filter by the exact source filename — so "Bil Pluss" pulls from
    # ONLY Bil-Pluss-alminnelige-vilkar.pdf, never Bil-Kasko or Bil-Ansvar.
    filter_ = None
    if product_label and product_label in LABEL_TO_FILE:
        filter_ = {"source": LABEL_TO_FILE[product_label]}

    docs = store.similarity_search(query, k=k, filter=filter_)

    if not docs:
        return {
            "answer": (
                f"{REFUSAL} Jeg finner ingen relevante avsnitt i vilkårene for "
                f"{product_label or 'dette produktet'}. Sjekk forsikringsbeviset "
                "ditt eller ring Gjensidige."
            ),
            "sources": [],
            "refused": True,
        }

    context = _format_context(docs)
    llm = ChatAnthropic(
        model="claude-sonnet-4-6", temperature=0.3, max_tokens=1024,
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            (
                f"Kunden har **{product_label or 'forsikring'}**.\n\n"
                f"Vilkår:\n{context}\n\n"
                f"Spørsmål: {question}"
            ),
        ),
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
    result = answer(q, product_label="Bil Pluss")
    print(result["answer"])
    print("\nKilder:")
    for s in result["sources"]:
        print(f"  - {s['source']} (side {s['page']})")
