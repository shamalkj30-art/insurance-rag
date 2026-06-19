# insurance-rag — Q&A over Gjensidige vilkår, with citations and refusals

A retrieval-augmented chat assistant for **Norwegian insurance policy
documents** (Gjensidige vilkår). Built to demonstrate what it actually
takes to make a RAG system you'd trust in front of customers, not just
in a demo.

**Live demo:** _<add Streamlit / HF Spaces URL>_

---

## What makes this more than a RAG demo

1. **Product-filtered retrieval.** The customer picks their insurance
   product (Bil / Innbo / Hus / Hytte / Reise / Helse / Person) and the
   retriever is restricted to *only that product's vilkår*. A bil question
   can't accidentally be answered with an innbo clause.

2. **Strict grounding + honest refusal.** The model is instructed to answer
   *only* from the retrieved passages. When the passages don't contain
   the answer, it responds with a humble refusal and points the customer
   to a saksbehandler — never fabricates. The eval harness measures this
   behaviour explicitly.

3. **Citations on every answer.** Each response shows the source PDF and
   page so the customer (or a reviewer) can trace any claim back.

4. **Conversational UI.** First turn: *"Which insurance do you have?"*
   From then on, every retrieval is scoped to that product.

5. **Norwegian end-to-end.** The system prompt, refusal phrase, eval
   testset, and UI are all in Norwegian. Embeddings use a multilingual
   model that handles Scandinavian languages well.

---

## Architecture

```
                 ┌─────────────────────────────────────────────────┐
PDF vilkår ───► │ ingest.py                                        │
(21 docs)       │  load → chunk (800/120) → tag with product       │
                │  → embed (multilingual-e5-base) → FAISS          │
                └─────────────────────────────────────────────────┘
                                  │
                              vectorstore/
                                  │
User question ─► [select product] ─► rag.answer(question, product)
                                  │
                                  ▼
                    filter chunks WHERE product = X
                                  │
                                  ▼
                       top-K retrieval (k=5)
                                  │
                                  ▼
                Claude Sonnet 4.6 with strict system prompt
                  • answer only from retrieved chunks
                  • refuse if no answer in chunks
                  • cite sources
                                  │
                                  ▼
                  { answer, sources, refused }
```

## Stack

**Python · LangChain · Claude Sonnet 4.6 + Haiku 4.5 (judge) · sentence-transformers (intfloat/multilingual-e5-base) · FAISS · FastAPI · Streamlit · LangSmith**

### Why these choices

- **Claude Sonnet** for the answer model — strong Norwegian, calibrated
  refusals when instructed, lower cost than Opus.
- **Local multilingual embeddings** (`intfloat/multilingual-e5-base`)
  rather than a cloud embedding API — Gjensidige vilkår wouldn't be
  customer data, but the same architecture must work for documents that
  *are* sensitive. Local embeddings keep that path open.
- **FAISS** for the vector store — fast, file-based, no separate
  infrastructure to run; perfectly fine at this corpus size.

## Run it locally

```bash
git clone https://github.com/shamalkj30-art/insurance-rag.git
cd insurance-rag
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env       # add ANTHROPIC_API_KEY (and LangSmith if you want tracing)

# Drop your vilkår PDFs into data/documents/, then:
python src/ingest.py       # builds the FAISS index (first run downloads ~440 MB model)

# Run the chat UI
streamlit run src/app.py

# Or hit it as an API
uvicorn src.api:app --reload
curl -X POST localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Hva er egenandelen ved kollisjon?", "product": "Bil"}'

# Run the eval
python eval/run_eval.py
```

## Results

_(Numbers from `eval/run_eval.py` against the testset in `eval/testset.jsonl`,
21 real Gjensidige vilkår PDFs, Claude Sonnet 4.6 as the answer model and
Claude Sonnet 4.6 as judge.)_

| Metric | Score |
|---|---|
| **Accuracy on answerable Q's** | **7 / 10 (70%)** |
| **Correct refusal on unanswerable Q's** | **4 / 4 (100%)** |

The testset covers 10 answerable Norwegian Q's across products (Bil, Innbo,
Reise, Hus) and 4 unanswerable trick questions (*"Hvilken farge har bilen
min?"*, *"Hvor mye tjener direktøren?"*, etc.) that the agent must decline.

### The tuning journey (what these numbers cost)

- **k=5, rigid judge** → 1/10. Judge wouldn't accept paraphrased answers.
- **k=5, topic-focused judge** → 4/10. Real retrieval misses surfaced.
- **k=10, same judge** → 6/10. More chunks recovered some misses.
- **k=15** (current) → 7/10. Diminishing returns; the remaining 3 fails
  are recall-bound — the model correctly refuses rather than fabricates.

The fact that refusal correctness stayed at 100% throughout this is the
number I care about most: the system never makes up an answer to look
better on a metric. That's the difference between a chat toy and
something you'd put in front of a customer.

## Limitations & what I'd do differently

- **Reranking.** Single-stage retrieval. A cross-encoder reranker
  (e.g. `BAAI/bge-reranker-v2-m3`) would close the gap on
  multi-clause questions where the right chunk isn't the most similar.
- **Metadata-aware chunking.** Right now chunks are split by character
  count. Splitting on `§` markers would keep each clause whole and
  improve citation precision.
- **No reranking by recency.** Vilkår get updated; today there's no
  signal that picks the newest version when several exist.
- **Eval is small** (14 cases). For production I'd add 100+ real customer
  questions and have a saksbehandler grade them, then measure agreement
  with the LLM judge.

## Project structure

```
insurance-rag/
├── src/
│   ├── products.py    # product enum + filename → product mapping
│   ├── ingest.py      # PDF → chunk → tag with product → embed → FAISS
│   ├── rag.py         # filtered retrieval + grounded answer + refusal
│   ├── api.py         # FastAPI endpoint
│   └── app.py         # conversational Streamlit chat
├── eval/
│   ├── testset.jsonl  # Norwegian Q&A pairs (answerable + unanswerable)
│   └── run_eval.py    # accuracy + refusal measurement
├── data/documents/    # vilkår PDFs (gitignored — public PDFs only)
├── tests/
└── requirements.txt
```

## Notes on data & privacy

Vilkår PDFs are **publicly downloadable** from gjensidige.no. The repo
ships no customer data and no internal Gjensidige material. `.env` is
gitignored.
