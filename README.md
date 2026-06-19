# insurance-rag

A chat that lets you ask Gjensidige's real insurance terms (vilkår) in plain Norwegian and get back answers grounded in the actual document — with citations, and a refusal when the answer isn't there.

I built this to see what production-ish RAG over real Norwegian insurance vilkår actually looks like. I've worked at Gjensidige for over seven years, mostly close to claims, and I know how vilkår are written — nested clauses, exceptions on exceptions, the same word meaning different things in different products. Most chatbot demos I've seen would happily make stuff up. I wanted one that wouldn't.

**Live demo:** https://huggingface.co/spaces/Shamalkj30/insurance-rag

---

## What it does

Pick the exact product you have — Bil Pluss, Reise Student, Alvorlig sykdom, 21 to choose from. Ask a question in Norwegian. The agent reads only the vilkår for that one product, answers, and shows you the page it pulled from. If the answer isn't in the document, it says so instead of guessing.

The refusal-instead-of-guessing part is the one that mattered most to me. In insurance, "close enough" is worse than "I don't know."

## Under the hood

Each chunk in the vector store gets tagged with the exact product it came from (Bil Pluss, Innbo Ung, etc.), using a curated filename map. The retriever filters by that tag — so a Bil Pluss question can never pull a Bil Kasko clause. The two have different egenandel rules and crossing them would mislead the customer.

Claude Sonnet 4.6 does the answering, with a strict system prompt: answer only from the retrieved chunks, and if the answer isn't there, start with a specific refusal phrase. The eval harness measures both how often it gets the right answer AND how often it correctly refuses on unanswerable questions.

## Results

Eval is in `eval/run_eval.py` — 14 Norwegian Q&A cases against the real vilkår.

| Metric | Score |
|---|---|
| **Accuracy on answerable questions** | **7 / 10 (70%)** |
| **Refusal correctness on unanswerable** | **4 / 4 (100%)** |

The 70% isn't the model making things up — it's retrieval misses. When the right clause isn't in the top-K chunks, the model correctly refuses. That's the trade-off I'd rather have than a confident wrong answer.

### What I learned tuning this

- **Started at k=5, got 4/10.** Real recall misses on broad questions like *"dekker forsikringen brann?"* — the most semantically similar chunks weren't the most relevant ones.
- **k=15 got me to 7/10**, diminishing returns past there.
- **My first eval judge was too strict.** It expected specific NOK amounts that often aren't in vilkår — Gjensidige uses "avtalt egenandel" (varies per contract) a lot. Rewrote the judge to grade on topic-correctness and grounding, not on matching a rigid expected string.
- **A reranker would be the next lever.** Cross-encoder over the top-K, take top-3 — that's the natural next step.

## Stack

Python · LangChain · Claude Sonnet 4.6 · sentence-transformers (intfloat/multilingual-e5-base) · FAISS · FastAPI · Streamlit · LangSmith

A few choices worth calling out:

- **Local embeddings** instead of a cloud embedding API. Vilkår are public, but if I were shipping this against internal documents I'd want the embedding step local. Easier to keep that option open from day one than retrofit it later.
- **Claude as the LLM.** It's what I use day-to-day, the Norwegian quality is solid, and keeping the project on one model vendor simplifies the production story.
- **FAISS.** Corpus is small, it's just a file, no separate infrastructure to run.

## What I'd build next

- **Reranking** — cross-encoder over the top-K to push the right clause higher when retrieval is noisy.
- **Better chunking** — splitting on `§` markers instead of character count would keep each clause whole.
- **Bigger eval** — 14 cases is a sanity check. Production-ready would be 100+ real customer questions graded by a saksbehandler.
- **Recency awareness** — vilkår get updated, nothing today prefers the newest version.

## Run it locally

```bash
git clone https://github.com/shamalkj30-art/insurance-rag.git
cd insurance-rag
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env       # add ANTHROPIC_API_KEY (+ LangSmith if you want tracing)

# Drop your vilkår PDFs into data/documents/, then:
python src/ingest.py       # builds the FAISS index (first run downloads ~440 MB model)
streamlit run src/app.py
```

## Layout

```
insurance-rag/
├── src/
│   ├── products.py   # the 21 supported products + filename map
│   ├── ingest.py     # PDF → chunks → embed → FAISS
│   ├── rag.py        # filtered retrieval + grounded answer + refusal
│   ├── api.py        # FastAPI wrapper
│   └── app.py        # conversational Streamlit chat
├── eval/
│   ├── testset.jsonl # Q&A pairs (answerable + unanswerable)
│   └── run_eval.py   # accuracy + refusal measurement
├── data/documents/   # vilkår PDFs (gitignored)
└── tests/
```

## Notes

Vilkår PDFs are publicly downloadable from gjensidige.no. This repo doesn't ship customer data or anything internal. `.env` is gitignored.
