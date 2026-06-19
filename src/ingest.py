"""
ingest.py — build the vector store from policy documents.

Run once, or whenever the documents change:
    python src/ingest.py

What changed in this version:
- **Local multilingual embeddings** (intfloat/multilingual-e5-base) instead
  of a cloud embedding API. Norwegian works well out of the box, no extra
  API keys required, and the embedding step keeps customer-sensitive
  vilkår text local — defensible as a data-residency choice for an
  insurance company.
- **Product metadata on every chunk** so the retriever can filter to only
  the vilkår that match the customer's product (Bil / Innbo / Reise / ...).
- Source PDF + page number are still preserved so every answer can cite
  back to its origin.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from products import FILE_TO_LABEL

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"
INDEX_DIR = Path(__file__).parent.parent / "vectorstore"

# Chunk size is the single biggest lever on retrieval quality.
# 800 / 120 is a sensible default for legal-style policy text; we'll
# benchmark variants in the eval harness.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# multilingual-e5-base: 768-dim, ~440MB, strong on Scandinavian languages.
# The "passage:" / "query:" prefixes are part of the e5 family's
# instruction-tuned format — we apply them when ingesting and querying
# respectively. Without them, retrieval quality drops noticeably.
EMBED_MODEL = "intfloat/multilingual-e5-base"


def make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # required for cosine sim
    )


def load_documents():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(
            f"No PDFs found in {DOCS_DIR}. Drop some vilkår PDFs there first."
        )

    docs = []
    for pdf in pdfs:
        # The product label is taken from the curated map so the agent
        # always filters on a known product variant (e.g. "Bil Pluss"),
        # never a guessed category. Files that aren't in the map are
        # still ingested but get the filename as a fallback label.
        label = FILE_TO_LABEL.get(pdf.name, pdf.stem)
        loaded = PyPDFLoader(str(pdf)).load()  # one Document per page
        for d in loaded:
            d.metadata["source"] = pdf.name      # ← the field we filter on
            d.metadata["product"] = label        # ← human-readable label
            # E5 model wants the "passage:" prefix at index time
            d.page_content = f"passage: {d.page_content}"
        docs.extend(loaded)
        print(f"  {label:<22} {len(loaded):>3}p  {pdf.name}")
    return docs


def main() -> None:
    print("Loading PDFs...")
    docs = load_documents()
    print(f"  → {len(docs)} pages loaded across {len({d.metadata['source'] for d in docs})} files")

    print(f"\nSplitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} chunks")

    by_product = {}
    for c in chunks:
        by_product.setdefault(c.metadata["product"], 0)
        by_product[c.metadata["product"]] += 1
    print("  chunks per product:")
    for prod, n in sorted(by_product.items()):
        print(f"    {n:>4}  {prod}")

    print(f"\nEmbedding with {EMBED_MODEL} (first run downloads ~440 MB model)...")
    embeddings = make_embeddings()
    store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"\n✓ Saved FAISS index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
