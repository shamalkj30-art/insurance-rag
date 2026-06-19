"""
api.py — FastAPI wrapper around the RAG pipeline.

Demonstrates that this is a service, not a notebook. Run with:
    uvicorn src.api:app --reload

Then POST to /ask:
    curl -X POST localhost:8000/ask \\
         -H "Content-Type: application/json" \\
         -d '{"question": "Hva er egenandelen ved kollisjon?", "product": "Bil"}'
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="Insurance Policy Q&A API")


class AskRequest(BaseModel):
    question: str
    product: Optional[str] = None  # exact product label, e.g. "Bil Pluss"
    k: Optional[int] = None


class Source(BaseModel):
    source: Optional[str]
    page: Optional[int]


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    refused: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return answer(req.question, product_label=req.product, k=req.k or 15)
