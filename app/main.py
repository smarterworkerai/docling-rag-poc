"""FastAPI service: ingest documents, hybrid-search + rerank, grounded Q&A."""

import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from ingestion import ingest_document
from retrieval import hybrid_search, answer_question

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("docling-rag")

app = FastAPI(title="docling-rag-poc", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    # Minimum reranker score for a chunk to be used as context (0..1 roughly)
    min_score: float = 0.3


class SearchRequest(QueryRequest):
    pass


class AnswerResponse(BaseModel):
    answer: Optional[str]
    answered: bool
    reason: str
    citations: list[dict]


@app.on_event("startup")
async def startup() -> None:
    from storage import ensure_collection

    ensure_collection()
    log.info(
        "collection '%s' ready on Qdrant %s (dense %dd + sparse)",
        settings.qdrant_collection,
        settings.qdrant_url,
        settings.embed_dim,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Ingest a PDF/DOCX/... file: Docling parse -> chunk -> BGE-M3 embed -> Qdrant."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    supported = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md", ".txt", ".csv"}
    if suffix and suffix not in supported:
        raise HTTPException(400, f"unsupported file type '{suffix}', supported: {sorted(supported)}")

    with tempfile.NamedTemporaryFile(suffix=suffix or None, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        stats = ingest_document(tmp_path, filename=file.filename or os.path.basename(tmp_path))
        return {"status": "ok", "filename": file.filename, **stats}
    finally:
        os.unlink(tmp_path)


@app.post("/search")
def search(req: SearchRequest) -> dict:
    """Hybrid search + rerank, returns ranked chunks. No LLM call."""
    hits = hybrid_search(req.question, top_k=req.top_k, min_score=req.min_score)
    return {"hits": hits}


@app.post("/query")
def query(req: QueryRequest) -> AnswerResponse:
    """Hybrid search + rerank + grounded LLM answer with citations."""
    hits = hybrid_search(req.question, top_k=req.top_k, min_score=req.min_score)
    answer, answered, reason = answer_question(req.question, hits)
    return AnswerResponse(
        answer=answer,
        answered=answered,
        reason=reason,
        citations=[
            {
                "doc": h["doc"],
                "page": h.get("page"),
                "chunk": h["chunk_id"],
                "score": round(h["score"], 4),
                "text": h["text"][:500],
            }
            for h in hits
        ],
    )
