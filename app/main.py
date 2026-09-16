"""FastAPI service: ingest documents, hybrid-search + rerank, grounded Q&A."""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from ingestion import ingest_document
from retrieval import hybrid_search, answer_question

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("docling-rag")

app = FastAPI(title="docling-rag-poc", version="0.1.0")
_STATIC_DIR = Path(__file__).parent / "static"
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
        # keep the original for citation deep-links (file.pdf#page=N)
        try:
            files_dir = Path("/data/files")
            files_dir.mkdir(parents=True, exist_ok=True)
            dest = files_dir / Path(file.filename or "upload").name  # sanitize: name only
            shutil.copyfile(tmp_path, dest)
        except OSError:
            log.exception("could not retain original file for page links")
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
                "pages": h.get("pages") or ([h["page"]] if h.get("page") else []),
                "chunk": h["chunk_id"],
                "score": round(h["score"], 4),
                "text": h["text"][:500],
            }
            for h in hits
        ],
    )


# ---- UI + document management ----

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/documents")
def list_docs() -> dict:
    """List ingested documents with chunk counts (for the UI sidebar)."""
    from storage import list_documents

    return {"documents": list_documents()}


@app.delete("/documents")
def delete_doc(doc: str = Query(...)) -> dict:
    """Remove a document and all its chunks from the index."""
    from storage import delete_document

    n = delete_document(doc)
    # also drop the retained original (page-link source), if present
    try:
        f = Path("/data/files") / Path(doc).name
        if f.is_file():
            f.unlink()
    except OSError:
        pass
    return {"status": "ok", "deleted_chunks": n, "doc": doc}


@app.get("/files/{name}")
def get_file(name: str) -> FileResponse:
    """Serve a retained original document (for citation page deep-links)."""
    safe = Path(name).name  # no path traversal
    f = Path("/data/files") / safe
    if not f.is_file():
        raise HTTPException(404, "original file not retained (re-ingest to enable page links)")
    return FileResponse(f)
