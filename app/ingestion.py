"""Docling parse -> chunk -> BGE-M3 embed (dense+sparse) -> Qdrant upsert."""

import logging
from pathlib import Path

from config import settings
from retrieval import get_embedder

log = logging.getLogger("docling-rag.ingest")


def parse_document(path: str) -> str:
    """Docling: PDF/DOCX/... -> markdown with layout awareness."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    # A4000 Mobile tuning: defaults are 4; raise to fill free VRAM (~5 GB spare
    # after the embedding/rerank models take ~2.5 GB).
    pipeline_options.layout_batch_size = settings.docling_layout_batch_size
    pipeline_options.ocr_batch_size = settings.docling_ocr_batch_size
    if settings.docling_table_batch_size:
        pipeline_options.table_batch_size = settings.docling_table_batch_size
    device = settings.docling_device
    if device and device != "auto":
        pipeline_options.accelerator_options.device = device

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(Path(path))
    return result.document.export_to_markdown()


def chunk_markdown(md: str) -> list[str]:
    """Simple recursive chunker on markdown headings/paragraphs.

    BGE-M3 has 8k context, so chunk size is about retrieval granularity,
    not model limits. ~800 tokens with overlap is a solid default.
    """
    size, overlap = settings.chunk_size, settings.chunk_overlap
    # split on paragraphs first, keeping headings with their section
    paragraphs = [p.strip() for p in md.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        plen = len(para)
        if current_len + plen > size and current:
            text = "\n\n".join(current)
            chunks.append(text)
            # keep tail as overlap
            tail = text[-overlap:] if overlap > 0 else ""
            current = [tail] if tail else []
            current_len = len(tail)
        current.append(para)
        current_len += plen
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c.strip()) > 20]


def ingest_document(path: str, filename: str) -> dict:
    md = parse_document(path)
    chunks = chunk_markdown(md)
    log.info("parsed %s -> %d chunks", filename, len(chunks))

    embedder = get_embedder()
    dense, sparse = embedder.embed(chunks)  # dense: list[vec], sparse: list[{id:weight}]

    from storage import upsert_chunks

    upsert_chunks(chunks, dense, sparse, doc=filename)
    return {"chunks": len(chunks), "chars": len(md), "doc": filename}
