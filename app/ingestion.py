"""Docling parse -> chunk -> BGE-M3 embed (dense+sparse) -> Qdrant upsert."""

import logging
from pathlib import Path

from config import settings
from retrieval import get_embedder

log = logging.getLogger("docling-rag.ingest")


def _ensure_docling_artifacts() -> "Path":
    """docling>=2.1xx requires artifacts_path to already contain the models
    (no auto-download). Download them on first use into the artifacts dir
    (a mounted volume, so this happens once per host)."""
    from pathlib import Path

    from docling.datamodel.settings import settings as docling_settings

    art = docling_settings.artifacts_path
    if art is None:
        art = Path(docling_settings.cache_dir) / "models"
    art = Path(art)
    art.mkdir(parents=True, exist_ok=True)
    if _artifacts_complete(art):
        return art
    log.info("docling artifacts incomplete at %s - downloading models (incremental)", art)
    from docling.utils.model_downloader import download_models

    download_models(output_dir=art, progress=True)
    return art


def _artifacts_complete(art: "Path") -> bool:
    """Check for the actual files the pipeline resolves, not just a non-empty dir.
    A previous crashed download can leave the dir partially populated."""
    rapid = art / "RapidOcr"
    needed_rapid = [
        "PP-OCRv6_det_small.pth",
        "ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "PP-OCRv6_rec_small.pth",
        "ppocrv6_dict.txt",
    ]
    if not all((rapid / f).exists() for f in needed_rapid):
        return False
    # layout model lands in '<org>--<repo>' (hf repo id with / replaced)
    layout_dirs = [d for d in art.glob("*--*") if d.is_dir()]
    if not layout_dirs or not any(layout_dirs[0].glob("**/*")):
        return False
    return True


def parse_document(path: str):
    """Docling parse -> list of (paragraph_text, page_no|None) units.

    Chunking from items (instead of flat markdown export) keeps page
    provenance: every docling item carries .prov with the source page.
    Returns None to signal "use the flat markdown fallback".
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

    artifacts_path = _ensure_docling_artifacts()

    pipeline_options = ThreadedPdfPipelineOptions()
    pipeline_options.artifacts_path = artifacts_path
    # A4000 Mobile tuning: defaults are 4; raise to fill free VRAM (~5 GB spare
    # after the embedding/rerank models take ~2.5 GB). Requires docling>=2.4x
    # (ThreadedPdfPipelineOptions); these fields sit on PdfPipelineOptions and
    # are inherited here.
    pipeline_options.layout_batch_size = settings.docling_layout_batch_size
    pipeline_options.ocr_batch_size = settings.docling_ocr_batch_size
    pipeline_options.table_batch_size = settings.docling_table_batch_size
    device = settings.docling_device
    if device and device != "auto":
        from docling.datamodel.pipeline_options import AcceleratorDevice

        pipeline_options.accelerator_options.device = AcceleratorDevice(device)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(Path(path))
    # release the pipeline's GPU memory (layout/OCR models, ~2-3 GB) so the
    # embedder/reranker have room; docling re-inits pipelines on next convert
    try:
        for _fmt, pipeline in list(getattr(converter, "initialized_pipelines", {}).items()):
            del pipeline
        converter.initialized_pipelines.clear()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    try:
        return _extract_paragraphs(result.document)
    except Exception:  # noqa: BLE001 - fall back to flat markdown (no pages)
        log.exception("item-level extraction failed; falling back to flat markdown")
        return None


def _extract_paragraphs(doc) -> list[tuple[str, int | None]]:
    """Walk docling items -> (text, page) paragraph units."""
    from docling_core.types.doc.labels import DocItemLabel
    from docling.document_converter import ConversionResult  # noqa: F401 (typing hint)

    text_labels = {
        DocItemLabel.PARAGRAPH,
        DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.LIST_ITEM,
    }
    units: list[tuple[str, int | None]] = []
    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label == DocItemLabel.TABLE:
            try:
                text = item.export_to_markdown(doc)
            except Exception:  # noqa: BLE001
                text = getattr(item, "text", None) or ""
            if text.strip():
                prov = getattr(item, "prov", None)
                page = prov[0].page_no if prov else None
                units.append((text.strip(), page))
        elif label in text_labels or label == DocItemLabel.PARAGRAPH:
            text = (getattr(item, "text", None) or "").strip()
            if text:
                prov = getattr(item, "prov", None)
                page = prov[0].page_no if prov else None
                units.append((text, page))
        elif label == DocItemLabel.PICTURE:
            # keep captions/grouped text; skip the image itself
            continue
    return units


def chunk_markdown(md: str) -> list[str]:
    """Flat fallback: chunk raw markdown text (no page info)."""
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


def chunk_units(units: list[tuple[str, int | None]]) -> list[dict]:
    """Chunk (paragraph_text, page) units; a chunk keeps the list of source pages."""
    size, overlap = settings.chunk_size, settings.chunk_overlap
    chunks: list[dict] = []
    current: list[tuple[str, int | None]] = []
    current_len = 0
    for text, page in units:
        plen = len(text)
        if current_len + plen > size and current:
            chunks.append(_mk_chunk(current))
            tail = "\n\n".join(t for t, _ in current)[-overlap:] if overlap > 0 else ""
            current = [(tail, current[-1][1])] if tail else []
            current_len = len(tail)
        current.append((text, page))
        current_len += plen
    if current:
        chunks.append(_mk_chunk(current))
    return [c for c in chunks if len(c["text"].strip()) > 20]


def _mk_chunk(current: list[tuple[str, int | None]]) -> dict:
    pages = sorted({p for _, p in current if p is not None})
    return {
        "text": "\n\n".join(t for t, _ in current),
        "page": pages[0] if pages else None,
        "pages": pages,
    }


def ingest_document(path: str, filename: str) -> dict:
    units = parse_document(path)
    if units is None:
        md = result_markdown_fallback(path)
        chunks = chunk_markdown(md)
        chunks = [{"text": c, "page": None, "pages": []} for c in chunks]
    else:
        chunks = chunk_units(units)
    log.info("parsed %s -> %d chunks", filename, len(chunks))

    embedder = get_embedder()
    # BGE-M3 .encode() returns {'dense_vecs': ndarray, 'lexical_weights': [dict]}
    out = embedder.encode([c["text"] for c in chunks], return_sparse=True)
    dense = [v.tolist() for v in out["dense_vecs"]]
    sparse = [{int(k): float(w) for k, w in lw.items()} for lw in out["lexical_weights"]]

    from storage import upsert_chunks

    upsert_chunks([c["text"] for c in chunks], dense, sparse, doc=filename, pages=[c["pages"] for c in chunks])
    return {"chunks": len(chunks), "chars": sum(len(c["text"]) for c in chunks), "doc": filename}


def result_markdown_fallback(path: str) -> str:
    """Flat markdown export (used when item extraction fails)."""
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(Path(path))
    return result.document.export_to_markdown()
