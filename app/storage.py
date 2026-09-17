"""Qdrant collection management: named dense vector + sparse vector, RRF fusion."""

import logging
import uuid

from qdrant_client import QdrantClient, models

from config import settings

log = logging.getLogger("docling-rag.storage")

DENSE_NAME = "bge-m3-dense"
SPARSE_NAME = "bge-m3-sparse"


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=60)


def ensure_collection() -> None:
    c = _client()
    if not c.collection_exists(settings.qdrant_collection):
        c.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE_NAME: models.VectorParams(
                    size=settings.embed_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_NAME: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        log.info("created collection %s", settings.qdrant_collection)


def upsert_chunks(
    chunks: list[str],
    dense: list,
    sparse: list,
    doc: str,
    pages: list[list[int]] | None = None,
    metadata: dict | None = None,
) -> None:
    c = _client()
    points = []
    base_payload = metadata or {}
    for i in range(len(chunks)):
        payload = {**base_payload, "text": chunks[i], "doc": doc, "chunk_id": i}
        if pages and pages[i]:
            payload["page"] = pages[i][0]
            payload["pages"] = pages[i]
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc}:{i}")),
                vector={
                    DENSE_NAME: dense[i],
                    SPARSE_NAME: models.SparseVector(
                        indices=[int(k) for k in sparse[i]], values=[float(sparse[i][k]) for k in sparse[i]]
                    ),
                },
                payload=payload,
            )
        )
    c.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


def rrf_search(query: str, top_k: int) -> list[dict]:
    """Qdrant server-side hybrid: dense prefetch + sparse prefetch, RRF fusion."""
    from retrieval import embed_query

    dense_vec, sparse_vec = embed_query(query)
    c = _client()
    result = c.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=dense_vec, using=DENSE_NAME, limit=settings.retrieval_limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse_vec.keys()), values=list(sparse_vec.values())
                ),
                using=SPARSE_NAME,
                limit=settings.retrieval_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=settings.retrieval_limit,
        with_payload=True,
    )
    hits = []
    for p in result.points:
        payload = p.payload or {}
        hits.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "doc": payload.get("doc"),
                "document_id": payload.get("document_id"),
                "title": payload.get("title"),
                "publisher": payload.get("publisher"),
                "source_url": payload.get("source_url") or payload.get("final_url") or payload.get("original_url"),
                "language": payload.get("language"),
                "country_region": payload.get("country_region"),
                "document_type": payload.get("document_type"),
                "topics": payload.get("topics"),
                "flooring_types": payload.get("flooring_types"),
                "authority_score": payload.get("authority_score"),
                "page": payload.get("page"),
                "pages": payload.get("pages") or [],
                "text": payload.get("text", ""),
                "score": 0.0,
                "rrf": p.score,
            }
        )
    return hits


def list_documents() -> list[dict]:
    """Aggregate chunk counts per document via a payload index."""
    c = _client()
    # ensure a payload index on doc for efficient scrolling
    try:
        c.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="doc",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:  # noqa: BLE001  (already exists)
        pass
    docs: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["doc", "document_id", "title", "publisher", "language"],
            with_vectors=False,
            limit=512,
            offset=offset,
        )
        for point in points or []:
            payload = point.payload or {}
            doc = payload.get("doc")
            if not doc:
                continue
            entry = docs.setdefault(
                doc,
                {
                    "doc": doc,
                    "chunks": 0,
                    "document_id": payload.get("document_id"),
                    "title": payload.get("title"),
                    "publisher": payload.get("publisher"),
                    "language": payload.get("language"),
                },
            )
            entry["chunks"] += 1
        if offset is None:
            break
    return [docs[d] for d in sorted(docs)]


def delete_document(doc: str) -> int:
    """Delete all chunks of a document by its payload; returns deleted count."""
    c = _client()
    info = c.count(
        collection_name=settings.qdrant_collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="doc", match=models.MatchValue(value=doc))]
        ),
        exact=True,
    )
    n = info.count
    c.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc", match=models.MatchValue(value=doc))]
            )
        ),
    )
    return n
