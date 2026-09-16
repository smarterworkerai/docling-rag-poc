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


def upsert_chunks(chunks: list[str], dense: list, sparse: list, doc: str) -> None:
    c = _client()
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc}:{i}")),
            vector={
                DENSE_NAME: dense[i],
                SPARSE_NAME: models.SparseVector(
                    indices=[int(k) for k in sparse[i]], values=[float(sparse[i][k]) for k in sparse[i]]
                ),
            },
            payload={"text": chunks[i], "doc": doc, "chunk_id": i},
        )
        for i in range(len(chunks))
    ]
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
                "page": payload.get("page"),
                "text": payload.get("text", ""),
                "score": 0.0,
                "rrf": p.score,
            }
        )
    return hits
