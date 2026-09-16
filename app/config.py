"""Central config, everything overridable via environment (.env)."""

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass
class Settings:
    # --- Qdrant ---
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://qdrant:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "docs"))
    embed_dim: int = field(default_factory=lambda: _int("EMBED_DIM", 1024))  # BGE-M3 dense dim

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))

    # --- Retrieval ---
    retrieval_limit: int = field(default_factory=lambda: _int("RETRIEVAL_LIMIT", 50))  # candidates per leg
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))

    # --- Docling (A4000 Mobile tuned: big GPU batches) ---
    docling_layout_batch_size: int = field(default_factory=lambda: _int("DOCLING_LAYOUT_BATCH_SIZE", 32))
    docling_ocr_batch_size: int = field(default_factory=lambda: _int("DOCLING_OCR_BATCH_SIZE", 32))
    docling_table_batch_size: int = field(default_factory=lambda: _int("DOCLING_TABLE_BATCH_SIZE", 4))
    docling_device: str = field(default_factory=lambda: os.getenv("DOCLING_DEVICE", "auto"))

    # --- Models ---
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    )
    hf_cache: str = field(default_factory=lambda: os.getenv("HF_HOME", "/data/models"))

    # --- LLM (cloud provider; only outbound internet dependency) ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))

    # --- Misc ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_answer_tokens: int = field(default_factory=lambda: _int("MAX_ANSWER_TOKENS", 1024))


settings = Settings()
