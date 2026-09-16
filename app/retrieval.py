"""BGE-M3 embedding, Qdrant hybrid search (RRF), Qwen3 reranking, grounded LLM answer."""

import logging
from functools import lru_cache

from config import settings

log = logging.getLogger("docling-rag.retrieval")

SYSTEM_PROMPT = (
    "You answer questions STRICTLY from the provided context chunks. "
    "Rules: (1) Use only information present in the context. "
    "(2) If the context does not contain the answer, say you don't know. "
    "(3) Never use your own training knowledge to fill gaps. "
    "(4) Cite the [doc, chunk] you used."
)


@lru_cache(maxsize=1)
def get_embedder():
    from FlagEmbedding import BGEM3FlagModel

    log.info("loading embedder %s (hf cache %s)", settings.embed_model, settings.hf_cache)
    # fp16 on GPU; on CPU-only hosts fp32 is used (fp16 gives no speedup there)
    use_fp16 = settings.device.startswith("cuda")
    return BGEM3FlagModel(settings.embed_model, use_fp16=use_fp16)


@lru_cache(maxsize=1)
def get_reranker():
    """Dispatches on model family: Qwen3 (causal LM, yes/no logits) vs
    cross-encoders like bge-reranker / mxbai-rerank (sequence classification)."""
    import torch
    from transformers import AutoTokenizer

    model_name = settings.rerank_model
    log.info("loading reranker %s", model_name)
    device = settings.device if settings.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if "qwen3-reranker" in model_name.lower():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        kind = "qwen3"
    else:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
        kind = "cross-encoder"
    model = model.to(device).eval()
    return tok, model, kind, device


def embed_query(query: str):
    m = get_embedder()
    out = m.encode([query], return_sparse=True)
    dense = out["dense_vecs"][0].tolist()
    sparse = {int(k): float(w) for k, w in out["lexical_weights"][0].items()}
    return dense, sparse


def _rerank(query: str, texts: list[str]) -> list[float]:
    """Score query/doc pairs; returns relevance probabilities in [0, 1].

    Two model families are supported:
    - Qwen3-Reranker: causal LM scored via Yes/No token logits (official pattern)
    - standard cross-encoders (bge-reranker, mxbai-rerank, ...): sigmoid over
      the relevance logit — no Yes/No prompt template involved
    """
    tok, model, kind, device = get_reranker()
    import torch

    if kind == "qwen3":
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the query. " \
                 "Only answer Yes or No.<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        pairs = [
            f"{prefix}Query: {query}\nDocument: {t}{suffix}" for t in texts
        ]
        with torch.no_grad():
            inputs = tok(pairs, padding=True, truncation=True, max_length=2048,
                         return_tensors="pt").to(device)
            scores = model(**inputs).logits[:, -1, :]
        yes = tok("Yes", add_special_tokens=False).input_ids[0]
        no = tok("No", add_special_tokens=False).input_ids[0]
        logits = scores[:, [yes, no]].float()
        probs = torch.softmax(logits, dim=-1)[:, 0]
    else:
        pairs = [[query, t] for t in texts]
        with torch.no_grad():
            inputs = tok(pairs, padding=True, truncation=True, max_length=1024,
                         return_tensors="pt").to(device)
            logits = model(**inputs).logits.squeeze(-1).float()
            probs = torch.sigmoid(logits)
    return probs.tolist()


def hybrid_search(question: str, top_k: int = 5, min_score: float = 0.3) -> list[dict]:
    from storage import rrf_search

    hits = rrf_search(question, top_k=top_k)
    if not hits:
        return []
    scores = _rerank(question, [h["text"] for h in hits])
    for h, s in zip(hits, scores):
        h["score"] = s
    hits.sort(key=lambda h: h["score"], reverse=True)
    return [h for h in hits if h["score"] >= min_score][:top_k]


def answer_question(question: str, hits: list[dict]):
    if not hits:
        return None, False, "no sufficiently relevant chunks retrieved; refusing to answer"
    context = "\n\n".join(
        f"[doc={h['doc']}, chunk={h['chunk_id']}, page={h.get('page')}]\n{h['text']}" for h in hits
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer (cite [doc, chunk]):"

    import os

    provider = settings.llm_provider.lower()
    default_headers = None
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            return None, False, "no LLM API key set (OPENROUTER_API_KEY); retrieval-only mode"
        base_url = settings.llm_base_url or "https://openrouter.ai/api/v1"
        # optional attribution headers, appreciated by OpenRouter
        default_headers = {"HTTP-Referer": "https://github.com/smarterworkerai/docling-rag-poc",
                           "X-Title": "docling-rag-poc"}
    elif provider == "zai":
        # z.ai coding plan: OpenAI-compatible, endpoint is plan-scoped
        api_key = os.getenv("ZAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            return None, False, "no LLM API key set (ZAI_API_KEY); retrieval-only mode"
        base_url = settings.llm_base_url or "https://api.z.ai/api/coding/paas/v4"
        default_headers = {"Accept-Language": "en-US,en"}
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            return None, False, "no LLM API key set (OPENAI_API_KEY); retrieval-only mode"
        base_url = settings.llm_base_url or None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.max_answer_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content, True, "ok"
    except Exception as e:  # noqa: BLE001
        log.exception("LLM call failed")
        return None, False, f"LLM call failed: {e}"
