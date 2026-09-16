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
    return BGEM3FlagModel(settings.embed_model, use_fp16=True)


@lru_cache(maxsize=1)
def get_reranker():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = settings.rerank_model
    log.info("loading reranker %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, pad_token=None, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model


def embed_query(query: str):
    m = get_embedder()
    out = m.encode([query], return_sparse=True)
    dense = out["dense_vecs"][0].tolist()
    sparse = {int(k): float(w) for k, w in out["lexical_weights"][0].items()}
    return dense, sparse


def _rerank(query: str, texts: list[str]) -> list[float]:
    """Qwen3-Reranker: yes/no logit scoring (official usage pattern)."""
    tok, model = get_reranker()
    import torch

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the query. " \
             "Only answer Yes or No.<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pairs = [
        f"{prefix}Query: {query}\nDocument: {t}{suffix}" for t in texts
    ]
    with torch.no_grad():
        inputs = tok(pairs, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(model.device)
        scores = model(**inputs).logits[:, -1, :]
    yes = tok("Yes", add_special_tokens=False).input_ids[0]
    no = tok("No", add_special_tokens=False).input_ids[0]
    logits = scores[:, [yes, no]].float()
    probs = torch.softmax(logits, dim=-1)[:, 0]
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

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None, False, "no LLM API key set (OPENAI_API_KEY); retrieval-only mode"
    base_url = settings.llm_base_url or None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
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
