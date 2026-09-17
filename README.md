# Docling RAG PoC

Local, document-grounded Q&A with hybrid retrieval and reranking — tuned for a
laptop-class workstation (RTX A4000 Mobile 8 GB, 32 GB RAM). Everything runs
locally except the final LLM call, which uses a cloud provider API.

## What's inside

- **Docling** (MIT) — layout-aware parsing of PDF, DOCX, PPTX, XLSX, HTML, MD
- **BGE-M3** (MIT) — one model produces dense + sparse (lexical) embeddings → true hybrid search
- **Qdrant** (Apache-2.0) — vector DB with named dense + sparse vectors and server-side RRF fusion
- **Qwen3-Reranker-0.6B** (Apache-2.0) — second-stage reranking of retrieved candidates
- **Cloud LLM via API** — grounded answer generation, instructed to answer only from context and refuse otherwise (no internet knowledge)

## Architecture

Two Docker services; the model stages run in-process inside the `rag` service
(on a single 8 GB GPU they share VRAM, so one process controls when each model
holds memory):

```
                        ┌───────────────────────────── rag service (FastAPI, 1 GPU process) ─────────────────────────────┐
                        │                                                                                                  │
 PDF/DOCX/… ──upload──► │  Docling ──► chunker ──► BGE-M3 ──┐                                                            │
                        │  (parse,    (page-     (embed:    │                                                            │
                        │   GPU OCR   provenant  dense +    │ upsert                                                      │
                        │   + layout) chunks)    sparse)    ▼                                                            │
                        │                            ┌──────────────┐                                                    │
                        │  originals kept ─────────► │   Qdrant     │◄──────────────────────────────┐                   │
                        │  (/data/files, for         │  (separate   │  hybrid search: dense + sparse │                   │
                        │   file.pdf#page=N links)   │  service)    │  legs fused via RRF           │                   │
                        │                            └──────┬───────┘                                │                   │
                        │                                   │ top-50 candidates                     │                   │
                        │                                   ▼                                       │                   │
                        │                            Qwen3-Reranker ── all scores < 0.3? ──► refuse (no LLM call)      │
                        │                                   │ top-k relevant chunks                 │                   │
                        │                                   ▼                                       │                   │
                        │                            cloud LLM API (OpenAI / OpenRouter / z.ai / any compatible)       │
                        │                                   │                                                          │
 chat UI (built-in) ◄───┤                                   ▼                                                          │
 citations link to ─────┼────────── file.pdf#page=N ◄──── answer with [doc, chunk, page] citations                        │
 the original PDF page  │                                                                                                  │
                        └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

What each component does:

- **Docling** — layout-aware document parsing (GPU): OCR, reading order, tables.
  Produces text items that each know their source **page**.
- **Chunker** — groups paragraphs into ~800-char chunks, carrying their page
  numbers through.
- **BGE-M3** — embedding model; one pass yields a dense vector (1024d, meaning)
  and a sparse/lexical vector (exact terms) — the basis of hybrid search.
- **Qdrant** — vector database (the second Docker service). Stores both vector
  types per chunk and fuses dense+sparse search results server-side (RRF).
- **Qwen3-Reranker** — second-stage scorer over the fused top-50; keeps only
  chunks genuinely relevant to the question. Doubles as the **refusal gate**:
  if nothing scores ≥ 0.3, the LLM is never called.
- **Cloud LLM API** — the only internet dependency. Writes the final answer
  strictly from the surviving chunks, citing `[doc, chunk, page]`.
- **Web UI** — served by the same service: ingest via drag&drop, document list,
  chat; citations deep-link into the retained original PDF (`file.pdf#page=N`).

VRAM budget on 8 GB: BGE-M3 ≈ 1.1 GB + reranker ≈ 1.3 GB + Docling batches ≈ 3–4 GB
(released after parsing so query-time models have room).

## Quickstart

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY (mandatory), everything else has working defaults
docker compose up --build
```

First start downloads ~3 GB of models (BGE-M3, Qwen3-Reranker, Docling layout
models) into the `model-cache` volume; the container health check allows 30 min
for this. Subsequent starts are fast.

API then available at `http://localhost:8000` (docs at `/docs`),
Qdrant dashboard at `http://localhost:6333/dashboard`.

## Use it

### Web UI

Open `http://localhost:8000/` — a built-in chat UI:

- **left sidebar:** drag&drop (or click) files to ingest, live ingest status, list of ingested documents with chunk counts, per-document delete
- **chat:** ask questions, answers come back with clickable source citations (hover shows the chunk text); if retrieval finds nothing relevant the bot refuses instead of guessing

### API

```bash
# ingest a document
curl -s -X POST http://localhost:8000/documents -F "file=@some.pdf"

# list ingested documents
curl -s http://localhost:8000/documents

# delete a document (and its chunks) from the index
curl -s -X DELETE "http://localhost:8000/documents?doc=some.pdf"

# retrieval only (no LLM): hybrid search + rerank, ranked chunks
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is said about warranty?", "top_k": 5}'

# grounded Q&A with citations
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is said about warranty?"}'
```

Response shape of `/query`:

```json
{
  "answer": "...",
  "answered": true,
  "reason": "ok",
  "citations": [{"doc": "some.pdf", "chunk": 42, "score": 0.87, "text": "..."}]
}
```

If nothing relevant is retrieved (`min_score` not met), the service **refuses**
to answer rather than letting the LLM improvise from its training data.

## .env reference

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `openrouter`, or `zai` |
| `OPENAI_API_KEY` | — | **Required when `LLM_PROVIDER=openai`.** Cloud LLM key (the only internet dependency) |
| `OPENROUTER_API_KEY` | — | **Required when `LLM_PROVIDER=openrouter`.** Key from openrouter.ai/keys; then set `LLM_MODEL` to the prefixed name, e.g. `anthropic/claude-sonnet-4` |
| `ZAI_API_KEY` | — | **Required when `LLM_PROVIDER=zai`.** GLM Coding Plan key; base URL is set automatically to the coding endpoint. `LLM_MODEL` e.g. `glm-4.7` / `glm-5.3` |
| `LLM_BASE_URL` / `LLM_API_KEY` | empty | Any other OpenAI-compatible endpoint (Groq, vLLM, Ollama, …) |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model name |
| `RAG_PORT` / `QDRANT_PORT` | 8000 / 6333 | Host ports for API and Qdrant dashboard |
| `QDRANT_COLLECTION` | `docs` | Qdrant collection name |
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model (dense+sparse in one pass) |
| `RERANK_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Second-stage reranker |
| `EMBED_DIM` | 1024 | Dense vector dimension of `EMBED_MODEL` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Chunker settings (characters) |
| `RETRIEVAL_LIMIT` | 50 | Candidates fetched per retrieval leg before reranking |
| `DOCLING_LAYOUT_BATCH_SIZE` | 32 | Docling GPU batch — **raised from default 4** for A4000 Mobile |
| `DOCLING_OCR_BATCH_SIZE` | 32 | Docling OCR GPU batch — **raised from default 4** |
| `DOCLING_TABLE_BATCH_SIZE` | 4 | Table structure batch (not GPU-batched yet upstream) |
| `DOCLING_DEVICE` | `auto` | `auto` = CUDA when visible; `cpu` forces CPU parsing |
| `MAX_ANSWER_TOKENS` | 1024 | LLM answer length cap |
| `LOG_LEVEL` | `INFO` | App logging |

### Tuning notes

- **OOM during ingestion** → lower `DOCLING_LAYOUT_BATCH_SIZE`/`DOCLING_OCR_BATCH_SIZE` (32 → 16 → 8).
- **16 GB GPU** → swap `RERANK_MODEL=Qwen/Qwen3-Reranker-4B` for better ranking quality.
- **CPU-only host** → don't edit anything; use the CPU overlay instead (see
  "Running on a weak CPU-only machine" below). Expect ~10 pages/min parsing
  instead of ~1–2 pages/s.
- **Ingestion speed**: a mixed 300-page PDF takes roughly 5–10 min on GPU with these batch sizes.

## Running on a weak CPU-only machine (after embedding on the GPU box)

Once your corpus is embedded (Qdrant holds the index), you can move the whole
stack to a small fanless/office machine — 4 GB RAM, no GPU — and keep querying
it. Queries still need the embedder + reranker at *query time* (one sentence +
top-50 chunks), which runs on CPU in ~1–3 s at this scale. Ingestion on such a
box also works but is slow (~5–10 pages/min parsing); the intended flow is
ingest on the GPU machine, query anywhere.

The repo ships a CPU overlay for exactly this:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build -d
```

What the overlay (`docker-compose.cpu.yml`) changes:

- **No GPU reservation** — runs anywhere
- **`RERANK_MODEL=BAAI/bge-reranker-base`** — a ~110 MB cross-encoder instead of
  Qwen3-Reranker-0.6B (~2.4 GB fp32, too heavy next to BGE-M3 on 4 GB RAM).
  Ranking quality drops slightly vs Qwen3; the refusal threshold (`min_score`)
  may need recalibration because score distributions differ between reranker
  families — if valid questions get refused, lower it; if junk gets answered, raise it.
- **`DEVICE=cpu`** — embedder/reranker load fp32 without probing CUDA, and BGE-M3
  skips fp16 (no benefit on CPU)
- **Docling batches back to 4** — irrelevant without a GPU
- **`Dockerfile.cpu` + `requirements-cpu.txt`** — plain torch wheel, ~2 GB smaller
  image than the CUDA build

Both reranker families are auto-detected (`app/retrieval.py`): Qwen3 models use
their Yes/No-logit pattern, everything else (bge-reranker, mxbai-rerank, …) is
loaded as a standard cross-encoder with sigmoid scoring — so
`RERANK_MODEL=mixedbread/mxbai-rerank-xsmall` etc. also work with no code change.

### Migrating between machines (ingest on GPU box → query on weak box)

Only two volumes hold state; everything else regenerates. **Stop the stack on
both machines** (`docker compose stop`) while copying — Qdrant keeps state in
memory and a live copy can be inconsistent. Check the exact volume name first
with `docker volume ls | grep qdrant` (it is `<project-dir>_qdrant-data`, so it
differs if the checkout directory is named differently).

**1. Qdrant volume — mandatory. This IS your index**: all chunks, dense +
sparse vectors and payloads. The original PDFs are not needed for querying.

```bash
# on the ingestion (GPU) machine
docker run --rm -v docling-rag-poc_qdrant-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/qdrant-snapshot.tgz -C /data .

# after copying qdrant-snapshot.tgz to the target machine:
docker compose up -d qdrant                       # start the empty qdrant first
docker run --rm -v docling-rag-poc_qdrant-data:/data -v $(pwd):/backup alpine \
  sh -c "cd /data && tar xzf /backup/qdrant-snapshot.tgz"
docker compose restart qdrant
```

**2. Model-cache volume — optional** (~3 GB; skips the first-start model
download). Same commands with `docling-rag-poc_model-cache` instead of
`qdrant-data`. If you skip it, models re-download automatically. Note: when the
target uses a different reranker than the source (e.g. CPU overlay's
bge-reranker-base vs the GPU default Qwen3-0.6B), the cache holds whichever
models were loaded there — missing ones simply download on first use.

**3. Code + config:** `git clone` + `docker compose build` (plus
`-f docker-compose.cpu.yml` on the CPU target). Recreate `.env` from
`.env.example` — don't blindly copy it across profiles, since the reranker
default differs between GPU and CPU setups.

**Sanity check after migration:**

```bash
curl -s http://localhost:8000/documents        # should list all docs + chunk counts
curl -s -X POST http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"question": "<something you know is in the docs>"}'
```

`/search` is the cleanest check — it exercises embedding + Qdrant + reranker
without spending an LLM call. If it returns hits, the migration succeeded.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | built-in web UI (chat, ingest, document list) |
| GET | `/health` | liveness |
| POST | `/documents` | upload + ingest a file (multipart) |
| GET | `/documents` | list ingested documents with chunk counts |
| DELETE | `/documents?doc=…` | remove a document and all its chunks |
| POST | `/search` | hybrid + rerank, returns chunks (no LLM call) |
| POST | `/query` | full pipeline: retrieval + grounded answer + citations |

Interactive OpenAPI docs: `http://localhost:8000/docs`.

## Licenses of components

Docling MIT · BGE-M3 MIT · Qdrant Apache-2.0 · Qwen3-Reranker Apache-2.0 ·
this repo MIT.
