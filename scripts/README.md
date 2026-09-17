# Scripts

Helper scripts for ingesting the flooring corpus from `../rag_data` into the running Docling RAG API.

## Prerequisites

Start the stack first:

```bash
docker compose up --build -d
```

Recommended GPU-safe runtime settings:

```bash
RERANK_BATCH_SIZE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True docker compose up --build -d
```

## Pilot Ingestion

Indexes a small representative subset and skips documents already present:

```bash
scripts/ingest_flooring_pilot.sh
```

## Full Ingestion

Indexes all documents from `../rag_data/manifests/sources.jsonl` and skips already indexed documents:

```bash
scripts/ingest_flooring_corpus.sh
```

## Custom Paths

```bash
CORPUS_ROOT=/path/to/rag_data API=http://localhost:8000 scripts/ingest_flooring_corpus.sh
```

## Verify

```bash
curl -s http://localhost:8000/documents
```

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Welche Hinweise gibt es zu Parkettklebstoff und Fußbodenheizung?","top_k":5,"min_score":0.0}'
```

See `scripts/INGESTION_COMMANDS.md` for expanded command examples.
