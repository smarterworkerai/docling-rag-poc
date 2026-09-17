# Flooring Corpus Ingestion Commands

Run these from the `docling-rag-poc` repository while the Docker stack is running.

## Start / Restart Stack

```bash
docker compose up --build -d
```

For safer GPU memory behavior during heavy ingestion/querying:

```bash
RERANK_BATCH_SIZE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True docker compose up --build -d
```

## Pilot Ingestion

Ingests six representative documents and skips any already indexed documents:

```bash
scripts/ingest_flooring_pilot.sh
```

Equivalent explicit command:

```bash
python3 scripts/ingest_manifest.py \
  --manifest ../rag_data/manifests/sources.jsonl \
  --root ../rag_data \
  --api http://localhost:8000 \
  --skip-indexed \
  --timeout 2400 \
  --document-id uzin_004 \
  --document-id meister_001 \
  --document-id haro_001 \
  --document-id pallmann_001 \
  --document-id berger_seidle_002 \
  --document-id murexin_001
```

## Full Corpus Ingestion

Ingests all documents from `../rag_data/manifests/sources.jsonl`, skipping already indexed documents:

```bash
scripts/ingest_flooring_corpus.sh
```

Equivalent explicit command:

```bash
python3 scripts/ingest_manifest.py \
  --manifest ../rag_data/manifests/sources.jsonl \
  --root ../rag_data \
  --api http://localhost:8000 \
  --skip-indexed \
  --timeout 3600
```

## Background Full Ingestion

```bash
mkdir -p logs
scripts/ingest_flooring_corpus.sh > logs/flooring_ingest.log 2>&1 &
printf '%s\n' "$!" > logs/flooring_ingest.pid
```

Check progress:

```bash
tail -f logs/flooring_ingest.log
curl -s http://localhost:8000/documents
```

## Custom Paths

Override defaults with environment variables:

```bash
CORPUS_ROOT=/path/to/rag_data API=http://localhost:8000 scripts/ingest_flooring_corpus.sh
```

Available variables:

- `CORPUS_ROOT`, default `../rag_data`
- `MANIFEST`, default `$CORPUS_ROOT/manifests/sources.jsonl`
- `API`, default `http://localhost:8000`
- `TIMEOUT`, default `3600` for full ingestion and `2400` for pilot

## Verification

```bash
curl -s http://localhost:8000/documents
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Welche Hinweise gibt es zu Parkettklebstoff und Fußbodenheizung?","top_k":5,"min_score":0.0}'
```
