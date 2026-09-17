#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_ROOT="${CORPUS_ROOT:-$ROOT/../rag_data}"
MANIFEST="${MANIFEST:-$CORPUS_ROOT/manifests/sources.jsonl}"
API="${API:-http://localhost:8000}"

cd "$ROOT"

python3 scripts/ingest_manifest.py \
  --manifest "$MANIFEST" \
  --root "$CORPUS_ROOT" \
  --api "$API" \
  --skip-indexed \
  --timeout "${TIMEOUT:-2400}" \
  --document-id uzin_004 \
  --document-id meister_001 \
  --document-id haro_001 \
  --document-id pallmann_001 \
  --document-id berger_seidle_002 \
  --document-id murexin_001
