#!/usr/bin/env python3
"""Upload a manifest-backed corpus to the Docling RAG API."""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request, error


METADATA_FIELDS = [
    "document_id",
    "title",
    "publisher",
    "original_url",
    "final_url",
    "domain",
    "language",
    "country_region",
    "document_type",
    "flooring_types",
    "topics",
    "authority_score",
    "license_status",
    "copyright_notice",
    "retrieved_at",
    "file_size_bytes",
    "page_count",
    "sha256",
    "requires_ocr",
]


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def encode_multipart(fields, files):
    boundary = "----doclingragmanifestboundary"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")
    for name, filename, content_type, data in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_document(api: str, pdf: Path, metadata: dict, timeout: int):
    body, content_type = encode_multipart(
        {"metadata": json.dumps(metadata, ensure_ascii=False)},
        [("file", pdf.name, "application/pdf", pdf.read_bytes())],
    )
    req = request.Request(
        api.rstrip("/") + "/documents",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def indexed_document_ids(api: str) -> set[str]:
    try:
        with request.urlopen(api.rstrip("/") + "/documents", timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return set()
    return {d.get("document_id") for d in data.get("documents", []) if d.get("document_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", default=None, type=Path, help="root for relative filenames")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--publisher", action="append", default=[])
    ap.add_argument("--document-id", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--skip-indexed", action="store_true")
    args = ap.parse_args()

    root = args.root or args.manifest.resolve().parents[1]
    rows = list(iter_rows(args.manifest))
    if args.publisher:
        wanted = set(args.publisher)
        rows = [r for r in rows if r.get("publisher") in wanted]
    if args.document_id:
        wanted = set(args.document_id)
        rows = [r for r in rows if r.get("document_id") in wanted]
    if args.limit:
        rows = rows[: args.limit]
    if args.skip_indexed:
        indexed = indexed_document_ids(args.api)
        rows = [r for r in rows if r.get("document_id") not in indexed]

    ok = 0
    failed = 0
    for row in rows:
        pdf = root / row["filename"]
        metadata = {k: row.get(k) for k in METADATA_FIELDS if k in row}
        metadata["source_url"] = row.get("final_url") or row.get("original_url")
        try:
            print(f"ingest {row.get('document_id')} {pdf}", flush=True)
            result = post_document(args.api, pdf, metadata, args.timeout)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            ok += 1
        except (OSError, error.URLError, error.HTTPError, TimeoutError) as exc:
            print(f"ERROR {row.get('document_id')} {pdf}: {exc}", file=sys.stderr, flush=True)
            failed += 1
        if args.sleep:
            time.sleep(args.sleep)
    print(f"complete ok={ok} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
