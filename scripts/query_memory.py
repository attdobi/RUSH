#!/usr/bin/env python3
"""Query the local AI handoff memory embedding index."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "docs" / "ai-handoff" / "memory-embeddings" / "index.jsonl"

ENDPOINT = "http://127.0.0.1:1234/v1"
EMBEDDINGS_URL = f"{ENDPOINT}/embeddings"
MODEL = "text-embedding-embeddinggemma-300m-qat"
DIMS = 768


def embed_query(question: str, timeout: float = 60.0) -> np.ndarray:
    payload = json.dumps({"model": MODEL, "input": question}).encode("utf-8")
    request = urllib.request.Request(
        EMBEDDINGS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"embedding endpoint returned HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(f"embedding endpoint unreachable at {EMBEDDINGS_URL}: {exc}") from exc

    try:
        embedding = json.loads(body)["data"][0]["embedding"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid embedding response from {EMBEDDINGS_URL}: {exc}") from exc

    if len(embedding) != DIMS:
        raise RuntimeError(f"embedding dimension mismatch: expected {DIMS}, got {len(embedding)}")
    return np.asarray(embedding, dtype=np.float32)


def load_index() -> tuple[list[dict[str, object]], np.ndarray]:
    if not INDEX_PATH.exists():
        rel_path = INDEX_PATH.relative_to(REPO_ROOT)
        raise RuntimeError(
            f"missing {rel_path}; run './.venv/bin/python scripts/build_memory_embeddings.py' first"
        )

    records: list[dict[str, object]] = []
    vectors: list[list[float]] = []
    with INDEX_PATH.open("r", encoding="utf-8") as index_file:
        for line_number, line in enumerate(index_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            embedding = record.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != DIMS:
                raise RuntimeError(f"invalid embedding at {INDEX_PATH}:{line_number}")
            records.append(record)
            vectors.append(embedding)

    if not records:
        raise RuntimeError(f"{INDEX_PATH.relative_to(REPO_ROOT)} is empty")

    return records, np.asarray(vectors, dtype=np.float32)


def cosine_scores(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = float(np.linalg.norm(query))
    vector_norms = np.linalg.norm(vectors, axis=1)
    denom = np.maximum(vector_norms * query_norm, 1e-12)
    return (vectors @ query) / denom


def snippet(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 1:
        print("error: --k must be positive", file=sys.stderr)
        return 2

    try:
        records, vectors = load_index()
        query = embed_query(args.question)
        scores = cosine_scores(query, vectors)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    top_indices = np.argsort(scores)[::-1][: args.k]
    for rank, index in enumerate(top_indices, start=1):
        record = records[int(index)]
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        heading = metadata.get("heading") or "(no heading)"
        print(f"{rank}. score={scores[index]:.4f} {record['source_path']} {heading}")
        print(f"   {snippet(str(record['text']))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
