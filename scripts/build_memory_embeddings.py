#!/usr/bin/env python3
"""Build the local AI handoff memory embedding index."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "ai-handoff" / "memory-embeddings"
INDEX_PATH = OUTPUT_DIR / "index.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# Point at the LM Studio server (Gemma-embedding). Defaults to loopback, but any
# machine — including a networked agent reaching Attila's GPU host — can override
# with RUSH_LOCAL_BASE_URL=http://<host>:1234/v1 (same var the labeling pipeline
# uses in pipeline/providers/registry.py, so the whole repo honors one setting).
ENDPOINT = os.environ.get("RUSH_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
EMBEDDINGS_URL = f"{ENDPOINT}/embeddings"
MODEL = "text-embedding-embeddinggemma-300m-qat"
DIMS = 768

TARGET_CHARS = 1000
MIN_CHARS = 650
OVERLAP_CHARS = 140
BATCH_SIZE = 16
MAX_INDEX_BYTES = 10_000_000

CURATED_PATHS = [
    "docs/ai-handoff/HANDOFF.md",
    "docs/ai-handoff/SESSION-2026-07-05-fable5.md",
    "README.md",
    "docs/architecture.md",
    "docs/architecture-bulk-labeling.md",
    "docs/label-hierarchy.md",
    "docs/DESIGN-per-project-ontology.md",
    "docs/runbook-bulk-labeling.md",
    "docs/mnist-prompt-v0.md",
    "docs/mnist-benchmarks-v0.md",
    "docs/DEMO-REVIEW-2026-07-04.md",
]

CURATED_GLOBS = [
    "policy-graph/MNIST_Digits/v0.1/*.md",
    "policy-graph/Generative_AI/v0.1/*.md",
    # current GenAI policy (v0.3) so semantic recall returns live policy, not just the seed
    "policy-graph/Generative_AI/v0.3/*.md",
]


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def iter_source_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    for rel_path in CURATED_PATHS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            warn(f"missing curated source: {rel_path}")
            continue
        if path not in seen:
            paths.append(path)
            seen.add(path)

    for pattern in CURATED_GLOBS:
        matches = sorted(Path(match) for match in glob.glob(str(REPO_ROOT / pattern)))
        if not matches:
            warn(f"missing curated source glob: {pattern}")
            continue
        for path in matches:
            if path not in seen:
                paths.append(path)
                seen.add(path)

    return paths


def is_markdown_heading(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#") and len(stripped) > 1 and stripped[1] in "# \t"


def markdown_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    paragraph: list[str] = []
    heading: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            block = "\n".join(paragraph).strip()
            if block:
                blocks.append((heading, block))
            paragraph.clear()

    for line in text.splitlines():
        if is_markdown_heading(line):
            flush_paragraph()
            heading = line.strip()
            blocks.append((heading, heading))
            continue
        if line.strip():
            paragraph.append(line.rstrip())
        else:
            flush_paragraph()

    flush_paragraph()
    return blocks


def tail_overlap(text: str) -> str:
    collapsed = text[-OVERLAP_CHARS:].lstrip()
    first_space = collapsed.find(" ")
    if first_space > 0:
        collapsed = collapsed[first_space + 1 :]
    return collapsed.strip()


def split_long_block(block: str) -> list[str]:
    parts: list[str] = []
    start = 0
    while start < len(block):
        end = min(len(block), start + TARGET_CHARS)
        if end < len(block):
            cut = block.rfind(" ", start + MIN_CHARS, end)
            if cut > start:
                end = cut
        part = block[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(block):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return parts


def chunk_markdown(text: str) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    current_parts: list[str] = []
    current_heading: str | None = None

    def current_text() -> str:
        return "\n\n".join(part for part in current_parts if part).strip()

    def emit() -> None:
        nonlocal current_parts
        chunk_text = current_text()
        if not chunk_text:
            return
        chunks.append(
            {
                "text": chunk_text,
                "heading": current_heading,
                "chars": len(chunk_text),
            }
        )
        overlap = tail_overlap(chunk_text)
        current_parts = [overlap] if overlap else []

    for heading, block in markdown_blocks(text):
        if len(block) > TARGET_CHARS:
            if current_text():
                emit()
            current_parts = []
            for part in split_long_block(block):
                chunks.append({"text": part, "heading": heading, "chars": len(part)})
            current_heading = heading
            continue

        next_len = len(current_text()) + len(block) + 2
        if current_text() and next_len > TARGET_CHARS:
            emit()
        if heading is not None:
            current_heading = heading
        current_parts.append(block)

    if current_text():
        emit()

    return chunks


def embed_batch(inputs: list[str], timeout: float = 60.0) -> list[list[float]]:
    payload = json.dumps({"model": MODEL, "input": inputs}).encode("utf-8")
    request = urllib.request.Request(
        EMBEDDINGS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: BaseException | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            embeddings = [item["embedding"] for item in parsed["data"]]
            if len(embeddings) != len(inputs):
                raise RuntimeError(
                    f"embedding response count mismatch: expected {len(inputs)}, got {len(embeddings)}"
                )
            for embedding in embeddings:
                if len(embedding) != DIMS:
                    raise RuntimeError(
                        f"embedding dimension mismatch: expected {DIMS}, got {len(embedding)}"
                    )
            return [[float(value) for value in embedding] for embedding in embeddings]
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"embedding endpoint returned HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid embedding response from {EMBEDDINGS_URL}: {exc}") from exc

        if attempt < 4:
            time.sleep(0.8 * attempt)

    raise RuntimeError(
        f"embedding endpoint unreachable or transiently failing at {EMBEDDINGS_URL}: {last_error}"
    ) from last_error


def build_records() -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    embedded_sources: list[str] = []

    for path in iter_source_paths():
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)
        if not chunks:
            warn(f"no chunks produced for source: {rel_path}")
            continue
        embedded_sources.append(rel_path)
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{rel_path}#{chunk_index}",
                    "source_path": rel_path,
                    "chunk_index": chunk_index,
                    "text": chunk["text"],
                    "embedding": None,
                    "metadata": {
                        "heading": chunk["heading"],
                        "chars": chunk["chars"],
                    },
                }
            )

    if not records:
        raise RuntimeError("no source chunks found; cannot build memory index")

    return records, embedded_sources


def write_outputs(records: list[dict[str, object]], embedded_sources: list[str]) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_PATH.open("w", encoding="utf-8") as index_file:
        for record in records:
            index_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            index_file.write("\n")

    total_bytes = INDEX_PATH.stat().st_size
    if total_bytes > MAX_INDEX_BYTES:
        raise RuntimeError(
            f"{INDEX_PATH.relative_to(REPO_ROOT)} is {total_bytes} bytes, above {MAX_INDEX_BYTES}"
        )

    manifest = {
        "embedding_model": MODEL,
        "dims": DIMS,
        "endpoint": ENDPOINT,
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "chunk_params": {
            "target_chars": TARGET_CHARS,
            "min_chars": MIN_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "batch_size": BATCH_SIZE,
        },
        "source_files": embedded_sources,
        "chunk_count": len(records),
        "total_bytes": total_bytes,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("error: --batch-size must be positive", file=sys.stderr)
        return 2

    try:
        records, embedded_sources = build_records()
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset : offset + args.batch_size]
            embeddings = embed_batch([str(record["text"]) for record in batch])
            for record, embedding in zip(batch, embeddings, strict=True):
                record["embedding"] = embedding
            print(f"embedded {min(offset + len(batch), len(records))}/{len(records)} chunks")

        total_bytes = write_outputs(records, embedded_sources)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {INDEX_PATH.relative_to(REPO_ROOT)} "
        f"({total_bytes} bytes, {len(records)} chunks, {len(embedded_sources)} source files)"
    )
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
