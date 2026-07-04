#!/usr/bin/env python3
"""Compare MNIST decision quality for local Gemma single vs batched calls.

The runner now keeps local models single-image for normal runs, but this script
intentionally probes the local OpenAI-compatible client both ways so we can
choose the web default for non-local API batching with a concrete A/B signal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import queue
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.manifest import load_policy_markdown  # noqa: E402
from pipeline.providers.base import LabelRequest, LabelResponse  # noqa: E402
from pipeline.providers.registry import build_client, local_base_url  # noqa: E402
from pipeline.scoring.decision_quality_multiclass import compute_multiclass_metrics  # noqa: E402
from pipeline.scoring.tasks import MNIST_MULTICLASS  # noqa: E402

MODEL_ID = "local/gemma-4-26b-a4b-qat"
DEFAULT_BATCH_SIZE = 5
DEFAULT_N = 20


@dataclass(frozen=True)
class GoldImage:
    sample_id: str
    image_path: Path
    truth: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_gold_images(n: int) -> list[GoldImage]:
    """Load a balanced-ish MNIST gold slice from the sampled CSV manifests."""
    manifest_dir = ROOT / "data" / "images" / "mnist-classification" / "manifests"
    rows = _read_csv(manifest_dir / "val_labels.csv") + _read_csv(manifest_dir / "train_labels.csv")
    by_digit: dict[str, list[dict[str, str]]] = {str(d): [] for d in range(10)}
    for row in rows:
        label = str(row.get("label", "")).strip()
        image_path = ROOT / row.get("repo_rel_path", "")
        if label in by_digit and image_path.is_file():
            by_digit[label].append(row)

    picked: list[dict[str, str]] = []
    while len(picked) < n:
        changed = False
        for digit in map(str, range(10)):
            if len(picked) >= n:
                break
            bucket = by_digit[digit]
            if bucket:
                picked.append(bucket.pop(0))
                changed = True
        if not changed:
            break

    if not picked:
        raise FileNotFoundError(
            "No MNIST gold images found under data/images/mnist-classification; "
            "run scripts/sample_mnist_gold_sets.py first."
        )

    return [
        GoldImage(
            sample_id=row["sample_id"],
            image_path=ROOT / row["repo_rel_path"],
            truth=str(row["label"]).strip(),
        )
        for row in picked
    ]


def local_gemma_available(timeout_s: float) -> tuple[bool, str]:
    url = local_base_url().rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            if 200 <= response.status < 300:
                return True, f"{url} responded HTTP {response.status}"
            return False, f"{url} responded HTTP {response.status}"
    except (OSError, urllib.error.URLError) as exc:
        return False, f"{url} unavailable: {type(exc).__name__}: {exc}"


def make_requests(images: Iterable[GoldImage]) -> list[LabelRequest]:
    policy_markdown = load_policy_markdown(ROOT / "policy-graph" / "MNIST_Digits" / "v0.1")
    return [
        LabelRequest(
            image_path=image.image_path,
            image_id=image.sample_id,
            policy_markdown=policy_markdown,
            policy_graph_version="MNIST_Digits.v0.1",
            prompt_version="v0.1",
            model_id=MODEL_ID,
            area="MNIST_Digits",
        )
        for image in images
    ]


def chunked(items: list[LabelRequest], size: int) -> Iterable[list[LabelRequest]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def live_predictions(requests: list[LabelRequest], *, batch_size: int) -> tuple[list[str], list[str]]:
    client = build_client(MODEL_ID)
    responses: list[LabelResponse] = []
    if batch_size == 1:
        responses = [client.label(request) for request in requests]
    else:
        for batch in chunked(requests, batch_size):
            responses.extend(client.batch_label(batch))

    labels: list[str] = []
    errors: list[str] = []
    for response in responses:
        labels.append(str(response.label or "abstain").strip().lower())
        if response.error:
            errors.append(f"{response.image_id}: {response.error}")
    return labels, errors


def fallback_predictions(images: list[GoldImage]) -> tuple[list[str], list[str]]:
    """Deterministic illustrative predictions when local Gemma is offline."""
    single: list[str] = []
    batched: list[str] = []
    for image in images:
        digest = hashlib.sha256(image.sample_id.encode("utf-8")).hexdigest()
        pred = image.truth
        if int(digest[:2], 16) % 23 == 0:
            pred = str((int(image.truth) + 1) % 10)
        single.append(pred)
        batched.append(pred)
    return single, batched


def _live_worker(
    requests: list[LabelRequest],
    batch_size: int,
    out: mp.Queue,
) -> None:
    try:
        single, single_errors = live_predictions(requests, batch_size=1)
        batched, batched_errors = live_predictions(requests, batch_size=batch_size)
        out.put(("ok", single, batched, single_errors + batched_errors))
    except BaseException as exc:  # noqa: BLE001 - child reports and exits
        out.put(("error", f"{type(exc).__name__}: {exc}"))


def live_predictions_with_deadline(
    requests: list[LabelRequest],
    *,
    batch_size: int,
    deadline_s: int,
) -> tuple[list[str], list[str], list[str]]:
    out: mp.Queue = mp.Queue()
    proc = mp.Process(target=_live_worker, args=(requests, batch_size, out), daemon=True)
    proc.start()
    try:
        status, *payload = out.get(timeout=deadline_s if deadline_s > 0 else None)
    except queue.Empty as exc:
        proc.terminate()
        proc.join(timeout=2)
        raise TimeoutError(f"live Gemma A/B exceeded {deadline_s}s deadline") from exc
    finally:
        if proc.is_alive():
            proc.join(timeout=2)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
    if status == "ok":
        single, batched, errors = payload
        return single, batched, errors
    raise RuntimeError(str(payload[0] if payload else "unknown live worker error"))


def accuracy_report(predictions: list[str], truths: list[str]) -> dict[str, object]:
    metrics = compute_multiclass_metrics(
        predictions,
        truths,
        classes=MNIST_MULTICLASS.classes,
        abstain=MNIST_MULTICLASS.abstain,
    )
    correct = sum(1 for pred, truth in zip(predictions, truths) if pred == truth)
    return {
        "correct": correct,
        "total": len(truths),
        "accuracy": metrics["accuracy"],
        "n_decided": metrics["n"],
        "n_abstained": metrics["n_abstained"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="number of MNIST gold images to compare")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="batched images per request")
    parser.add_argument("--probe-timeout-s", type=float, default=2.0, help="local /v1/models probe timeout")
    parser.add_argument(
        "--live-deadline-s",
        type=int,
        default=90,
        help="maximum seconds to spend on live local Gemma before deterministic fallback",
    )
    parser.add_argument("--json", action="store_true", help="also print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be >= 2 for the A/B batched arm")

    images = load_gold_images(args.n)
    truths = [image.truth for image in images]
    available, availability_note = local_gemma_available(args.probe_timeout_s)
    live = False
    errors: list[str] = []

    if available:
        try:
            requests = make_requests(images)
            single, batched, errors = live_predictions_with_deadline(
                requests,
                batch_size=args.batch_size,
                deadline_s=args.live_deadline_s,
            )
            live = not errors
            if errors:
                single, batched = fallback_predictions(images)
                availability_note = (
                    f"{availability_note}; provider returned errors, using deterministic fallback"
                )
        except (RuntimeError, TimeoutError) as exc:
            single, batched = fallback_predictions(images)
            availability_note = f"{availability_note}; {exc}, using deterministic fallback"
    else:
        single, batched = fallback_predictions(images)

    single_report = accuracy_report(single, truths)
    batched_report = accuracy_report(batched, truths)
    flips = sum(1 for a, b in zip(single, batched) if a != b)
    single_acc = float(single_report["accuracy"] or 0.0)
    batched_acc = float(batched_report["accuracy"] or 0.0)
    accuracy_delta = batched_acc - single_acc
    recommended_default = args.batch_size if abs(accuracy_delta) <= (1 / len(truths)) and flips <= 1 else 1

    payload = {
        "model": MODEL_ID,
        "mode": "live" if live else "fallback",
        "availability_note": availability_note,
        "n_images": len(images),
        "batch_size": args.batch_size,
        "single": single_report,
        "batched": batched_report,
        "label_flips": flips,
        "accuracy_delta_batched_minus_single": round(accuracy_delta, 6),
        "recommended_default_images_per_call": recommended_default,
        "sample_ids": [image.sample_id for image in images],
    }

    mode_text = "LIVE local Gemma" if live else "FALLBACK deterministic (illustrative)"
    print("MNIST local Gemma batching A/B")
    print(f"mode: {mode_text}")
    print(f"note: {availability_note}")
    if errors:
        print(f"provider_errors: {len(errors)}; first={errors[0]}")
    print(f"images: {len(images)}")
    print(
        "single_image: "
        f"{single_report['correct']}/{single_report['total']} correct, "
        f"accuracy={float(single_report['accuracy'] or 0.0):.3f}, "
        f"abstained={single_report['n_abstained']}"
    )
    print(
        f"batched_{args.batch_size}: "
        f"{batched_report['correct']}/{batched_report['total']} correct, "
        f"accuracy={float(batched_report['accuracy'] or 0.0):.3f}, "
        f"abstained={batched_report['n_abstained']}"
    )
    print(f"label_flips: {flips}")
    print(f"accuracy_delta_batched_minus_single: {accuracy_delta:+.3f}")
    print(f"recommended_default_images_per_call: {recommended_default}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
