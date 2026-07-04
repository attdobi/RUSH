"""Stratified-by-gold-label sampling coverage (X1)."""
from __future__ import annotations

from collections import Counter

from pipeline.manifest import SampleRecord, select_samples


def _mnist_like_records(per_class: int = 30) -> list[SampleRecord]:
    """Build a 10-class (digits 0-9) dev_golden + holdout pool.

    We reuse the binary ``sme_label`` field as the gold-label stratification
    key; here it carries the digit string so we can assert per-class balance
    exactly the way MNIST would.
    """
    rows: list[SampleRecord] = []
    for split in ("dev_golden", "holdout"):
        for digit in range(10):
            for i in range(per_class):
                sid = f"{split}_{digit}_{i:04d}"
                rows.append(
                    SampleRecord(
                        sample_id=sid,
                        repo_rel_path=f"data/images/mnist/{split}/{digit}/{i}.png",
                        split=split,
                        sme_label_raw="ai_generated",
                        sme_label=str(digit),
                        dataset="mnist",
                        sha256=f"{abs(hash(sid)):064x}"[-64:],
                        sampling_version="mnist-sampling-v1",
                    )
                )
    return rows


def _binary_records(n: int = 40) -> list[SampleRecord]:
    rows: list[SampleRecord] = []
    for i in range(n):
        label = "gen_ai" if i % 2 == 0 else "not_gen_ai"
        sid = f"dev_golden_{i:04d}"
        rows.append(
            SampleRecord(
                sample_id=sid,
                repo_rel_path=f"data/images/x/{i}.jpg",
                split="dev_golden",
                sme_label_raw="ai_generated" if label == "gen_ai" else "not_ai_generated",
                sme_label=label,
                dataset="genai",
                sha256=f"{i:064x}"[-64:],
                sampling_version="genai-gold-sampling-v1",
            )
        )
    return rows


def test_mnist_k20_is_two_per_class():
    rows = _mnist_like_records()
    picked = select_samples(rows, split="dev_golden", limit=20)
    assert len(picked) == 20
    counts = Counter(r.sme_label for r in picked)
    assert set(counts) == {str(d) for d in range(10)}
    assert all(v == 2 for v in counts.values())


def test_mnist_k50_is_five_per_class():
    rows = _mnist_like_records()
    picked = select_samples(rows, split="dev_golden", limit=50)
    counts = Counter(r.sme_label for r in picked)
    assert all(v == 5 for v in counts.values())
    assert len(picked) == 50


def test_split_all_stratifies_each_portion():
    rows = _mnist_like_records()
    picked = select_samples(rows, split="all", limit=20)
    # 20 per split portion => 40 total, each portion 2-per-class.
    assert len(picked) == 40
    for split in ("dev_golden", "holdout"):
        counts = Counter(r.sme_label for r in picked if r.split == split)
        assert all(v == 2 for v in counts.values())


def test_deterministic_and_sorted():
    rows = _mnist_like_records()
    a = select_samples(rows, split="dev_golden", limit=20)
    b = select_samples(rows, split="dev_golden", limit=20)
    ids = [r.sample_id for r in a]
    assert ids == [r.sample_id for r in b]
    assert ids == sorted(ids)


def test_binary_balance_with_remainder():
    rows = _binary_records()
    picked = select_samples(rows, split="dev_golden", limit=5)
    assert len(picked) == 5
    counts = Counter(r.sme_label for r in picked)
    # 2 classes, k=5 -> 3 + 2 (remainder to the lexically-first class).
    assert sorted(counts.values()) == [2, 3]


def test_stratified_can_be_disabled():
    rows = _binary_records()
    picked = select_samples(rows, split="dev_golden", limit=4, stratified=False)
    assert [r.sample_id for r in picked] == [
        "dev_golden_0000",
        "dev_golden_0001",
        "dev_golden_0002",
        "dev_golden_0003",
    ]


def test_backfills_when_class_short():
    # Class "gen_ai" has only 1 row; k=4 across 2 classes wants 2 each.
    rows = [
        SampleRecord(
            sample_id="dev_golden_0000",
            repo_rel_path="a.jpg",
            split="dev_golden",
            sme_label_raw="ai_generated",
            sme_label="gen_ai",
            dataset="genai",
            sha256="0" * 64,
            sampling_version="v1",
        ),
    ] + [
        SampleRecord(
            sample_id=f"dev_golden_{i:04d}",
            repo_rel_path=f"{i}.jpg",
            split="dev_golden",
            sme_label_raw="not_ai_generated",
            sme_label="not_gen_ai",
            dataset="genai",
            sha256=f"{i:064x}"[-64:],
            sampling_version="v1",
        )
        for i in range(1, 6)
    ]
    picked = select_samples(rows, split="dev_golden", limit=4)
    assert len(picked) == 4  # back-filled from the majority class
