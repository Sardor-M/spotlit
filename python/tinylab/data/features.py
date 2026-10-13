"""Vectorized encoder for bulk training-data prep. Must match tinylab.encoder exactly."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

from ..encoder import FNV_OFFSET_BASIS, FNV_PRIME, SPACE, EncoderSpec, SPEC_V1, normalize

BASIS = np.uint64(FNV_OFFSET_BASIS)
PRIME = np.uint64(FNV_PRIME)
SIGN_BIT = np.uint64(63)


def counts_of(text: str, spec: EncoderSpec = SPEC_V1) -> np.ndarray:
    """Signed bucket counts as a dense int32 vector, same result as encoder.features()."""
    normalized = normalize(text, spec)
    counts = np.zeros(spec.num_buckets, dtype=np.int32)
    if not normalized:
        return counts

    padded = bytes([SPACE]) + normalized + bytes([SPACE])
    data = np.frombuffer(padded, dtype=np.uint8).astype(np.uint64)
    length = data.size

    for size in spec.ngram_sizes:
        if length < size:
            continue
        window = length - size + 1
        digest = np.full(window, BASIS, dtype=np.uint64)
        # One vectorized pass per byte offset; only 3-5 iterations regardless of text length.
        for offset in range(size):
            digest = (digest ^ data[offset : offset + window]) * PRIME
        digest = (digest ^ np.uint64(size)) * PRIME

        buckets = (digest % np.uint64(spec.num_buckets)).astype(np.int64)
        signs = np.where(digest >> SIGN_BIT, -1, 1).astype(np.int32)
        counts += np.bincount(buckets, weights=signs, minlength=spec.num_buckets).astype(np.int32)

    return counts


def sparse_of(text: str, spec: EncoderSpec = SPEC_V1) -> tuple[np.ndarray, np.ndarray]:
    """(buckets, counts) for the non-zero entries."""
    counts = counts_of(text, spec)
    buckets = np.nonzero(counts)[0].astype(np.int32)
    return buckets, counts[buckets]


DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)
# Few distinct batch widths, so MLX reuses buffers instead of caching one per shape.
PAD_MULTIPLE = 128
CACHE_ARRAYS = ("buckets", "counts", "offsets")


def file_digest(path: Path) -> str:
    """Streamed sha256 of a file, so a valid cache never requires loading the texts."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


class FeatureCache:
    """Sparse features of every text in CSR form, memory-mapped from one directory per bucket count."""

    def __init__(self, buckets: np.ndarray, counts: np.ndarray, offsets: np.ndarray):
        self.buckets, self.counts, self.offsets = buckets, counts, offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @classmethod
    def build(
        cls, texts: list[str], spec: EncoderSpec = SPEC_V1, workers: int = DEFAULT_WORKERS
    ) -> "FeatureCache":
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        if spec.num_buckets > 1 << 16:
            raise ValueError("bucket ids are stored as uint16")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            sparse = [
                (buckets.astype(np.uint16), counts.astype(np.int16))
                for buckets, counts in pool.map(partial(sparse_of, spec=spec), texts, chunksize=2048)
            ]
        offsets = np.zeros(len(sparse) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(buckets) for buckets, _ in sparse])
        buckets = np.concatenate([b for b, _ in sparse])
        counts = np.concatenate([c for _, c in sparse])
        return cls(buckets, counts, offsets)

    def save(self, directory: Path, digest: str) -> None:
        """Write to a temporary directory and rename, so an interrupted write never looks valid."""
        directory = Path(directory)
        staging = directory.with_name(directory.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for name in CACHE_ARRAYS:
            np.save(staging / f"{name}.npy", getattr(self, name))
        (staging / "digest").write_text(digest, encoding="utf-8")
        shutil.rmtree(directory, ignore_errors=True)
        staging.rename(directory)

    @classmethod
    def load(cls, directory: Path, digest: str) -> "FeatureCache | None":
        directory = Path(directory)
        marker = directory / "digest"
        if not marker.exists() or marker.read_text(encoding="utf-8") != digest:
            return None
        return cls(*(np.load(directory / f"{name}.npy", mmap_mode="r") for name in CACHE_ARRAYS))

    @classmethod
    def for_file(cls, path: Path, spec: EncoderSpec, workers: int = DEFAULT_WORKERS) -> "FeatureCache":
        """Load the cache for a jsonl of texts, building it once if missing or stale."""
        from .label_with_teacher import load_texts

        path = Path(path)
        directory = path.with_name(f"{path.stem}-features-{spec.num_buckets}")
        digest = file_digest(path)
        cache = cls.load(directory, digest)
        if cache is None:
            print(f"building feature cache {directory} with {workers} workers", flush=True)
            cls.build(load_texts(path), spec, workers).save(directory, digest)
            cache = cls.load(directory, digest)
        return cache

    def sparse(self, rows) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (
                np.asarray(self.buckets[self.offsets[row] : self.offsets[row + 1]], dtype=np.int32),
                np.asarray(self.counts[self.offsets[row] : self.offsets[row + 1]]),
            )
            for row in rows
        ]


def pad_batch(
    sparse: list[tuple[np.ndarray, np.ndarray]], multiple: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad a batch to equal width, rounded up to `multiple`; returns buckets, counts, mask."""
    width = max(1, max(len(buckets) for buckets, _ in sparse))
    width = -(-width // multiple) * multiple
    n = len(sparse)
    out_buckets = np.zeros((n, width), dtype=np.int32)
    out_counts = np.zeros((n, width), dtype=np.float32)
    mask = np.zeros((n, width), dtype=np.float32)
    for row, (buckets, counts) in enumerate(sparse):
        size = len(buckets)
        out_buckets[row, :size] = buckets
        out_counts[row, :size] = counts
        mask[row, :size] = 1.0
    return out_buckets, out_counts, mask
