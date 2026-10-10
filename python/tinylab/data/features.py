"""Vectorized encoder for bulk training-data prep. Must match tinylab.encoder exactly."""

from __future__ import annotations

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


def pad_batch(
    sparse: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad a batch to equal width; returns buckets, counts and a 0/1 mask."""
    width = max(1, max(len(buckets) for buckets, _ in sparse))
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
