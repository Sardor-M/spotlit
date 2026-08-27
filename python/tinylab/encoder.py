"""Tokenizer-free input encoding: text to hashed byte n-gram features. See ADR-0005.

Frozen contract. Must stay bit-identical to cpp/src/encoder.cpp and the browser port.
"""

from __future__ import annotations

from dataclasses import dataclass

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
UINT64_MASK = 0xFFFFFFFFFFFFFFFF

SPACE = 0x20


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    version: int = 1
    num_buckets: int = 4096
    ngram_sizes: tuple[int, ...] = (3, 4, 5)
    max_input_bytes: int = 512

    def describe(self) -> str:
        sizes = ",".join(str(n) for n in self.ngram_sizes)
        return (
            f"buckets={self.num_buckets} ngrams={sizes} "
            f"max_input_bytes={self.max_input_bytes}"
        )


SPEC_V1 = EncoderSpec()


def normalize(text: str, spec: EncoderSpec = SPEC_V1) -> bytes:
    """UTF-8 bytes, ASCII-lowercased, non-alphanumeric collapsed to single spaces.

    ASCII-only lowercasing is deliberate: Unicode case folding differs between Python
    and C++ and would silently desynchronize the two encoders.
    """
    out = bytearray()
    previous_was_space = True
    for byte in text.encode("utf-8"):
        if 0x41 <= byte <= 0x5A:
            current = byte + 0x20
        elif 0x61 <= byte <= 0x7A or 0x30 <= byte <= 0x39 or byte >= 0x80:
            current = byte
        else:
            current = SPACE

        if current == SPACE:
            if previous_was_space:
                continue
            previous_was_space = True
        else:
            previous_was_space = False
        out.append(current)

    if out and out[-1] == SPACE:
        out.pop()
    return bytes(out[: spec.max_input_bytes])


def hash_ngram(gram: bytes, size: int) -> int:
    """FNV-1a 64, with the n-gram length mixed in so sizes cannot collide."""
    digest = FNV_OFFSET_BASIS
    for byte in gram:
        digest = ((digest ^ byte) * FNV_PRIME) & UINT64_MASK
    return ((digest ^ size) * FNV_PRIME) & UINT64_MASK


def features(text: str, spec: EncoderSpec = SPEC_V1) -> list[tuple[int, int]]:
    """Sorted (bucket, signed count) pairs; integers only, so parity is exact."""
    normalized = normalize(text, spec)
    if not normalized:
        return []

    padded = bytes([SPACE]) + normalized + bytes([SPACE])
    counts = [0] * spec.num_buckets
    for size in spec.ngram_sizes:
        for start in range(len(padded) - size + 1):
            digest = hash_ngram(padded[start : start + size], size)
            bucket = digest % spec.num_buckets
            counts[bucket] += -1 if digest >> 63 else 1

    return [(bucket, count) for bucket, count in enumerate(counts) if count]


def dense(text: str, spec: EncoderSpec = SPEC_V1) -> list[int]:
    """Full bucket vector. Convenience for inspection; training uses features()."""
    vector = [0] * spec.num_buckets
    for bucket, count in features(text, spec):
        vector[bucket] = count
    return vector
