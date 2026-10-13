"""Word-sequence input for the contextual student: hashed pieces per word, in order. See ADR-0011."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np

from ..encoder import EncoderSpec, hash_ngram, normalize
from .features import DEFAULT_WORKERS, file_digest

WHOLE_WORD = 0xFF
SHORT_WIDTH = 16
# Few distinct batch shapes, so MLX reuses buffers instead of caching one per shape.
UNIQUE_MULTIPLE = 1024
CACHE_ARRAYS = ("vocab_grams", "vocab_offsets", "words", "offsets")


@dataclass(frozen=True, slots=True)
class WordSpec:
    num_buckets: int = 8192
    ngram_sizes: tuple[int, ...] = (3, 4, 5)
    max_input_bytes: int = 512
    max_words: int = 64
    max_word_bytes: int = 32
    max_grams: int = 24

    def describe(self) -> str:
        return (
            f"buckets={self.num_buckets} ngrams={','.join(map(str, self.ngram_sizes))} "
            f"max_input_bytes={self.max_input_bytes} max_words={self.max_words} "
            f"max_word_bytes={self.max_word_bytes} max_grams={self.max_grams}"
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "ngram_sizes": list(self.ngram_sizes)}

    @classmethod
    def from_dict(cls, raw: dict) -> "WordSpec":
        return cls(**{**raw, "ngram_sizes": tuple(raw["ngram_sizes"])})


def words_of(text: str, spec: WordSpec) -> list[bytes]:
    normalized = normalize(text, EncoderSpec(max_input_bytes=spec.max_input_bytes))
    return [word[: spec.max_word_bytes] for word in normalized.split(b" ") if word][: spec.max_words]


def word_grams(word: bytes, spec: WordSpec) -> list[int]:
    """Bucket ids of one word: the whole padded word, then its n-grams by size, capped."""
    padded = b" " + word + b" "
    ids = [hash_ngram(padded, WHOLE_WORD) % spec.num_buckets]
    for size in spec.ngram_sizes:
        ids.extend(
            hash_ngram(padded[start : start + size], size) % spec.num_buckets
            for start in range(len(padded) - size + 1)
        )
    return ids[: spec.max_grams]


@dataclass(slots=True)
class WordBatch:
    grams: np.ndarray
    gram_mask: np.ndarray
    words: np.ndarray
    word_mask: np.ndarray


def assemble(
    sequences: list[np.ndarray], grams_of: Callable[[np.ndarray], list], spec: WordSpec
) -> WordBatch:
    """Batch texts given as word keys; each distinct word's pieces are looked up once, slot 0 is padding."""
    lengths = np.array([len(sequence) for sequence in sequences])
    flat = np.concatenate(sequences) if lengths.sum() else np.zeros(0, dtype=np.int64)
    unique, inverse = np.unique(flat, return_inverse=True)

    slots = -(-(len(unique) + 1) // UNIQUE_MULTIPLE) * UNIQUE_MULTIPLE
    grams = np.zeros((slots, spec.max_grams), dtype=np.int32)
    gram_mask = np.zeros((slots, spec.max_grams), dtype=np.float32)
    for slot, ids in enumerate(grams_of(unique), start=1):
        grams[slot, : len(ids)] = ids
        gram_mask[slot, : len(ids)] = 1.0

    width = SHORT_WIDTH if lengths.max(initial=0) <= SHORT_WIDTH else spec.max_words
    words = np.zeros((len(sequences), width), dtype=np.int32)
    word_mask = np.zeros((len(sequences), width), dtype=np.float32)
    rows = np.repeat(np.arange(len(sequences)), lengths)
    cols = np.concatenate([np.arange(n) for n in lengths]) if len(lengths) else np.zeros(0, dtype=np.int64)
    words[rows, cols] = inverse + 1
    word_mask[rows, cols] = 1.0
    return WordBatch(grams, gram_mask, words, word_mask)


def encode_texts(texts: list[str], spec: WordSpec) -> WordBatch:
    """Batch raw texts; the reference path used at evaluation and for held-out monitoring."""
    cached = _cached_grams(spec)
    vocabulary: dict[bytes, int] = {}
    sequences = [
        np.array([vocabulary.setdefault(word, len(vocabulary)) for word in words_of(text, spec)], dtype=np.int64)
        for text in texts
    ]
    by_index = list(vocabulary)
    return assemble(sequences, lambda unique: [cached(by_index[key]) for key in unique], spec)


@lru_cache(maxsize=8)
def _cached_grams(spec: WordSpec) -> Callable[[bytes], list[int]]:
    @lru_cache(maxsize=1 << 18)
    def grams(word: bytes) -> list[int]:
        return word_grams(word, spec)

    return grams


class WordCache:
    """A file's vocabulary with each word's bucket ids, and every text as word indices; memory-mapped."""

    def __init__(self, vocab_grams, vocab_offsets, words, offsets):
        self.vocab_grams, self.vocab_offsets, self.words, self.offsets = vocab_grams, vocab_offsets, words, offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def sequence(self, row: int) -> np.ndarray:
        return np.asarray(self.words[self.offsets[row] : self.offsets[row + 1]], dtype=np.int64)

    def grams(self, word: int) -> np.ndarray:
        return self.vocab_grams[self.vocab_offsets[word] : self.vocab_offsets[word + 1]]

    @classmethod
    def build(cls, texts: list[str], spec: WordSpec, workers: int = DEFAULT_WORKERS) -> "WordCache":
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        with ProcessPoolExecutor(max_workers=workers) as pool:
            sequences = list(pool.map(partial(words_of, spec=spec), texts, chunksize=2048))
        index: dict[bytes, int] = {}
        words = np.fromiter(
            (index.setdefault(word, len(index)) for sequence in sequences for word in sequence), dtype=np.int32
        )
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(sequence) for sequence in sequences])
        del sequences
        with ProcessPoolExecutor(max_workers=workers) as pool:
            grams = list(pool.map(partial(word_grams, spec=spec), list(index), chunksize=4096))
        vocab_offsets = np.zeros(len(grams) + 1, dtype=np.int64)
        vocab_offsets[1:] = np.cumsum([len(ids) for ids in grams])
        vocab_grams = np.fromiter(chain.from_iterable(grams), dtype=np.int32, count=int(vocab_offsets[-1]))
        return cls(vocab_grams, vocab_offsets, words, offsets)

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
    def load(cls, directory: Path, digest: str) -> "WordCache | None":
        marker = Path(directory) / "digest"
        if not marker.exists() or marker.read_text(encoding="utf-8") != digest:
            return None
        return cls(*(np.load(Path(directory) / f"{name}.npy", mmap_mode="r") for name in CACHE_ARRAYS))

    @classmethod
    def for_file(cls, path: Path, spec: WordSpec, workers: int = DEFAULT_WORKERS) -> "WordCache":
        """Load the cache for a jsonl of texts, building it once if missing, stale or for another spec."""
        from .label_with_teacher import load_texts

        path = Path(path)
        directory = path.with_name(f"{path.stem}-words-{spec.num_buckets}")
        digest = f"{file_digest(path)} {spec.describe()}"
        cache = cls.load(directory, digest)
        if cache is None:
            print(f"building word cache {directory} with {workers} workers", flush=True)
            cls.build(load_texts(path), spec, workers).save(directory, digest)
            cache = cls.load(directory, digest)
        return cache
