"""The contextual student: hashed word pieces read in order by a small transformer. See ADR-0011."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..data.words import WordBatch, WordSpec, encode_texts
from .student import EPS

MASKED = -1e9


@dataclass(frozen=True, slots=True)
class ContextualConfig:
    num_buckets: int = 8192
    width: int = 128
    layers: int = 4
    heads: int = 4
    max_words: int = 64
    d_out: int = 384
    kind: str = "contextual"

    @property
    def parameters(self) -> int:
        w = self.width
        block = 4 * w * w + (w * 4 * w + 4 * w) + (4 * w * w + w) + 4 * w
        return self.num_buckets * w + self.max_words * w + self.layers * block + 2 * w + w * self.d_out

    def describe(self) -> str:
        return (
            f"contextual buckets={self.num_buckets} width={self.width} layers={self.layers} "
            f"heads={self.heads} d_out={self.d_out} params={self.parameters:,}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def word_spec(self) -> WordSpec:
        return WordSpec(num_buckets=self.num_buckets, max_words=self.max_words)


SIZES = {
    "S": ContextualConfig(num_buckets=8192, width=128),
    "M": ContextualConfig(num_buckets=16384, width=256),
}


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiHeadAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.attention(h, h, h, mask)
        return x + self.down(nn.gelu(self.up(self.norm2(x))))


class ContextualStudent(nn.Module):
    def __init__(self, config: ContextualConfig):
        super().__init__()
        self.config = config
        self.table = nn.Embedding(config.num_buckets, config.width)
        self.position = nn.Embedding(config.max_words, config.width)
        self.blocks = [Block(config.width, config.heads) for _ in range(config.layers)]
        self.norm = nn.LayerNorm(config.width)
        self.out = nn.Linear(config.width, config.d_out, bias=False)

    def __call__(self, grams: mx.array, gram_mask: mx.array, words: mx.array, word_mask: mx.array) -> mx.array:
        pieces = (self.table(grams) * gram_mask[..., None]).sum(axis=1)
        vectors = pieces / mx.maximum(gram_mask.sum(axis=1, keepdims=True), 1.0)
        x = mx.take(vectors, words, axis=0) + self.position(mx.arange(words.shape[1]))
        mask = ((1.0 - word_mask) * MASKED)[:, None, None, :]
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        pooled = (x * word_mask[..., None]).sum(axis=1) / mx.maximum(word_mask.sum(axis=1, keepdims=True), 1.0)
        out = self.out(pooled)
        return out / mx.maximum(mx.linalg.norm(out, axis=-1, keepdims=True), EPS)


def run(model: ContextualStudent, batch: WordBatch) -> mx.array:
    return model(
        mx.array(batch.grams), mx.array(batch.gram_mask), mx.array(batch.words), mx.array(batch.word_mask)
    )


def embed(model: ContextualStudent, texts: list[str], batch_size: int = 256) -> np.ndarray:
    spec = model.config.word_spec()
    chunks = []
    for start in range(0, len(texts), batch_size):
        out = run(model, encode_texts(texts[start : start + batch_size], spec))
        mx.eval(out)
        chunks.append(np.array(out))
    return np.concatenate(chunks, axis=0)
