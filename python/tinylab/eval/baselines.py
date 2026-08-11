"""Baselines to beat: random floor, lexical overlap, BM25 incumbent."""

from __future__ import annotations

import math
import random
import re
from collections import Counter

from .schema import Doc

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs; no stemming, no stopwords."""
    return TOKEN_RE.findall(text.lower())


def _rank_by_score(scores: dict[str, float], top_k: int) -> list[str]:
    """Score descending, doc_id ascending, so results never depend on corpus order."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, score in ordered[:top_k] if score > 0.0]


class RandomRetriever:
    def __init__(self, seed: int = 0) -> None:
        self.name = f"random(seed={seed})"
        self.seed = seed
        self._doc_ids: list[str] = []

    def index(self, docs: list[Doc]) -> None:
        self._doc_ids = [doc.doc_id for doc in docs]

    def rank(self, query_text: str, top_k: int) -> list[str]:
        rng = random.Random(f"{self.seed}:{query_text}")
        shuffled = list(self._doc_ids)
        rng.shuffle(shuffled)
        return shuffled[:top_k]


class LexicalOverlapRetriever:
    name = "lexical-overlap"

    def __init__(self) -> None:
        self._sets: dict[str, set[str]] = {}

    def index(self, docs: list[Doc]) -> None:
        self._sets = {doc.doc_id: set(tokenize(doc.text)) for doc in docs}

    def rank(self, query_text: str, top_k: int) -> list[str]:
        query_tokens = set(tokenize(query_text))
        if not query_tokens:
            return []
        scores = {
            doc_id: len(query_tokens & tokens) / len(query_tokens | tokens)
            for doc_id, tokens in self._sets.items()
        }
        return _rank_by_score(scores, top_k)


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.name = f"bm25(k1={k1},b={b})"
        self.k1 = k1
        self.b = b
        self._tf: dict[str, Counter[str]] = {}
        self._length: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_length = 0.0

    def index(self, docs: list[Doc]) -> None:
        self._tf = {doc.doc_id: Counter(tokenize(doc.text)) for doc in docs}
        self._length = {doc_id: sum(tf.values()) for doc_id, tf in self._tf.items()}
        self._avg_length = sum(self._length.values()) / len(self._length)

        document_frequency: Counter[str] = Counter()
        for tf in self._tf.values():
            document_frequency.update(tf.keys())

        n_docs = len(self._tf)
        self._idf = {
            term: max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0))
            for term, df in document_frequency.items()
        }

    def rank(self, query_text: str, top_k: int) -> list[str]:
        query_terms = tokenize(query_text)
        scores: dict[str, float] = {}
        for doc_id, tf in self._tf.items():
            length_norm = self.k1 * (
                1 - self.b + self.b * self._length[doc_id] / self._avg_length
            )
            score = 0.0
            for term in query_terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                weight = frequency * (self.k1 + 1) / (frequency + length_norm)
                score += self._idf.get(term, 0.0) * weight
            if score:
                scores[doc_id] = score
        return _rank_by_score(scores, top_k)


REGISTRY = {
    "random": RandomRetriever,
    "lexical-overlap": LexicalOverlapRetriever,
    "bm25": BM25Retriever,
}
