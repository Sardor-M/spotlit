"""Dense and hybrid retrievers. Needs the `teacher` extra; see ADR-0003."""

from __future__ import annotations

from .baselines import BM25Retriever
from .schema import Doc

DEFAULT_TEACHER = "BAAI/bge-small-en-v1.5"
RRF_K = 60
FUSION_DEPTH = 100

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Cosine similarity over sentence-embedding vectors."""

    def __init__(self, model_name: str = DEFAULT_TEACHER, query_prefix: str = QUERY_PREFIX):
        self.name = f"dense({model_name})"
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = None
        self._doc_ids: list[str] = []
        self._matrix = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, docs: list[Doc]) -> None:
        model = self._load()
        self._doc_ids = [doc.doc_id for doc in docs]
        self._matrix = model.encode(
            [doc.text for doc in docs],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def rank(self, query_text: str, top_k: int) -> list[str]:
        model = self._load()
        vector = model.encode(
            [self.query_prefix + query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        scores = self._matrix @ vector
        ordered = sorted(
            zip(self._doc_ids, scores.tolist()), key=lambda item: (-item[1], item[0])
        )
        return [doc_id for doc_id, _ in ordered[:top_k]]


class RRFRetriever:
    """Reciprocal rank fusion, the shape Lumen's retrieval already uses."""

    def __init__(self, retrievers: list, label: str, k: int = RRF_K):
        self.name = f"rrf(k={k}, {label})"
        self.retrievers = retrievers
        self.k = k

    def index(self, docs: list[Doc]) -> None:
        for retriever in self.retrievers:
            retriever.index(docs)

    def rank(self, query_text: str, top_k: int) -> list[str]:
        scores: dict[str, float] = {}
        for retriever in self.retrievers:
            ranking = retriever.rank(query_text, max(top_k, FUSION_DEPTH))
            for position, doc_id in enumerate(ranking, 1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + position)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [doc_id for doc_id, _ in ordered[:top_k]]


SYSTEMS = ("teacher", "hybrid")


def build(system: str, model_name: str = DEFAULT_TEACHER):
    if system == "teacher":
        return DenseRetriever(model_name)
    if system == "hybrid":
        return RRFRetriever([BM25Retriever(), DenseRetriever(model_name)], label="bm25+dense")
    raise ValueError(f"unknown dense system {system!r}; known: {', '.join(SYSTEMS)}")
