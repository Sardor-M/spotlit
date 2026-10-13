"""Student retriever, scored by the same harness as every baseline. See ADR-0006."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Doc

BATCH = 256


class StudentRetriever:
    """Cosine similarity over the distilled student's vectors."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self.name = f"student({self.checkpoint.stem})"
        self._model = None
        self._spec = None
        self._doc_ids: list[str] = []
        self._matrix = None

    def _load(self):
        if self._model is not None:
            return self._model

        import mlx.core as mx
        from mlx.utils import tree_unflatten

        from ..encoder import EncoderSpec
        from ..models.student import MLX_CACHE_LIMIT, Student, StudentConfig

        mx.set_cache_limit(MLX_CACHE_LIMIT)

        meta = json.loads(self.checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        config = StudentConfig(**meta["config"])
        model = Student(config)
        weights = mx.load(str(self.checkpoint.with_suffix(".safetensors")))
        model.update(tree_unflatten(list(weights.items())))
        mx.eval(model.parameters())

        self._model = model
        self._spec = EncoderSpec(num_buckets=config.num_buckets)
        return model

    def _embed(self, texts: list[str]):
        import mlx.core as mx
        import numpy as np

        from ..data.features import PAD_MULTIPLE, pad_batch, sparse_of

        model = self._load()
        chunks = []
        for start in range(0, len(texts), BATCH):
            window = texts[start : start + BATCH]
            sparse = [sparse_of(text, self._spec) for text in window]
            buckets, counts, mask = pad_batch(sparse, PAD_MULTIPLE)
            out = model(mx.array(buckets), mx.array(counts), mx.array(mask))
            mx.eval(out)
            chunks.append(np.array(out))
        return np.concatenate(chunks, axis=0)

    def index(self, docs: list[Doc]) -> None:
        self._doc_ids = [doc.doc_id for doc in docs]
        self._matrix = self._embed([doc.text for doc in docs])

    def rank(self, query_text: str, top_k: int) -> list[str]:
        vector = self._embed([query_text])[0]
        scores = self._matrix @ vector
        ordered = sorted(
            zip(self._doc_ids, scores.tolist()), key=lambda item: (-item[1], item[0])
        )
        return [doc_id for doc_id, _ in ordered[:top_k]]
