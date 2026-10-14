"""Stage 2: a pretrained cross-encoder that re-scores stage 1's top elements. See ADR-0013."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rank import cosine_ranking, load_steps, query_of, student_embedder

BASE = "google/bert_uncased_L-2_H-128_A-2"
DEPTH = 400
MAX_TOKENS = 128
SCORE_BATCH = 512


def device_for(torch):
    from ..data.label_with_teacher import limit_mps_memory

    return torch.device("mps") if limit_mps_memory(torch) else torch.device("cpu")


def stage1_lists(steps_path: Path, checkpoint: Path, depth: int = DEPTH) -> dict[str, list[str]]:
    """Stage 1's top `depth` node ids per step, cached beside the steps file."""
    cache = steps_path.with_name(f"{steps_path.stem}-stage1-{checkpoint.name}-{depth}.jsonl")
    if cache.exists():
        with cache.open(encoding="utf-8") as handle:
            return {row["action_uid"]: row["top"] for row in map(json.loads, handle)}
    embed = student_embedder(checkpoint)
    lists = {step["action_uid"]: cosine_ranking(step, embed)[:depth] for step in load_steps(steps_path)}
    partial = cache.with_suffix(".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for uid, top in lists.items():
            handle.write(json.dumps({"action_uid": uid, "top": top}) + "\n")
    partial.rename(cache)
    return lists


class CrossEncoder:
    """Relevance of (goal, element) pairs."""

    def __init__(self, tokenizer, model, device):
        import torch

        self.torch, self.tokenizer, self.model, self.device = torch, tokenizer, model, device

    @classmethod
    def load(cls, directory: Path | str) -> "CrossEncoder":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = device_for(torch)
        model = AutoModelForSequenceClassification.from_pretrained(directory, num_labels=1)
        return cls(AutoTokenizer.from_pretrained(directory), model.to(device).eval(), device)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for start in range(0, len(texts), SCORE_BATCH):
                window = texts[start : start + SCORE_BATCH]
                batch = self.tokenizer(
                    [query] * len(window), window, truncation=True, max_length=MAX_TOKENS,
                    padding="max_length", return_tensors="pt",
                ).to(self.device)
                out.append(self.model(**batch).logits[:, 0].float().cpu().numpy())
        if self.device.type == "mps":
            self.torch.mps.empty_cache()
        return np.concatenate(out) if out else np.zeros(0)


def rerank(step: dict, stage1: list[str], encoder: CrossEncoder, depth: int = DEPTH) -> list[str]:
    head = stage1[:depth]
    text = dict(step["candidates"])
    scores = encoder.score(query_of(step), [text[node_id] for node_id in head])
    order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
    return [head[i] for i in order] + stage1[depth:]
