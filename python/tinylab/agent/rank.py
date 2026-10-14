"""Recall@k of element rankers on compact Mind2Web steps: BM25, students, fusions, the teacher. See ADR-0012."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval.baselines import BM25Retriever

KS = (1, 5, 10, 20, 50, 100, 200)
RRF_K = 60
FUSION_DEPTH = 100
CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class Element:
    doc_id: str
    text: str


def query_of(step: dict) -> str:
    history = "; ".join(step["history"])
    return step["task"] + (f" Previous actions: {history}" if history else "")


def load_steps(path: Path, limit: int = 0) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle]
    return steps[:limit] if limit else steps


def bm25_ranking(step: dict) -> list[str]:
    retriever = BM25Retriever()
    retriever.index([Element(node_id, text) for node_id, text in step["candidates"]])
    return retriever.rank(query_of(step), len(step["candidates"]))


def cosine_ranking(step: dict, embed) -> list[str]:
    ids = [node_id for node_id, _ in step["candidates"]]
    vectors = embed([text for _, text in step["candidates"]])
    scores = vectors @ embed([query_of(step)], query=True)[0]
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [ids[i] for i in order]


def fuse(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, node_id in enumerate(ranking[:FUSION_DEPTH], 1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (RRF_K + position)
    return [node_id for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def student_embedder(checkpoint: Path):
    from ..eval.student import StudentRetriever

    retriever = StudentRetriever(checkpoint)
    return lambda texts, query=False: retriever._embed(texts)


def teacher_embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    from ..data.label_with_teacher import limit_mps_memory
    from ..eval.dense import DEFAULT_TEACHER, QUERY_PREFIX

    mps = limit_mps_memory(torch)
    model = SentenceTransformer(DEFAULT_TEACHER)

    def embed(texts, query=False):
        out = model.encode(
            [QUERY_PREFIX + t for t in texts] if query else texts,
            batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        if mps:
            torch.mps.empty_cache()
        return out

    return embed


def scores_of(step: dict, ranking: list[str]) -> dict[str, float]:
    gold = set(step["gold_described"])
    first = next((rank for rank, node_id in enumerate(ranking, 1) if node_id in gold), None)
    out = {f"recall@{k}": float(first is not None and first <= k) for k in KS}
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def token_saving(step: dict, ranking: list[str], k: int = 50) -> float:
    text = dict(step["candidates"])
    kept = sum(len(text[node_id]) for node_id in ranking[:k]) or 1
    return step["html_chars"] / kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-rank-elements", description=__doc__)
    parser.add_argument("--steps", type=Path, default=Path("data/mind2web/dev.jsonl"))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="student; repeatable")
    parser.add_argument("--teacher", action="store_true", help="also score the 33M teacher (uses the GPU)")
    parser.add_argument("--reranker", type=Path, help="stage 2; re-ranks the first --checkpoint's list")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    steps = load_steps(args.steps, args.limit)
    reachable = sum(bool(step["gold_described"]) for step in steps)
    median = statistics.median(len(step["candidates"]) for step in steps)
    print(f"{len(steps):,} steps from {args.steps}; gold element described in {reachable:,} "
          f"({reachable / len(steps):.1%}); median {median:.0f} candidates")

    embedders = {path.name: student_embedder(path) for path in args.checkpoint}
    if args.teacher:
        embedders["teacher"] = teacher_embedder()

    encoder = None
    if args.reranker:
        if args.reranker.suffix == ".onnx":
            from transformers import AutoTokenizer

            from .export_reranker import OnnxEncoder

            encoder = OnnxEncoder(args.reranker, AutoTokenizer.from_pretrained(args.reranker.parent))
            reranker_name = f"{args.reranker.parent.name}/{args.reranker.name}"
        else:
            from .rerank import CrossEncoder

            encoder = CrossEncoder.load(args.reranker)
            reranker_name = args.reranker.name
        stage1 = args.checkpoint[0].name

    results: dict[str, list[dict]] = {}
    savings: dict[str, list[float]] = {}
    start = time.time()
    for step in steps:
        rankings = {"bm25": bm25_ranking(step)}
        for name, embed in embedders.items():
            rankings[name] = cosine_ranking(step, embed)
            rankings[f"bm25+{name}"] = fuse(rankings["bm25"], rankings[name])
        if encoder is not None:
            from .rerank import rerank

            rankings[f"{stage1} > {reranker_name}"] = rerank(step, rankings[stage1], encoder)
        for name, ranking in rankings.items():
            results.setdefault(name, []).append(scores_of(step, ranking))
            savings.setdefault(name, []).append(token_saving(step, ranking))

    print(f"scored in {time.time() - start:.0f}s")
    header = "  ".join(f"{metric:>9}" for metric in [*(f"recall@{k}" for k in KS), "mrr"])
    print(f"{'system':<44}{header}  html/top50")
    metrics = [*(f"recall@{k}" for k in KS), "mrr"]
    for name, rows in results.items():
        means = "  ".join(f"{np.mean([row[metric] for row in rows]):9.3f}" for metric in metrics)
        print(f"{name:<44}{means}  {statistics.median(savings[name]):8.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
