"""Fine-tune the stage 2 cross-encoder: gold vs stage 1's hard negatives, softmax per group. See ADR-0013."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..gpu_lock import exclusive_gpu
from ..train.distill import clock
from .rank import KS, load_steps, query_of
from .rerank import BASE, DEPTH, MAX_TOKENS, CrossEncoder, device_for, rerank, stage1_lists


def group_of(step: dict, top: list[str], size: int, rng: np.random.Generator) -> list[str]:
    """Gold text first, then `size - 1` negatives from stage 1's top list (topped up from the page)."""
    gold = set(step["gold_described"])
    text = dict(step["candidates"])
    negatives = [node_id for node_id in top if node_id not in gold]
    if len(negatives) < size - 1:
        negatives += [node_id for node_id, _ in step["candidates"] if node_id not in gold and node_id not in top]
    picked = rng.choice(len(negatives), size=size - 1, replace=len(negatives) < size - 1)
    return [text[rng.choice(sorted(gold))]] + [text[negatives[i]] for i in picked]


def dev_recall(encoder: CrossEncoder, steps: list[dict], lists: dict) -> dict:
    hits = {k: 0 for k in KS}
    for step in steps:
        ranking = rerank(step, lists[step["action_uid"]], encoder)
        gold = set(step["gold_described"])
        first = next((rank for rank, node_id in enumerate(ranking, 1) if node_id in gold), None)
        for k in KS:
            hits[k] += first is not None and first <= k
    return {f"r@{k}": round(hits[k] / len(steps), 4) for k in KS if k <= 50}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-train-reranker", description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--stage1", type=Path, default=Path("checkpoints/ranker-linear-6144x320-v1"))
    parser.add_argument("--train", type=Path, default=Path("data/mind2web/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/mind2web/dev.jsonl"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--group", type=int, default=32, help="gold + negatives per step")
    parser.add_argument("--steps-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dev-steps", type=int, default=200, help="dev steps re-ranked at each log line")
    parser.add_argument("--log-every", type=int, default=250, help="batches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    with exclusive_gpu(f"tinylab-train-reranker {args.tag}".strip()):
        try:
            return train(args)
        except KeyboardInterrupt:
            print("\nstopped by Ctrl+C; no checkpoint written", flush=True)
            return 130


def train(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = device_for(torch)

    print("stage 1 lists (cached after the first run)...", flush=True)
    train_lists, dev_lists = stage1_lists(args.train, args.stage1), stage1_lists(args.dev, args.stage1)
    steps = [step for step in load_steps(args.train) if step["gold_described"]]
    dev = load_steps(args.dev)
    dev_sample = dev[:: max(1, len(dev) // args.dev_steps)][: args.dev_steps]

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)
    batches = int(args.epochs * len(steps) / args.steps_per_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, batches // 10, batches)
    params = sum(p.numel() for p in model.parameters())
    print(f"{len(steps):,} train steps, {batches:,} batches of {args.steps_per_batch} x {args.group} pairs; "
          f"{args.base} ({params:,} params) on {device.type}; stage 1 {args.stage1.name}, depth {DEPTH}", flush=True)

    stem = f"reranker-{args.base.split('/')[-1]}" + (f"-{args.tag}" if args.tag else "")
    history, running, start = [], None, time.time()
    order = np.concatenate([rng.permutation(len(steps)) for _ in range(int(np.ceil(args.epochs)) + 1)])
    for batch_index in range(1, batches + 1):
        model.train()
        chosen = order[(batch_index - 1) * args.steps_per_batch : batch_index * args.steps_per_batch]
        queries, elements = [], []
        for index in chosen:
            step = steps[index]
            group = group_of(step, train_lists[step["action_uid"]], args.group, rng)
            queries += [query_of(step)] * len(group)
            elements += group
        encoded = tokenizer(queries, elements, truncation=True, max_length=MAX_TOKENS,
                            padding="max_length", return_tensors="pt").to(device)
        logits = model(**encoded).logits.view(len(chosen), args.group)
        target = torch.zeros(len(chosen), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        value = loss.item()
        if not np.isfinite(value):
            raise FloatingPointError(f"loss became {value} at batch {batch_index}; stopping without saving")
        running = value if running is None else 0.98 * running + 0.02 * value
        if batch_index % args.log_every == 0 or batch_index == batches:
            model.eval()
            report = dev_recall(CrossEncoder(tokenizer, model, device), dev_sample, dev_lists)
            elapsed = time.time() - start
            gpu = f"  gpu {torch.mps.driver_allocated_memory() / 1e9:.1f} GB" if device.type == "mps" else ""
            left = clock(elapsed / batch_index * (batches - batch_index))
            print(f"  batch {batch_index:>5}/{batches}  loss {running:.4f}  dev {report}{gpu}"
                  f"  elapsed {clock(elapsed)}  left {left}", flush=True)
            history.append({"batch": batch_index, "loss": running, **report})

    directory = args.out / stem
    partial = args.out / f"{stem}.partial"
    model.save_pretrained(partial)
    tokenizer.save_pretrained(partial)
    (partial / "tinylab.json").write_text(json.dumps({
        "base": args.base, "stage1": str(args.stage1), "depth": DEPTH, "group": args.group,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed,
        "history": history, "seconds": round(time.time() - start, 1),
    }, indent=2), encoding="utf-8")
    if directory.exists():
        raise FileExistsError(f"{directory} exists; pass another --tag")
    partial.rename(directory)
    print(f"saved {directory} in {clock(time.time() - start)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
