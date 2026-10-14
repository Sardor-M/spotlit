"""Mind2Web as compact element-ranking records: train (split by website into train/dev) and test. See ADR-0012."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dom import describe_candidates

REPO = "osunlp/Mind2Web"
TRAIN_FILES = [f"data/train/train_{index}.json" for index in range(11)]
TEST_ZIP = "test.zip"
TEST_SPLITS = ("test_task", "test_website", "test_domain")
DEV_BUCKETS = 15
HISTORY_STEPS = 3
TASK_KEYS = ("annotation_id", "website", "domain", "confirmed_task", "action_reprs")
# raw_html is never sent to the workers: it is twice the size of cleaned_html and unused.
ACTION_KEYS = ("action_uid", "cleaned_html", "pos_candidates", "neg_candidates", "operation")
WORKERS = max(1, (os.cpu_count() or 2) // 2)


def split_of(website: str) -> str:
    bucket = int(hashlib.sha256(f"tinylab-m2w-v1:{website}".encode()).hexdigest(), 16) % 100
    return "dev" if bucket < DEV_BUCKETS else "train"


def step_record(job: tuple[dict, int, dict]) -> dict:
    task, index, action = job
    candidates = action["pos_candidates"] + action["neg_candidates"]
    described = describe_candidates(action["cleaned_html"], [c["backend_node_id"] for c in candidates])
    gold = [c["backend_node_id"] for c in action["pos_candidates"]]
    return {
        "annotation_id": task["annotation_id"],
        "action_uid": action["action_uid"],
        "website": task["website"],
        "domain": task["domain"],
        "task": task["confirmed_task"],
        "history": task["action_reprs"][max(0, index - HISTORY_STEPS) : index],
        "op": action["operation"]["op"],
        "value": action["operation"]["value"],
        "candidates": [[node_id, text] for node_id, text in described.items()],
        "gold": gold,
        "gold_described": [node_id for node_id in gold if node_id in described],
        "html_chars": len(action["cleaned_html"]),
    }


def process(path: Path, part: Path, split: str | None = None) -> int:
    with path.open(encoding="utf-8") as handle:
        tasks = json.load(handle)
    jobs = []
    for task in tasks:
        meta = {key: task[key] for key in TASK_KEYS}
        for index, action in enumerate(task["actions"]):
            jobs.append((meta, index, {key: action[key] for key in ACTION_KEYS}))
    del tasks
    staging = part.with_suffix(".partial")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool, staging.open("w", encoding="utf-8") as out:
        for record in pool.map(step_record, jobs, chunksize=8):
            record["split"] = split or split_of(record["website"])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    staging.rename(part)
    return len(jobs)


def build_test(out_dir: Path, password: str, keep_raw: bool) -> None:
    """Evaluation only: the authors ask that the unzipped test data is never redistributed or trained on."""
    from huggingface_hub import hf_hub_download

    parts = out_dir / "parts"
    archive_path = Path(hf_hub_download(REPO, TEST_ZIP, repo_type="dataset", local_dir=out_dir / "raw"))
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
            split = next((s for s in TEST_SPLITS if s in name), None)
            if split is None:
                continue
            part = parts / f"{split}__{Path(name).stem}.jsonl"
            if part.exists():
                continue
            extracted = Path(archive.extract(name, out_dir / "raw" / "test", pwd=password.encode()))
            steps = process(extracted, part, split)
            extracted.unlink()
            print(f"  {name}: {steps:,} steps", flush=True)
    if not keep_raw:
        archive_path.unlink()
    for split in TEST_SPLITS:
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for part in sorted(parts.glob(f"{split}__*.jsonl")):
                out.write(part.read_text(encoding="utf-8"))
        count = sum(1 for _ in (out_dir / f"{split}.jsonl").open(encoding="utf-8"))
        print(f"wrote {count:,} steps to {out_dir / split}.jsonl", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinylab-build-mind2web", description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/mind2web"))
    parser.add_argument("--keep-raw", action="store_true", help="keep the 600 MB source files after processing")
    parser.add_argument("--test-password", help="build the test splits (password: Mind2Web repo README)")
    args = parser.parse_args(argv)
    if args.test_password:
        build_test(args.out_dir, args.test_password, args.keep_raw)
        return 0

    from huggingface_hub import hf_hub_download

    parts = args.out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_FILES:
        part = parts / (Path(name).stem + ".jsonl")
        if part.exists():
            print(f"  {name}: already processed", flush=True)
            continue
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=args.out_dir / "raw"))
        steps = process(path, part)
        if not args.keep_raw:
            path.unlink()
        print(f"  {name}: {steps:,} steps", flush=True)

    counts = {"train": 0, "dev": 0}
    with (args.out_dir / "train.jsonl").open("w", encoding="utf-8") as train, (args.out_dir / "dev.jsonl").open(
        "w", encoding="utf-8"
    ) as dev:
        for part in sorted(parts.glob("train_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
            for line in part.open(encoding="utf-8"):
                split = json.loads(line)["split"]
                (dev if split == "dev" else train).write(line)
                counts[split] += 1
    print(f"wrote {counts['train']:,} train and {counts['dev']:,} dev steps to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
