"""Append-only run ledger in docs/all/experiments/. Never edit past entries."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .harness import EvalResult


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiments_dir() -> Path:
    return repo_root() / "docs" / "all" / "experiments"


def ledger_path() -> Path:
    return experiments_dir() / "ledger.jsonl"


def runs_dir() -> Path:
    return experiments_dir() / "runs"


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def already_recorded(run_id: str) -> bool:
    path = ledger_path()
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as handle:
        return any(json.loads(line).get("run_id") == run_id for line in handle if line.strip())


def load_run(run_id: str) -> EvalResult:
    """Rehydrate a past run so it can serve as an incumbent."""
    path = runs_dir() / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no recorded run {run_id!r} at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EvalResult(
        system=raw["system"],
        split=raw["split"],
        n_queries=raw["n_queries"],
        n_docs=raw["n_docs"],
        metrics=raw["metrics"],
        per_query=raw["per_query"],
    )


def record(result: EvalResult, config: dict, notes: str = "", force: bool = False) -> str:
    """Append one run and return its run_id; refuses a config already recorded."""
    run_id = config_hash(config)
    if already_recorded(run_id) and not force:
        raise RuntimeError(
            f"config {run_id} has already been run (see {ledger_path()}). "
            "Change something, or pass --force if you genuinely mean to re-run it."
        )

    runs_dir().mkdir(parents=True, exist_ok=True)
    detail = {
        "run_id": run_id,
        "system": result.system,
        "split": result.split,
        "n_queries": result.n_queries,
        "n_docs": result.n_docs,
        "metrics": result.metrics,
        "per_query": result.per_query,
    }
    (runs_dir() / f"{run_id}.json").write_text(
        json.dumps(detail, indent=2, sort_keys=True), encoding="utf-8"
    )

    entry = {
        "run_id": run_id,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": result.system,
        "split": result.split,
        "n_queries": result.n_queries,
        "n_docs": result.n_docs,
        "metrics": {name: round(value, 6) for name, value in sorted(result.metrics.items())},
        "config": config,
        "notes": notes,
        "harness_frozen": _harness_state(),
    }
    with ledger_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return run_id


def record_comparison(comparison_dict: dict, notes: str = "") -> None:
    experiments_dir().mkdir(parents=True, exist_ok=True)
    entry = {
        "kind": "comparison",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "harness_frozen": _harness_state(),
        **comparison_dict,
    }
    with ledger_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def _harness_state() -> str:
    """Whether the frozen files were intact when this number was produced."""
    from .frozen import verify

    if os.environ.get("TINYLAB_SKIP_FROZEN_CHECK"):
        return "unchecked"
    problems = verify()
    return "intact" if not problems else f"MODIFIED: {', '.join(sorted(problems))}"


def asdict_comparison(comparison) -> dict:
    return asdict(comparison)
