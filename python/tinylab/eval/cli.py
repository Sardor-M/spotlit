"""`tinylab-eval` command line: splits, run, compare, verify-harness, bless-harness, selfcheck."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from . import baselines, frozen, ledger, rigor
from .harness import DEFAULT_KS, DEFAULT_TOP_K, EvalSet, evaluate
from .splits import Split, summarize

DEFAULT_SET = Path(__file__).resolve().parent / "fixtures"
SEALED_OPT_IN = "--yes-i-am-opening-the-sealed-set"


def _load(args: argparse.Namespace) -> EvalSet:
    return EvalSet.load(args.set)


def cmd_splits(args: argparse.Namespace) -> int:
    eval_set = _load(args)
    counts = summarize(eval_set.docs, eval_set.queries)
    print(f"evaluation set: {args.set}")
    print(f"{'':10}{'units':>8}{'docs':>8}{'queries':>9}")
    for split in Split:
        print(
            f"{split.value:10}"
            f"{counts['units'][split.value]:>8}"
            f"{counts['docs'][split.value]:>8}"
            f"{counts['queries'][split.value]:>9}"
        )
    empty = [name for name in ("val", "test") if counts["queries"][name] == 0]
    if empty:
        print(f"\nwarning: no queries in {', '.join(empty)} — too few units to split meaningfully")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.split == Split.SEALED and not args.yes_i_am_opening_the_sealed_set:
        print(
            "refusing to open the sealed split.\n"
            "It is opened once, after the model and every hyperparameter are frozen.\n"
            f"See docs/all/experiments/SEALED.md. If that is truly where you are, pass {SEALED_OPT_IN}",
            file=sys.stderr,
        )
        return 2

    eval_set = _load(args)
    factory = baselines.REGISTRY.get(args.system)
    if factory is None:
        print(
            f"unknown system {args.system!r}; known: {', '.join(sorted(baselines.REGISTRY))}",
            file=sys.stderr,
        )
        return 2

    retriever = factory(seed=args.seed) if args.system == "random" else factory()
    result = evaluate(retriever, eval_set, args.split, ks=DEFAULT_KS, top_k=DEFAULT_TOP_K)

    config = {
        "system": retriever.name,
        "split": str(args.split),
        "eval_set": str(args.set),
        "n_docs": result.n_docs,
        "n_queries": result.n_queries,
        "top_k": DEFAULT_TOP_K,
        "ks": list(DEFAULT_KS),
    }
    try:
        run_id = ledger.record(result, config, notes=args.notes, force=args.force)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print(f"run {run_id}  {retriever.name}  split={result.split}  n={result.n_queries}")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name:<12} {value:.4f}")
    problems = frozen.verify()
    if problems:
        print(f"\nWARNING: frozen harness files differ: {', '.join(problems)}", file=sys.stderr)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    incumbent = [ledger.load_run(run_id) for run_id in args.incumbent]
    candidate = [ledger.load_run(run_id) for run_id in args.candidate]
    comparison = rigor.compare(
        incumbent,
        candidate,
        metric=args.metric,
        min_delta=args.min_delta,
        stochastic=args.stochastic,
        seed=args.seed,
    )
    seeds = f"{comparison.n_seeds[0]} vs {comparison.n_seeds[1]}"
    print(f"{comparison.incumbent}  ->  {comparison.candidate}")
    print(f"  metric        {comparison.metric}")
    print(f"  n queries     {comparison.n_queries}  (seeds: {seeds})")
    print(f"  incumbent     {comparison.incumbent_mean:.4f}")
    print(f"  candidate     {comparison.candidate_mean:.4f}")
    print(
        f"  delta         {comparison.mean_delta:+.4f}"
        f"  95% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]"
    )
    print(f"  P(better)     {comparison.p_better:.3f}   (need >= {rigor.REQUIRED_P_BETTER})")
    print(f"  min effect    {comparison.min_delta:+.4f}")
    print(f"\n  {comparison.verdict}: {comparison.reason}")
    if args.record:
        ledger.record_comparison(asdict(comparison), notes=args.notes)
        print("  recorded in the ledger")
    return 0 if comparison.verdict == "KEEP" else 1


def cmd_verify_harness(args: argparse.Namespace) -> int:
    problems = frozen.verify()
    if not problems:
        print(f"frozen harness intact ({len(frozen.FROZEN_FILES)} files)")
        return 0
    print("frozen harness has changed:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        "\nEvery number recorded before this change came from different code.\n"
        "Either revert, or `bless-harness` and write an ADR saying why.",
        file=sys.stderr,
    )
    return 1


def cmd_bless_harness(args: argparse.Namespace) -> int:
    entries = frozen.bless()
    print(f"wrote {frozen.manifest_path()}")
    for name, sha in sorted(entries.items()):
        print(f"  {sha[:16]}  {name}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Assert bm25 > lexical-overlap > random; if that breaks, the harness is wrong."""
    eval_set = _load(args)
    scores = {}
    for name in ("random", "lexical-overlap", "bm25"):
        factory = baselines.REGISTRY[name]
        retriever = factory(seed=0) if name == "random" else factory()
        result = evaluate(retriever, eval_set, None)
        scores[name] = result.metrics["ndcg@10"]
        print(f"  {retriever.name:<28} ndcg@10 {scores[name]:.4f}")

    print()
    if scores["bm25"] > scores["lexical-overlap"] > scores["random"]:
        print("selfcheck PASS — bm25 > lexical-overlap > random")
        return 0
    print("selfcheck FAIL — expected bm25 > lexical-overlap > random", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinylab-eval", description=__doc__)
    parser.add_argument(
        "--set",
        type=Path,
        default=DEFAULT_SET,
        help="directory holding corpus.jsonl and queries.jsonl (default: bundled fixture)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_splits = sub.add_parser("splits", help="show the unit/doc/query split distribution")
    p_splits.set_defaults(func=cmd_splits)

    p_run = sub.add_parser("run", help="score a system and record it in the ledger")
    p_run.add_argument("--system", required=True, choices=sorted(baselines.REGISTRY))
    p_run.add_argument("--split", type=Split, default=Split.VAL, choices=list(Split))
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--notes", default="")
    p_run.add_argument("--force", action="store_true", help="re-run an already-recorded config")
    p_run.add_argument(SEALED_OPT_IN, action="store_true", dest="yes_i_am_opening_the_sealed_set")
    p_run.set_defaults(func=cmd_run)

    p_compare = sub.add_parser("compare", help="run the keep/discard gate on recorded runs")
    p_compare.add_argument("--incumbent", required=True, nargs="+", metavar="RUN_ID")
    p_compare.add_argument("--candidate", required=True, nargs="+", metavar="RUN_ID")
    p_compare.add_argument("--metric", default="ndcg@10")
    p_compare.add_argument("--min-delta", type=float, default=0.0)
    p_compare.add_argument(
        "--stochastic",
        action="store_true",
        help=f"candidate is a trained model; require {rigor.MIN_SEEDS_STOCHASTIC} seeds",
    )
    p_compare.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p_compare.add_argument("--record", action="store_true")
    p_compare.add_argument("--notes", default="")
    p_compare.set_defaults(func=cmd_compare)

    p_verify = sub.add_parser("verify-harness", help="check frozen files against the manifest")
    p_verify.set_defaults(func=cmd_verify_harness)

    p_bless = sub.add_parser("bless-harness", help="rewrite the frozen manifest")
    p_bless.set_defaults(func=cmd_bless_harness)

    p_self = sub.add_parser("selfcheck", help="prove the harness orders known baselines correctly")
    p_self.set_defaults(func=cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
