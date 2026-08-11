"""Retrieval evaluation harness. See README.md and ADR-0002."""

from .harness import EvalResult, EvalSet, Retriever, evaluate
from .schema import Doc, Query
from .splits import Split

__all__ = ["Doc", "EvalResult", "EvalSet", "Query", "Retriever", "Split", "evaluate"]
