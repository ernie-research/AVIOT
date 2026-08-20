"""Evaluation entry points for AVIOT."""

from .evaluate import (
    EvaluationExample,
    evaluate,
    extract_choice,
    normalize_answer,
    parse_example,
    read_examples,
    score_prediction,
)

__all__ = [
    "EvaluationExample",
    "evaluate",
    "extract_choice",
    "normalize_answer",
    "parse_example",
    "read_examples",
    "score_prediction",
]
