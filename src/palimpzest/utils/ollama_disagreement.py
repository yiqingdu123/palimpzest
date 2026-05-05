"""
Per-row difficulty from disagreement between two local Ollama models.

Use with :func:`palimpzest.utils.difficulty_sampling.stratified_sample_dataframe` by
passing a factory from :func:`ollama_disagreement_difficulty_fn` or by precomputing
:func:`ollama_disagreement_scores` and then using your own ``difficulty_fn`` that
returns the stored series.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def ollama_chat(
    api_base: str,
    model: str,
    prompt: str,
    *,
    timeout_s: float = 120.0,
) -> str:
    """
    Run a single-turn chat against a local Ollama server and return assistant text.
    """
    base = api_base.rstrip("/")
    url = f"{base}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def _normalize_answer(s: str) -> str:
    return " ".join(s.strip().lower().split())


def disagreement_score_for_texts(
    text: str,
    model_a: str,
    model_b: str,
    *,
    task_instruction: str = "Extract the primary medical condition:",
    truncate_chars: int = 1000,
    api_base: str = "http://localhost:11434",
    normalize_answers: bool = True,
    on_error_score: float = 1.0,
    timeout_s: float = 120.0,
) -> float:
    """
    Return ``0.0`` if both models agree on a non-empty answer, else ``1.0`` (hard).

    Matches the usual proxy: easy only when outputs match and are non-empty after optional
    normalization. Both empty → treated as hard (1.0), same as separate-empty mismatch logic.
    """
    if truncate_chars is not None and truncate_chars > 0:
        truncated = text[:truncate_chars]
    else:
        truncated = text
    prompt = f"{task_instruction}\n\nText: {truncated}\n\nOutput only the condition name."

    try:
        ans_a = ollama_chat(api_base, model_a, prompt, timeout_s=timeout_s)
        ans_b = ollama_chat(api_base, model_b, prompt, timeout_s=timeout_s)
    except Exception as e:
        logger.warning("Ollama disagreement step failed: %s", e)
        return float(on_error_score)

    if normalize_answers:
        ca, cb = _normalize_answer(ans_a), _normalize_answer(ans_b)
    else:
        ca, cb = ans_a.strip(), ans_b.strip()

    if ca == cb and ca != "":
        return 0.0
    return 1.0


def ollama_disagreement_scores(
    df: pd.DataFrame,
    *,
    text_column: str,
    model_a: str,
    model_b: str,
    task_instruction: str = "Extract the primary medical condition:",
    truncate_chars: int = 1000,
    api_base: str = "http://localhost:11434",
    normalize_answers: bool = True,
    on_error_score: float = 1.0,
    timeout_s: float = 120.0,
) -> pd.Series:
    """
    Run two-model disagreement on each row; returns a ``Series`` aligned with ``df``.

    This performs ``2 * len(df)`` local inference calls — run once on a candidate pool,
    then pass scores into stratified sampling or cache them to disk for reuse.
    """
    if text_column not in df.columns:
        raise KeyError(f"column {text_column!r} not in DataFrame")

    scores: list[float] = []
    for _, row in df.iterrows():
        raw = row[text_column]
        text = "" if pd.isna(raw) else str(raw)
        s = disagreement_score_for_texts(
            text,
            model_a,
            model_b,
            task_instruction=task_instruction,
            truncate_chars=truncate_chars,
            api_base=api_base,
            normalize_answers=normalize_answers,
            on_error_score=on_error_score,
            timeout_s=timeout_s,
        )
        scores.append(s)

    return pd.Series(scores, index=df.index, dtype=float)


def ollama_disagreement_difficulty_fn(
    text_column: str,
    model_a: str,
    model_b: str,
    *,
    task_instruction: str = "Extract the primary medical condition:",
    truncate_chars: int = 1000,
    api_base: str = "http://localhost:11434",
    normalize_answers: bool = True,
    on_error_score: float = 1.0,
    timeout_s: float = 120.0,
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Build a ``difficulty_fn`` for :func:`~palimpzest.utils.difficulty_sampling.stratified_sample_dataframe`.

    Example::

        from palimpzest.utils.difficulty_sampling import stratified_sample_dataframe
        from palimpzest.utils.ollama_disagreement import ollama_disagreement_difficulty_fn

        fn = ollama_disagreement_difficulty_fn(
            "fulltext",
            model_a="llama3.2:3b",
            model_b="phi3:mini",
            api_base="http://localhost:11434",
        )
        subset = stratified_sample_dataframe(pool_df, 25, num_bins=3, seed=42, difficulty_fn=fn)
    """

    def _fn(df: pd.DataFrame) -> pd.Series:
        return ollama_disagreement_scores(
            df,
            text_column=text_column,
            model_a=model_a,
            model_b=model_b,
            task_instruction=task_instruction,
            truncate_chars=truncate_chars,
            api_base=api_base,
            normalize_answers=normalize_answers,
            on_error_score=on_error_score,
            timeout_s=timeout_s,
        )

    return _fn
