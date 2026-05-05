"""Unit tests for MAB train-index stratification and difficulty proxies."""

import numpy as np
import pytest

from palimpzest.core.data.iter_dataset import MemoryDataset
from palimpzest.policy import MaxQuality
from palimpzest.query.execution.mab_execution_strategy import (
    MABExecutionStrategy,
    _strip_ollama_model_tag,
    _token_density_difficulty_proxy,
    stratify_source_indices_by_scores,
)


def test_strip_ollama_model_tag():
    assert _strip_ollama_model_tag("ollama/llama3.2:3b") == "llama3.2:3b"
    assert _strip_ollama_model_tag("phi3:mini") == "phi3:mini"


def test_token_density_difficulty_proxy_word_length():
    assert _token_density_difficulty_proxy({"abstract": "one two three"}, "abstract") == 3.0
    assert _token_density_difficulty_proxy({"abstract": ""}, "abstract") == 0.0
    assert _token_density_difficulty_proxy({"title": "hello world", "abstract": "ignored"}, "title") == 2.0


def test_stratify_source_indices_by_scores_empty():
    rng = np.random.default_rng(0)
    assert stratify_source_indices_by_scores([], rng) == []


class _RngNoShuffle:
    """Stand-in for ``np.random.Generator``: shuffle is a no-op (order stays tertile-sorted)."""

    def shuffle(self, _x) -> None:
        return None


def test_stratify_source_indices_round_robin_no_shuffle():
    """With shuffle disabled, n=9 yields strict easy/med/hard round-robin order."""
    rng = _RngNoShuffle()

    pairs = [(f"ds---{i}", float(i)) for i in range(9)]
    out = stratify_source_indices_by_scores(pairs, rng)

    assert out == [
        "ds---0",
        "ds---3",
        "ds---6",
        "ds---1",
        "ds---4",
        "ds---7",
        "ds---2",
        "ds---5",
        "ds---8",
    ]


def test_stratify_source_indices_is_permutation():
    rng = np.random.default_rng(12345)
    pairs = [(f"id-{i}", float(i * i)) for i in range(12)]
    out = stratify_source_indices_by_scores(pairs, rng)
    assert len(out) == 12
    assert sorted(out) == sorted(p[0] for p in pairs)


def _make_mab(**kwargs):
    defaults = dict(
        policy=MaxQuality(),
        max_workers=2,
        stratify_source_indices_by_difficulty=False,
        difficulty_proxy="token_density",
        stratify_difficulty_text_column="abstract",
        ollama_disagreement_task_instruction=None,
        ollama_models=None,
        ollama_api_base="http://localhost:11434",
    )
    defaults.update(kwargs)
    return MABExecutionStrategy(**defaults)


def test_difficulty_score_token_density_default_column():
    mab = _make_mab(stratify_difficulty_text_column="abstract")
    assert mab._difficulty_score_for_stratify({"abstract": "hello world"}) == 2.0


def test_difficulty_score_ollama_fallback_when_single_model():
    mab = _make_mab(
        difficulty_proxy="ollama_disagreement",
        ollama_models=["ollama/llama3.2:3b"],
        stratify_difficulty_text_column="abstract",
    )
    assert mab._difficulty_score_for_stratify({"abstract": "one two three"}) == 3.0


def test_difficulty_score_ollama_disagreement_uses_helper(mocker):
    mab = _make_mab(
        difficulty_proxy="ollama_disagreement",
        ollama_models=["ollama/model-a", "ollama/model-b"],
        ollama_api_base="http://127.0.0.1:11434",
        stratify_difficulty_text_column="abstract",
    )
    mock_disc = mocker.patch(
        "palimpzest.query.execution.mab_execution_strategy.disagreement_score_for_texts",
        return_value=1.0,
    )
    score = mab._difficulty_score_for_stratify({"abstract": "some clinical text"})
    assert score == 1.0
    mock_disc.assert_called_once()
    _, kwargs = mock_disc.call_args
    assert kwargs["api_base"] == "http://127.0.0.1:11434"
    ca, cb = mock_disc.call_args[0][1], mock_disc.call_args[0][2]
    assert ca == "model-a"
    assert cb == "model-b"


def test_memory_dataset_token_density_ordering_integration():
    """End-to-end: MemoryDataset rows scored by token proxy produce stratified index list."""
    rows = [
        {"abstract": "a"},  # 1 word — easy
        {"abstract": "b c d"},  # 3 words — hard among three
        {"abstract": "e f"},  # 2 words — medium
    ]
    ds = MemoryDataset(id="mem", vals=rows)
    mab = _make_mab(seed=99)
    rng = np.random.default_rng(99)

    scored = []
    for idx in range(len(ds)):
        scored.append(
            (
                f"{ds.id}---{idx}",
                mab._difficulty_score_for_stratify(ds[idx]),
            )
        )
    out = stratify_source_indices_by_scores(scored, rng)

    assert set(out) == {f"{ds.id}---0", f"{ds.id}---1", f"{ds.id}---2"}
    assert len(out) == 3
