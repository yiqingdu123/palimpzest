from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np
from litellm import embedding as litellm_embedding

from palimpzest.core.data.dataset import Dataset

logger = logging.getLogger(__name__)


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        chunks = []
        for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
            v_str = _safe_text(v)
            if v_str:
                chunks.append(f"{k}: {v_str}")
        return "\n".join(chunks)
    if isinstance(value, (list, tuple, set)):
        chunks = [_safe_text(v) for v in value]
        return "\n".join([c for c in chunks if c])
    if hasattr(value, "to_dict"):
        try:
            return _safe_text(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _record_to_text(record) -> str:
    if isinstance(record, dict):
        return _safe_text(record)
    if hasattr(record, "to_dict"):
        try:
            return _safe_text(record.to_dict())
        except Exception:
            return str(record)
    return _safe_text(record)


def _build_texts(dataset: Dataset, source_indices: list[str]) -> list[str]:
    texts = []
    for source_idx in source_indices:
        row_idx = int(str(source_idx).split("---")[-1])
        row = dataset[row_idx]
        texts.append(_record_to_text(row))
    return texts


def _build_rows_and_texts(dataset: Dataset, source_indices: list[str]) -> tuple[list[object], list[str]]:
    rows, texts = [], []
    for source_idx in source_indices:
        row_idx = int(str(source_idx).split("---")[-1])
        row = dataset[row_idx]
        rows.append(row)
        texts.append(_record_to_text(row))
    return rows, texts


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = np.std(arr)
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


def _centroid_outlier_scores(embeddings: np.ndarray) -> np.ndarray:
    centroid = np.mean(embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0.0:
        return np.zeros(embeddings.shape[0], dtype=np.float32)
    centroid = centroid / centroid_norm
    return 1.0 - (embeddings @ centroid)


def _nearest_scalar_distances(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if selected.size == 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    return np.min(np.abs(values[:, None] - values[selected][None, :]), axis=1)


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _missingness_ratio(record) -> float:
    if not isinstance(record, dict):
        return 0.0

    total_fields = len(record)
    if total_fields == 0:
        return 0.0

    missing_fields = 0
    for value in record.values():
        if value is None:
            missing_fields += 1
        elif isinstance(value, str) and value.strip() == "":
            missing_fields += 1
        elif isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            missing_fields += 1

    return float(missing_fields / total_fields)


def _format_noise_ratio(text: str) -> float:
    if len(text) == 0:
        return 0.0

    raw_tokens = text.split()
    if len(raw_tokens) == 0:
        return 0.0

    symbol_ratio = sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / len(text)
    digit_heavy_ratio = sum(
        1
        for token in raw_tokens
        if len(token) > 0 and sum(ch.isdigit() for ch in token) / len(token) >= 0.5
    ) / len(raw_tokens)
    uppercase_ratio = sum(
        1
        for token in raw_tokens
        if len(token) >= 3 and any(ch.isalpha() for ch in token) and token.upper() == token
    ) / len(raw_tokens)

    return float(0.5 * symbol_ratio + 0.3 * digit_heavy_ratio + 0.2 * uppercase_ratio)


def _compute_generic_difficulty_scores(rows: list[object], texts: list[str]) -> np.ndarray:
    tokenized_rows = [_tokenize_text(text) for text in texts]
    document_frequency: dict[str, int] = {}
    for tokens in tokenized_rows:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    scores = []
    for row, text, tokens in zip(rows, texts, tokenized_rows, strict=False):
        if len(tokens) == 0:
            rare_token_ratio = 0.0
        else:
            rare_token_ratio = sum(1 for token in tokens if document_frequency.get(token, 0) == 1) / len(tokens)

        format_noise = _format_noise_ratio(text)
        missingness = _missingness_ratio(row)
        scores.append(0.5 * rare_token_ratio + 0.3 * format_noise + 0.2 * missingness)

    return np.asarray(scores, dtype=np.float32)


def _texts_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def _compute_embeddings_openai(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    vectors = []
    for start_idx in range(0, len(texts), batch_size):
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]
        response = litellm_embedding(input=batch, model=model)
        vectors.extend([item["embedding"] for item in response.data])
    return np.asarray(vectors, dtype=np.float32)


def _compute_embeddings_local(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    # Lazy import so local mode does not add startup cost/dependency for default mode.
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model)
    vectors = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def _get_cached_embeddings(
    texts: list[str],
    provider: str,
    model: str,
    batch_size: int,
    cache_dir: str | None,
) -> np.ndarray:
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".palimpzest", "sampling-cache")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key_material = f"provider={provider}|model={model}|batch={batch_size}|texts={_texts_hash(texts)}"
    cache_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")

    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            return embeddings

    if provider == "openai":
        embeddings = _compute_embeddings_openai(texts, model, batch_size)
    elif provider == "local":
        embeddings = _compute_embeddings_local(texts, model, batch_size)
    else:
        raise ValueError(f"Unsupported sampling embedding provider: {provider}")

    np.save(cache_path, embeddings)
    return embeddings


def _greedy_hybrid_order(
    embeddings: np.ndarray,
    lengths: np.ndarray,
    difficulties: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    rng: np.random.Generator,
) -> list[int]:
    n = embeddings.shape[0]
    if n <= 1:
        return list(range(n))

    # Normalize vectors so cosine similarity is a dot product.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    emb = embeddings / norms

    length_score = _zscore(lengths)
    difficulty_score = _zscore(difficulties)
    embedding_seed_score = _centroid_outlier_scores(emb)
    feature_seed_score = alpha * embedding_seed_score + beta * np.abs(length_score) + gamma * np.abs(difficulty_score)

    selected = []
    remaining = set(range(n))

    # Select the first point from the combined outlier signal.
    first_scores = feature_seed_score + rng.random(n) * 1e-9
    first_idx = int(np.argmax(first_scores))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while remaining:
        rem_list = np.asarray(sorted(remaining), dtype=np.int32)
        sims = emb[rem_list] @ emb[np.asarray(selected, dtype=np.int32)].T
        selected_idx = np.asarray(selected, dtype=np.int32)
        embedding_dist = 1.0 - np.max(sims, axis=1)
        length_dist = _nearest_scalar_distances(length_score, selected_idx)[rem_list]
        difficulty_dist = _nearest_scalar_distances(difficulty_score, selected_idx)[rem_list]
        total = (
            alpha * embedding_dist
            + beta * length_dist
            + gamma * difficulty_dist
            + rng.random(len(rem_list)) * 1e-9
        )
        next_idx = int(rem_list[int(np.argmax(total))])
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected


def order_source_indices_with_embedding_hybrid(
    dataset_id: str,
    dataset: Dataset,
    source_indices: list[str],
    seed: int,
    embedding_provider: str = "openai",
    embedding_model: str = "openai/text-embedding-3-small",
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.0,
    batch_size: int = 128,
    cache_dir: str | None = None,
) -> list[str]:
    """Order source indices using weighted diversity coverage over embeddings, length, and difficulty."""
    if len(source_indices) <= 1:
        return source_indices

    rng = np.random.default_rng(seed=seed)
    if alpha == 0.0 and beta == 0.0 and gamma == 0.0:
        shuffled = list(source_indices)
        rng.shuffle(shuffled)
        return shuffled

    rows, texts = _build_rows_and_texts(dataset, source_indices)
    lengths = np.asarray([len(text) for text in texts], dtype=np.float32)
    difficulties = _compute_generic_difficulty_scores(rows, texts)

    try:
        embeddings = _get_cached_embeddings(
            texts=texts,
            provider=embedding_provider,
            model=embedding_model,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
    except Exception as e:
        logger.warning(
            "Embedding-hybrid sampling failed for dataset %s (%s). Falling back to random shuffle.",
            dataset_id,
            e,
        )
        shuffled = list(source_indices)
        rng.shuffle(shuffled)
        return shuffled

    order = _greedy_hybrid_order(
        embeddings=embeddings,
        lengths=lengths,
        difficulties=difficulties,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        rng=rng,
    )
    return [source_indices[idx] for idx in order]
