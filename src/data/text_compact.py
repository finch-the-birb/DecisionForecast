"""Compact 15D text: FinBERT sentiment plus train-only event prototypes.

Phase 4, Sprint 10. Each article becomes 5 sentiment coordinates and 10
cosine similarities to spherical k-means centroids. Centroids and the
neutral vector are fit on the train split only. The daily series reuses
the next-session binding and missing-day decay of ``build_daily_series``
(default λ = 0.03).

Sentiment columns are FinBERT probabilities in the order positive,
negative, neutral, then polarity ``P_pos - P_neg`` and Shannon entropy in
nats. This module does not download a model unless
:func:`finbert_probabilities` is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.data.text_series import build_daily_series

SENTIMENT_DIM = 5
EVENT_DIM = 10
COMPACT_DIM = SENTIMENT_DIM + EVENT_DIM
N_PROTOTYPES = EVENT_DIM
DECAY_LAMBDA = 0.03
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_LABELS = ("positive", "negative", "neutral")
COMPACT_COLUMNS: tuple[str, ...] = (
    "sent_pos",
    "sent_neg",
    "sent_neu",
    "sent_polarity",
    "sent_entropy",
    *(f"event_p{i}" for i in range(EVENT_DIM)),
)
_PROB_SUM_ATOL = 1e-4


@dataclass(frozen=True)
class CompactTextState:
    """Train-only prototypes and the 15D neutral vector."""

    centroids: np.ndarray
    mu: np.ndarray

    def __post_init__(self) -> None:
        centroids = np.array(self.centroids, dtype=np.float64, copy=True)
        mu = np.array(self.mu, dtype=np.float64, copy=True).reshape(-1)
        expected_width = SENTIMENT_DIM + (centroids.shape[0] if centroids.ndim == 2 else -1)
        if centroids.ndim != 2 or mu.shape != (expected_width,):
            raise ValueError(
                f"expected centroids [K, D] and mu [5+K], got {centroids.shape} and {mu.shape}"
            )
        object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "mu", mu)


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax. Columns stay in the caller order."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError("logits must have shape [N, 3]")
    if scores.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("logits must be finite")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def sentiment_features(probs: np.ndarray) -> np.ndarray:
    """Map FinBERT probabilities to ``[N, 5]``: pos, neg, neu, polarity, entropy."""
    probabilities = np.asarray(probs, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            "sentiment probabilities must have shape [N, 3] "
            "in order positive, negative, neutral"
        )
    if probabilities.shape[0] == 0:
        return np.zeros((0, SENTIMENT_DIM), dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("sentiment probabilities must be finite and non-negative")
    row_sum = probabilities.sum(axis=1)
    if np.any(np.abs(row_sum - 1.0) > _PROB_SUM_ATOL):
        raise ValueError("sentiment probability rows must sum to 1")
    positive = probabilities[:, 0]
    negative = probabilities[:, 1]
    neutral = probabilities[:, 2]
    polarity = positive - negative
    logged = np.zeros_like(probabilities)
    nonzero = probabilities > 0.0
    logged[nonzero] = probabilities[nonzero] * np.log(probabilities[nonzero])
    entropy = -logged.sum(axis=1)
    return np.column_stack([positive, negative, neutral, polarity, entropy])


def fit_spherical_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = N_PROTOTYPES,
    *,
    n_init: int = 10,
    max_iter: int = 50,
    seed: int = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Unit centroids ``[K, D]`` from spherical k-means on these rows only.

    Assignment uses cosine similarity. Each update is the L2-normalized mean
    of the members. ``n_init`` restarts use seeds ``seed, seed+1, ...`` and
    the restart with the highest mean cosine is kept.
    """
    points = _unit_rows(embeddings)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_init < 1:
        raise ValueError("n_init must be >= 1")
    if len(points) < n_clusters:
        raise ValueError(
            f"need at least {n_clusters} embeddings to fit {n_clusters} prototypes, got {len(points)}"
        )
    best_centers: np.ndarray | None = None
    best_score = -np.inf
    for restart in range(n_init):
        centers = _spherical_kmeans_once(
            points,
            n_clusters,
            max_iter=max_iter,
            seed=int(seed) + restart,
            tol=tol,
        )
        score = float((points @ centers.T).max(axis=1).mean())
        if score > best_score:
            best_score = score
            best_centers = centers
    if best_centers is None:
        raise RuntimeError("spherical k-means produced no centroids")
    return best_centers


def event_prototype_features(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine similarity of each embedding to every centroid, shape ``[N, K]``."""
    points = _unit_rows(embeddings, allow_empty=True)
    centers = _unit_rows(centroids)
    if points.shape[0] == 0:
        return np.zeros((0, centers.shape[0]), dtype=np.float64)
    if points.shape[1] != centers.shape[1]:
        raise ValueError(
            f"embedding dim {points.shape[1]} does not match centroid dim {centers.shape[1]}"
        )
    return points @ centers.T


def article_compact_features(
    embeddings: np.ndarray,
    sentiment_probs: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Stack sentiment and event coordinates into ``[N, 15]`` when ``K=10``."""
    sentiment = sentiment_features(sentiment_probs)
    events = event_prototype_features(embeddings, centroids)
    if sentiment.shape[0] != events.shape[0]:
        raise ValueError("embeddings and sentiment probabilities must have the same number of rows")
    return np.concatenate([sentiment, events], axis=1)


def fit_compact_state(
    embeddings: np.ndarray,
    sentiment_probs: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    n_prototypes: int = N_PROTOTYPES,
    n_init: int = 10,
    max_iter: int = 50,
    seed: int = 0,
) -> CompactTextState:
    """Fit prototypes and μ on the train rows. Other rows are ignored."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    probs = np.asarray(sentiment_probs, dtype=np.float64)
    if vectors.ndim != 2 or probs.ndim != 2 or vectors.shape[0] != probs.shape[0]:
        raise ValueError("embeddings [N, D] and sentiment probabilities [N, 3] must share N")
    if train_mask is None:
        mask = np.ones(vectors.shape[0], dtype=bool)
    else:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape != (vectors.shape[0],):
            raise ValueError("train_mask must be a boolean mask of length N")
    if not mask.any():
        raise ValueError("train split is empty")
    centroids = fit_spherical_kmeans(
        vectors[mask],
        n_clusters=n_prototypes,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
    )
    train_compact = article_compact_features(vectors[mask], probs[mask], centroids)
    mu = train_compact.mean(axis=0)
    return CompactTextState(centroids=centroids, mu=mu)


def build_compact_daily_series(
    trading_dates,
    article_dates: np.ndarray,
    embeddings: np.ndarray,
    sentiment_probs: np.ndarray,
    state: CompactTextState,
    lam: float = DECAY_LAMBDA,
    missing_policy: str = "decay",
) -> tuple[np.ndarray, np.ndarray]:
    """15D daily series: next-session binding, train-mean centering, then decay.

    An article timestamp in ``[previous trading day, this trading day)`` is
    attributed to this trading day. Days without news before the first article
    stay at 0. Later gaps multiply the centered vector by ``exp(-lam)``.
    """
    vectors = np.asarray(embeddings, dtype=np.float64)
    probs = np.asarray(sentiment_probs, dtype=np.float64)
    n_articles = len(np.asarray(article_dates))
    if vectors.shape[0] != n_articles or probs.shape[0] != n_articles:
        raise ValueError("article embeddings, probabilities, and dates must have the same length")
    compact = article_compact_features(vectors, probs, state.centroids)
    return build_daily_series(
        trading_dates,
        article_dates,
        compact,
        state.mu,
        lam,
        missing_policy=missing_policy,
    )


def finbert_probabilities(
    texts: Sequence[str],
    *,
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 512,
) -> np.ndarray:
    """Run frozen FinBERT and return probabilities ``[N, 3]``.

    The checkpoint is downloaded on first use. Callers that only have cached
    probabilities should use :func:`sentiment_features` instead.
    """
    rows = [str(text) for text in texts]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = model.config.id2label
    labels = tuple(str(id2label.get(i, id2label.get(str(i)))).lower() for i in range(3))
    if labels != FINBERT_LABELS:
        raise ValueError(f"expected FinBERT labels {FINBERT_LABELS}, got {labels}")
    model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            chunks.append(probabilities_from_logits(logits))
    return np.concatenate(chunks, axis=0)


def _unit_rows(embeddings: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    rows = np.asarray(embeddings, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape {rows.shape}")
    if rows.shape[0] == 0 and allow_empty:
        return rows
    if rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("embeddings must contain at least one non-zero row")
    if not np.isfinite(rows).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings must be non-zero")
    return rows / norms[:, None]


def _spherical_kmeans_once(
    points: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int,
    seed: int,
    tol: float,
) -> np.ndarray:
    centers = _kmeans_plus_plus(points, n_clusters, seed)
    labels = np.full(points.shape[0], -1, dtype=int)
    for _ in range(max_iter):
        assigned = np.argmax(points @ centers.T, axis=1)
        updated = _update_centers(points, assigned, n_clusters)
        shift = float(np.linalg.norm(updated - centers))
        unchanged = np.array_equal(assigned, labels)
        centers = updated
        labels = assigned
        if unchanged or shift < tol:
            break
    return centers


def _kmeans_plus_plus(points: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows = points.shape[0]
    first = int(rng.integers(n_rows))
    centers = np.empty((n_clusters, points.shape[1]), dtype=np.float64)
    centers[0] = points[first]
    closest = _squared_chord(points, centers[0])
    closest[first] = 0.0
    chosen = {first}
    for cluster in range(1, n_clusters):
        total = float(closest.sum())
        if total <= 1e-12:
            pick = next(index for index in range(n_rows) if index not in chosen)
        else:
            pick = int(rng.choice(n_rows, p=closest / total))
        centers[cluster] = points[pick]
        chosen.add(pick)
        closest = np.minimum(closest, _squared_chord(points, centers[cluster]))
        closest[pick] = 0.0
    return centers


def _squared_chord(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    cosine = points @ center
    return np.maximum(0.0, 2.0 - 2.0 * cosine)


def _update_centers(points: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = np.zeros((n_clusters, points.shape[1]), dtype=np.float64)
    empty: list[int] = []
    for cluster in range(n_clusters):
        members = points[labels == cluster]
        if len(members) == 0:
            empty.append(cluster)
            continue
        mean = members.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        centers[cluster] = members[0] if norm <= 1e-12 else mean / norm
    if not empty:
        return centers
    isolation = (points @ centers.T).max(axis=1)
    order = np.argsort(isolation, kind="mergesort")
    for cluster, point_index in zip(empty, order, strict=False):
        centers[cluster] = points[int(point_index)]
    return centers
