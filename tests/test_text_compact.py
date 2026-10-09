"""Compact 15D text: sentiment, spherical prototypes, train-only fit, decay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.text_compact import (
    COMPACT_COLUMNS,
    COMPACT_DIM,
    DECAY_LAMBDA,
    article_compact_features,
    build_compact_daily_series,
    event_prototype_features,
    fit_compact_state,
    fit_spherical_kmeans,
    probabilities_from_logits,
    sentiment_features,
)


def _dirichlet(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(3), size=n)


def test_sentiment_polarity_and_entropy() -> None:
    probs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )
    features = sentiment_features(probs)
    assert features.shape == (3, 5)
    np.testing.assert_allclose(features[0], [1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(features[1], [0.0, 1.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(features[2, :4], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(features[2, 4], np.log(3.0))


def test_softmax_matches_a_stable_manual_softmax() -> None:
    logits = np.array([[5.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    got = probabilities_from_logits(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(got.sum(axis=1), 1.0)


def test_one_prototype_is_the_normalized_train_mean() -> None:
    train = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    held_out = np.array([[0.0, -4.0]])
    embeddings = np.vstack([train, held_out])
    probs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mask = np.array([True, True, False])
    state = fit_compact_state(
        embeddings,
        probs,
        train_mask=mask,
        n_prototypes=1,
        n_init=1,
        seed=0,
    )
    expected_centroid = np.array([0.5, 0.5])
    expected_centroid /= np.linalg.norm(expected_centroid)
    np.testing.assert_allclose(state.centroids[0], expected_centroid, atol=1e-8)
    leaked = embeddings.mean(axis=0)
    leaked /= np.linalg.norm(leaked)
    assert abs(float(state.centroids[0] @ leaked)) < 0.99

    cosine = float(expected_centroid[0])
    np.testing.assert_allclose(state.mu[:5], [0.5, 0.5, 0.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(state.mu[5], cosine, atol=1e-8)
    assert state.mu.shape == (6,)
    assert len(COMPACT_COLUMNS) == COMPACT_DIM == 15


def test_held_out_rows_do_not_move_centroids_or_mu() -> None:
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(16, 8))
    probs = _dirichlet(16, seed=2)
    mask = np.arange(16) < 10
    kwargs = dict(train_mask=mask, n_prototypes=3, n_init=2, seed=4)
    original = fit_compact_state(embeddings, probs, **kwargs)
    edited = embeddings.copy()
    edited[10:] += 7.0
    edited_probs = probs.copy()
    edited_probs[10:] = np.array([0.05, 0.05, 0.90])
    again = fit_compact_state(edited, edited_probs, **kwargs)
    np.testing.assert_allclose(original.centroids, again.centroids)
    np.testing.assert_allclose(original.mu, again.mu)


def test_orthogonal_points_become_distinct_prototypes() -> None:
    points = np.eye(4)
    centers = fit_spherical_kmeans(points, n_clusters=4, n_init=1, seed=0)
    assert centers.shape == (4, 4)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0)
    similarity = centers @ points.T
    np.testing.assert_allclose(similarity.max(axis=1), 1.0, atol=1e-8)
    assert len(np.unique(similarity.argmax(axis=1))) == 4


def test_two_separated_clusters_are_recovered() -> None:
    rng = np.random.default_rng(3)
    left = rng.normal(scale=0.05, size=(25, 12))
    left[:, 0] += 6.0
    right = rng.normal(scale=0.05, size=(25, 12))
    right[:, 1] += 6.0
    centers = fit_spherical_kmeans(np.vstack([left, right]), n_clusters=2, n_init=4, seed=1)
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)
    left_mean /= np.linalg.norm(left_mean)
    right_mean /= np.linalg.norm(right_mean)
    poles = np.vstack([left_mean, right_mean])
    similarity = np.abs(centers @ poles.T)
    assert similarity.max(axis=0).min() > 0.95
    assert similarity.argmax(axis=0)[0] != similarity.argmax(axis=0)[1]


def test_event_feature_is_cosine_to_each_centroid() -> None:
    embeddings = np.array([[3.0, 0.0], [0.0, 4.0]])
    centroids = np.array([[1.0, 0.0], [0.0, 2.0]])
    features = event_prototype_features(embeddings, centroids)
    np.testing.assert_allclose(features, [[1.0, 0.0], [0.0, 1.0]])


def test_daily_series_uses_next_session_binding_and_decay() -> None:
    dates = pd.date_range("2021-01-04", periods=4, freq="B")
    rng = np.random.default_rng(5)
    train_embeddings = rng.normal(size=(12, 6))
    train_probs = _dirichlet(12, seed=6)
    state = fit_compact_state(
        train_embeddings,
        train_probs,
        n_prototypes=10,
        n_init=1,
        seed=7,
    )
    article_embeddings = train_embeddings[:2]
    article_probs = train_probs[:2]
    # Sunday noon binds to Monday; Tuesday noon binds to Wednesday.
    article_dates = np.array(["2021-01-03T12:00:00", "2021-01-05T12:00:00"], dtype="datetime64[ns]")
    series, has_news = build_compact_daily_series(
        dates,
        article_dates,
        article_embeddings,
        article_probs,
        state,
    )
    assert series.shape == (4, COMPACT_DIM)
    assert has_news.tolist() == [True, False, True, False]
    compact = article_compact_features(article_embeddings, article_probs, state.centroids)
    np.testing.assert_allclose(series[0], compact[0] - state.mu, atol=1e-5)
    np.testing.assert_allclose(series[1], series[0] * np.exp(-DECAY_LAMBDA), atol=1e-5)
    np.testing.assert_allclose(series[2], compact[1] - state.mu, atol=1e-5)
    np.testing.assert_allclose(series[3], series[2] * np.exp(-DECAY_LAMBDA), atol=1e-5)

    edited = article_embeddings.copy()
    edited[1] += 2.5
    edited_series, _ = build_compact_daily_series(
        dates,
        article_dates,
        edited,
        article_probs,
        state,
    )
    np.testing.assert_allclose(series[:2], edited_series[:2], atol=1e-5)
    assert not np.allclose(series[2], edited_series[2])


def test_compact_columns_follow_sentiment_then_events() -> None:
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    centroids = np.eye(3)[:2]
    compact = article_compact_features(embeddings, probs, centroids)
    sentiment = sentiment_features(probs)
    events = event_prototype_features(embeddings, centroids)
    np.testing.assert_allclose(compact[:, :5], sentiment)
    np.testing.assert_allclose(compact[:, 5:], events)
    assert list(COMPACT_COLUMNS[:5]) == [
        "sent_pos",
        "sent_neg",
        "sent_neu",
        "sent_polarity",
        "sent_entropy",
    ]
    assert COMPACT_COLUMNS[5] == "event_p0"
    assert COMPACT_COLUMNS[-1] == "event_p9"
