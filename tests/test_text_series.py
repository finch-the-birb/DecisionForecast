"""Synthetic tests for daily text series (no FinLang download)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.embeddings import article_cache_key, resolve_article_text
from src.data.text_series import build_daily_series, pool_window


def test_decay_and_news_mean_minus_mu() -> None:
    dates = pd.date_range("2021-01-04", periods=4, freq="B")  # Mon-Thu
    mu = np.array([1.0, 0.0], dtype=np.float32)
    lam = 0.03
    # News only on Tuesday 00:00 → assigned to Wednesday (interval [Tue, Wed))
    art_dates = np.array(["2021-01-05T12:00:00"], dtype="datetime64[ns]")
    art_vecs = np.array([[4.0, 0.0]], dtype=np.float32)
    e, has = build_daily_series(dates, art_dates, art_vecs, mu, lam, missing_policy="decay")
    assert e.shape == (4, 2)
    assert not has[0]
    np.testing.assert_array_equal(e[0], np.zeros(2, dtype=np.float32))
    # Tue 12:00 is in [Mon, Tue)? Mon=01-04, Tue=01-05 → [Mon, Tue) does not include Tue noon.
    assert not has[1]
    np.testing.assert_array_equal(e[1], np.zeros(2, dtype=np.float32))
    # Wed 01-06: interval [Tue, Wed) includes Tue noon.
    assert has[2]
    np.testing.assert_allclose(e[2], np.array([3.0, 0.0], dtype=np.float32))
    # Thu: no news → decay
    assert not has[3]
    np.testing.assert_allclose(e[3], e[2] * np.exp(-lam), rtol=1e-5)


def test_article_on_end_idx_date_not_in_window() -> None:
    dates = pd.date_range("2021-01-04", periods=4, freq="B")
    mu = np.zeros(2, dtype=np.float32)
    end_idx = 2
    art_dates = np.array([dates[end_idx].to_datetime64()])
    art_vecs = np.array([[9.0, 9.0]], dtype=np.float32)
    e, has = build_daily_series(dates, art_dates, art_vecs, mu, 0.03)
    window = e[0:end_idx]
    assert not np.allclose(window, 9.0)
    # Article dated dates[end_idx] lands on the next trading day (end_idx+1).
    assert has[end_idx + 1]
    np.testing.assert_allclose(e[end_idx + 1], np.array([9.0, 9.0], dtype=np.float32))
    assert not has[end_idx]


def test_window_agg_three_modes() -> None:
    series = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    lam = 0.03
    last = pool_window(series, 0, 3, "last", lam)
    mean = pool_window(series, 0, 3, "mean", lam)
    rec = pool_window(series, 0, 3, "recency_weighted", lam, renormalize=False)
    np.testing.assert_allclose(last, [3.0])
    np.testing.assert_allclose(mean, [2.0])
    ages = np.array([2.0, 1.0, 0.0])
    w = np.exp(-lam * ages)
    expected = float((w * np.array([1.0, 2.0, 3.0])).sum() / w.sum())
    np.testing.assert_allclose(rec, [expected], rtol=1e-5)
    rec_n = pool_window(series, 0, 3, "recency_weighted", lam, renormalize=True)
    np.testing.assert_allclose(np.linalg.norm(rec_n), 1.0, rtol=1e-5)


def test_article_fallback_skip_vs_title() -> None:
    row = pd.Series({"Lsa_summary": "", "Article_title": "Hello title", "Url": "http://x"})
    assert resolve_article_text(row, "Lsa_summary", "skip") == ""
    assert resolve_article_text(row, "Lsa_summary", "title") == "Hello title"
    row2 = pd.Series({"Lsa_summary": "Body", "Article_title": "Hello title"})
    assert resolve_article_text(row2, "Lsa_summary", "skip") == "Body"
    assert article_cache_key("http://x", "Hello title", "2021-01-01") == article_cache_key(
        "http://x", "other", "2099"
    )
    k1 = article_cache_key("", "Hello title", "2021-01-01")
    k2 = article_cache_key("", "Hello title", "2021-01-02")
    assert k1 != k2


def test_first_day_lo_is_one_calendar_day() -> None:
    dates = pd.date_range("2021-01-04", periods=3, freq="B")
    mu = np.zeros(2, dtype=np.float32)
    far = np.array(["2020-12-05T12:00:00"], dtype="datetime64[ns]")
    e, has = build_daily_series(dates, far, np.array([[9.0, 9.0]], dtype=np.float32), mu, 0.03)
    assert not has[0]
    np.testing.assert_array_equal(e[0], np.zeros(2, dtype=np.float32))
    near = np.array(["2021-01-03T12:00:00"], dtype="datetime64[ns]")
    e2, has2 = build_daily_series(dates, near, np.array([[4.0, 0.0]], dtype=np.float32), mu, 0.03)
    assert has2[0]
    np.testing.assert_allclose(e2[0], np.array([4.0, 0.0], dtype=np.float32))


def test_ticker_series_path_includes_lam_and_policy(tmp_path) -> None:
    from src.data.text_series import ticker_series_path

    p1 = ticker_series_path(tmp_path, "enc", "AAPL", lam=0.03, missing_policy="decay")
    p2 = ticker_series_path(tmp_path, "enc", "AAPL", lam=0.10, missing_policy="decay")
    p3 = ticker_series_path(tmp_path, "enc", "AAPL", lam=0.03, missing_policy="zero")
    assert p1.name == "AAPL_decay_lam0.03.npz"
    assert p2.name == "AAPL_decay_lam0.1.npz"
    assert p3.name == "AAPL_zero_lam0.03.npz"
    assert p1 != p2
    assert p1 != p3


