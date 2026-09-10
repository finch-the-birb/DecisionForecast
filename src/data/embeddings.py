"""Frozen per-article sentence embeddings (one vector per news row)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

_EMPTY = {"", "nan", "none", "nat"}


def encoder_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def article_cache_key(url: object, title: object, date: object) -> str:
    url_s = str(url or "").strip()
    if url_s and url_s.lower() not in _EMPTY:
        return hashlib.sha1(url_s.encode("utf-8", errors="ignore")).hexdigest()
    payload = f"{title or ''}{date or ''}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def resolve_article_text(
    row: pd.Series,
    article_field: str,
    article_fallback: str,
) -> str:
    """Article-level text. Fallback is skip|title — not a decay/discount policy."""
    primary = str(row.get(article_field, "") or "").strip()
    if primary.lower() not in _EMPTY:
        return primary
    if str(article_fallback).lower() == "title":
        title = str(row.get("Article_title", "") or row.get("article_title", "") or "").strip()
        if title.lower() not in _EMPTY:
            return title
    return ""


def _row_url(row: pd.Series) -> object:
    for col in ("Url", "URL", "url"):
        if col in row.index:
            return row.get(col)
    return ""


class TextEmbeddingCache:
    """Per-ticker npz of sha1(article) → embedding. Empty strings are not encoded."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        dim: int,
        encode_batch_size: int = 64,
        max_seq_tokens: int = 512,
        device: str | torch.device | None = None,
        prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.encode_batch_size = int(encode_batch_size)
        self.max_seq_tokens = int(max_seq_tokens)
        self.prefix = prefix
        self.device_str = _resolve_encoder_device(device)
        self.cache_dir = Path(cache_dir) / encoder_slug(model_name) / "articles"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._index: dict[str, dict[str, int]] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def _ticker_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.npz"

    def _load_ticker(self, ticker: str) -> None:
        if ticker in self._index:
            return
        path = self._ticker_path(ticker)
        if not path.exists():
            self._index[ticker] = {}
            self._vectors[ticker] = np.zeros((0, self.dim), dtype=np.float32)
            return
        with np.load(path, allow_pickle=True) as data:
            keys = [str(k) for k in data["keys"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[-1] != self.dim:
            log.warning("Article cache dim mismatch %s; rebuilding", path)
            path.unlink(missing_ok=True)
            self._index[ticker] = {}
            self._vectors[ticker] = np.zeros((0, self.dim), dtype=np.float32)
            return
        self._index[ticker] = {k: i for i, k in enumerate(keys)}
        self._vectors[ticker] = vectors

    def _save_ticker(self, ticker: str) -> None:
        index = self._index[ticker]
        vectors = self._vectors[ticker]
        keys = np.empty(len(index), dtype=object)
        ordered = np.zeros((len(index), self.dim), dtype=np.float32)
        for key, i in index.items():
            keys[i] = key
            ordered[i] = vectors[i]
        path = self._ticker_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=keys, vectors=ordered)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading frozen text encoder %s on %s", self.model_name, self.device_str)
            self._model = SentenceTransformer(self.model_name, device=self.device_str)
            self._model.max_seq_length = self.max_seq_tokens
            st_device = getattr(self._model, "device", self.device_str)
            log.info("Text encoder ready: %s device=%s", self.model_name, st_device)
            if self.device_str.startswith("cuda") and "cpu" in str(st_device).lower():
                raise RuntimeError(
                    f"Requested text encoder on {self.device_str} but SentenceTransformer is on {st_device}"
                )
            probe = np.asarray(
                self._model.encode(
                    ["dimension probe"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=self.device_str,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if probe.shape[-1] != self.dim:
                raise ValueError(
                    f"Encoder {self.model_name} dim={probe.shape[-1]}, cfg dim={self.dim}"
                )
        return self._model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payloads = [f"{self.prefix}{t}" if self.prefix else t for t in texts]
        vecs = self.model.encode(
            payloads,
            batch_size=self.encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device_str,
        )
        return np.asarray(vecs, dtype=np.float32)

    def ensure_encoded(self, ticker: str, keys: list[str], texts: list[str]) -> None:
        if len(keys) != len(texts):
            raise ValueError("keys/texts length mismatch")
        self._load_ticker(ticker)
        index = self._index[ticker]
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        seen: set[str] = set()
        for key, text in zip(keys, texts, strict=True):
            if not text or key in index or key in seen:
                continue
            seen.add(key)
            missing_keys.append(key)
            missing_texts.append(text)
        if not missing_keys:
            return
        log.info("Encoding %d new articles for %s", len(missing_keys), ticker)
        new_vecs = self.encode_texts(missing_texts)
        old = self._vectors[ticker]
        start = old.shape[0]
        self._vectors[ticker] = np.concatenate([old, new_vecs], axis=0)
        for i, key in enumerate(missing_keys):
            index[key] = start + i
        self._save_ticker(ticker)

    def lookup(self, ticker: str, keys: list[str]) -> np.ndarray:
        self._load_ticker(ticker)
        index = self._index[ticker]
        vectors = self._vectors[ticker]
        out = np.zeros((len(keys), self.dim), dtype=np.float32)
        for i, key in enumerate(keys):
            j = index.get(key)
            if j is not None:
                out[i] = vectors[j]
        return out

    def ticker_ready(self, ticker: str) -> bool:
        return self._ticker_path(ticker).exists()


def extract_article_payloads(
    news: pd.DataFrame,
    article_field: str,
    article_fallback: str,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return keys, texts (empty skipped), and boolean mask over news rows."""
    if news.empty:
        return [], [], np.zeros(0, dtype=bool)
    keys: list[str] = []
    texts: list[str] = []
    mask = np.zeros(len(news), dtype=bool)
    for i, (_, row) in enumerate(news.iterrows()):
        text = resolve_article_text(row, article_field, article_fallback)
        if not text:
            continue
        date = row.get("Date", "")
        key = article_cache_key(_row_url(row), row.get("Article_title", ""), date)
        keys.append(key)
        texts.append(text)
        mask[i] = True
    return keys, texts, mask


def _resolve_encoder_device(device: str | torch.device | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Text encoder requested CUDA but torch.cuda.is_available() is False")
    return str(resolved)
