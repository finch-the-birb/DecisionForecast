from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbeddingCache:
    """Offline frozen sentence embeddings keyed by (ticker, window_end_date)."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        dim: int,
        max_chars: int = 8000,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.dim = dim
        self.max_chars = max_chars
        self._model: SentenceTransformer | None = None
        self._memory: dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _key(self, ticker: str, end_date: str, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{ticker}_{end_date}_{digest}"

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def encode(self, ticker: str, end_date: str, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        text = text[: self.max_chars]
        key = self._key(ticker, end_date, text)
        if key in self._memory:
            return self._memory[key]
        path = self._path(key)
        if path.exists():
            vec = np.load(path).astype(np.float32)
        else:
            vec = self.model.encode(text, normalize_embeddings=True)
            vec = np.asarray(vec, dtype=np.float32)
            np.save(path, vec)
        self._memory[key] = vec
        return vec

    def encode_batch(self, items: list[tuple[str, str, str]]) -> np.ndarray:
        return np.stack([self.encode(t, d, x) for t, d, x in items], axis=0)
