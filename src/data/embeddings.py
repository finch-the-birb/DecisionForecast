from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def _encoder_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


class TextEmbeddingCache:
    """Offline frozen sentence embeddings keyed by (ticker, window_end_date)."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        dim: int,
        max_chars: int = 8000,
        device: str | torch.device | None = None,
        prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.max_chars = max_chars
        self.prefix = prefix
        self.device_str = _resolve_encoder_device(device)
        self.cache_dir = Path(cache_dir) / _encoder_slug(model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer | None = None
        self._memory: dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading frozen text encoder %s on %s", self.model_name, self.device_str)
            self._model = SentenceTransformer(self.model_name, device=self.device_str)
            st_device = getattr(self._model, "device", self.device_str)
            log.info("Text encoder ready: %s device=%s", self.model_name, st_device)
            if self.device_str.startswith("cuda") and "cpu" in str(st_device).lower():
                raise RuntimeError(
                    f"Requested text encoder on {self.device_str} but SentenceTransformer is on {st_device}"
                )
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
        if self.prefix:
            text = f"{self.prefix}{text}"
        key = self._key(ticker, end_date, text)
        if key in self._memory:
            return self._memory[key]
        path = self._path(key)
        if path.exists():
            vec = np.load(path).astype(np.float32)
            if vec.shape[-1] != self.dim:
                log.warning("Cache dim mismatch for %s (%s vs %s); recomputing", path, vec.shape, self.dim)
                path.unlink(missing_ok=True)
                vec = self._embed(text)
                np.save(path, vec)
        else:
            vec = self._embed(text)
            np.save(path, vec)
        self._memory[key] = vec
        return vec

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            text,
            normalize_embeddings=True,
            device=self.device_str,
            show_progress_bar=False,
        )
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[-1] != self.dim:
            raise ValueError(
                f"Encoder {self.model_name} returned dim={vec.shape[-1]}, cfg dim={self.dim}"
            )
        return vec

    def encode_batch(self, items: list[tuple[str, str, str]]) -> np.ndarray:
        return np.stack([self.encode(t, d, x) for t, d, x in items], axis=0)


def _resolve_encoder_device(device: str | torch.device | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Text encoder requested CUDA but torch.cuda.is_available() is False")
    return str(resolved)
