from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(device_cfg: str) -> torch.device:
    requested = str(device_cfg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device=cuda but torch.cuda.is_available() is False")
    return device


def log_torch_device(device: torch.device, *, role: str = "train") -> None:
    log.info("%s device: %s", role, device)
    log.info("torch.cuda.is_available=%s", torch.cuda.is_available())
    if device.type != "cuda":
        log.warning("%s is on CPU — embeddings/train will not use the GPU", role)
        return
    idx = device.index if device.index is not None else 0
    log.info("GPU name: %s", torch.cuda.get_device_name(idx))
    cap = torch.cuda.get_device_capability(idx)
    log.info("CUDA capability: %s.%s", cap[0], cap[1])


def log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        log.info("%s CUDA mem: n/a (no CUDA)", tag)
        return
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    log.info("%s CUDA mem allocated=%.1f MiB reserved=%.1f MiB", tag, allocated, reserved)
    if allocated < 1.0:
        log.warning("%s CUDA allocated < 1 MiB — tensors may still be on CPU", tag)
