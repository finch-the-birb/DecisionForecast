# Report: phase1-finance-encoder-gpu

- authored_by: runner
- created_at: 2026-09-08T19:12:01Z
- request_folder: agent-handoff/2026-09-08_1904_phase1-finance-encoder-gpu/
- tested_ref: feat/phase1-data@7f0fd7ecf3e351ea6463bd684e23dee784618bf8
- status: blocked

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: 7f0fd7e (code 95deb30). Skipped MiniLM fusion-smoke (no report started).
# skipped download_fnspid.py
uv sync
uv run python scripts/precache_fnspid_news.py --ticker-set=dev
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```

## Outcome
- exit_code: uv sync=0; precache=0; train=1
- duration: uv sync ~1s; precache ~1m 45s (parquet hits); train ~2m 09s until HF 401
- host: runpod pod `p5cnr4o8bnd4c4`
- device: cuda (no CPU fallback)

CUDA check passed. Datasets built with NVDA and recent-window caps. Train died on the **first** `__getitem__` when loading `FinLang/investopedia_embedding`: Hub **401 Unauthorized** / `RepositoryNotFoundError` (gated or private; unauthenticated). No `HF_TOKEN` on this pod. Did not store keys.

Skipped MiniLM request `2026-09-08_1854_phase1-fusion-smoke` as instructed.

## Key signals
- GPU name: **NVIDIA L4** (capability 8.9); `torch.cuda.is_available=True`; train device `cuda`
- encoder device: **n/a** — log `Loading frozen text encoder FinLang/investopedia_embedding on cuda` then 401; never reached `Text encoder ready`
- CUDA MiB after model.to: allocated=**0.6** reserved=**2.0** (warning: allocated < 1 MiB). Model parameters on **cuda:0**. First-batch CUDA mem: n/a
- text L2: **n/a** (encoder never encoded a batch)
- test mse/mae: **n/a**
- oom: no
- traceback_summary: `huggingface_hub.errors.RepositoryNotFoundError: 401` on `https://huggingface.co/FinLang/investopedia_embedding/resolve/main/modules.json`. HF: gated/private repo needs an authenticated token.

Splits (NVDA present):
- train: 64 windows, end_date 2021-12-16 .. 2021-12-22, tickers include NVDA (`NVDA.csv -> nvda.csv`)
- val: 32 windows, 2022-12-16 .. 2022-12-21
- test: 32 windows, 2023-12-13 .. 2023-12-18
- sizes: train 64, val 32, test 32

## Artifacts (paths on Runner disk — do not commit binaries)
- hydra run (failed): `outputs/2026-09-08/19-11-09/train.log`
- explain / checkpoint / projection: not written
- mlflow_run_id: n/a (failed before first batch; run may have started empty)

## Conclusions for Dev
1. CUDA path is fail-closed correctly: L4, `train.device=cuda`, no CPU fallback. TimeXL landed on `cuda:0`. Encoder CUDA was requested; download never completed.
2. Blocker is **HF auth**, not GPU. `FinLang/investopedia_embedding` returns 401 without a token. Runner needs `HF_TOKEN` (or a public encoder override). Do not commit the token; set it in the pod env.
3. `after model.to` allocated **0.6 MiB** (< 1 MiB success bar) with Model A still tiny and encoder not loaded. Re-check after a successful encoder load + first batch.
4. NVDA casing and recent-window caps look fixed vs the previous pass report.

## Suggested next command (optional)
```bash
# after HF_TOKEN is in the pod environment (not in git)
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```
