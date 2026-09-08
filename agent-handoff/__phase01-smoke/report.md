# Report: phase01-smoke

- authored_by: runner
- created_at: 2026-09-08T16:57:19Z
- request_folder: agent-handoff/__phase01-smoke/
- tested_ref: main@1d9140d29e8335b06dee5927c7dc62140f40fe6e
- status: blocked

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# python: /workspace/DecisionForecast/.venv (CPython 3.12.3), uv 0.9.0
uv sync
uv run python scripts/download_fnspid.py
uv run python scripts/precache_fnspid_news.py --ticker-set=dev
uv run python -m src.training.train \
  model=a \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```

## Outcome
- exit_code: uv sync=0; download_fnspid.py=0; precache=1; train=1
- duration: uv sync ~14m; FNSPID download+extract ~8m; precache fail immediate; train fail after ~2m torch import
- host: runpod pod `p5cnr4o8bnd4c4` (DecisionForecastPod), volume `/workspace`
- device: NVIDIA L4 23034 MiB (CUDA 13.0); unused — train never reached model build

`uv sync` installed 140 packages into `.venv` on the volume (torch 2.13.0 + CUDA 13 wheels). FNSPID (`Zihan1004/FNSPID`) is **not gated**; downloaded without `HF_TOKEN` (unauthenticated Hub warning only). Layout on volume:

- `Data/FNSPID/` ≈ 31G
- `Stock_news/nasdaq_exteral_data.csv` ≈ 22G
- `Stock_news/All_external.csv` ≈ 5.4G
- `Stock_price/full_history.zip` extracted to `Stock_price/full_history/` (~7700 CSVs under nested `full_history/`)

Pre-cache and train both abort at import: **`ModuleNotFoundError: No module named 'src.data'`**.

Hydra `configs/config.yaml` defaults to `data: fnspid`, but **`configs/data/` is missing** (`configs/data/fnspid.yaml` not in the tree). `src/training/train.py` and `src/explain/projection.py` import `src.data.collate` / `src.data.dataset`; that package is not in `1d9140d`.

No Hydra `outputs/` run dir was created.

## Key signals
- metrics: n/a (train did not start)
- oom: no
- traceback_summary: precache `scripts/precache_fnspid_news.py:9` → `from src.data.dataset import TickerDataStore` → `ModuleNotFoundError: No module named 'src.data'`. train `src/training/train.py:12` → `from src.data.collate import forecast_collate` → same.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: n/a (no `outputs/` run)
- explain: n/a
- mlflow_run_id: n/a
- data: `/workspace/DecisionForecast/Data/FNSPID` (gitignored; on network volume)
- venv: `/workspace/DecisionForecast/.venv`

## Conclusions for Dev
1. Land **phase 1 data package** before the next Runner smoke: `src/data/` (`dataset.py` with `TickerDataStore` / `FNSPIDForecastDataset` / `build_datasets`, `collate.py` with `forecast_collate`) matching the imports in `src/training/train.py` and `src/explain/projection.py`.
2. Add **`configs/data/fnspid.yaml`** (T/P/H, split dates, `tickers.dev` + `tickers.paper`, `root`/`news_source`/`features`/`target`/`text.dim`) so Hydra `defaults: data: fnspid` resolves. Precache defaults to this path.
3. Environment on this pod is ready: uv env, L4 GPU, FNSPID on volume. Re-run the same capped `model=a` command after those files exist; do not re-download FNSPID unless the tree is wiped.

## Suggested next command (optional)
```bash
# after src/data + configs/data/fnspid.yaml are on main
uv run python scripts/precache_fnspid_news.py --ticker-set=dev
uv run python -m src.training.train \
  model=a \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```
