# Report: phase1-data-smoke

- authored_by: runner
- created_at: 2026-09-08T18:48:47Z
- request_folder: agent-handoff/2026-09-08_1834_phase1-data-smoke/
- tested_ref: feat/phase1-data@32d9ef44bec4749cb56b8bd7a148563ce64d830a
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD during run: 32d9ef4 (handoff request; descendant of f3e69fd)
# skipped download_fnspid.py — Data/FNSPID already on volume
uv sync
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
- exit_code: uv sync=0; precache=0; train=0
- duration: uv sync ~1s (already installed); precache ~5m 37s; train ~2m 49s (includes first MiniLM download after epoch 1)
- host: runpod pod `p5cnr4o8bnd4c4`, volume `/workspace`
- device: cuda (NVIDIA L4)

Dataset sizes: train 64, val 32, test 32. Hydra experiment `a_H7`.

Precache (one-pass, all 15 dev tickers including FB): AAPL 9338, MSFT 8737, GOOGL 1754, AMZN 5060, FB 433, NVDA 11862, TSLA 10587, JPM 2883, BAC 1811, WMT 8686, JNJ 2928, PG 1312, XOM 7346, DIS 9654, NFLX 3028.

Train warning: `Skipping ticker NVDA: No price file for NVDA: .../full_history/full_history/NVDA.csv`. On this volume the file is `nvda.csv` (lowercase). FB prices exist as `FB.csv`; META is absent.

## Key signals
- metrics: epoch1 train_loss=1.9090 val_mse=4.6214 val_mae=2.1457; test mse=2.7462 mae=1.6483
- oom: no
- traceback_summary: n/a

Window caps take the **first** N indices. Those are the earliest AAPL bars (`end_date` 1981-03-11 in projection JSON). News aggregation for those windows is empty, so `TextEmbeddingCache.encode` returns zeros **without loading MiniLM during train/val/test**. The encoder only loaded afterward for `save_projection_examples`. This smoke is therefore price-only (zero text). Metrics are not a text-fusion signal.

## Artifacts (paths on Runner disk — do not commit binaries)
- hydra run: `outputs/2026-09-08/18-46-59/`
- train_log: `outputs/2026-09-08/18-46-59/train.log`
- checkpoint: `outputs/2026-09-08/18-46-59/checkpoints/best.pt`
- explain: `outputs/2026-09-08/18-46-59/explain/projection_examples_a.json`
- config: `outputs/2026-09-08/18-46-59/config_resolved.yaml`
- mlflow_run_id: `be709ed259b04251b0507a8f82ca5385` (experiment `decision-forecast`, sqlite `mlflow.db`)
- news parquet cache: `Data/FNSPID/cache/nasdaq_exteral_data/`

## Conclusions for Dev
1. Phase 1 data package unblocks Model A: precache + capped train complete on L4; projection JSON written. Previous `__phase01-smoke` blocker is gone.
2. Price loader is case-sensitive. FNSPID extracts `nvda.csv` not `NVDA.csv` — NVDA was dropped from this run. Normalize ticker filenames (or look up case-insensitively) before paper ticker sets.
3. `max_*_windows` slices from the start of the index list, so this cap is 1981 AAPL, empty news, zero text vectors. For a real fusion smoke, cap from the **end** of each split (or sample recent windows) so MiniLM actually runs in `__getitem__`.
4. FB news cached (433 rows); the META rename is not the skip that fired here — NVDA casing is.

## Suggested next command (optional)
```bash
# after case-insensitive prices + recent-window cap
uv run python -m src.training.train \
  model=a \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```
