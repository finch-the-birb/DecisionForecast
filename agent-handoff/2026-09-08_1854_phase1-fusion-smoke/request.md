# Request: phase1-fusion-smoke

- authored_by: dev
- created_at: 2026-09-08T18:54:49Z
- target_ref: feat/phase1-data@ed00044b8be982d7b0068c4ae95cbeaf67192f0d
- phase: 1
- priority: high

## Goal
Re-run capped Model A after the two findings in `2026-09-08_1834_phase1-data-smoke` (status pass, but price-only). Confirm: (1) NVDA loads via case-insensitive `nvda.csv`; (2) `max_*_windows` takes **recent** windows so news/MiniLM actually run; (3) first-batch text L2 is **not** ~0.

Success: train exit 0; no OOM; split logs show 2019–2023-ish dates and NVDA among tickers; `First batch text L2 mean` > 0; test mse/mae logged; projection JSON written.

## Commands
FNSPID and news parquet cache are already on the volume — **do not re-download**. Precache should be a cache-hit.

```bash
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

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (`Data/FNSPID` + `Data/FNSPID/cache/nasdaq_exteral_data/`)
- gpu: optional (auto); MiniLM will load during train this time
- approx_ram_gb: 16+ is enough if news parquet is cached (skip the 22G CSV scan)
- git: checkout `feat/phase1-data` at this request commit (descendant of `ed00044`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.

Please record in Key signals: split date ranges, whether NVDA was resolved, first-batch text L2, and test mse/mae.
