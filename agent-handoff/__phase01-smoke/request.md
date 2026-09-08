# Request: phase01-smoke

- authored_by: runner
- created_at: 2026-09-08T16:57:19Z
- target_ref: main@1d9140d29e8335b06dee5927c7dc62140f40fe6e
- phase: 0/1
- priority: normal

## Goal
Capped Model A smoke on the Runpod host (dev tickers, 1 epoch) after FNSPID download + news precache. Confirm Hydra train entry, dataset construction, and one train/val/test pass without OOM.

## Commands
No Dev `request.md` was on `main`. Commands below are the user-specified fallback.

```bash
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

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes after download (Data/FNSPID on volume)
- gpu: NVIDIA L4 23 GiB (optional for this smoke; train.device=auto)
- approx_ram_gb: host RAM available; FNSPID news CSV ~22G on disk

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
