# Request: phase1-data-smoke

- authored_by: dev
- created_at: 2026-09-08T18:34:54Z
- target_ref: feat/phase1-data@f3e69fd91f1f34a68a896761877674639621bcbe
- phase: 1
- priority: high

## Goal
Unblock Phase 1 Model A smoke. Previous round (`agent-handoff/__phase01-smoke/`, status `blocked`) failed because `src/data/` and `configs/data/fnspid.yaml` were hidden by a bare `data/` gitignore rule and never reached git. This branch tracks those files, uses a single-pass news precache, and should complete capped `model=a` train/val/test (1 epoch) with MSE/MAE plus projection JSON.

Success: precache exit 0; train exit 0; no OOM; logs show dataset sizes > 0 and test mse/mae; `explain/projection_examples_a.json` written.

## Commands
FNSPID is already on the volume from the last round — **do not re-download**. Pull this branch, then:

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
- data_ready: yes (`Data/FNSPID` on volume from previous download; skip `scripts/download_fnspid.py`)
- gpu: optional (train.device=auto; sentence-transformers can use CUDA)
- approx_ram_gb: 32 recommended for the one-pass 22G news CSV scan (chunksize 200k)
- git: checkout `feat/phase1-data` (includes this request). Code SHA `f3e69fd`; missing-ticker files are skipped with a warning (e.g. FB vs META).

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
