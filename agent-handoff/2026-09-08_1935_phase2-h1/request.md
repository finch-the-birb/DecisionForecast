# Request: phase2-h1

- authored_by: dev
- created_at: 2026-09-08T19:35:19Z
- target_ref: feat/phase1-data@88faf1a54bf02a2a57e5e4e2a7ae4dd150cfe627
- phase: 2
- priority: normal

## Goal
Capped H1 smoke: **Model A vs Model B** on the same protocol (dev tickers, H=7, 1 epoch, same window caps, `train.device=cuda`). B is TimeXer + G1+G2 prototypes + **late** text fusion (`FinLang/finance-embeddings-investopedia`).

Success: both trains exit 0; no OOM; no HF 401; CUDA for TimeXL **and** TimeXer; first-batch text L2 > 0; each run logs `H1_ROW model=… mse=… mae=…` and writes `metrics.json`; projection JSON for a and b. Fill the A vs B table in this report from those numbers only (Δ(A→B) = B − A). This is a smoke table, not a paper result.

## Commands
FNSPID, news parquet, and the public encoder cache are already on the volume — **do not re-download**, **no HF_TOKEN**.

```bash
uv sync
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
uv run python -m src.training.train \
  model=b \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```

After both runs, grep logs for `H1_ROW` and copy `metrics.json` from each Hydra `outputs/` dir into Key signals as:

```text
| model | mse | mae |
| A | … | … |
| B | … | … |
| Δ(A→B) | … | … |
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes
- gpu: **required** (`train.device=cuda`)
- approx_ram_gb: 16+
- git: checkout `feat/phase1-data` at this request commit (descendant of `88faf1a`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.
- Do not implement C0/C1. Do not invent metrics.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
