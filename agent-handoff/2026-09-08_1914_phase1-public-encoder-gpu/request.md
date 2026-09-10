# Request: phase1-public-encoder-gpu

- authored_by: dev
- created_at: 2026-09-08T19:14:07Z
- target_ref: feat/phase1-data@1553671ef3b124ea12b7751ff83066eaa0b4b307
- phase: 1
- priority: high

## Goal
Re-run the CUDA Model A smoke after `2026-09-08_1904_phase1-finance-encoder-gpu` **blocked** on Hub 401. Encoder id is now the **public** repo `FinLang/finance-embeddings-investopedia` (`gated: false`). Do **not** set `HF_TOKEN`. Same GPU asserts as last round.

Success:
- encoder downloads without 401
- logs: `Text encoder ready: FinLang/finance-embeddings-investopedia device=cuda`
- L4, model params on cuda, first-batch CUDA mem after encoder load
- first-batch text L2 > 0
- train exit 0; test mse/mae; projection JSON

## Commands
FNSPID + news parquet already on volume. No download. No HF token.

```bash
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

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes
- gpu: **required** (`train.device=cuda`)
- approx_ram_gb: 16+
- Hub: public encoder, unauthenticated download expected
- git: `feat/phase1-data` at this request commit (descendant of `1553671`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.
- Do not request or commit `HF_TOKEN`.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.

Key signals: GPU name, encoder device, CUDA MiB after model.to **and** first batch (encoder should bump VRAM), text L2, test mse/mae.
