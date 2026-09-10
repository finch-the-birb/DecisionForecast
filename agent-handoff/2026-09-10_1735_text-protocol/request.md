# Request: text-protocol

- authored_by: dev
- created_at: 2026-09-10T17:35:21Z
- target_ref: feat/phase1-data@d97b1faa9088eacefb7cd9d1d7e27781f1bd67ba
- phase: 1
- priority: normal

## Goal
Capped Model A smoke on the **new text protocol**: per-article embeddings → daily series (next-session bind, train-only `mu`, missing-day decay) → pooled `text` `[B,768]` and `text_seq` `[B,60,768]`. Old concat+`max_chars` is gone.

Success: precache exit 0; `mu_dev.npy` and per-ticker `.npz` under `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/`; news-coverage table in logs (AAPL/MSFT/NVDA/TSLA high, FB/BAC lower expected); `% windows with has_news_frac==0` < 100%; train exit 0; first-batch shapes `text [B,768]`, `text_seq [B,60,768]`; MSE/MAE + `explain/projection_examples_a.json`. Smoke only — not paper.

## Commands
FNSPID + news parquet already on volume `/workspace` — **do not re-download**, **no HF_TOKEN**, **not** `ticker_set=paper`. Public encoder `FinLang/finance-embeddings-investopedia`. Old `cache/embeddings/` window `.npy` is unused; new cache is `cache/text_series/`.

cwd: `/workspace/DecisionForecast`. Deploy key in `~/.ssh` before `git push` of `report.md`.

```bash
git fetch origin
git checkout feat/phase1-data
git pull --ff-only origin feat/phase1-data
git rev-parse HEAD   # must contain d97b1fa

uv sync
uv run python -m src.training.train --cfg job >/dev/null
uv run python scripts/precache_fnspid_news.py --ticker-set=dev
uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```

Grep logs for:
- `TEXT_COVERAGE`
- `First batch shapes`
- `METRICS_ROW`
- `Saved projection examples`

Copy the coverage table and first-batch shapes verbatim. Confirm files:

```text
Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev.npy
Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/articles/<ticker>.npz
Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/<ticker>.npz
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes
- gpu: optional (needed for fast encode; `precache_text_embeddings.py --device=cuda`)
- approx_ram_gb: 32
- git: checkout `feat/phase1-data` at this request commit (descendant of `d97b1fa`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run B/C0/C1, H5/LLM, `ticker_set=paper`, or uncapped trains.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.
- Do not invent metrics. Do not commit `outputs/`, caches, or checkpoints.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
