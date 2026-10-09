# Request: protocol-norm

- authored_by: dev
- created_at: 2026-09-10T20:49:29Z
- target_ref: feat/phase2-timexer@2c733ae
- phase: 1
- priority: high

## Goal
Sprint 1 + Sprint 3 sanity on the **new data protocol**: `split.train_start=2015-01-01`, `normalize=per_window`, new mu file `mu_dev_2015-01-01_2021-12-31.npy`. Uncapped **dev** windows, H=7, 20 epochs. Models: **dlinear**, **timexer_plain**, **a** only.

Success:
- all three trains exit 0, CUDA, `METRICS_ROW` + `test_mae_denorm`
- `TEXT_COVERAGE` train `zero_windows` **noticeably below 0.42** (old AAPL-from-1980 figure)
- for **each** model, val-MSE and test-MSE are **the same order of magnitude** (no ~0.4 val vs ~5–18 test)
- sanity band `0.7 ≤ mse_plain/mse_dlinear ≤ 1.3` — **record** pass or fail, **do not tune**
- log epoch time and VRAM (`log_cuda_memory` / nvidia-smi)

Not paper. Do not invent metrics.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Do **not** cap windows. Do **not** run b/c0/c1.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 7b68aaf

uv sync
uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
# expect a NEW file:
# Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy
# old mu_dev.npy may remain; do not delete it

SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7"
for m in dlinear timexer_plain a; do
  uv run python -m src.training.train model=$m $SHARED
done
```

Grep: `TEXT_COVERAGE`, `METRICS_ROW`, `n_params`, `mae_denorm`, `First batch shapes`, val_mse on last epoch.

Table from the three `metrics.json` + coverage logs only:

```text
| model | val_mse (last epoch) | test_mse | test_mae | test_mae_denorm | n_params |
| dlinear | … | … | … | … | … |
| timexer_plain | … | … | … | … | … |
| a | … | … | … | … | … |
| ratio plain/dlinear test_mse | … |  |  |  |  |
```

Also paste train/val/test `zero_windows` from TEXT_COVERAGE.

If val/test MSE still differ by an order of magnitude → `fail` (protocol not fixed). If trains ok but sanity band out → `partial`. If both protocol (same-order MSE) and trains green, band optional → `pass` even if band fails **only if** you still write the ratio; prefer `partial` when band fails but protocol looks fixed.

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (prices + article caches); **mu file must be rebuilt**
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `7b68aaf`
- windows: **uncapped** on 15 dev tickers from 2015 — longer than previous 64-window smokes; L4 should hold batch 32

## Out of scope
- Do not change product code unless a trivial one-liner unblocks.
- Do not run `model=b/c0/c1`, H={14,30}, seeds, or paper.
- Do not tune lr/epochs/caps to chase the DLinear band.
- Do not commit `outputs/`, caches, `mlflow.db`.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
