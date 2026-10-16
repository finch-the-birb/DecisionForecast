# Request: phase2-sanity

- authored_by: dev
- created_at: 2026-09-10T19:41:01Z
- target_ref: feat/phase2-timexer@a2815ce637b8552c6b39e839531cb170b092e5e7
- phase: 2
- priority: normal

## Goal
Sprint 3 sanity: **DLinear** vs **TimeXer_plain** (no prototypes, no text in the model) on the same capped FNSPID protocol. This is a pipeline check, not H1/H2. TimeXer uses multivariate PatchEmbed over all OHLCV channels and a single \(G_{en}\) with `exo=None`.

Success: both trains exit 0; CUDA; `METRICS_ROW` for each; `n_params` logged; TimeXer_plain test MSE within **±30%** of DLinear (i.e. `0.7 ≤ mse_plain/mse_dlinear ≤ 1.3`) as a sanity band. If outside the band, still write the numbers — status `partial` if trains succeeded but the ratio is out of band. Smoke / sanity only — not paper.

## Commands
Text-series cache from `text-protocol` is on the volume — **do not re-download FNSPID**, **no HF_TOKEN**, **not** `ticker_set=paper`. Re-run text precache only if `mu_dev.npy` is missing.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain a2815ce

uv sync
for m in dlinear timexer_plain; do
  uv run python -m src.training.train \
    model=$m \
    train.device=cuda \
    train.ticker_set=dev \
    train.max_train_windows=64 \
    train.max_val_windows=32 \
    train.max_test_windows=32 \
    train.epochs=10 \
    data.horizon=7
done
```

Grep `METRICS_ROW` and `n_params`. Table from the two `metrics.json` files only:

```text
| model | mse | mae | n_params |
| dlinear | … | … | … |
| timexer_plain | … | … | … |
| ratio plain/dlinear mse | … |  |  |
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (`cache/text_series/` from text-protocol)
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+
- git: `feat/phase2-timexer` at this request commit (descendant of `a2815ce`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock.
- Do not run `model=a/b/c0/c1`, H5, or `ticker_set=paper`.
- Do not invent metrics. Do not commit `outputs/` or caches.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
