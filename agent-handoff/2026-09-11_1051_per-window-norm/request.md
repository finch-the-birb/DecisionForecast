# Request: per-window-norm

- authored_by: dev
- created_at: 2026-09-11T10:51:00Z
- target_ref: feat/phase2-timexer@9da44d1
- phase: 1
- priority: high

## Goal
Confirm **per-window** (RevIN-style) price normalize + `train_start=2015-01-01` on uncapped **dev**. Models: **dlinear**, **a**, **timexer_plain**. H=7, 10 epochs. Rebuild μ if the dated file is missing.

This protocol is already on the branch (since `2c733ae`). `protocol-norm` on 2026-09-10 was `partial` only because of `zero_windows`; val/test scale and DLinear band already passed. This round is a **confirmation** on current HEAD (includes `tests/test_per_window_norm.py`). Do not treat as a new algorithm.

Success:
- all three trains exit 0, CUDA, `METRICS_ROW` + `test_mae_denorm`
- `test_mse` in **0.1–1.5** (not 5–18)
- val-MSE and test-MSE **same order of magnitude**
- log `data.normalize=per_window` / `train_start` (config_resolved or MLflow)

Not paper. Do not invent metrics.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Do **not** cap windows. Do **not** run b/c0/c1.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 9da44d1

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

SHARED="train.device=cuda train.ticker_set=dev train.epochs=10 data.horizon=7"
uv run python -m src.training.train model=dlinear $SHARED
uv run python -m src.training.train model=a $SHARED
uv run python -m src.training.train model=timexer_plain $SHARED
```

Grep: `TEXT_COVERAGE`, `METRICS_ROW`, `mae_denorm`, `train device:`, last-epoch `val_mse`.

Table from the three `metrics.json` + logs:

```text
| model | val_mse (last epoch) | test_mse | test_mae | test_mae_denorm | n_params |
| dlinear | … | … | … | … | … |
| a | … | … | … | … | … |
| timexer_plain | … | … | … | … | … |
```

Status:
- `fail` — crash, or val vs test still an order of magnitude apart, or test_mse still 5–18
- `partial` — trains green but one model outside 0.1–1.5
- `pass` — all three in 0.1–1.5 and same-order val/test

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if the volume still has FNSPID + text caches; **μ must exist or be rebuilt**
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `9da44d1` (code protocol `2c733ae`)
- prior: `protocol-norm` already showed DLinear test_mse **0.67**, A **0.98**, plain **0.72** (uncapped, 20-epoch budget, early stop). Expect the same order.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run `model=b/c0/c1`, seeds, H∈{14,30}, or paper.
- Do not tune lr. Do not commit `outputs/`, caches, `mlflow.db`.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
