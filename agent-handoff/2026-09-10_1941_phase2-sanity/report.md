# Report: phase2-sanity

- authored_by: runner
- created_at: 2026-09-10T19:59:10Z
- request_folder: agent-handoff/2026-09-10_1941_phase2-sanity/
- tested_ref: feat/phase2-timexer@55adf8931a0870d3fa0b8933a68b1b00695b8d58 (contains a2815ce)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not a/b/c0/c1. Not paper.
# mu_dev.npy already on volume — text precache not re-run.
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
# HEAD=55adf89 (branch tip after request commit c8ed0fb)
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

## Outcome
- exit_code: uv sync=0; dlinear=0; timexer_plain=0
- duration: both trains ~3m 19s (198868 ms)
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4. First batch `x=cuda:0`. No OOM. No CPU fallback. Not paper.

METRICS_ROW (verbatim):
```
METRICS_ROW model=dlinear horizon=7 mse=1.6030 mae=0.7331
METRICS_ROW model=timexer_plain horizon=7 mse=7.2417 mae=1.5336
```

n_params (verbatim logs):
```
Model parameters on cuda:0 n_params=854          # dlinear
Model parameters on cuda:0 n_params=146695       # timexer_plain
```

Table from the two `metrics.json` files (n_params from logs; json has no n_params):

| model | mse | mae | n_params |
| dlinear | 1.6030 | 0.7331 | 854 |
| timexer_plain | 7.2417 | 1.5336 | 146695 |
| ratio plain/dlinear mse | 4.5175 |  |  |

Sanity band `0.7 ≤ mse_plain/mse_dlinear ≤ 1.3`: **out of band** (4.5175). Trains succeeded → `partial`.

## Key signals
- metrics: see table (metrics.json: dlinear mse=1.6030339 mae=0.7330509; timexer_plain mse=7.2417192 mae=1.5336193)
- epoch-10 val (logs, not json): dlinear val_mse=0.4618; timexer_plain val_mse=0.4172
- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- dlinear: `outputs/2026-09-10/19-56-15/` (`metrics.json`, `checkpoints/best.pt`)
- timexer_plain: `outputs/2026-09-10/19-57-49/` (`metrics.json`, `checkpoints/best.pt`)

## Conclusions for Dev
1. Pipeline check **ok**: both models train on CUDA, log `n_params` / `METRICS_ROW`, H=7, ticker_set=dev, 10 epochs, capped windows.
2. Sanity band **failed**: TimeXer_plain test MSE is **4.52×** DLinear (need ≤1.3×). Not a crash.
3. Val MSE is close (plain 0.417 vs DLinear 0.462); the gap is **test**. Plain train_loss still falling at epoch 10 (2.045) with 146k params vs DLinear 854 — likely overfit on 64 windows, not a proven backbone bug.
4. Dataloader still emits `text`/`text_seq` (dataset protocol); models are prototype-free / text-unused by config.

## Suggested next command (optional)
```bash
# only if Dev wants a less overfit sanity (still not paper); new request.md
# raise max_*_windows or epochs; do not run a/b/c0/c1 here
```
