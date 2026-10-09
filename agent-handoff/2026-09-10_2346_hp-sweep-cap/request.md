# Request: hp-sweep-cap

- authored_by: dev
- created_at: 2026-09-10T23:46:00Z
- target_ref: feat/phase2-timexer@2c733ae
- phase: adhoc
- priority: normal

## Goal
Capped **hyperparameter sweep** on the accepted protocol (`train_start=2015-01-01`, `normalize=per_window`). Not paper. Not a replacement for R3.

Grid (Hydra `-m`, **12 sequential** jobs): `model=a,b,c1` × `train.lr=0.0003,0.001` × `model.d_model=64,128`. Caps **256/64/64** (not the smoke 64/32/32 — that is only two train batches). H=7, epochs=20, patience=5, seed=42. `d_ff` stays 256 even when `d_model=128`.

Do **not** mix these runs into the uncapped R3 table. Do **not** run `ablation_table` (it will pick up seed=42 / uncapped jobs). Build the table from **this** sweep's `METRICS_ROW` / `metrics.json` only.

Success: 12/12 exit 0, CUDA, val/test same order, table of all 12 rows + which (model, lr, d_model) won **val_mse**. Negative / tiny Δ vs R3 is valid. HP on 256 windows must not be treated as paper HP.

## Waiting (do not block the chat on GPU)

**Do not** run the sweep in the foreground. Cursor cannot subscribe to “process on the pod finished”; it can subscribe to a **timer** or to a **git push**.

1. Start the sweep with `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately with `status: partial`, pid, log path, `tested_ref`. Commit + push (`handoff: hp-sweep-cap report (started)`). Then **end the turn**.
3. Subscribe a **one-shot timer** (~90 min, `delaySeconds=5400`): if the log has no `SWEEP_DONE` yet, re-subscribe once more (same delay) and end the turn again. **Do not** `sleep` / `nvidia-smi` poll loops.
4. When `SWEEP_DONE` is in the log (or the process is gone and the log is complete): harvest metrics, overwrite `report.md` with the final table, `status: pass|fail|partial`, commit + push (`handoff: hp-sweep-cap report (<status>)`).

If `nohup` quoting fails, start the same `uv run … -m …` in `tmux` and still follow steps 2–4. Do not sit in the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential only. Precache only if the dated mu is missing. Do not change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 2c733ae

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/hp-sweep-cap.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
uv run python -m src.training.train -m \
  model=a,b,c1 \
  train.lr=0.0003,0.001 \
  model.d_model=64,128 \
  train.device=cuda \
  train.ticker_set=dev \
  train.epochs=20 \
  data.horizon=7 \
  train.max_train_windows=256 \
  train.max_val_windows=64 \
  train.max_test_windows=64
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID in the first turn.

Grep the finished log: `METRICS_ROW`, `train device:`, `First batch tensors`, `Error`, `SWEEP_DONE`. Hydra sweep dir is `outputs/multirun/<date>/<time>/`.

Final table (from **this** sweep only):

```text
| model | lr | d_model | val_mse | test_mse | test_mae | mae_denorm | n_params |
| a | 0.0003 | 64 | … | … | … | … | … |
| … | … | … | … | … | … | … | … |
```

Mark the row with best **val_mse**. Also paste `train device:` / `x=cuda:0` from one job (GPU check).

Status:
- `fail` — majority crash, or all jobs on CPU
- `partial` — started (first push), **or** some jobs failed, **or** table incomplete
- `pass` — 12/12, CUDA, full table, winner by val_mse named

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (dated mu from protocol-norm)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `2c733ae`
- prior: R3 `dev-full-ablation` `partial` (n=4 table mix). This sweep is separate.
- windows: `_cap_recent` 256/64/64 — not smoke 64/32/32, not uncapped

## Out of scope
- Do not change product code unless a one-liner unblocks Hydra `-m`.
- Do not run `model=c0`, `dlinear`, `timexer_plain`, extra seeds, H∈{14,30}, or `ticker_set=paper`.
- Do not expand the grid. Do not run `ablation_table`.
- Do not commit `outputs/`, caches, `mlflow.db`.

## After the run
Same folder `report.md`, commit on this branch, push. First push = started; last push = metrics.
