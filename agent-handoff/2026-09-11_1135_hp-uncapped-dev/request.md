# Request: hp-uncapped-dev

- authored_by: dev
- created_at: 2026-09-11T11:35:00Z
- target_ref: feat/phase2-timexer@79d2ff7
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** HP sweep on the accepted protocol (`train_start=2015-01-01`, `normalize=per_window`). **8 sequential jobs:** `model=a,b,c0,c1` × `train.lr=0.0003,0.001`. H=7, seed=42, epochs=20, patience=5. **No window caps.**

Do **not** sweep `d_model`, `d_ff`, H, lookback, dropout, extra seeds, or `ticker_set=paper`. Do **not** run `ablation_table` (it will mix R3 / cap-sweep). Table from **this** sweep's logs / `metrics.json` only.

Choose the winner by **val_mse** (last epoch or best.pt val — use last-epoch val from the log, plus test from `METRICS_ROW`). Test is recorded, not used to pick HP. Capped 256-window sweep is **invalid** for this decision.

Success: 8/8 exit 0, CUDA, 8-row table, best val_mse named. Not paper.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: hp-uncapped-dev report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=5400` (90 min, safety net only). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the table, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 79d2ff7

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/hp-uncapped-dev.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7 train.seed=42 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
for m in a b c0 c1; do
  for lr in 0.0003 0.001; do
    echo "=== START model=$m lr=$lr $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    uv run python -m src.training.train model=$m train.lr=$lr $SHARED
    echo "=== DONE  model=$m lr=$lr $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  done
done
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `val_mse`, `train device:`, `First batch tensors`, `=== START`, `=== DONE`, `SWEEP_DONE`.

Final table (**this sweep only**):

```text
| model | lr | val_mse | test_mse | test_mae | mae_denorm | n_params |
| a | 0.0003 | … | … | … | … | … |
| a | 0.001 | … | … | … | … | … |
| b | 0.0003 | … | … | … | … | … |
| b | 0.001 | … | … | … | … | … |
| c0 | 0.0003 | … | … | … | … | … |
| c0 | 0.001 | … | … | … | … | … |
| c1 | 0.0003 | … | … | … | … | … |
| c1 | 0.001 | … | … | … | … | … |
```

Mark the row with best **val_mse**. Paste `train device:` / `x=cuda:0` from one job.

Status:
- `fail` — majority crash or CPU-only
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 8/8, CUDA, full table, val winner named

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if volume has FNSPID + dated μ (confirmed on per-window-norm)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `79d2ff7`
- budget: ~40–80 min (R3 was ~5 min/job × 8)
- prior: `per-window-norm` `pass`; R3 uncapped seeds 0/1/2 at lr=0.001. This grid is seed=42 × two lrs. Do not mix tables.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run dlinear / timexer_plain / paper / extra seeds / d_model / caps 256 or 64.
- Do not run `ablation_table`. Do not commit `outputs/` or `mlflow.db`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
