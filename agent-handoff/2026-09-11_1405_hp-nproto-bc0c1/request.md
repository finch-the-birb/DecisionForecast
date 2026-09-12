# Request: hp-nproto-bc0c1

- authored_by: dev
- created_at: 2026-09-11T14:05:00Z
- target_ref: feat/phase2-timexer@02e91e0959ffe32b380e880a1535125aef8b502f
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** HP sweep: **n_prototypes on TimeXer models only**. **9 sequential jobs:** `model=b,c0,c1` × `model.n_prototypes=5,10,20`. Freeze `train.lr=0.0003`, `d_model=64`, `d_ff=256`, seed=42, H=7, epochs=20, patience=5, `train.num_workers=4`. **No window caps.**

Width and lr are already decided (`hp-width-nproto` / `hp-uncapped-dev`): `(128,512)` lost last-epoch val; lr=0.001 lost val. A already ran proto 5 vs 10 (10 won) — **do not re-run A**. Re-run proto=10 for B/C0/C1 **in this log** (do not copy rows from `hp-width-nproto`).

Winner = **last-epoch val_mse**. Test is recorded, not used to pick HP. Table from **this** sweep only. Not paper.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: hp-nproto-bc0c1 report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=5400` (90 min, safety net only). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the table, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not override `d_model` / `d_ff` (yaml 64/256).

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 02e91e0 (and 03f123b Coverage + DataLoader)

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/hp-nproto-bc0c1.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7 train.seed=42 train.lr=0.0003 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}
for m in b c0 c1; do
  for p in 5 10 20; do
    run model=$m model.n_prototypes=$p
  done
done
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `val_mse`, `train device:`, `First batch tensors`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

Final table (**this sweep only**):

```text
| model | n_prototypes | val_mse | test_mse | test_mae | mae_denorm | n_params |
| b | 5 | … | … | … | … | … |
| b | 10 | … | … | … | … | … |
| b | 20 | … | … | … | … | … |
| c0 | 5 | … | … | … | … | … |
| c0 | 10 | … | … | … | … | … |
| c0 | 20 | … | … | … | … | … |
| c1 | 5 | … | … | … | … | … |
| c1 | 10 | … | … | … | … | … |
| c1 | 20 | … | … | … | … | … |
```

Mark the row with best **last-epoch val_mse**. Paste one `train device:` / `x=cuda:0` line and one `DataLoader num_workers=` line. Paste the three `TEXT_COVERAGE split=` lines (once).

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 9/9, CUDA, workers=4, full table, val winner named

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if volume has FNSPID + dated μ (confirmed on hp-width-nproto)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `02e91e0` (must include `03f123b` Coverage + DataLoader)
- budget: ~30–50 min (hp-width-nproto 8 jobs was ~27 min)
- prior: `hp-width-nproto` `pass` — val winner B 64/256 proto=10; A proto=5 lost to 10; width 128/512 lost val. Do not mix tables.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run A / dlinear / timexer_plain / paper / extra seeds / lr=0.001 / d_model / d_ff / caps.
- Do not run `ablation_table`. Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
