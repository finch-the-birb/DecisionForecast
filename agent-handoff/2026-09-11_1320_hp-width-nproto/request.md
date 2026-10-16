# Request: hp-width-nproto

- authored_by: dev
- created_at: 2026-09-11T13:20:00Z
- target_ref: feat/phase2-timexer@03f123b4cd86f0ed319d8c0368a221667d2be399
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** HP sweep on current HEAD (Coverage + DataLoader prefetch). **8 sequential jobs**, `train.lr=0.0003`, seed=42, H=7, epochs=20, patience=5. **No window caps.**

Grid (do **not** cartesian `d_model` × `d_ff`):

1. **Paired width** on TimeXer models: `model=b,c0,c1` × `(d_model=64, d_ff=256)` vs `(d_model=128, d_ff=512)` → 6 jobs.
2. **n_prototypes** on A (no `d_ff`): `model=a` × `n_prototypes=5,10` → 2 jobs.

Freeze lr at **0.0003** (hp-uncapped-dev: 0.001 was worse last-epoch val on all four models). Re-run the default `(64, 256)` / A `n_prototypes=10` on this HEAD — `num_workers=4` changes shuffle, so **do not mix** with the previous 8-row table.

Winner = **last-epoch val_mse**. Test is recorded, not used to pick HP. Table from **this** sweep's logs only. Not paper.

Also confirm: `TEXT_COVERAGE` still prints (same format); `DataLoader num_workers=4 pin_memory=True`; coverage is fast (not minutes of CPU before the first batch). Coverage numbers may match the known uncapped line (`train windows=23478 … zero_windows=0.565`); record them, do not fail on 0.42.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: hp-width-nproto report (started)`).
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
git rev-parse HEAD   # must contain 03f123b4cd86f0ed319d8c0368a221667d2be399 (Coverage + DataLoader)

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/hp-width-nproto.log
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
run model=a model.n_prototypes=5
run model=a model.n_prototypes=10
for m in b c0 c1; do
  run model=$m model.d_model=64 model.d_ff=256
  run model=$m model.d_model=128 model.d_ff=512
done
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `val_mse`, `train device:`, `First batch tensors`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

Final table (**this sweep only**):

```text
| model | d_model | d_ff | n_prototypes | val_mse | test_mse | test_mae | mae_denorm | n_params |
| a | 64 | n/a | 5 | … | … | … | … | … |
| a | 64 | n/a | 10 | … | … | … | … | … |
| b | 64 | 256 | 10 | … | … | … | … | … |
| b | 128 | 512 | 10 | … | … | … | … | … |
| c0 | 64 | 256 | 10 | … | … | … | … | … |
| c0 | 128 | 512 | 10 | … | … | … | … | … |
| c1 | 64 | 256 | 10 | … | … | … | … | … |
| c1 | 128 | 512 | 10 | … | … | … | … | … |
```

Mark the row with best **val_mse**. Paste one `train device:` / `x=cuda:0` line and one `DataLoader num_workers=` line. Paste the three `TEXT_COVERAGE split=` lines (once).

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 8/8, CUDA, workers=4, full table, val winner named

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if volume has FNSPID + dated μ (confirmed on hp-uncapped-dev)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `03f123b4cd86f0ed319d8c0368a221667d2be399` (must include Coverage + DataLoader; default `train.num_workers=4`)
- budget: ~40–80 min (previous 8-job uncapped was ~57 min with the old coverage loop)
- prior: `hp-uncapped-dev` `pass` @ `2f85763` / code `b082360`. lr=0.001 not in this grid. Capped 256-window sweep is invalid. Do not mix tables.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run dlinear / timexer_plain / paper / extra seeds / lr=0.001 / unpaired `d_model`×`d_ff` / caps 256 or 64.
- Do not run `ablation_table`. Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
