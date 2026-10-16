# Request: paper-timexer-5ts-compact

- authored_by: dev
- created_at: 2026-10-05T23:40:00Z
- target_ref: feat/phase2-timexer@a9006d25f91bd102ac5d8f1a0f958921b4dced46
- phase: 4
- priority: high

## Goal
Paper grid of TimeXer on the 5 OHLCV channels plus the frozen 15D compact text. Do not rebuild that text cache. `data=fnspid_compact` reads prices through the usual per-window RevIN and aligns `cache/text_compact/{ticker}.parquet` onto the price calendar. `y` is scaled by the close channel.

`model=c1_compact` uses `PatchEmbed` on 5 channels (`5 * 12 = 60 -> 64`), `Linear(15, 64)` for the text, and a global-to-patch bridge on every encoder layer. `model.use_prototypes=false` returns `proto_losses=None` and trains with Huber δ=0.5 only. `model.use_prototypes=true` adds `PrototypeResidual(n_prototypes=10)` before the encoder and the three prototype penalties (`lambda_c=0.1`, `lambda_e=0.1`, `lambda_d=0.01`). Head pool is `last`.

Twelve jobs: `e_layers` in {1, 2} × `use_prototypes` in {false, true} × seeds 0, 1, 2. Each job must see `x` `[B, 60, 5]` and `text_seq` `[B, 60, 15]`. The code raises if it does not.

Success: all 12 trains exit 0; no OOM; each run logs `x` last dim 5 and `text_seq` last dim 15; a `METRICS_ROW` per job. Status `partial` if some trains succeed. This is the paper protocol (epochs 20, uncapped windows), not a smoke.

## Commands
If `git fetch` fails authentication, copy the deploy key already on the volume (`/workspace/.ssh/id_ed25519_github` and `.pub`) into `~/.ssh` the same way as `phase4-pipeline-smoke`. Do not print the private key. Do not commit it.

FNSPID and `cache/text_compact` are already on the volume from `paper-timexer-selected-26ts`. Do not run `scripts/download_fnspid.py`. Do not run `scripts/prepare_selected_data.py`. Do not set `HF_TOKEN`. Do not re-encode FinBERT or FinLang.

```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git merge-base --is-ancestor a9006d25f91bd102ac5d8f1a0f958921b4dced46 HEAD
uv sync

mkdir -p outputs
LOG=outputs/paper-timexer-5ts-compact.log
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
echo "=== SWEEP $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

SHARED="train.device=cuda train.ticker_set=paper train.epochs=20 train.patience=5 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
BASE="model=c1_compact data=fnspid_compact model.loss.kind=huber model.loss.delta=0.5 model.head.pool=last"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED $BASE "$@"
  echo "=== DONE $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

for e in 1 2; do
  for s in 0 1 2; do
    run model.e_layers=$e model.use_prototypes=false model.loss.lambda_c=0 model.loss.lambda_e=0 model.loss.lambda_d=0 train.seed=$s
  done
done
for e in 1 2; do
  for s in 0 1 2; do
    run model.e_layers=$e model.use_prototypes=true model.n_prototypes=10 model.loss.lambda_c=0.1 model.loss.lambda_e=0.1 model.loss.lambda_d=0.01 train.seed=$s
  done
done
echo "=== SWEEP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
' > "$LOG" 2>&1 &
echo "PID $! LOG $LOG"
```

## Waiting (do not monitor the GPU)

Do not run the grid in the foreground. Do not poll `nvidia-smi`, `tail -f`, or `sleep` loops.

1. Start the `nohup` above. Print PID and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: paper-timexer-5ts-compact report (started)`).
3. Subscribe one one-shot timer: `delaySeconds=14400` (4 h). Then end the turn.
4. If training finishes earlier, the human will ping this chat — then harvest. Do not watch the process.
5. Harvest only when the timer fires and the log contains `SWEEP_DONE` (or the PID is dead and the log is complete), or the human asks. Overwrite `report.md` with the table below, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is still running: one line (`still running`, jobs completed N/12), push, and stop. Do not start another timer.

## Report table
From the log only. Do not invent metrics.

```text
| e_layers | use_prototypes | seed | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
```

Also list any ticker the log skips, and the first-batch shapes. `best_val_mse` is the minimum epoch `val_mse`. `test_mse` / `test_mae` are the `METRICS_ROW` of `best.pt`.

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes. Prices are the FNSPID csv files. Compact text is `Data/FNSPID/cache/text_compact/{ticker}.parquet` from the previous paper prepare. A missing parquet skips that ticker; do not rebuild it.
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+ (no news-csv scan)
- git: `feat/phase2-timexer`, descendant of `a9006d25f91bd102ac5d8f1a0f958921b4dced46`

## Out of scope
- Do not change product code unless a trivial one-liner unblocks the run; prefer reporting the blocker.
- Do not run `model=a`, `model=b`, `model=c0`, `model=c1`, `model=dlinear`, `model=timexer_plain`, `model=timexer_selected`, or `ticker_set=dev`.
- Do not run `scripts/prepare_selected_data.py` or `scripts/download_fnspid.py`.
- Do not refit compact prototypes on 2022/2023.
- Do not commit `outputs/`, `mlruns/`, `mlflow.db`, checkpoints, `.env`, `.ssh/`, or anything under `Data/` / `cache/`.

## After the run
Write `report.md` in this same folder, commit on `feat/phase2-timexer`, push.
