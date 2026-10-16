# Request: paper-timexer-selected-26ts

- authored_by: dev
- created_at: 2026-10-05T21:20:00Z
- target_ref: feat/phase2-timexer@9bf6a81aa29f5ec12b57685fe3079b02ead74c9b
- phase: 4
- priority: high

## Goal
Recompute the paper fold-1 signature with the price level pinned, then train only `timexer_selected` at two depths.

`scripts/prepare_selected_data.py --ticker-set paper --fold 1` freezes FinBERT+FinLang prototypes and the TreeSHAP Top-25 on **walk-forward fold 1 train only** (2015-01-01 .. 2019-12-31, purge `H-1`, embargo 14). `close` is not a booster candidate. The frozen numeric block is `("close", *top_25)` — 26 channels. Compact text stays 15D. `build_datasets` keeps the existing paper split (train through 2021-12-31, val through 2022-12-31) and scales `y` by the lookback mean and std of channel 0. Do not refit the signature on 2020–2023.

The previous paper signature has 25 technical names and no `close`. `read_signature` rejects it. Prepare must rewrite `cache/selected_signatures/paper_fold1_signature.json` before any train.

`timexer_selected` must see `x` `[B, 60, 26]` and `text_seq` `[B, 60, 15]`. The code raises if it does not. Huber δ=0.5, `pool=last`. Run both capacities: `model.e_layers=1` and `model.e_layers=2`, seeds 0, 1, 2. Six jobs total. Each encoder layer has its own global-to-patch bridge.

Success: prepare exit 0; signature `ts_columns[0]` is `close` and `len(ts_columns)==26`; all 6 trains exit 0; no OOM; each run logs `x` last dim 26 and `text_seq` last dim 15; a `METRICS_ROW` per job. Status `partial` if prepare and some trains succeed. This is the paper protocol (epochs 20, uncapped windows), not a smoke.

## Commands
If `git fetch` fails authentication, copy the deploy key already on the volume (`/workspace/.ssh/id_ed25519_github` and `.pub`) into `~/.ssh` the same way as `phase4-pipeline-smoke`. Do not print the private key. Do not commit it.

FNSPID is already on the volume. Do not run `scripts/download_fnspid.py`. Do not set `HF_TOKEN`. FinBERT (`ProsusAI/finbert`) and the FinLang encoder may download on first use; that is expected. Reuse `cache/text_series` article npz and `cache/text_compact/sentiment` if they exist. Do not reuse the old 25-column signature.

```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git merge-base --is-ancestor 9bf6a81aa29f5ec12b57685fe3079b02ead74c9b HEAD
uv sync

mkdir -p outputs
LOG=outputs/paper-timexer-selected-26ts.log
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
echo "=== PREPARE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python scripts/prepare_selected_data.py --ticker-set paper --fold 1 --device cuda
echo "=== PREPARE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

SHARED="train.device=cuda train.ticker_set=paper train.epochs=20 train.patience=5 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

for s in 0 1 2; do
  run model=timexer_selected data=fnspid_selected model.e_layers=1 model.loss.kind=huber model.loss.delta=0.5 train.seed=$s
done
for s in 0 1 2; do
  run model=timexer_selected data=fnspid_selected model.e_layers=2 model.loss.kind=huber model.loss.delta=0.5 train.seed=$s
done
echo "=== SWEEP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
' > "$LOG" 2>&1 &
echo "PID $! LOG $LOG"
```

## Waiting (do not monitor the GPU)

Do not run the grid in the foreground. Do not poll `nvidia-smi`, `tail -f`, or `sleep` loops.

1. Start the `nohup` above. Print PID and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: paper-timexer-selected-26ts report (started)`).
3. Subscribe one one-shot timer: `delaySeconds=14400` (4 h). Then end the turn.
4. If training finishes earlier, the human will ping this chat — then harvest. Do not watch the process.
5. Harvest only when the timer fires and the log contains `SWEEP_DONE` (or the PID is dead and the log is complete), or the human asks. Overwrite `report.md` with the table below, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is still running: one line (`still running`, jobs completed N/6, whether `PREPARE_DONE` is in the log), push, and stop. Do not start another timer.

## Report table
From the log only. Do not invent metrics.

```text
| e_layers | seed | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
```

Also copy the prepare lines: signature path, `n_train_rows`, `label_dates`, and the 26 column names. The first name must be `close`. Note any skipped ticker.

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID prices + news parquet). Article embeddings and FinBERT probabilities are built by the prepare script if missing.
- gpu: required (`--device cuda`, `train.device=cuda`)
- approx_ram_gb: 32+ (news scan if a ticker parquet is missing, then LightGBM on the pooled train rows)
- git: `feat/phase2-timexer`, descendant of `9bf6a81aa29f5ec12b57685fe3079b02ead74c9b`

## Out of scope
- Do not change product code unless a trivial one-liner unblocks the run; prefer reporting the blocker.
- Do not run `model=a`, `model=b`, `model=c0`, `model=c1`, `model=dlinear`, `model=timexer_plain`, or `ticker_set=dev`.
- Do not refit prototypes or the signature on the 2022/2023 split.
- Do not commit `outputs/`, `mlruns/`, `mlflow.db`, checkpoints, `.env`, `.ssh/`, or anything under `Data/` / `cache/`.

## After the run
Write `report.md` in this same folder, commit on `feat/phase2-timexer`, push.
