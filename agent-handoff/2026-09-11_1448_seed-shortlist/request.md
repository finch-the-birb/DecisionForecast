# Request: seed-shortlist

- authored_by: dev
- created_at: 2026-09-11T14:48:00Z
- target_ref: feat/phase2-timexer@a93af33656eea62bd26f9f8947ebb5a8fd850567
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** 3-seed confirmation of the HP shortlist (not a new HP grid). **9 sequential jobs:**

| model | n_prototypes | why |
| b | 10 | best val among B in `hp-nproto-bc0c1`; val winner of `hp-width-nproto` |
| c0 | 5 | best val among C0 in `hp-nproto-bc0c1` |
| c1 | 5 | val winner of `hp-nproto-bc0c1` |

Seeds **0,1,2 only** (not 42). Freeze `train.lr=0.0003`, yaml `d_model=64` `d_ff=256`, H=7, epochs=20, patience=5, `train.num_workers=4`. **No window caps.**

Winner = **mean last-epoch val_mse over the three seeds** (not test, not a single seed). Report mean ± std for val_mse and test_mse. Table from **this** sweep's logs only.

Do **not** mix with R3 (that was lr=0.001 and proto=10 for all). Do **not** mix seed=42 HP rows. Do **not** run `ablation_table` (it will mix old MLflow runs). Not paper.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: seed-shortlist report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=5400` (90 min, safety net only). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the table, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not override `d_model` / `d_ff`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain a93af33 (and 03f123b Coverage + DataLoader)

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/seed-shortlist.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7 train.lr=0.0003 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}
for spec in "b 10" "c0 5" "c1 5"; do
  set -- $spec
  m=$1 p=$2
  for s in 0 1 2; do
    run model=$m model.n_prototypes=$p train.seed=$s
  done
done
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `val_mse`, `train device:`, `First batch tensors`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `H3_ROW`, `=== START`, `=== DONE`, `SWEEP_DONE`.

Final tables (**this sweep only**):

Per-seed:

```text
| model | n_prototypes | seed | val_mse | test_mse | test_mae | mae_denorm |
| b | 10 | 0 | … | … | … | … |
| b | 10 | 1 | … | … | … | … |
| b | 10 | 2 | … | … | … | … |
| c0 | 5 | 0 | … | … | … | … |
| c0 | 5 | 1 | … | … | … | … |
| c0 | 5 | 2 | … | … | … | … |
| c1 | 5 | 0 | … | … | … | … |
| c1 | 5 | 1 | … | … | … | … |
| c1 | 5 | 2 | … | … | … | … |
```

Means (n=3, last-epoch val; test from `METRICS_ROW`):

```text
| model | n_prototypes | val_mse mean±std | test_mse mean±std |
| b | 10 | … | … |
| c0 | 5 | … | … |
| c1 | 5 | … | … |
```

Mark the **mean val_mse** winner. Paste one `train device:` / `x=cuda:0` line, one `DataLoader num_workers=` line, and the three `TEXT_COVERAGE split=` lines (once). If `H3_ROW` `text_zero` appears for the three C1 jobs, paste those three lines (record only; do not fail on Δ).

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table / missing seed
- `pass` — 9/9, CUDA, workers=4, both tables, mean-val winner named

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if volume has FNSPID + dated μ (confirmed on hp-nproto-bc0c1)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `a93af33` (must include `03f123b` Coverage + DataLoader)
- budget: ~30–50 min (hp-nproto-bc0c1 9 jobs was ~31 min)
- prior: `hp-nproto-bc0c1` `pass` — seed=42 val winner C1 proto=5; B proto=10; C0 proto=5. This round is seeds 0/1/2 at those three configs only.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run A / dlinear / timexer_plain / paper / seed=42 / lr=0.001 / d_model / d_ff / extra proto values / caps.
- Do not run `ablation_table`. Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
