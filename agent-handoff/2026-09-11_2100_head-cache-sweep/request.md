# Request: head-cache-sweep

- authored_by: dev
- created_at: 2026-09-11T21:00:00Z
- target_ref: feat/phase2-timexer@7f61ab9683ded50ff6f7a18b0f161c24e4c69f76
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** 72-job overnight sweep (~3 hours on Runpod L4) validating two architectural fixes:
1. **FlattenHead** (PatchTST/TimeXer style) preserving temporal order of patches vs temporal mean pooling.
2. **Daily text series cache key fix**: `lam` and `missing_policy` in filename enabling honest news decay sweeps.

The 72 jobs are organized into 4 sequential blocks:
- **Block 1: Linear vs DLinear control anchors (6 jobs)**
  `dlinear` at `train.lr=0.0003, 0.001` × `train.seed=0, 1, 2`
- **Block 2: Head architecture on timexer_plain (18 jobs)**
  `model=timexer_plain` × `model.head.type=linear, mlp` × `model.head.dropout=0.1, 0.2` (for linear: `model.head.dropout` applies to input dropout; for mlp: input dropout) × `train.seed=0, 1, 2` (note: when head.type=linear, dropout=0.1 vs 0.2 checks regularizing linear head; 12 linear + 6 mlp = 12 total distinct configs × seeds = 18 jobs: `head.type=linear` with dropout 0.1, 0.2 [6 jobs], `head.type=mlp` with dropout 0.1, 0.2 [6 jobs] — 12 jobs total if 2 types × 2 dropouts × 3 seeds = 12 jobs; plus baseline anchors and full coverage: 2 head.type × 2 head.dropout × 3 seeds = 12 jobs).
  *Exact 72-job grid structure:*
  - Block 1 (6 jobs): `model=dlinear` × `train.lr ∈ {0.0003, 0.001}` × `seed ∈ {0, 1, 2}`
  - Block 2 (18 jobs): `model=timexer_plain` × `head.type ∈ {linear, mlp}` × `model.head.dropout ∈ {0.0, 0.1, 0.2}` × `seed ∈ {0, 1, 2}` (2 types × 3 dropouts × 3 seeds = 18 jobs)
  - Block 3 (24 jobs): Models `timexer_plain`, `c0` (proto=5), `c1` (proto=5), `b` (proto=10) × `model.head.dropout ∈ {0.1, 0.2}` × `seed ∈ {0, 1, 2}` on `head.type=linear` (4 models × 2 dropouts × 3 seeds = 24 jobs)
  - Block 4 (24 jobs): Models `c0` (proto=5), `c1` (proto=5), `b` (proto=10), `a` (proto=10) × `data.text.decay_lambda ∈ {0.01, 0.10}` × `seed ∈ {0, 1, 2}` on `head.type=linear` (4 models × 2 lambdas × 3 seeds = 24 jobs)

Total jobs: 6 + 18 + 24 + 24 = 72 jobs.

Default settings across all runs:
`train.device=cuda train.ticker_set=dev train.epochs=20 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null`
Default `train.lr=0.0003` (except Block 1 where lr=0.001 is compared).
Default `d_model=64 d_ff=256` (yaml).

Winner per question:
- Head type: does `head.type=linear` outperform `mlp` and old mean pooling on plain?
- Head vs DLinear: does any model with FlattenHead beat DLinear (0.6739 test / 0.7237 best_val)?
- Decay sensitivity: does news decay rate $\lambda \in \{0.01, 0.10\}$ change C0/C1/B/A performance now that cache correctly distinguishes $\lambda$?

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: head-cache-sweep report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=10800` (180 min, 3 hours). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`, jobs completed N/72), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not override `d_model` / `d_ff`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 7f61ab9683ded50ff6f7a18b0f161c24e4c69f76

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/head-cache-sweep.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

# --- BLOCK 1: Linear vs DLinear control anchors (6 jobs) ---
for lr in 0.0003 0.001; do
  for s in 0 1 2; do
    run model=dlinear train.lr=$lr train.seed=$s
  done
done

# --- BLOCK 2: Head architecture on timexer_plain (18 jobs) ---
for htype in linear mlp; do
  for drop in 0.0 0.1 0.2; do
    for s in 0 1 2; do
      run model=timexer_plain train.lr=0.0003 model.head.type=$htype model.head.dropout=$drop train.seed=$s
    done
  done
done

# --- BLOCK 3: Main models with FlattenHead (24 jobs) ---
for spec in "timexer_plain null" "c0 5" "c1 5" "b 10"; do
  set -- $spec
  m=$1 proto=$2
  for drop in 0.1 0.2; do
    for s in 0 1 2; do
      if [ "$proto" != "null" ]; then
        run model=$m model.n_prototypes=$proto train.lr=0.0003 model.head.type=linear model.head.dropout=$drop train.seed=$s
      else
        run model=$m train.lr=0.0003 model.head.type=linear model.head.dropout=$drop train.seed=$s
      fi
    done
  done
done

# --- BLOCK 4: Honest text decay influence (24 jobs) ---
for spec in "c0 5" "c1 5" "b 10" "a 10"; do
  set -- $spec
  m=$1 proto=$2
  for lam in 0.01 0.10; do
    for s in 0 1 2; do
      run model=$m model.n_prototypes=$proto train.lr=0.0003 data.text.decay_lambda=$lam model.head.type=linear model.head.dropout=0.1 train.seed=$s
    done
  done
done

echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

### Reporting format in report.md
Group results into 4 concise summary tables (mean ± std over seeds 0, 1, 2) for:
1. Block 1: DLinear lr 0.0003 vs 0.001
2. Block 2: Plain FlattenHead (linear vs mlp × dropout)
3. Block 3: Plain vs C0 vs C1 vs B with linear head
4. Block 4: Decay lambda sensitivity (0.01 vs 0.10) for C0, C1, B, A

Include:
- Confirmation that daily cache files created contain `_lam0.01.npz` and `_lam0.1.npz`.
- Answer to key questions: (1) Does FlattenHead beat mean pooling? (2) Does any model beat DLinear? (3) Does text decay matter?

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 72/72 (or all available completed), CUDA, workers=4, summary tables provided

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID + dated μ on disk)
- gpu: required (L4)
- approx_ram_gb: 16+
- git: descendant of `7f61ab9683ded50ff6f7a18b0f161c24e4c69f76`
- budget: ~180 min (72 jobs × ~2.5 min/job)
- prior: `baseline-seed` showed DLinear (0.6739) beating old mean-pooling C0 (0.7018) and plain (0.6887). This sweep investigates whether FlattenHead bridges or reverses this gap.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run paper / seed=42 / extra T/H combinations / caps / ablation_table.
- Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
