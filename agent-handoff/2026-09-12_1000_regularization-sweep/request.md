# Request: regularization-sweep

- authored_by: dev
- created_at: 2026-09-12T10:00:00Z
- target_ref: feat/phase2-timexer@7681745f6072c55b5069744e97a5db59d8cd5a9a
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** 60-job overnight sweep (~2.5–3 h on Runpod L4) on the C1.1 last-patch setup from `head-pooling-sweep` (test 0.6862±0.0056, last_val 0.8531, best_val 0.7641, overfit gap ~0.089). DLinear remains the anchor (last 0.7272 / test 0.6739).

Test four regularization levers, **without** changing T/H/split/tickers:

1. Task loss instead of pure MSE: Huber (δ∈{1.0, 0.5}), directional (γ∈{0.1, 0.5}), correlation (α=0.3).
2. Prototype weights: λ_c=λ_e ∈ {0, 0.3, 0.5} vs yaml default 0.1 (job 1).
3. AdamW weight decay 0.01 / 0.05 and dropout 0.2.
4. Capacity: `e_layers=1`, `d_model=32 d_ff=128`.

**Val/test stay MSE** (`evaluate` is unchanged). Train may optimize Huber / dir / corr; still pick on last-epoch `val_mse` (MSE). Record `best_val`, `test_mse` from `best.pt`, and `overfit_gap = last_val − best_val`.

60 jobs = 20 configs × seeds `{0,1,2}`.

Shared: `train.device=cuda train.ticker_set=dev train.epochs=20 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_*_windows=null`.  
`SHARED_C1='model=c1 model.n_prototypes=5 model.head.pool=last'`  
yaml `d_model=64 d_ff=256`, `head.type=linear`, default `head.dropout=0.1`, default `train.weight_decay=0.0001` unless overridden. Do **not** override T / H / decay_lambda / ticker lists.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: regularization-sweep report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=9000` (150 min). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`, jobs completed N/60), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not run flatten / A / B / extra T/H / `ablation_table`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 7681745f6072c55b5069744e97a5db59d8cd5a9a

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/regularization-sweep.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
C1="model=c1 model.n_prototypes=5 model.head.pool=last"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

# --- BLOCK 1: task losses on C1.1 last (18 jobs) ---
for s in 0 1 2; do
  run $C1 model.loss.kind=mse train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.kind=huber model.loss.delta=1.0 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.kind=huber model.loss.delta=0.5 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.kind=directional model.loss.gamma_dir=0.1 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.kind=directional model.loss.gamma_dir=0.5 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.kind=correlation model.loss.alpha_corr=0.3 train.seed=$s
done

# --- BLOCK 2: prototype λ (18 jobs) ---
for s in 0 1 2; do
  run $C1 model.loss.lambda_c=0.0 model.loss.lambda_e=0.0 model.loss.lambda_d=0.0 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.lambda_c=0.3 model.loss.lambda_e=0.3 model.loss.lambda_d=0.01 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.loss.lambda_c=0.5 model.loss.lambda_e=0.5 model.loss.lambda_d=0.01 train.seed=$s
done
for s in 0 1 2; do
  run model=c0 model.n_prototypes=5 model.head.pool=mean model.loss.lambda_c=0.0 model.loss.lambda_e=0.0 model.loss.lambda_d=0.0 train.seed=$s
done
for s in 0 1 2; do
  run model=c0 model.n_prototypes=5 model.head.pool=mean model.loss.lambda_c=0.3 model.loss.lambda_e=0.3 model.loss.lambda_d=0.01 train.seed=$s
done
for s in 0 1 2; do
  run model=timexer_plain model.head.pool=last train.seed=$s
done

# --- BLOCK 3: capacity + weight decay + dropout (18 jobs) ---
for s in 0 1 2; do
  run $C1 train.weight_decay=0.01 train.seed=$s
done
for s in 0 1 2; do
  run $C1 train.weight_decay=0.05 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.e_layers=1 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.d_model=32 model.d_ff=128 train.seed=$s
done
for s in 0 1 2; do
  run $C1 model.dropout=0.2 model.head.dropout=0.2 train.seed=$s
done
for s in 0 1 2; do
  run model=timexer_plain model.head.pool=last train.weight_decay=0.01 train.seed=$s
done

# --- BLOCK 4: combo + DLinear (6 jobs) ---
for s in 0 1 2; do
  run $C1 model.loss.kind=huber model.loss.delta=0.5 train.weight_decay=0.01 train.seed=$s
done
for s in 0 1 2; do
  run model=dlinear train.seed=$s
done

echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `n_params=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

### Reporting format in report.md
Means n=3, sample std. `best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt).

**Table 1 — losses on C1.1** (jobs 1–6): label, last_val, best_val, test_mse, overfit_gap.  
**Table 2 — prototype strength** (jobs 1, 7–11): C1.1 λ∈{0.0, 0.1, 0.3, 0.5} and C0 mean λ∈{0.0, 0.3}. Job 1 is C1.1 λ=0.1. Include job 12 (plain last) as a no-proto reference row.  
**Table 3 — capacity / decay / dropout** (jobs 12–18): plain last, C1.1 wd 0.01 / 0.05, e_layers=1, d=32, dropout 0.2, plain last wd 0.01.  
**Table 4 — combo vs DLinear** (jobs 19–20) plus C1.1 MSE (job 1) and C1.1 Huber δ=0.5 (job 3) for context.

Also a short per-seed table for any config whose test_mse mean **beats** C1.1 MSE (0.6862) or last_val **beats** DLinear (0.7272).

Confirm CUDA + `num_workers=4` on all 60. Confirm DLinear last/test match 0.7272 / 0.6739.

Answer in **Conclusions for Dev**:
1. Does any loss beat C1.1 MSE on last_val and/or shrink overfit_gap without wrecking test?
2. Do stronger / zero proto λ help C1.1 or C0?
3. Does wd / shallower / narrower / dropout close the last_val−best_val gap?
4. Does the Huber+wd combo beat DLinear on last_val or only on test?

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 60/60, CUDA, workers=4, four tables provided

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID + dated μ on disk)
- gpu: required (L4)
- approx_ram_gb: 16+
- git: descendant of `7681745f6072c55b5069744e97a5db59d8cd5a9a`
- budget: ~150 min (60 jobs × ~2.5 min/job)
- prior: `head-pooling-sweep` C1.1 last test 0.6862, last_val 0.8531, overfit ~0.089. DLinear 0.6739 / 0.7272.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run paper / seed=42 / flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
- Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
