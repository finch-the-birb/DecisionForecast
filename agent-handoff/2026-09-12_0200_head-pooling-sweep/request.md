# Request: head-pooling-sweep

- authored_by: dev
- created_at: 2026-09-12T02:00:00Z
- target_ref: feat/phase2-timexer@93437e7c1c2d0c5571e80adad794137e8fd040ac
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** 24-job sweep (~45–60 min on Runpod L4) after reverting the default head to compact **mean-pool** and adding two compact alternatives (`last`, `global`). `flatten` is **not** in this grid (already rejected: plain test 0.724 vs mean-pool 0.689).

Questions (pick on **mean last-epoch val_mse**, n=3 seeds 0/1/2; record test from `best.pt`):
1. Does `model.head.pool=mean` on `timexer_plain` reproduce the mean-pool control **test 0.6887±0.0273** / last_val **0.7679±0.0084** from `baseline-seed`?
2. Does compact `last` beat `mean` on plain and/or C0?
3. On C1: **C1 mean** vs **C1.1 last** vs **C1.2 global** (`G_en` after text cross-attn). Does any compact C1 beat C0 mean / DLinear?
4. DLinear anchor still **test 0.6739±0.0013** / last_val **0.7272±0.0035**?

24 jobs = 8 configs × seeds `{0,1,2}`:

| # | label | Hydra extras |
|---|--------|----------------|
| 1 | plain mean | `model=timexer_plain model.head.pool=mean` |
| 2 | plain last | `model=timexer_plain model.head.pool=last` |
| 3 | C0 mean | `model=c0 model.n_prototypes=5 model.head.pool=mean` |
| 4 | C0 last | `model=c0 model.n_prototypes=5 model.head.pool=last` |
| 5 | C1 mean | `model=c1 model.n_prototypes=5 model.head.pool=mean` |
| 6 | C1.1 last | `model=c1 model.n_prototypes=5 model.head.pool=last` |
| 7 | C1.2 global | `model=c1 model.n_prototypes=5 model.head.pool=global` |
| 8 | DLinear | `model=dlinear` |

Shared: `train.device=cuda train.ticker_set=dev train.epochs=20 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_*_windows=null`. yaml `d_model=64 d_ff=256`, `head.type=linear`, `head.dropout=0.1`. Do **not** override `d_model` / `d_ff` / decay / T / H.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: head-pooling-sweep report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=5400` (90 min). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`, jobs completed N/24), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not run `flatten`, A, B, extra λ, or `ablation_table`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 93437e7c1c2d0c5571e80adad794137e8fd040ac

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/head-pooling-sweep.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

# 1 plain mean
for s in 0 1 2; do
  run model=timexer_plain model.head.pool=mean train.seed=$s
done
# 2 plain last
for s in 0 1 2; do
  run model=timexer_plain model.head.pool=last train.seed=$s
done
# 3 C0 mean
for s in 0 1 2; do
  run model=c0 model.n_prototypes=5 model.head.pool=mean train.seed=$s
done
# 4 C0 last
for s in 0 1 2; do
  run model=c0 model.n_prototypes=5 model.head.pool=last train.seed=$s
done
# 5 C1 mean
for s in 0 1 2; do
  run model=c1 model.n_prototypes=5 model.head.pool=mean train.seed=$s
done
# 6 C1.1 last
for s in 0 1 2; do
  run model=c1 model.n_prototypes=5 model.head.pool=last train.seed=$s
done
# 7 C1.2 global (G_en)
for s in 0 1 2; do
  run model=c1 model.n_prototypes=5 model.head.pool=global train.seed=$s
done
# 8 DLinear anchor
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
One summary table (mean ± sample std over seeds 0, 1, 2) with columns: `label | pool | last_val | best_val | test_mse | n_params`.

Rows in this order: plain mean, plain last, C0 mean, C0 last, C1 mean, C1.1 last, C1.2 global, DLinear.

Also a short per-seed table (stop_ep, last_val, test_mse) so Dev can see overfit (`last_val − best_val`).

Confirm:
- `train device: cuda` and `num_workers=4` on every job
- `n_params` for compact TimeXer heads is much smaller than FlattenHead (~4k head weights, not 576→7)
- C1.2 `pool=global` did not crash (`g_en` passed)

Answer the four questions in **Conclusions for Dev**. Compare `last` and `global` explicitly to the same-model `mean` row.

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 24/24, CUDA, workers=4, both tables provided

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID + dated μ on disk)
- gpu: required (L4)
- approx_ram_gb: 16+
- git: descendant of `93437e7c1c2d0c5571e80adad794137e8fd040ac`
- budget: ~90 min (24 jobs × ~2.5 min/job)
- prior: `head-cache-sweep` rejected FlattenHead (plain test 0.724 vs old mean-pool 0.689). This sweep restores mean as default and tests compact last / C1 global vs DLinear 0.6739.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run paper / seed=42 / flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
- Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
