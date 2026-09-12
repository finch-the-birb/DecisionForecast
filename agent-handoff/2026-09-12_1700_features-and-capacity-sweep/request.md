# Request: features-and-capacity-sweep

- authored_by: dev
- created_at: 2026-09-12T17:00:00Z
- target_ref: feat/phase2-timexer@798bb80fc0ad50d84505704b3d8073096bce7570
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** 36-job sweep (~1.5–2 h on Runpod L4). Hypothesis: OHLCV PatchEmbed is multicollinear (`z(Open)≈z(High)≈z(Low)≈z(Close)` under `per_window`), so dropping O/H/L and using **F1=[close]** or **F2=[close,volume]** closes the remaining gap from C1 `e_layers=1` + `pool=last` (test 0.6800) to DLinear (0.6739).

Feature tags (Hydra lists, quote them):
- `F1`: `'data.features=[close]'`
- `F2`: `'data.features=[close,volume]'`
- `F5`: `'data.features=[close,volume,open,high,low]'` (yaml default)

36 jobs = 12 configs × seeds `{0,1,2}`:

| # | label | extras |
|---|--------|--------|
| 1 | C1.1 MSE F1 | `model=c1 model.n_prototypes=5 model.e_layers=1 model.head.pool=last` + F1 |
| 2 | C1.1 MSE F2 | same + F2 |
| 3 | C1.1 MSE F5 | same + F5 |
| 4 | C1.1 Huber δ=0.5 F1 | C1.1 + `model.loss.kind=huber model.loss.delta=0.5` + F1 |
| 5 | C1.1 Huber F2 | same + F2 |
| 6 | C1.1 Huber F5 | same + F5 |
| 7 | plain last e=1 F1 | `model=timexer_plain model.e_layers=1 model.head.pool=last` + F1 |
| 8 | plain last e=1 F2 | same + F2 |
| 9 | plain last e=1 F5 | same + F5 |
| 10 | C0 mean e=1 λ=0 F1 | `model=c0 model.n_prototypes=5 model.e_layers=1 model.head.pool=mean model.loss.lambda_c=0.0 model.loss.lambda_e=0.0 model.loss.lambda_d=0.0` + F1 |
| 11 | C0 mean e=1 λ=0 F2 | same + F2 |
| 12 | DLinear F1 | `model=dlinear` + F1 |

`per_window` z-scores **per channel**, so Close scale is the same for F1/F2/F5. DLinear F1 should **match** prior F5 DLinear (last 0.7272 / test 0.6739) — if it does not, stop and report (feature load bug).

Val/test stay MSE. Pick on mean last-epoch `val_mse`. Record `best_val`, `test_mse` (`best.pt`), `overfit_gap = last_val − best_val`, `n_params`.

Shared: `train.device=cuda train.ticker_set=dev train.epochs=20 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_*_windows=null`. yaml `d_model=64 d_ff=256` unless a job says otherwise. Do **not** override T / H / decay_lambda / tickers / `d_model`.

C1 yaml default is now `e_layers=1` `head.pool=last` (`798bb80`); still pass them explicitly.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: features-and-capacity-sweep report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=7200` (120 min). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update (`still running`, jobs completed N/36), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not run flatten / A / B / extra T/H / `ablation_table`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 798bb80fc0ad50d84505704b3d8073096bce7570

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/features-and-capacity-sweep.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
C1="model=c1 model.n_prototypes=5 model.e_layers=1 model.head.pool=last"
PLAIN="model=timexer_plain model.e_layers=1 model.head.pool=last"
C0="model=c0 model.n_prototypes=5 model.e_layers=1 model.head.pool=mean model.loss.lambda_c=0.0 model.loss.lambda_e=0.0 model.loss.lambda_d=0.0"
F1="data.features=[close]"
F2="data.features=[close,volume]"
F5="data.features=[close,volume,open,high,low]"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

# --- BLOCK 1: C1.1 MSE × F1/F2/F5 (9) ---
for f in "$F1" "$F2" "$F5"; do
  for s in 0 1 2; do
    run $C1 $f train.seed=$s
  done
done

# --- BLOCK 2: C1.1 Huber δ=0.5 × F1/F2/F5 (9) ---
for f in "$F1" "$F2" "$F5"; do
  for s in 0 1 2; do
    run $C1 model.loss.kind=huber model.loss.delta=0.5 $f train.seed=$s
  done
done

# --- BLOCK 3: plain last e=1 × F1/F2/F5 (9) ---
for f in "$F1" "$F2" "$F5"; do
  for s in 0 1 2; do
    run $PLAIN $f train.seed=$s
  done
done

# --- BLOCK 4: C0 mean e=1 λ=0 × F1/F2 (6) ---
for f in "$F1" "$F2"; do
  for s in 0 1 2; do
    run $C0 $f train.seed=$s
  done
done

# --- BLOCK 5: DLinear F1 (3) ---
for s in 0 1 2; do
  run model=dlinear $F1 train.seed=$s
done

echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `First batch shapes`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `n_params=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

Confirm first-batch `x` last dim is 1 / 2 / 5 for F1 / F2 / F5.

### Reporting format in report.md
Means n=3, sample std. `best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt).

**Table 1 — C1.1 MSE:** F1 vs F2 vs F5 (last_val, best_val, test_mse, overfit_gap, n_params).  
**Table 2 — C1.1 Huber δ=0.5:** F1 vs F2 vs F5 (same columns).  
**Table 3 — plain last e=1:** F1 vs F2 vs F5 (same columns).  
**Table 4 — ranking vs DLinear:** all 12 configs, sorted by last_val, plus DLinear F1. Highlight any TimeXer that beats DLinear last 0.7272 or test 0.6739.

Also note: C1.1 MSE F5 should reproduce regularization-sweep `e_layers=1` (last 0.7415 / test 0.6800). DLinear F1 should match 0.7272 / 0.6739.

Status:
- `fail` — majority crash, CPU-only, `num_workers=0`, or DLinear F1 far from 0.6739
- `partial` — started / still running / incomplete table
- `pass` — 36/36, CUDA, workers=4, four tables

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID + dated μ on disk)
- gpu: required (L4)
- approx_ram_gb: 16+
- git: descendant of `798bb80fc0ad50d84505704b3d8073096bce7570`
- budget: ~120 min (36 jobs × ~2.5–3 min/job)
- prior: `regularization-sweep` C1.1 e_layers=1 F5 test 0.6800, last 0.7415; DLinear 0.6739 / 0.7272; C0 mean λ=0 F5 test 0.6777 (2 layers).

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run paper / seed=42 / flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
- Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
