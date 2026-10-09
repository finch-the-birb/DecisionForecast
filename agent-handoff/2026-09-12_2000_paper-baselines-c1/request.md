# Request: paper-baselines-c1

- authored_by: dev
- created_at: 2026-09-12T20:00:00Z
- target_ref: feat/phase2-timexer@089a1fe707f227206fec659f8bc0a95617bdee6d
- phase: adhoc
- priority: high

## Goal
First **uncapped `ticker_set=paper`** run (63 tickers). Dev (15 names) is no longer the scoreboard. Answer:

1. What is DLinear’s true last_val / test on paper F1 `[close]`? (dev 0.7272 / 0.6739 does **not** transfer.)
2. Does **plain e=1 pool=last F2** keep a test edge over DLinear (dev test 0.6697 vs 0.6739)?
3. On ~4× more windows, does **C1.1 e=2** beat **C1.1 e=1** (both Huber δ=0.5, F5, proto=5), or is one layer still better?

12 jobs = 4 configs × seeds `{0,1,2}`. **No Model A** this round (keep the 3.5 h budget).

| # | label | Hydra extras |
|---|--------|----------------|
| 1 | DLinear F1 | `model=dlinear 'data.features=[close]'` |
| 2 | plain last e=1 F2 | `model=timexer_plain model.e_layers=1 model.head.pool=last 'data.features=[close,volume]'` |
| 3 | C1.1 Huber e=1 F5 | `model=c1 model.n_prototypes=5 model.e_layers=1 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5 'data.features=[close,volume,open,high,low]'` |
| 4 | C1.1 Huber e=2 F5 | same as 3 with `model.e_layers=2` |

Shared: `train.device=cuda train.ticker_set=paper train.epochs=20 train.patience=5 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null`.

Val/test stay MSE. Pick on **mean last-epoch val_mse**. Record `best_val`, `test_mse` (`best.pt`), `overfit_gap = last_val − best_val`, `n_params`, and **actual** train/val/test window counts from the log (do not guess ~100k).

**HF_TOKEN stays unset.** Encoder remains public `FinLang/finance-embeddings-investopedia`. Precache paper news + μ + daily `.npz` only if `mu_paper_2015-01-01_2021-12-31.npy` is missing.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: paper-baselines-c1 report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=12600` (210 min). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update (`still running`, jobs completed N/12, whether precache finished), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download of the raw dump if data already on disk. **This request is `ticker_set=paper`.** Sequential (sqlite). Do not change lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not run flatten / B / C0 / extra T/H / `ablation_table` / Model A.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 089a1fe707f227206fec659f8bc0a95617bdee6d

uv sync

mkdir -p outputs
LOG=outputs/paper-baselines-c1.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
MU_PAPER="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_paper_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU_PAPER" ]; then
  echo "=== PRECACHE paper news + embeddings $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python scripts/precache_fnspid_news.py --ticker-set=paper
  uv run python scripts/precache_text_embeddings.py --ticker-set=paper --device=cuda
  echo "=== PRECACHE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
else
  echo "=== PRECACHE skip mu_paper exists ==="
fi

SHARED="train.device=cuda train.ticker_set=paper train.epochs=20 train.patience=5 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
F1="data.features=[close]"
F2="data.features=[close,volume]"
F5="data.features=[close,volume,open,high,low]"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

for s in 0 1 2; do
  run model=dlinear $F1 train.seed=$s
done
for s in 0 1 2; do
  run model=timexer_plain model.e_layers=1 model.head.pool=last $F2 train.seed=$s
done
for s in 0 1 2; do
  run model=c1 model.n_prototypes=5 model.e_layers=1 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5 $F5 train.seed=$s
done
for s in 0 1 2; do
  run model=c1 model.n_prototypes=5 model.e_layers=2 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5 $F5 train.seed=$s
done

echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `First batch shapes`, `Dataset sizes`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `n_params=`, `PRECACHE`, `=== START`, `=== DONE`, `SWEEP_DONE`.

Confirm first-batch `x` last dim: DLinear=1, plain=2, C1=5.

### Reporting format in report.md
`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt). Means n=3, sample std.

Record paper split sizes (verbatim `Dataset sizes` + `TEXT_COVERAGE`) — first job is enough if they match.

**Table 1 — per-seed:** label, seed, stop_ep, last_val, best_val, overfit_gap, test_mse.  
**Table 2 — means n=3:** the four configs (last_val, best_val, test_mse, overfit_gap, n_params).

**Conclusions for Dev** must answer:
1. DLinear paper last/test (new scoreboard, not 0.6739).
2. Does plain F2 still beat DLinear on last_val and/or test?
3. Does C1.1 e=2 beat C1.1 e=1 on last_val (and test) at this scale?
4. CUDA / `num_workers=4` / precache skip-or-ran.

Status:
- `fail` — majority crash, CPU-only, `num_workers=0`, missing paper μ after precache, or empty splits
- `partial` — started / still running / incomplete table
- `pass` — 12/12, CUDA, workers=4, both tables, window counts logged

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: FNSPID prices on disk; **paper μ / news parquet may be missing** — precache in the nohup script
- gpu: required (L4) for train **and** embedding precache
- approx_ram_gb: 16+
- git: descendant of `089a1fe707f227206fec659f8bc0a95617bdee6d`
- budget: ~210 min (precache if needed + 12 jobs; paper windows much larger than dev 23478)
- prior: `features-and-capacity-sweep` (dev): DLinear F1 0.7272 / 0.6739; plain e=1 F2 test 0.6697 last 0.7512; C1.1 Huber e=1 F5 test 0.6729 last 0.7363.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run Model A / B / C0 / flatten / extra T/H / decay_lambda / caps / ablation_table / `ticker_set=dev`.
- Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
