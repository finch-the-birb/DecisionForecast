# Request: paper-ablation-a-b-c0

- authored_by: dev
- created_at: 2026-09-12T23:00:00Z
- target_ref: feat/phase2-timexer@033996f48aa5bc3376f0dc099e1f14c0a89f5f8d
- phase: adhoc
- priority: high

## Goal
Fill the missing paper ablation cells for H1 (A vs B) and H2 (B vs C0 vs C1) on the **same 61-ticker protocol** as `paper-baselines-c1` (train 98236 / val 12048 / test 11904; UNH/VZ skipped).

Fusion-only contrast: **Huber δ=0.5**, **F5 OHLCV**, TimeXer **e_layers=1** + **pool=last**. A stays TimeXL CNN with **proto=10** and **pool=mean**.

9 jobs = 3 configs × seeds `{0,1,2}`:

| # | label | Hydra extras |
|---|--------|----------------|
| 1 | A late CNN proto=10 | `model=a model.n_prototypes=10 model.loss.kind=huber model.loss.delta=0.5` + F5 |
| 2 | B late e=1 last proto=10 | `model=b model.n_prototypes=10 model.e_layers=1 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5` + F5 |
| 3 | C0 mid-add e=1 last proto=5 | `model=c0 model.n_prototypes=5 model.e_layers=1 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5` + F5 |

`F5`: `'data.features=[close,volume,open,high,low]'`

Shared: `train.device=cuda train.ticker_set=paper train.epochs=20 train.patience=5 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null`.

Val/test stay MSE. Pick on **mean last-epoch val_mse**. Record `best_val`, `test_mse` (`best.pt`), `overfit_gap`, `n_params`.

**Do not precache.** `mu_paper_2015-01-01_2021-12-31.npy` and daily `.npz` are already on the Runner disk from `paper-baselines-c1`. If μ is missing, `status: blocked` — do not download FNSPID or set `HF_TOKEN`.

Confirm splits match 98236 / 12048 / 11904 (or report the new counts if they differ). First-batch `x` last dim must be 5.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: paper-ablation-a-b-c0 report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=9000` (150 min). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update (`still running`, jobs completed N/9), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **no precache**. Sequential (sqlite). Do not change lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not run DLinear / plain / C1 / flatten / extra T/H / `ablation_table` / `ticker_set=dev`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 033996f48aa5bc3376f0dc099e1f14c0a89f5f8d

uv sync
MU_PAPER="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_paper_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU_PAPER" ]; then
  echo "BLOCKED: missing $MU_PAPER — do not precache in this request"
  exit 2
fi

mkdir -p outputs
LOG=outputs/paper-ablation-a-b-c0.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=paper train.epochs=20 train.patience=5 train.lr=0.0003 data.lookback_T=60 data.horizon=7 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
F5="data.features=[close,volume,open,high,low]"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

for s in 0 1 2; do
  run model=a model.n_prototypes=10 model.loss.kind=huber model.loss.delta=0.5 $F5 train.seed=$s
done
for s in 0 1 2; do
  run model=b model.n_prototypes=10 model.e_layers=1 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5 $F5 train.seed=$s
done
for s in 0 1 2; do
  run model=c0 model.n_prototypes=5 model.e_layers=1 model.head.pool=last model.loss.kind=huber model.loss.delta=0.5 $F5 train.seed=$s
done

echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `First batch shapes`, `Dataset sizes`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `n_params=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

### Reporting format in report.md
`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt). Means n=3, sample std.

Record paper split sizes (verbatim). Confirm `x=(32, 60, 5)`.

**Table 1 — per-seed:** label, seed, stop_ep, last_val, best_val, overfit_gap, test_mse.  
**Table 2 — means n=3:** A, B, C0 (last_val, best_val, test_mse, overfit_gap, n_params).  
**Table 3 — paper scoreboard (61 tickers):** merge this run with `paper-baselines-c1` (do **not** re-run those jobs):

| label | last_val | test_mse | source |
| DLinear F1 | 0.6985±0.0033 | 0.7535±0.0009 | paper-baselines-c1 |
| TimeXer plain e=1 F2 | 0.7263±0.0231 | 0.7650±0.0091 | paper-baselines-c1 |
| Model A late CNN | *this run* | *this run* | this |
| Model B late e=1 | *this run* | *this run* | this |
| Model C0 mid-add e=1 | *this run* | *this run* | this |
| Model C1.1 Huber e=1 F5 | 0.7173±0.0065 | 0.7543±0.0026 | paper-baselines-c1 |

**Conclusions for Dev:**
1. H1 on paper last_val: A vs B (who wins)?
2. H2 on paper last_val: B vs C0 vs locked C1.1 0.7173 (does mid-add or cross-attn beat late)?
3. Does any of A/B/C0 beat DLinear last 0.6985 or test 0.7535?
4. CUDA / `num_workers=4` / no precache.

Status:
- `fail` — majority crash, CPU-only, `num_workers=0`, or empty splits
- `blocked` — missing `mu_paper` (do not precache)
- `partial` — started / still running / incomplete table
- `pass` — 9/9, CUDA, workers=4, three tables

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (paper μ + daily npz from `paper-baselines-c1`; prices on disk; UNH/VZ missing is expected)
- gpu: required (L4)
- approx_ram_gb: 16+
- git: descendant of `033996f48aa5bc3376f0dc099e1f14c0a89f5f8d`
- budget: ~150 min (9 jobs × ~10–12 min on paper)
- prior: `paper-baselines-c1` DLinear last/test 0.6985 / 0.7535; C1.1 Huber e=1 0.7173 / 0.7543; plain F2 0.7263 / 0.7650.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not precache, do not set `HF_TOKEN`, do not download FNSPID.
- Do not run DLinear / plain / C1 / flatten / extra T/H / decay_lambda / caps / ablation_table / `ticker_set=dev`.
- Do not commit `outputs/` or `mlflow.db`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
