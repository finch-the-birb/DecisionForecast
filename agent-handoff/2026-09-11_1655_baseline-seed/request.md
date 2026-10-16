# Request: baseline-seed

- authored_by: dev
- created_at: 2026-09-11T16:55:00Z
- target_ref: feat/phase2-timexer@dc19237f54418408b575b4fc2a66ce82061251dc
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** 3-seed **baselines** at the locked HP (not a new architecture grid). **9 sequential jobs:** `model=dlinear`, `timexer_plain`, `a` (n_prototypes=10) × seeds **0,1,2**.

Freeze: `train.lr=0.0003`, T=60, H=7, yaml `d_model=64` `d_ff=256` (plain/A), epochs=20, patience=5, `train.num_workers=4`. **No window caps.**

Do **not** re-run B / C0 / C1 (already in `seed-shortlist` at this HP). Do **not** mix those rows into this sweep's 9-job table. After the means, paste a **scoreboard footnote** copying `seed-shortlist` mean val/test (n=3) so Dev can compare C0/5 vs these baselines — label it prior log, not this run.

Winner **among these three models** = **mean last-epoch val_mse** (n=3), same rule as `seed-shortlist`. Also report mean **best_val** (min Epoch val_mse) and mean test (`METRICS_ROW` = best.pt). Test is recorded, not used to pick. Not paper.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: baseline-seed report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=5400` (90 min, safety net only). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`. Do not override `d_model` / `d_ff`. Do not pass `model.n_prototypes` on dlinear / timexer_plain.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain dc19237 (and 03f123b Coverage + DataLoader)

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/baseline-seed.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.lookback_T=60 data.horizon=7 train.lr=0.0003 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}
for s in 0 1 2; do
  run model=dlinear train.seed=$s
  run model=timexer_plain train.seed=$s
  run model=a model.n_prototypes=10 train.seed=$s
done
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `=== START`, `=== DONE`, `SWEEP_DONE`.

`best_val` = min `val_mse` among `Epoch N — … val_mse=` in that job. `last_val` = last Epoch val_mse. `stop_ep` = `Early stopping at epoch N` (else last Epoch N). Test = `METRICS_ROW` (best.pt).

### Table 1 — per-seed (this sweep only)

```text
| model | seed | stop_ep | best_val | last_val | test_mse | test_mae | mae_denorm |
| dlinear | 0 | … | … | … | … | … | … |
| dlinear | 1 | … | … | … | … | … | … |
| dlinear | 2 | … | … | … | … | … | … |
| timexer_plain | 0 | … | … | … | … | … | … |
| timexer_plain | 1 | … | … | … | … | … | … |
| timexer_plain | 2 | … | … | … | … | … | … |
| a | 0 | … | … | … | … | … | … |
| a | 1 | … | … | … | … | … | … |
| a | 2 | … | … | … | … | … | … |
```

### Table 2 — means n=3 (this sweep only)

```text
| model | last_val mean±std | best_val mean±std | test_mse mean±std |
| dlinear | … | … | … |
| timexer_plain | … | … | … |
| a | … | … | … |
```

Mark the **mean last_val** winner among these three. Paste one `train device:` / `x=cuda:0`, one `DataLoader num_workers=`, three `TEXT_COVERAGE split=` (once).

### Footnote — prior `seed-shortlist` (do not re-run; copy from git)

From `agent-handoff/2026-09-11_1448_seed-shortlist/report.md` means (n=3, last-epoch val / test):

```text
| model | proto | last_val mean±std | test_mse mean±std |
| b | 10 | 0.8816±0.0153 | 0.7244±0.0275 |
| c0 | 5 | 0.8338±0.0473 | 0.7018±0.0067 |
| c1 | 5 | 0.9072±0.0639 | 0.7018±0.0176 |
```

One sentence: whether this sweep's best mean last_val / mean test beats **C0 proto=5** from that footnote (C0 is the locked candidate, not a row in Table 1).

Status:
- `fail` — majority crash, CPU-only, or `num_workers=0` on this HEAD
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 9/9, CUDA, workers=4, both tables, mean-last_val winner named among dlinear/plain/A

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if volume has FNSPID + dated μ (confirmed on c0c1-windows)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `dc19237` (must include `03f123b` Coverage + DataLoader)
- budget: ~25–45 min (dlinear shorter than TimeXer; 9 jobs)
- prior: `seed-shortlist` `pass` locked C0 proto=5; `c0c1-windows` C1 not a stable win. Protocol-norm DLinear test 0.673 was **lr=0.001 / seed=42** — not this comparison.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run B / C0 / C1 / paper / seed=42 / lr=0.001 / d_model / extra proto / other T/H / decay_lambda / caps.
- Do not run `ablation_table`. Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
