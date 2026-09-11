# Request: c0c1-windows

- authored_by: dev
- created_at: 2026-09-11T15:40:00Z
- target_ref: feat/phase2-timexer@149c06b09eee45d113d99b1f2ae6d5e8947359f0
- phase: adhoc
- priority: high

## Goal
Uncapped **dev** screen: does **C1 cross-attn** beat **C0** (same TimeXer + proto, no cross-attn) at windows other than T=60 H=7? Also harvest **train vs val** to see overfit.

**8 sequential jobs:** `model=c0,c1` × four `(lookback_T, horizon)` pairs from the protocol sensitivity lists. Both models **`n_prototypes=5`**. Freeze `train.lr=0.0003`, yaml `d_model=64` `d_ff=256`, seed=42, epochs=20, patience=5, `train.num_workers=4`. **No window caps.**

Pairs (do **not** cartesian all T×H; skip T=60 H=7 — already have 3-seed at that cell):

| lookback_T | horizon | why |
| 60 | 14 | longer H, core T |
| 60 | 30 | longest H, core T |
| 36 | 7 | shorter T, core H |
| 96 | 7 | longer T, core H |

Do **not** sweep `data.text.decay_lambda` this round: daily-series `.npz` cache is **not** keyed by λ, so a lambda override would silently keep λ=0.03 carry/decay in `E` and only change window pooling. Leave decay for a later request after a cache-key fix.

Do **not** mix with `seed-shortlist` / R3 / seed=42 H=7 tables (different H or seeds). Not paper. Do **not** pick one global winner across H (mse scale changes with H). Per pair: who wins **best.pt test** (`METRICS_ROW`) and who wins **min epoch val_mse**. Record last-epoch val for overfit.

## Waiting (do not monitor the GPU)

**Do not** run the grid in the foreground. **Do not** poll `nvidia-smi`, `tail -f`, or `sleep` loops. **Do not** re-subscribe timers or keep checking the log.

1. Start `nohup` (commands below). Print `PID` and log path.
2. Write **this folder** `report.md` immediately: `status: partial`, pid, log, `tested_ref`. Commit + push (`handoff: c0c1-windows report (started)`).
3. Subscribe **one** one-shot timer: `delaySeconds=5400` (90 min, safety net only). Then **end the turn**.
4. **If training finishes earlier, the human will ping this Runner chat** — then harvest (step 5). Do not watch the process yourself.
5. Harvest **only** when (a) the timer fires **and** the log contains `SWEEP_DONE` / the PID is dead and the log is complete, **or** (b) the human asks you to check. Then overwrite `report.md` with the tables, `status: pass|fail|partial`, commit + push.

If the timer fires and the sweep is **still running**: write a one-line update in the report (`still running`), push, and **stop**. Do not start another timer. Wait for the human.

If `nohup` quoting fails, use `tmux`; still do not wait on the train process.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Sequential (sqlite). Precache only if the dated μ is missing. Do not change `data.text.decay_lambda`, `window_agg`, patch, ticker lists, `train_end` / `val_end`. Do not override `d_model` / `d_ff`.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 149c06b (and 03f123b Coverage + DataLoader)

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

mkdir -p outputs
LOG=outputs/c0c1-windows.log
cd /workspace/DecisionForecast
nohup bash -lc '
set -euo pipefail
cd /workspace/DecisionForecast
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 train.seed=42 train.lr=0.0003 train.num_workers=4 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null model.n_prototypes=5"
run() {
  echo "=== START $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  uv run python -m src.training.train $SHARED "$@"
  echo "=== DONE  $* $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}
for spec in "60 14" "60 30" "36 7" "96 7"; do
  set -- $spec
  T=$1 H=$2
  for m in c0 c1; do
    run model=$m data.lookback_T=$T data.horizon=$H
  done
done
echo SWEEP_DONE
' > "$LOG" 2>&1 &
echo "PID=$! log=$LOG"
```

Do **not** `wait` on that PID.

Grep the finished log: `METRICS_ROW`, `Epoch `, `Early stopping`, `val_mse`, `train device:`, `First batch tensors`, `First batch shapes`, `DataLoader num_workers=`, `TEXT_COVERAGE split=`, `H3_ROW`, `=== START`, `=== DONE`, `SWEEP_DONE`.

`lookback_T` is **not** in `METRICS_ROW` — take T/H from the matching `=== START` line. Test is **best.pt** (train reloads the best checkpoint). Last `Epoch … val_mse` is the early-stop epoch, often worse than min val.

### Table 1 — metrics (this sweep only)

```text
| model | lookback_T | horizon | stop_ep | best_val | last_val | test_mse | test_mae | mae_denorm |
| c0 | 60 | 14 | … | … | … | … | … | … |
| c1 | 60 | 14 | … | … | … | … | … | … |
| c0 | 60 | 30 | … | … | … | … | … | … |
| c1 | 60 | 30 | … | … | … | … | … | … |
| c0 | 36 | 7 | … | … | … | … | … | … |
| c1 | 36 | 7 | … | … | … | … | … | … |
| c0 | 96 | 7 | … | … | … | … | … | … |
| c1 | 96 | 7 | … | … | … | … | … | … |
```

`best_val` = min `val_mse` among `Epoch N — … val_mse=` lines in that job. `last_val` = val_mse of the last Epoch line. `stop_ep` = last Epoch N (or the `Early stopping at epoch N` number).

Per pair (T,H): name who wins **best_val** and who wins **test_mse**. Do not declare a single winner across H.

### Table 2 — overfit (same 8 jobs)

```text
| model | T | H | stop_ep | train_loss_at_best | train_loss_last | last_val - best_val |
| … | … | … | … | … | … | … |
```

`train_loss_at_best` from the Epoch line where val_mse = best_val. One paragraph: if train_loss keeps falling while val rises until patience=5, early stop is catching overfit; if stop_ep is 6–8 and train/val still moving together, more likely underfit/noise than memorization.

### C1 (and C0) text_zero

Paste all `H3_ROW … variant=text_zero` lines. Δmse ≈ 0 means cross-attn/text is not used for the forecast even if C1 wins mse.

Paste one `train device:` / `x=cuda:0`, one `DataLoader num_workers=`, and `TEXT_COVERAGE split=` **per distinct (T,H)** (window counts change). Paste one `First batch shapes` per (T,H) (text_seq length must match T).

Status:
- `fail` — majority crash, CPU-only, `num_workers=0`, or lookback/patch mismatch that aborts the grid
- `partial` — started (first push), still running (timer, not done), or incomplete table
- `pass` — 8/8, CUDA, workers=4, both tables, C0 vs C1 named per pair, overfit paragraph, text_zero pasted

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes if volume has FNSPID + dated μ (confirmed on seed-shortlist)
- gpu: required
- approx_ram_gb: 16+
- git: descendant of `149c06b` (must include `03f123b` Coverage + DataLoader)
- budget: ~25–45 min (recent 8–9 job uncapped ~27–31 min; H=30 similar x, longer y)
- prior: `seed-shortlist` `pass` — mean-val winner C0 proto=5 at T=60 H=7; C1 proto=5 worst mean val; C1 text_zero Δ≈0. This round tests whether that holds at other T/H.

## Out of scope
- Do not change product code unless a one-liner unblocks.
- Do not run A / B / dlinear / timexer_plain / paper / extra seeds / lr=0.001 / d_model / proto≠5 / T=60 H=7 / decay_lambda / window_agg / caps.
- Do not delete `*.npz` text-series caches. Do not run `ablation_table`. Do not commit `outputs/` or `mlflow.db`.
- Do not set `HF_TOKEN`. Encoder stays public `FinLang/finance-embeddings-investopedia`.

## After the run
Same folder `report.md`. First push = started. Last push = metrics (on timer **or** human ping).
