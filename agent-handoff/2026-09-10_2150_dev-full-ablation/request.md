# Request: dev-full-ablation

- authored_by: dev
- created_at: 2026-09-10T21:50:00Z
- target_ref: feat/phase2-timexer@2c733ae
- phase: 3
- priority: high

## Goal
Uncapped **dev** H2+H3 on the accepted protocol (`train_start=2015-01-01`, `normalize=per_window`, mu `mu_dev_2015-01-01_2021-12-31.npy`). Grid: **A / B / C0 / C1 × seed 0,1,2**, H=7, 20 epochs (early stopping OK). Then `ablation_table` (mean±std over seeds, paired Δ).

`c1-rewrite` is `pass`. `protocol-norm` protocol is accepted. **Do not** re-run dlinear / timexer_plain. **Do not** start `ticker_set=paper`. This is the last remote job on this ladder.

Success:
- all **12** trains exit 0, CUDA, `METRICS_ROW` + `mae_denorm` per (model, seed)
- val-MSE and test-MSE same order of magnitude on each job
- `ablation_table` markdown pasted in the report; **n=3** per model (seeds 0/1/2). Do not mix seed=42 capped smokes into the narrative table — the script already keeps latest-per-(model,H,seed)
- H3 block from the same script (mean±std). C1 `text_zero` mean Δmse **≠ 0**
- Negative Δ is valid. Not paper. Do not invent metrics.

## Commands
**no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Do **not** cap windows. Do **not** change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`. Sequential only (sqlite MLflow). Precache only if the dated mu file is missing.

R1 uncapped ~5 min/model with patience=5; budget **~1–2 h** for 12 jobs. Run the loops in the **foreground** (same as protocol-norm / c1-rewrite). If the chat may die, wrap the two `for` loops in `nohup` yourself and wait on the log; do not start other trains.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 2c733ae

uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
if [ ! -f "$MU" ]; then
  uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
fi

# explicit null caps — do not copy 64/32/32 from c1-rewrite
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
for m in a b c0 c1; do
  for s in 0 1 2; do
    echo "=== START model=$m seed=$s $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    uv run python -m src.training.train model=$m train.seed=$s $SHARED
    echo "=== DONE  model=$m seed=$s $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  done
done

uv run python -m src.evaluation.ablation_table \
  --tracking-uri sqlite:///mlflow.db \
  --experiment decision-forecast \
  --out-dir outputs/tables \
  --ticker-set dev \
  --lookback 60 \
  --normalize per_window \
  --train-start 2015-01-01
cat outputs/tables/ablation_a_b_c0_c1.md
```

Grep logs: `METRICS_ROW`, last-epoch `val_mse`, `=== START`, `=== DONE`.

Paste **verbatim** `outputs/tables/ablation_a_b_c0_c1.md` (H2 mean±std + H3 block). Also a compact 12-row scratch table from logs / `metrics.json` (do not commit the markdown file):

```text
| model | seed | val_mse | test_mse | test_mae | mae_denorm |
| a | 0 | … | … | … | … |
| … | … | … | … | … | … |
```

Status:
- `fail` — majority crash, or val vs test still an order of magnitude apart
- `partial` — table exists but n<3 for any of A/B/C0/C1, or C1 text_zero mean Δmse is 0.0000
- `pass` — 12/12, n=3, same-order val/test, table + H3 pasted, C1 text_zero ≠ 0

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (prices + text_series + dated mu)
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+
- git: descendant of `2c733ae` (HEAD may be `6c1c0bb` or later)
- prior: `c1-rewrite` `pass`; `protocol-norm` protocol accepted
- windows: **uncapped** 15 dev tickers from 2015 (~23k/2.5k/2.5k on R1)

## Out of scope
- Do not change product code unless a trivial one-liner unblocks.
- Do not run `model=dlinear`, `model=timexer_plain`, H∈{14,30}, or `ticker_set=paper`.
- Do not tune lr/epochs. Do not parallelize jobs.
- Do not commit `outputs/`, caches, `mlflow.db`, or `outputs/tables/*.md`.
- After this report: **stop**. Do not start a paper run.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
