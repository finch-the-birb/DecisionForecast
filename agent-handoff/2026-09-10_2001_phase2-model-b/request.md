# Request: phase2-model-b

- authored_by: dev
- created_at: 2026-09-10T20:01:08Z
- target_ref: feat/phase2-timexer@55adf8931a0870d3fa0b8933a68b1b00695b8d58
- phase: 2
- priority: normal

## Goal
Sprint 4 / H1 smoke: **Model A vs Model B** on the **same** capped FNSPID protocol (new daily text series). B is TimeXer + G1+G2 prototypes + **late** fusion (`fusion.kind=late`, text at head, encoder `exo=None`, `text_seq` ignored). OHLCV is multivariate PatchEmbed.

Success: both trains exit 0; CUDA; `METRICS_ROW` for each; MLflow params include `fusion.*` (B) and `n_params`; `explain/projection_examples_a.json` and `explain/projection_examples_b.json`. Negative result (B worse or ≈ A) is **valid** — still `pass` if both runs are green. Metrics only from logs / `metrics.json`. Not paper.

## Commands
Same overrides for A and B. Do **not** change `data.text.*`, lookback, seed, H, or window caps between models. Text-series cache is on the volume — **do not re-download FNSPID**, **no HF_TOKEN**, **not** `ticker_set=paper`. Precache only if `mu_dev.npy` is missing.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 55adf89

uv sync
SHARED="train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7"
for m in a b; do
  uv run python -m src.training.train model=$m $SHARED
done
```

Grep `METRICS_ROW`, `n_params`, and (for B) MLflow / resolved cfg `fusion.kind`, `fusion.text_at_head`, `fusion.text_to_patches`, `fusion.text_as_exogenous`. Confirm projection JSON paths exist. Table from the two `metrics.json` files only:

```text
| model | mse | mae | n_params |
| a | … | … | … |
| b | … | … | … |
| Δ(A→B) mse | … |  |  |
| Δ(A→B) mae | … |  |  |
```

Δ = B − A (from the json numbers).

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (`cache/text_series/` from text-protocol)
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+
- git: `feat/phase2-timexer` descendant of `55adf89`
- prior round: `agent-handoff/2026-09-10_1941_phase2-sanity/` status `partial` (pipeline ok; do not re-run dlinear/timexer_plain here)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock.
- Do not run `model=c0`, `model=c1`, `model=dlinear`, `model=timexer_plain`, H5, or `ticker_set=paper`.
- Do not invent metrics. Do not commit `outputs/` or caches.
- Do not retune 2.1 or change epochs/caps to chase the DLinear band.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
