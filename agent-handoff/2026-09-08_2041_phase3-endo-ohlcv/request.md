# Request: phase3-endo-ohlcv

- authored_by: dev
- created_at: 2026-09-08T20:41:58Z
- target_ref: feat/phase1-data@7e662756804e5bd9f580d4148a1c1a5c2e17bc08
- phase: 3
- priority: normal

## Goal
Re-smoke **A / B / C0 / C1** after an architecture fix: all OHLCV channels are endogenous TimeXer patches (M-style, per-variate \(G_{en}\)). Price channels are **not** inverted exo tokens. C1 exo is **text only**; B/C0 use `exo=None`. Head mean-pools the `close` channel.

Success: all four trains exit 0; no OOM; CUDA; `METRICS_ROW` (or `H1_ROW`) for each model; `metrics.json` + projection JSON. Fill the ablation table from those numbers only (Δ = later − earlier).

**Do not compare** these B/C0/C1 numbers to `2026-09-08_2001_phase3-ablation` — that smoke used inverted price-exo. Model A is unchanged; B/C0/C1 are a new backbone. Smoke only — not a paper H2 / paper-ticker run.

## Commands
FNSPID, news parquet, and encoder npy cache are on the volume — **do not re-download**, **no HF_TOKEN**, **not** `ticker_set=paper`.

```bash
uv sync
for m in a b c0 c1; do
  uv run python -m src.training.train \
    model=$m \
    train.device=cuda \
    train.ticker_set=dev \
    train.max_train_windows=64 \
    train.max_val_windows=32 \
    train.max_test_windows=32 \
    train.epochs=1
done
```

Grep logs for `METRICS_ROW` (fallback `H1_ROW`). Build the table with `format_ablation_table` on the four `metrics.json` objects:

```text
| model | mse | mae |
| A | … | … |
| B | … | … |
| C0 | … | … |
| C1 | … | … |
| Δ(A→B) | … | … |
| Δ(B→C0) | … | … |
| Δ(C0→C1) | … | … |
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes
- gpu: **required** (`train.device=cuda`)
- approx_ram_gb: 16+
- git: checkout `feat/phase1-data` at this request commit (descendant of `7e66275`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) or `ticker_set=paper` trains unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.
- Do not invent metrics.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
