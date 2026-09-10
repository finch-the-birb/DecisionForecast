# Request: phase3-ablation

- authored_by: dev
- created_at: 2026-09-08T20:01:24Z
- target_ref: feat/phase1-data@0258be7ed1f53d0b7dacaea72138619af9ecf8a6
- phase: 3
- priority: normal

## Goal
Capped H2 smoke: **A / B / C0 / C1** on the same protocol (dev tickers, H=7, 1 epoch, 64/32/32 windows, `train.device=cuda`). Fusion factor only: A/B late; C0 mid-add to patches (no G_en→text); C1 text as exo token via G_en cross-attn.

Success: all four trains exit 0; no OOM; CUDA; `METRICS_ROW` (or `H1_ROW`) for each model; `metrics.json` + projection JSON. Fill the ablation table from those numbers only (Δ = later − earlier). Smoke only — not a paper H2 / paper-ticker run.

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
- git: checkout `feat/phase1-data` at this request commit (descendant of `0258be7`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) or `ticker_set=paper` trains unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.
- Do not invent metrics.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
