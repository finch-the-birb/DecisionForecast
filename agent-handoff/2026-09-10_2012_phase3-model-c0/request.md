# Request: phase3-model-c0

- authored_by: dev
- created_at: 2026-09-10T20:12:07Z
- target_ref: feat/phase2-timexer@1e0aaffe0b6992d7d3613129828c73f8015731c0
- phase: 3
- priority: normal

## Goal
Sprint 5 / H2 factor **B vs C0** on the same capped FNSPID protocol as `phase2-model-b`. C0 injects `text_seq` as per-patch add **after** `proto(P)` (`days_to_patches`); encoder `exo=None`; **no** text at the head. Nested `fusion.*` must be asserted and logged.

Success: both trains exit 0; CUDA; `METRICS_ROW`; MLflow `fusion.*` + `n_params` for C0; `explain/projection_examples_b.json` and `explain/projection_examples_c0.json`. Negative Δ(B→C0) is valid. Metrics only from logs / `metrics.json`. Not paper.

## Commands
**Identical** overrides for B and C0 (seed stays 42 from config). Do **not** change `data.text.*`, lookback, seed, H, or window caps. **no HF_TOKEN**, no FNSPID download, not `ticker_set=paper`. Precache only if `mu_dev.npy` is missing.

```bash
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD   # must contain 1e0aaff

uv sync
SHARED="train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7"
for m in b c0; do
  uv run python -m src.training.train model=$m $SHARED
done
```

Grep `METRICS_ROW`, `n_params`. For C0 confirm MLflow / `config_resolved.yaml`:
`fusion.kind=mid_no_attn`, `text_at_head=false`, `text_to_patches=true`, `text_align=per_patch`, `text_inject=add`, `inject_layers=0` (or `[0]`), `text_as_exogenous=false`.

Table from the two `metrics.json` files only (Δ = C0 − B):

```text
| model | mse | mae | n_params |
| b | … | … | … |
| c0 | … | … | … |
| Δ(B→C0) mse | … |  |  |
| Δ(B→C0) mae | … |  |  |
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (`cache/text_series/` from text-protocol)
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+
- git: descendant of `1e0aaff`
- prior: `phase2-model-b` status `pass` (do not re-run a / dlinear / timexer_plain / c1)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock.
- Do not run `model=a`, `model=c1`, `model=dlinear`, `model=timexer_plain`, H5, or `ticker_set=paper`.
- Do not invent metrics. Do not commit `outputs/` or caches.
- Optional after trains: `uv run python -m src.evaluation.ablation_table --out-dir outputs/tables` (do not commit the markdown).

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
