# Request: c1-rewrite

- authored_by: dev
- created_at: 2026-09-10T21:35:00Z
- target_ref: feat/phase2-timexer@2c733ae
- phase: 3
- priority: high

## Goal
Sprint 5–6 / H2 on the **new data protocol** (accepted after `protocol-norm`: `train_start=2015-01-01`, `normalize=per_window`, mu `mu_dev_2015-01-01_2021-12-31.npy`). Capped smoke: **B vs C0 vs rewritten C1**. C1 must use shared `timexer_backbone` with **per-day text exo** and cross-attn only \(G_{en} \leftarrow Ex\); patches unmodified by text; head has no text concat.

`protocol-norm` was `partial` **only** because train `zero_windows=0.565` missed the old 0.42 bar (that bar was a late-2021 64-window slice). Val/test same order and DLinear band 1.07 are accepted. **Do not re-run** dlinear / timexer_plain / a. **Do not** chase `zero_windows`.

Success:
- all three trains exit 0, CUDA, `METRICS_ROW` + `mae_denorm`
- val-MSE and test-MSE **same order of magnitude** for each model (capped is OK; still no ~0.4 vs ~5)
- C1 fusion logged: `kind=mid_cross_attn`, `text_at_head=false`, `text_to_patches=false`, `text_as_exogenous=true`, `exo_tokens=per_day`
- `n_params_backbone_g12` **equal** across B / C0 / C1 (exo_embed / text_mlp sit outside backbone+g12)
- proto kmeans++ line present: `proto kmeans++ init bank=… nn_dist_mean=… min_pairwise=…`; epoch logs have `train_l_*`, `proto_nn_dist_mean`, `proto_min_pairwise_dist`
- projection JSON for each model; paste `n_distinct_nearest`
- `H3_ROW` for **text_zero** on B, C0, C1; C1 `delta_mse` **≠ 0** (text used via exo). Magnitude may be small on 64 windows; identically `0.0000` is a fail for the rewrite, not a pass
- Negative Δ(B→C0) / Δ(C0→C1) is valid

Not paper. Do not invent metrics.

## Commands
**Identical** overrides for B, C0, C1 (seed stays 42 from config). Do **not** change `data.text.*`, lookback, patch, H, ticker lists, `train_end` / `val_end`. **no HF_TOKEN**, no FNSPID download, **not** `ticker_set=paper`. Precache **only if** the new mu file is missing.

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

SHARED="train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7"
for m in b c0 c1; do
  uv run python -m src.training.train model=$m $SHARED
done
```

Grep: `TEXT_COVERAGE`, `METRICS_ROW`, `n_params`, `n_params_backbone_g12`, `mae_denorm`, `fusion.`, `proto kmeans++`, `proto_nn_dist_mean`, `proto_min_pairwise_dist`, `train_l_pred`, `H3_ROW`, `n_distinct_nearest`, last-epoch `val_mse`.

Confirm C1 MLflow / `config_resolved.yaml`:
`fusion.kind=mid_cross_attn`, `text_at_head=false`, `text_to_patches=false`, `text_as_exogenous=true`, `exo_tokens=per_day`.

Table from the three `metrics.json` + logs (`Δ = later − earlier`):

```text
| model | val_mse (last epoch) | test_mse | test_mae | test_mae_denorm | n_params | n_params_backbone_g12 |
| b | … | … | … | … | … | … |
| c0 | … | … | … | … | … | … |
| c1 | … | … | … | … | … | … |
| Δ(B→C0) mse | … |  |  |  |  |  |
| Δ(C0→C1) mse | … |  |  |  |  |  |
```

H3 from each train-run `explain/faithfulness_{m}.json` (not a later `explain.run` dir). Paste `H3_ROW` for `text_zero` and:

```text
| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |
| b | … | … | … | … | … |
| c0 | … | … | … | … | … |
| c1 | … | … | … | … | … |
```

Also paste train/val/test `zero_windows` (record only; do not fail on 0.42).

Status:
- `fail` — crash / fusion assert / val vs test still an order of magnitude apart
- `partial` — trains green but C1 `text_zero` Δmse is 0.0000, **or** `n_params_backbone_g12` differs across B/C0/C1
- `pass` — trains 0, fusion C1 correct, backbone+g12 parity, proto diagnostics present, C1 text_zero Δmse ≠ 0

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (prices + `cache/text_series/` + new mu from protocol-norm)
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+
- git: descendant of `2c733ae` (HEAD may be `2f3e7b0` or later; code SHA is `2c733ae`)
- prior: `protocol-norm` status `partial` (protocol accepted). Do not re-run a / dlinear / timexer_plain
- windows: capped 64/32/32 via `_cap_recent` (late-train slice, not 1980 prefix)

## Out of scope
- Do not change product code unless a trivial one-liner unblocks.
- Do not run `model=a`, `model=dlinear`, `model=timexer_plain`, H∈{14,30}, seeds, or `ticker_set=paper`.
- Do not tune lr/epochs/caps. Do not rebuild mu if the dated file exists.
- Do not commit `outputs/`, caches, `mlflow.db`.
- Optional: `uv run python -m src.evaluation.ablation_table --out-dir outputs/tables` (do not commit the markdown).

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
