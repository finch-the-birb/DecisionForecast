# Report: phase2-model-b

- authored_by: runner
- created_at: 2026-09-10T20:07:20Z
- request_folder: agent-handoff/2026-09-10_2001_phase2-model-b/
- tested_ref: feat/phase2-timexer@7be1c5e2b5e34ace8a2063976040425ebfc9707c (contains 55adf89)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not c0/c1. Not dlinear/timexer_plain. Not paper.
# mu_dev.npy already on volume.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
uv sync
SHARED="train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7"
for m in a b; do uv run python -m src.training.train model=$m $SHARED; done
```

## Outcome
- exit_code: uv sync=0; a=0; b=0
- duration: both trains ~3m 25s (204654 ms)
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4. First batch `x=cuda:0 text=cuda:0`. No OOM. Not paper.

Same Hydra overrides on both (only `model=` differs):
`train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7`

METRICS_ROW (verbatim):
```
METRICS_ROW model=a horizon=7 mse=5.5359 mae=1.6813
METRICS_ROW model=b horizon=7 mse=7.0935 mae=1.4745
```

n_params (verbatim logs / MLflow):
```
Model parameters on cuda:0 n_params=144199   # a
Model parameters on cuda:0 n_params=213063   # b
```

B MLflow params (`mlflow.db` run `a4525cf214334c3a9c90a720ab48fc65`; also `outputs/2026-09-10/20-05-57/config_resolved.yaml`):
```
fusion.kind=late
fusion.text_at_head=True
fusion.text_to_patches=False
fusion.text_as_exogenous=False
n_params=213063
```
A MLflow: `fusion.kind=late` (string `fusion: late` in cfg), `n_params=144199` (run `e10992622817496a8d95ef0484dfa13a`).

Table from the two `metrics.json` files (n_params from logs; json has no n_params). Δ = B − A:

| model | mse | mae | n_params |
| a | 5.5359 | 1.6813 | 144199 |
| b | 7.0935 | 1.4745 | 213063 |
| Δ(A→B) mse | +1.5575 |  |  |
| Δ(A→B) mae | −0.2068 |  |  |

json: A mse=5.5359483 mae=1.6812979; B mse=7.0934720 mae=1.4745053; Δmse=+1.5575237 Δmae=−0.2067926.

Projection JSON confirmed:
- `outputs/2026-09-10/20-04-16/explain/projection_examples_a.json`
- `outputs/2026-09-10/20-05-57/explain/projection_examples_b.json`

## Key signals
- metrics: B worse MSE than A on this cap; B better MAE. Negative H1-MSE result is valid smoke.
- epoch-10 val (logs): A val_mse=1.0528; B val_mse=0.4290 (B fits val better, test MSE still higher).
- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- A: `outputs/2026-09-10/20-04-16/` (`metrics.json`, `checkpoints/best.pt`, `explain/projection_examples_a.json`)
- B: `outputs/2026-09-10/20-05-57/` (`metrics.json`, `checkpoints/best.pt`, `explain/projection_examples_b.json`)
- mlflow_run_id: A `e10992622817496a8d95ef0484dfa13a`; B `a4525cf214334c3a9c90a720ab48fc65`

## Conclusions for Dev
1. Sprint 4 / H1 smoke **pass**: both trains CUDA exit 0; B late-fusion flags match the request; projection JSON written for A and B.
2. **Δmse B−A = +1.56** (B worse). **Δmae = −0.21** (B better). Valid negative on MSE; do not retune caps to chase DLinear.
3. B has more params (213k vs 144k) and lower epoch-10 val MSE (0.43 vs 1.05) with higher test MSE — same 64-window overfit pattern as `timexer_plain` in the sanity round.
4. Did not run c0/c1/dlinear/timexer_plain/paper.

## Suggested next command (optional)
```bash
# only if Dev wants C0/C1 on this protocol — new request.md
```
