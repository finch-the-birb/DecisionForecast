# Report: phase3-model-c0

- authored_by: runner
- created_at: 2026-09-10T20:45:40Z
- request_folder: agent-handoff/2026-09-10_2012_phase3-model-c0/
- tested_ref: feat/phase2-timexer@2c7ed896f29d9e77f5b0518d9674ca3937df32dd (contains 1e0aaff)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not a/c1/dlinear/timexer_plain. Not paper.
# mu_dev.npy already on volume.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
uv sync
SHARED="train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7"
for m in b c0; do uv run python -m src.training.train model=$m $SHARED; done
```

## Outcome
- exit_code: uv sync=0; b=0; c0=0
- duration: both trains ~2m 24s (143638 ms)
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4. First batch `x=cuda:0 text=cuda:0`. No OOM. Not paper.

Same Hydra overrides on both (only `model=` differs):
`train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7`

METRICS_ROW (verbatim):
```
METRICS_ROW model=b horizon=7 mse=7.0935 mae=1.4745
METRICS_ROW model=c0 horizon=7 mse=6.3575 mae=1.3941
```

n_params (verbatim logs / MLflow):
```
Model parameters on cuda:0 n_params=213063   # b
Model parameters on cuda:0 n_params=200711   # c0
```

C0 MLflow (`mlflow.db` run `0774b6c7ae03419891ca069cddca6924`) and `outputs/2026-09-10/20-43-59/config_resolved.yaml`:
```
fusion.kind=mid_no_attn
fusion.text_at_head=False
fusion.text_to_patches=True
fusion.text_align=per_patch
fusion.text_inject=add
fusion.inject_layers=0          # yaml / resolved: [0]
fusion.text_as_exogenous=False
n_params=200711
```

Table from the two `metrics.json` files (n_params from logs; json has no n_params). Δ = C0 − B:

| model | mse | mae | n_params |
| b | 7.0935 | 1.4745 | 213063 |
| c0 | 6.3575 | 1.3941 | 200711 |
| Δ(B→C0) mse | −0.7360 |  |  |
| Δ(B→C0) mae | −0.0804 |  |  |

json: B mse=7.0934744 mae=1.4745089; C0 mse=6.3574557 mae=1.3941408; Δmse=−0.7360187 Δmae=−0.0803680.

B test numbers match the prior `phase2-model-b` B run (seed 42).

Projection JSON confirmed:
- `outputs/2026-09-10/20-42-45/explain/projection_examples_b.json`
- `outputs/2026-09-10/20-43-59/explain/projection_examples_c0.json`

## Key signals
- metrics: C0 better than B on this cap (Δmse −0.74, Δmae −0.08). Smoke only.
- epoch-10 val (logs): B val_mse=0.4290; C0 val_mse=0.4598
- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- B: `outputs/2026-09-10/20-42-45/` (`metrics.json`, `checkpoints/best.pt`, `explain/projection_examples_b.json`)
- C0: `outputs/2026-09-10/20-43-59/` (`metrics.json`, `checkpoints/best.pt`, `explain/projection_examples_c0.json`)
- mlflow_run_id: B `8a0b4d3970f24b3e9ad433e190c232da`; C0 `0774b6c7ae03419891ca069cddca6924`

## Conclusions for Dev
1. Sprint 5 / H2 factor B vs C0 smoke **pass**: both CUDA exit 0; C0 fusion flags match the request (`mid_no_attn`, per-patch add, no head text, exo=false, inject `[0]`).
2. **Δmse C0−B = −0.74**, **Δmae = −0.08** (C0 better on this 64-window cap). Not paper H2.
3. Did not run a/c1/dlinear/timexer_plain/paper. Did not run optional `ablation_table`.

## Suggested next command (optional)
```bash
# only if Dev wants C1 on this protocol — new request.md
```
