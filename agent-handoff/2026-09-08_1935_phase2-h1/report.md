# Report: phase2-h1

- authored_by: runner
- created_at: 2026-09-08T19:45:06Z
- request_folder: agent-handoff/2026-09-08_1935_phase2-h1/
- tested_ref: feat/phase1-data@9e127d3e0d081def1f98589ee39011febb6a7e25
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: 9e127d3 (code 88faf1a). HF_TOKEN unset. No FNSPID/encoder re-download.
uv sync
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
uv run python -m src.training.train \
  model=b \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```

## Outcome
- exit_code: uv sync=0; train A=0; train B=0
- duration: A ~2m 28s; B ~2m 24s
- host: runpod pod `p5cnr4o8bnd4c4`
- device: cuda:0, NVIDIA L4 (capability 8.9). No CPU fallback. No HF 401. No OOM.

Both runs: 64/32/32 windows, H=7, NVDA resolved, params on cuda:0, first-batch text L2 > 0. Encoder npy cache hit (ST not reloaded this round).

## Key signals
H1_ROW:
```
H1_ROW model=a horizon=7 mse=18.1757 mae=3.2777
H1_ROW model=b horizon=7 mse=17.3772 mae=3.1577
```

metrics.json (A `outputs/2026-09-08/19-41-45/metrics.json`, B `outputs/2026-09-08/19-44-18/metrics.json`):
A mse=18.17574691772461 mae=3.27769136428833  
B mse=17.377195358276367 mae=3.1576571464538574

Table via `format_ab_table` on those JSON objects (Δ = B − A):

| model | mse | mae |
|-------|-----|-----|
| A | 18.1757 | 3.2777 |
| B | 17.3772 | 3.1577 |
| Δ(A→B) | -0.7986 | -0.1200 |

- A: first-batch text L2=0.6250; CUDA after model.to 0.6 MiB / first batch 0.7 MiB
- B: first-batch text L2=0.5625; CUDA after model.to 2.0 MiB / first batch 2.2 MiB
- A epoch1 train_loss=10.3847 val_mse=5.5947 val_mae=2.1391
- B epoch1 train_loss=34.3720 val_mse=5.0432 val_mae=2.0130
- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- A hydra: `outputs/2026-09-08/19-41-45/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_a.json)
- B hydra: `outputs/2026-09-08/19-44-18/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_b.json)

## Conclusions for Dev
1. H1 smoke path works: A and B both train on L4 with late fusion embeddings from cache; projection JSON for both.
2. On this capped 1-epoch slice B is slightly better on test (Δ mse −0.7986, Δ mae −0.1200). Smoke only — same tiny recent-window cap as phase1, not a paper H1.
3. First-batch CUDA mem stayed ~0.7–2.2 MiB because encoder weights were already on disk as npy (no ST reload). TimeXL/TimeXer params were on cuda:0.

## Suggested next command (optional)
```bash
# larger cap / more epochs when Dev wants a less noisy H1
uv run python -m src.training.train model=a train.device=cuda train.ticker_set=dev train.epochs=1
uv run python -m src.training.train model=b train.device=cuda train.ticker_set=dev train.epochs=1
```
