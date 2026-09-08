# Report: phase3-ablation

- authored_by: runner
- created_at: 2026-09-08T20:15:06Z
- request_folder: agent-handoff/2026-09-08_2001_phase3-ablation/
- tested_ref: feat/phase1-data@58d1141cc5fa4f897d22cdcbc21bf9bbd7da911f
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: 58d1141 (code 0258be7). HF_TOKEN unset. ticker_set=dev only. No FNSPID re-download.
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

## Outcome
- exit_code: uv sync=0; a=0; b=0; c0=0; c1=0
- duration: loop ~9m 31s wall
- host: runpod pod `p5cnr4o8bnd4c4`
- device: cuda:0, NVIDIA L4. No CPU fallback. No OOM. No HF 401. Not `ticker_set=paper`.

All four: 64/32/32 windows, H=7, params on cuda:0, first-batch text L2 > 0, METRICS_ROW + metrics.json + projection JSON.

## Key signals
METRICS_ROW:
```
METRICS_ROW model=a horizon=7 mse=18.1757 mae=3.2777
METRICS_ROW model=b horizon=7 mse=17.3772 mae=3.1577
METRICS_ROW model=c0 horizon=7 mse=16.8116 mae=3.0798
METRICS_ROW model=c1 horizon=7 mse=16.8367 mae=3.0844
```

metrics.json:
- A `outputs/2026-09-08/20-06-19/metrics.json` mse=18.17574691772461 mae=3.27769136428833
- B `outputs/2026-09-08/20-08-46/metrics.json` mse=17.377195358276367 mae=3.1576571464538574
- C0 `outputs/2026-09-08/20-11-04/metrics.json` mse=16.811582565307617 mae=3.079775094985962
- C1 `outputs/2026-09-08/20-13-24/metrics.json` mse=16.836687088012695 mae=3.0843565464019775

Table via `format_ablation_table` on those four objects (Δ = later − earlier):

| model | mse | mae |
|-------|-----|-----|
| A | 18.1757 | 3.2777 |
| B | 17.3772 | 3.1577 |
| C0 | 16.8116 | 3.0798 |
| C1 | 16.8367 | 3.0844 |
| Δ(A→B) | -0.7986 | -0.1200 |
| Δ(B→C0) | -0.5656 | -0.0779 |
| Δ(C0→C1) | +0.0251 | +0.0046 |

- oom: no
- traceback_summary: n/a
- first-batch text L2: A 0.6250; B/C0/C1 0.5625

## Artifacts (paths on Runner disk — do not commit binaries)
- A: `outputs/2026-09-08/20-06-19/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_a.json)
- B: `outputs/2026-09-08/20-08-46/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_b.json)
- C0: `outputs/2026-09-08/20-11-04/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_c0.json)
- C1: `outputs/2026-09-08/20-13-24/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_c1.json)

## Conclusions for Dev
1. Phase 3 smoke path works: C0 and C1 train on the same capped CUDA protocol as A/B.
2. On this 1-epoch / 64-window slice: A→B and B→C0 improve test mse/mae; C0→C1 is a small regression (+0.0251 mse). Smoke only — not paper H2.
3. Encoder npy cache hit again (no ST reload). Fusion difference is in the TimeXer path, not a new Hub download.

## Suggested next command (optional)
```bash
# larger cap / more epochs when Dev wants a less noisy H2 (still ticker_set=dev)
for m in a b c0 c1; do
  uv run python -m src.training.train model=$m train.device=cuda train.ticker_set=dev train.epochs=1
done
```
