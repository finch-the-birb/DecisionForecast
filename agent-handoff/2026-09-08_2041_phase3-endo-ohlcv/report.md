# Report: phase3-endo-ohlcv

- authored_by: runner
- created_at: 2026-09-08T20:58:13Z
- request_folder: agent-handoff/2026-09-08_2041_phase3-endo-ohlcv/
- tested_ref: feat/phase1-data@430d91d12cf0295c4f3fed2244cfe0350fb553b9
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: 430d91d (contains code 7e66275). HF_TOKEN unset. ticker_set=dev. No FNSPID/encoder re-download.
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
- duration: loop ~9m 29s wall (568516 ms)
- host: runpod pod `p5cnr4o8bnd4c4`
- device: cuda:0, NVIDIA L4 (capability 8.9). `train.device=cuda`; no CPU fallback. No OOM. No HF 401. Not `ticker_set=paper`.

All four: 64/32/32 windows, H=7, NVDA resolved, params on cuda:0, first-batch text L2 > 0, METRICS_ROW + metrics.json + projection JSON.

## Key signals
METRICS_ROW (verbatim):
```
METRICS_ROW model=a horizon=7 mse=18.1757 mae=3.2777
METRICS_ROW model=b horizon=7 mse=16.3450 mae=3.0208
METRICS_ROW model=c0 horizon=7 mse=16.9846 mae=3.1155
METRICS_ROW model=c1 horizon=7 mse=16.9914 mae=3.1174
```

metrics.json paths (this run only):
- A `outputs/2026-09-08/20-49-56/metrics.json` mse=18.17574691772461 mae=3.27769136428833
- B `outputs/2026-09-08/20-52-20/metrics.json` mse=16.345046997070312 mae=3.020841598510742
- C0 `outputs/2026-09-08/20-54-37/metrics.json` mse=16.984609603881836 mae=3.115532398223877
- C1 `outputs/2026-09-08/20-56-56/metrics.json` mse=16.99135398864746 mae=3.1173954010009766

Table via `format_ablation_table` on those four objects (Δ = later − earlier):

| model | mse | mae |
|-------|-----|-----|
| A | 18.1757 | 3.2777 |
| B | 16.3450 | 3.0208 |
| C0 | 16.9846 | 3.1155 |
| C1 | 16.9914 | 3.1174 |
| Δ(A→B) | -1.8307 | -0.2568 |
| Δ(B→C0) | +0.6396 | +0.0947 |
| Δ(C0→C1) | +0.0067 | +0.0019 |

- oom: no
- traceback_summary: n/a
- first-batch text L2: A 0.6250; B 0.5000; C0 0.5938; C1 0.5938

## Artifacts (paths on Runner disk — do not commit binaries)
- A: `outputs/2026-09-08/20-49-56/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_a.json)
- B: `outputs/2026-09-08/20-52-20/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_b.json)
- C0: `outputs/2026-09-08/20-54-37/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_c0.json)
- C1: `outputs/2026-09-08/20-56-56/` (metrics.json, train.log, checkpoints/best.pt, explain/projection_examples_c1.json)

## Conclusions for Dev
1. New TimeXer (all OHLCV endogenous, C1 exo = text only) trains on L4 for A/B/C0/C1 under the capped protocol.
2. On this 1-epoch / 64-window slice: B is best (Δ A→B mse −1.8307); C0 and C1 are close to each other and worse than B (Δ B→C0 +0.6396; Δ C0→C1 +0.0067). Smoke only — not paper H2.
3. Do not mix these B/C0/C1 numbers with `2026-09-08_2001_phase3-ablation` (inverted price-exo). This table is from `7e66275` only.

## Suggested next command (optional)
```bash
# still ticker_set=dev; raise caps/epochs only if Dev asks
for m in a b c0 c1; do
  uv run python -m src.training.train model=$m train.device=cuda train.ticker_set=dev train.epochs=1
done
```
