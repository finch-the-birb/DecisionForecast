# Report: protocol-norm

- authored_by: runner
- created_at: 2026-09-10T21:32:10Z
- request_folder: agent-handoff/2026-09-10_2049_protocol-norm/
- tested_ref: feat/phase2-timexer@29909e1629252b9dad95d772cba048696258c562 (contains 2c733ae)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not b/c0/c1. Not paper. No window caps.
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
# HEAD=29909e1 (request commit; code 2c733ae)
uv sync
uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7"
for m in dlinear timexer_plain a; do
  uv run python -m src.training.train model=$m $SHARED
done
```

## Outcome
- exit_code: uv sync=0; text_precache=0; dlinear=0; timexer_plain=0; a=0
- duration: precache ~58s; three trains ~15m 48s (947611 ms). Early stopping (patience=5): dlinear ep8, plain ep6, a ep11 (not full 20).
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4 23034 MiB. First batch `x=cuda:0 text=cuda:0`. No OOM. Not paper. Uncapped: train 23478 / val 2510 / test 2480.

Epoch time (from Epoch N timestamps): dlinear ~24 s/ep; timexer_plain ~30 s/ep; a ~28 s/ep.

VRAM (`log_cuda_memory` first train batch after warmup; nvidia-smi between jobs is 0 MiB because the process has exited):
- dlinear: allocated=22.0 MiB reserved=42.0 MiB
- timexer_plain: allocated=25.2 MiB reserved=46.0 MiB
- a: allocated=24.3 MiB reserved=48.0 MiB

New mu (old `mu_dev.npy` left in place):
`Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy` L2=0.4390

First batch shapes (verbatim, all three):
```
First batch shapes x=(32, 60, 5) text=(32, 768) text_seq=(32, 60, 768)
```

## Key signals
TEXT_COVERAGE split (verbatim):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```
Ticker `days_with_news` (verbatim): AAPL 0.198, MSFT 0.183, GOOGL 0.288, AMZN 0.097, FB 0.053, NVDA 0.631, TSLA 0.283, JPM 0.190, BAC 0.595, WMT 0.555, JNJ 0.353, PG 0.347, XOM 0.535, DIS 0.500, NFLX 0.588.

METRICS_ROW (verbatim):
```
METRICS_ROW model=dlinear horizon=7 mse=0.6733 mae=0.5762 mae_denorm=5.2572
METRICS_ROW model=timexer_plain horizon=7 mse=0.7174 mae=0.5978 mae_denorm=5.4950
METRICS_ROW model=a horizon=7 mse=0.9835 mae=0.7215 mae_denorm=7.2259
```

n_params (verbatim):
```
Model parameters on cuda:0 n_params=854        # dlinear
Model parameters on cuda:0 n_params=146695     # timexer_plain
Model parameters on cuda:0 n_params=144199     # a
```

Last-epoch val_mse (logs, not json): dlinear ep8 val_mse=0.7241; timexer_plain ep6 val_mse=0.7661; a ep11 val_mse=1.0546.

Table from the three `metrics.json` + last-epoch val from logs (n_params from logs):

| model | val_mse (last epoch) | test_mse | test_mae | test_mae_denorm | n_params |
| dlinear | 0.7241 | 0.6733 | 0.5762 | 5.2572 | 854 |
| timexer_plain | 0.7661 | 0.7174 | 0.5978 | 5.4950 | 146695 |
| a | 1.0546 | 0.9835 | 0.7215 | 7.2259 | 144199 |
| ratio plain/dlinear test_mse | 1.0655 |  |  |  |  |

json: dlinear mse=0.6733139 mae=0.5761713 mae_denorm=5.2572427; plain mse=0.7173833 mae=0.5978170 mae_denorm=5.4950423; a mse=0.9835436 mae=0.7215474 mae_denorm=7.2259388; ratio=0.7173833/0.6733139=1.06545.

Sanity band `0.7 ≤ mse_plain/mse_dlinear ≤ 1.3`: **in band** (1.0655). Not tuned.

Val vs test: same order of magnitude for all three (no ~0.4 val vs ~5–18 test). Protocol gap looks fixed.

Coverage: train `zero_windows=0.565` is **not** below 0.42 (old capped-64 figure). That is why status is `partial` despite green trains + in-band ratio + same-order val/test.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- dlinear: `outputs/2026-09-10/21-15-38/` (`metrics.json`, `train.log`)
- timexer_plain: `outputs/2026-09-10/21-20-11/`
- a: `outputs/2026-09-10/21-24-35/` (`explain/projection_examples_a.json`)
- mu: `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy`
- old mu kept: `.../mu_dev.npy`

## Conclusions for Dev
1. **per_window + train_start=2015** fixes the val/test scale bug: test MSE is 0.67–0.98 vs last-epoch val 0.72–1.05.
2. TimeXer_plain vs DLinear ratio **1.07** (in ±30%). Do not tune further for the band.
3. Train `zero_windows=0.565` missed the “below 0.42” bar. Ticker `days_with_news` did jump (AAPL 0.198 vs old 0.041); the 0.42 number was a late-2021 64-window slice, not a full 2015–2021 train.
4. Did not run b/c0/c1/paper. Did not cap windows. Did not change lr/epochs.

## Suggested next command (optional)
```bash
# only if Dev wants B/C0/C1 on this protocol — new request.md
```
