# Report: hp-sweep-cap

- authored_by: runner
- created_at: 2026-09-11T00:48:30Z
- request_folder: agent-handoff/2026-09-10_2346_hp-sweep-cap/
- tested_ref: feat/phase2-timexer@c43648231a81b627baeb1ceb9e9ea3d844351880 (contains 2c733ae)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not c0/dlinear/timexer_plain. Not paper. Not ablation_table.
# Sweep was nohup PID 90833; harvest after SWEEP_DONE (process already dead).
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
uv sync
nohup bash -lc 'uv run python -m src.training.train -m model=a,b,c1 train.lr=0.0003,0.001 model.d_model=64,128 train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7 train.max_train_windows=256 train.max_val_windows=64 train.max_test_windows=64; echo SWEEP_DONE'
```

## Outcome
- exit_code: uv sync=0; Hydra 12/12 jobs completed; log ends with `SWEEP_DONE`; nohup PID 90833 gone
- duration: ~3 m (00:02:47 → 00:05:44Z)
- host: runpod pod `i6lv3n0222en43`
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (job #0 and the rest). NVIDIA L4. No OOM. Not paper.

GPU check (verbatim, job #0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
```

METRICS_ROW (verbatim, 12/12):
```
METRICS_ROW model=a horizon=7 mse=2.3420 mae=1.3588 mae_denorm=14.4241
METRICS_ROW model=a horizon=7 mse=2.3613 mae=1.3661 mae_denorm=14.5353
METRICS_ROW model=a horizon=7 mse=2.8019 mae=1.3481 mae_denorm=13.2110
METRICS_ROW model=a horizon=7 mse=1.9848 mae=1.1648 mae_denorm=11.4147
METRICS_ROW model=b horizon=7 mse=0.6047 mae=0.6304 mae_denorm=7.5740
METRICS_ROW model=b horizon=7 mse=0.9496 mae=0.8417 mae_denorm=10.5601
METRICS_ROW model=b horizon=7 mse=1.7236 mae=1.1681 mae_denorm=14.5098
METRICS_ROW model=b horizon=7 mse=0.5065 mae=0.5717 mae_denorm=5.4714
METRICS_ROW model=c1 horizon=7 mse=0.9251 mae=0.8169 mae_denorm=9.8572
METRICS_ROW model=c1 horizon=7 mse=0.6217 mae=0.6175 mae_denorm=6.8861
METRICS_ROW model=c1 horizon=7 mse=1.0657 mae=0.8544 mae_denorm=9.9425
METRICS_ROW model=c1 horizon=7 mse=1.0702 mae=0.9005 mae_denorm=10.8567
```

Table from this sweep only (`outputs/multirun/2026-09-11/00-02-45/<0–11>/metrics.json` + last-epoch `val_mse` / `n_params` from each `train.log`). **Winner = best last-epoch val_mse**: **a / lr=0.001 / d_model=128** (job #3, val_mse=0.9932).

| model | lr | d_model | val_mse | test_mse | test_mae | mae_denorm | n_params |
| a | 0.0003 | 64 | 2.4479 | 2.3420 | 1.3588 | 14.4241 | 144199 |
| a | 0.0003 | 128 | 2.4436 | 2.3613 | 1.3661 | 14.5353 | 226951 |
| a | 0.001 | 64 | 1.5153 | 2.8019 | 1.3481 | 13.2110 | 144199 |
| **a** | **0.001** | **128** | **0.9932** | **1.9848** | **1.1648** | **11.4147** | **226951** |
| b | 0.0003 | 64 | 1.5921 | 0.6047 | 0.6304 | 7.5740 | 213063 |
| b | 0.0003 | 128 | 1.8543 | 0.9496 | 0.8417 | 10.5601 | 514631 |
| b | 0.001 | 64 | 1.0805 | 1.7236 | 1.1681 | 14.5098 | 213063 |
| b | 0.001 | 128 | 1.2465 | 0.5065 | 0.5717 | 5.4714 | 514631 |
| c1 | 0.0003 | 64 | 1.6188 | 0.9251 | 0.8169 | 9.8572 | 200711 |
| c1 | 0.0003 | 128 | 2.2294 | 0.6217 | 0.6175 | 6.8861 | 539143 |
| c1 | 0.001 | 64 | 1.1594 | 1.0657 | 0.8544 | 9.9425 | 200711 |
| c1 | 0.001 | 128 | 2.0249 | 1.0702 | 0.9005 | 10.8567 | 539143 |

Val vs test same order of magnitude on all 12 (no ~0.4 vs ~5–18). Caps 256/64/64. seed=42. `d_ff` stayed 256.

No Traceback / Error in the log. `SWEEP_DONE` present.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-sweep-cap.log`
- hydra sweep: `outputs/multirun/2026-09-11/00-02-45/` (jobs 0–11, each `metrics.json`)
- mlflow_run_id: n/a (not harvested; do not mix into R3 ablation_table)

## Conclusions for Dev
1. Capped HP sweep **pass**: 12/12 CUDA, full table. Best **val_mse** is **A, lr=0.001, d_model=128** (0.9932). Lowest **test_mse** is B lr=0.001 d_model=128 (0.5065) — not the val winner. Not paper HP.
2. Did not run c0 / dlinear / timexer_plain / ablation_table / paper.

## Suggested next command (optional)
```bash
# stop — do not start paper
```
