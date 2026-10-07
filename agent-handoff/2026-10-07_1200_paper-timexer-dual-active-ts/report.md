# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T11:33:24Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@39ee9f0d58e9bf62155e17723b2209b4aa7fb7f8
- status: fail

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git pull --ff-only origin feat/phase2-timexer  (c6fdfe5..39ee9f0)
# uv sync
# HF_TOKEN unset. No prepare. nohup 24 jobs, ticker_set=paper.
# Sweep aborted by set -e on job 4. No restart.
```

## Outcome
- exit_code: sweep PID 4271 is dead; SWEEP_DONE absent; 3/24 DONE; job 4 raised ValueError
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: 3 finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 4271 (dead)
- log: outputs/paper-timexer-dual-active-ts.log
- started_at: 2026-10-07T10:51:23Z
- died_at: job 4 started 2026-10-07T11:15:44Z, traceback at 11:17:16Z
- failed_job: Block 1 F2 [close, volume], n_features=2, e_layers=1, use_prototypes=true, n_prototypes=10, seed=1, n_params=126921
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes on the finished jobs: x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25)
- indicators: 25, tickers=61, endogenous [close, volume]
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW.

### Finished jobs

| features | n_features | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7508 | 0.7641 | 0.6012 |
| close,volume | 2 | 1 | false | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7462 | 0.7692 | 0.6026 |
| close,volume | 2 | 1 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7775 | 0.7633 | 0.5998 |

Mean n=3, sample std. Only the F2 e=1 no-prototype cell is complete.

| features | n_features | e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 122121 | 0.7582±0.0169 | 0.7655±0.0032 | 0.6012±0.0014 |

- oom: no
- traceback_summary: src/explain/bank.py collect_segment_bank calls model(x, text, text_seq=text_seq) and does not pass ts. src/models/timexer_dual.py forward then raises ValueError: TimeXerDual with n_ts_features=25 requires ts [B, T, 25] or [B, 25]. The three no-prototype jobs never entered that path. set -e stopped Block 1 e=2 and all of F1 and F5.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-dual-active-ts.log

## Error for Dev

Prototype init drops the 25D indicator tensor. The no-prototype train path does not.

`configs/train/default.yaml` sets `train.proto.init: kmeans`. `use_prototypes=true` makes `TimeXerDual.proto` exist, so `src/training/train.py` `run_training` calls `collect_segment_bank` before the first epoch. That helper never reads `batch["ts"]`:

```python
# src/explain/bank.py, collect_segment_bank
out = model(x, text, text_seq=text_seq)
```

`TimeXerDual.forward` rejects that call when `n_ts_features=25` and `ts is None` (`src/models/timexer_dual.py`, around the `n_ts_features > 0 and ts is None` check).

The epoch loop is fine. `_forecast_inputs` takes `batch.get("ts")`, and `_forecast_forward` passes `ts=ts` when the tensor is present. The three finished jobs logged `ts=(32, 60, 25)` and never entered the bank. `hasattr(model, "proto")` is false while `use_prototypes=false`, so those jobs skipped kmeans init.

Same hole: `src/explain/projection.py` also calls `collect_segment_bank`.

Needed change: pass `batch["ts"]` into the model from `collect_segment_bank`, the same way `_forecast_forward` does. Until that lands, every `c1_dual` job with `n_ts_features=25` and `use_prototypes=true` dies in bank init, and `set -e` drops the rest of the sweep. Runner did not patch product code and did not restart.

## Conclusions for Dev
1. Sweep is dead at 3/24. Job 4 is the first `use_prototypes=true` job. It built n_params=126921, then died in kmeans bank init. Block 1 e=2 and all of F1 and F5 never started.
2. The only complete cell, F2 e=1 without prototypes, has test 0.7655±0.0032 and mae 0.6012±0.0014. That does not beat DLinear test 0.7535 or 0.7547. Shapes on those jobs are x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25).
