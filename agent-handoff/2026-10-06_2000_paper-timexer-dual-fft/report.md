# Report: paper-timexer-dual-fft

- authored_by: runner
- created_at: 2026-10-06T17:34:45Z
- updated_at: 2026-10-06T18:44:00Z
- request_folder: agent-handoff/2026-10-06_2000_paper-timexer-dual-fft/
- tested_ref: feat/phase2-timexer@33e4a41d2493381cbc72c88635a3a1ebf09641e1
- status: partial (Block 1/4, 6/48 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. No old baselines. Human ping while PID 2777 still alive.
# Compact text parquet count on disk: 61
```

## Outcome
- exit_code: in progress; 6/48 DONE; `SWEEP_DONE` absent; no Traceback
- host: runpod pod `sw7txpqcwtk7tz` (hostname `86df0010ddba`)
- device: finished jobs are `train device: cuda`, `DataLoader num_workers=4`. NVIDIA L4 assumed from the running job; not polled.
- pid: 2777
- log: `outputs/paper-timexer-dual-fft.log`
- current_job: Block 1, stride=6, d_model=64, e_layers=2, use_prototypes=false, seed=0 (started 2026-10-06T18:40:17Z)
- splits on finished jobs: train 98236 / val 12048 / test 11904
- first-batch shapes on finished jobs: `x=(32, 60, 5)`, `text_seq=(32, 60, 15)`
- skipped price files: UNH, VZ

`best_val_mse` = min Epoch `val_mse`. `stop_ep` = early-stop epoch, else last logged epoch. `test_*` = `METRICS_ROW`.

n_params so far: e=1 no-proto 122761; e=1 proto=10 127561.

### Finished jobs

| stride | d_model | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
| 6 | 64 | 1 | false | 0 | 9 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7252 | 0.7636 | 0.6002 |
| 6 | 64 | 1 | false | 1 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7179 | 0.7718 | 0.6049 |
| 6 | 64 | 1 | false | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7322 | 0.7641 | 0.6008 |
| 6 | 64 | 1 | true | 0 | 15 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7037 | 0.7583 | 0.5977 |
| 6 | 64 | 1 | true | 1 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7134 | 0.7850 | 0.6167 |
| 6 | 64 | 1 | true | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7201 | 0.7715 | 0.6040 |

Partial means, only the two configs that already have n=3. Sample std.

| stride | d_model | e_layers | use_proto | best_val_mse | test_mse | test_mae |
| 6 | 64 | 1 | false | 0.7251±0.0072 | 0.7665±0.0046 | 0.6020±0.0026 |
| 6 | 64 | 1 | true | 0.7124±0.0082 | 0.7716±0.0134 | 0.6061±0.0097 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-dual-fft.log`

## Conclusions for Dev
1. Block 1 is still running: 6/48 done, job 7 in progress (`e_layers=2`, no prototypes, seed=0). Blocks 2–4 have not started.
2. On the finished e=1 stride=6 d_model=64 pair, prototypes lower best_val (0.7124 vs 0.7251) and do not lower test (0.7716 vs 0.7665). Both tests sit above the prior DLinear F1 0.7535. Incomplete grid; do not rank the sweep yet.
