# Report: paper-timexer-dual-fft

- authored_by: runner
- created_at: 2026-10-06T17:34:45Z
- updated_at: 2026-10-06T20:22:06Z
- request_folder: agent-handoff/2026-10-06_2000_paper-timexer-dual-fft/
- tested_ref: feat/phase2-timexer@33e4a41d2493381cbc72c88635a3a1ebf09641e1
- status: partial (Block 2/4, 12/48 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Timer harvest while PID 2777 is alive. No new train.
```

## Outcome
- exit_code: in progress; Block 1 DONE at 2026-10-06T20:19:01Z; 12/48 DONE; `SWEEP_DONE` absent; no Traceback
- host: runpod pod `sw7txpqcwtk7tz` (hostname `86df0010ddba`)
- device: all 12 finished jobs `train device: cuda`, `DataLoader num_workers=4`. No OOM.
- pid: 2777
- log: `outputs/paper-timexer-dual-fft.log`
- current_job: Block 2, stride=6, d_model=128, e_layers=1, use_prototypes=false, seed=0 (started 2026-10-06T20:19:01Z)
- splits: train 98236 / val 12048 / test 11904
- first-batch shapes: `x=(32, 60, 5)`, `text_seq=(32, 60, 15)`
- skipped price files: UNH, VZ

`best_val_mse` = min Epoch `val_mse`. `stop_ep` = early-stop epoch, else last logged epoch. `test_*` = `METRICS_ROW`.

### Finished jobs (Block 1, stride=6, d_model=64)

| stride | d_model | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
| 6 | 64 | 1 | false | 0 | 9 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7252 | 0.7636 | 0.6002 |
| 6 | 64 | 1 | false | 1 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7179 | 0.7718 | 0.6049 |
| 6 | 64 | 1 | false | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7322 | 0.7641 | 0.6008 |
| 6 | 64 | 1 | true | 0 | 15 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7037 | 0.7583 | 0.5977 |
| 6 | 64 | 1 | true | 1 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7134 | 0.7850 | 0.6167 |
| 6 | 64 | 1 | true | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7201 | 0.7715 | 0.6040 |
| 6 | 64 | 2 | false | 0 | 13 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7293 | 0.7768 | 0.6053 |
| 6 | 64 | 2 | false | 1 | 6 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7249 | 0.7661 | 0.6003 |
| 6 | 64 | 2 | false | 2 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7345 | 0.7779 | 0.6074 |
| 6 | 64 | 2 | true | 0 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 244491 | 0.7241 | 0.7642 | 0.6014 |
| 6 | 64 | 2 | true | 1 | 11 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 244491 | 0.7113 | 0.7825 | 0.6142 |
| 6 | 64 | 2 | true | 2 | 16 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 244491 | 0.7130 | 0.7728 | 0.6096 |

Means n=3, sample std. Block 1 only.

| stride | d_model | e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| 6 | 64 | 1 | false | 122761 | 0.7251±0.0072 | 0.7665±0.0046 | 0.6020±0.0026 |
| 6 | 64 | 1 | true | 127561 | 0.7124±0.0082 | 0.7716±0.0134 | 0.6061±0.0097 |
| 6 | 64 | 2 | false | 239691 | 0.7296±0.0048 | 0.7736±0.0065 | 0.6043±0.0036 |
| 6 | 64 | 2 | true | 244491 | 0.7161±0.0070 | 0.7732±0.0092 | 0.6084±0.0065 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-dual-fft.log`

## Conclusions for Dev
1. Block 1 is complete (12/12). Block 2 (stride=6, d_model=128) has just started. 12/48 overall.
2. Inside Block 1, no config beats DLinear test 0.7535. Best test is e=1 without prototypes, 0.7665. Prototypes lower best_val at both depths and do not lower test. e=2 test is slightly worse than e=1 (0.7736 / 0.7732 vs 0.7665 / 0.7716).
