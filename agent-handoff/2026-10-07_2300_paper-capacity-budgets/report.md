# Report: paper-capacity-budgets

- authored_by: runner
- created_at: 2026-10-07T22:12:16Z
- updated_at: 2026-10-07T23:15:04Z
- request_folder: agent-handoff/2026-10-07_2300_paper-capacity-budgets/
- tested_ref: feat/phase2-timexer@ddfa2bf89f4e632537cc92236851f57cc9326100
- status: partial (Block 2/3, 7/18 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Status harvest while PID 123275 is alive. No new train.
```

## Outcome
- exit_code: in progress; Block 1 DONE at 2026-10-07T22:59:19Z; 7/18 DONE; SWEEP_DONE absent; no Traceback
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 7 finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 123275
- log: outputs/paper-capacity-budgets.log
- started_at: 2026-10-07T22:11:58Z
- current_job: Block 2 c1_dual e_layers=2, d_model=56, n_heads=2, d_ff=200, seed=1 (started 2026-10-07T23:11:36Z)
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25)

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Means are n=3, sample std. Incomplete cells are omitted.

### Finished jobs

| model | e_layers | d_model | n_heads | d_ff | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | expected | best_val_mse | test_mse | test_mae |
| c1_dual | 1 | 56 | 2 | 200 | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7719 | 0.7779 | 0.6086 |
| c1_dual | 1 | 56 | 2 | 200 | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7262 | 0.7681 | 0.6004 |
| c1_dual | 1 | 56 | 2 | 200 | 2 | 8 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7526 | 0.7545 | 0.5935 |
| c1_dual | 1 | 48 | 3 | 96 | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 60777 | 60777 | 0.7621 | 0.7616 | 0.5976 |
| c1_dual | 1 | 48 | 3 | 96 | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 60777 | 60777 | 0.7349 | 0.7539 | 0.5932 |
| c1_dual | 1 | 48 | 3 | 96 | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 60777 | 60777 | 0.7431 | 0.7738 | 0.6065 |
| c1_dual | 2 | 56 | 2 | 200 | 0 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 178659 | 178659 | 0.7970 | 0.7830 | 0.6118 |

### Means

| model | e_layers | d_model | n_heads | d_ff | n_params | expected | best_val_mse | test_mse | test_mae |
| c1_dual | 1 | 56 | 2 | 200 | 91601 | 91601 | 0.7502±0.0229 | 0.7668±0.0118 | 0.6008±0.0076 |
| c1_dual | 1 | 48 | 3 | 96 | 60777 | 60777 | 0.7467±0.0140 | 0.7631±0.0100 | 0.5991±0.0068 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-capacity-budgets.log

## Conclusions for Dev
1. Block 1 is complete. 7/18 overall. Block 2 (dual e=2, d_model=56) has seed=0 done and seed=1 running. Measured n_params match the locked budgets: 91601, 60777, and 178659.
2. Dual e=1 at 75% (91601 params): test 0.7668±0.0118. At 50% (60777 params): test 0.7631±0.0100. The 50% mean is slightly under the full dual F2 e=1 mean 0.7655±0.0032. Both stay above DLinear F1 0.7535±0.0009. Closest single seeds are 50% seed=1 at 0.7539 and 75% seed=2 at 0.7545.
3. The first dual e=2 75% seed is test 0.7830, above the full dual F2 e=2 mean 0.7738±0.0075. That cell's mean waits on seeds 1 and 2.
