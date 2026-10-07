# Report: paper-capacity-budgets

- authored_by: runner
- created_at: 2026-10-07T22:12:16Z
- updated_at: 2026-10-07T22:33:51Z
- request_folder: agent-handoff/2026-10-07_2300_paper-capacity-budgets/
- tested_ref: feat/phase2-timexer@ddfa2bf89f4e632537cc92236851f57cc9326100
- status: partial (Block 1/3, 2/18 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Status harvest while PID 123275 is alive. No new train.
```

## Outcome
- exit_code: in progress; 2/18 DONE; SWEEP_DONE absent; no Traceback; Hydra accepted d_ff
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: both finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 123275
- log: outputs/paper-capacity-budgets.log
- started_at: 2026-10-07T22:11:58Z
- current_job: Block 1 c1_dual e_layers=1, d_model=56, n_heads=2, d_ff=200, seed=2 (started 2026-10-07T22:26:36Z)
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25)

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Means wait until a cell has 3 seeds.

### Finished jobs

| model | e_layers | d_model | n_heads | d_ff | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | expected | best_val_mse | test_mse | test_mae |
| c1_dual | 1 | 56 | 2 | 200 | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7719 | 0.7779 | 0.6086 |
| c1_dual | 1 | 56 | 2 | 200 | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7262 | 0.7681 | 0.6004 |

- oom: no
- traceback_summary: n/a
- n_params: first cell matches 91601. The 50% dual e=1 cell has not started.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-capacity-budgets.log

## Conclusions for Dev
1. Block 1 is 2/6 jobs done. 2/18 overall. Current job is the third seed of dual e=1 at d_model=56. n_params=91601 matches the 75% budget.
2. The two finished tests are 0.7779 and 0.7681. Both sit above the full dual F2 e=1 mean 0.7655±0.0032 and above DLinear F1 0.7535±0.0009. The 3-seed mean waits on seed=2.
