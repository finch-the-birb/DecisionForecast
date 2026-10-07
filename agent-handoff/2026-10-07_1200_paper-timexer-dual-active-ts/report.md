# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T13:04:00Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@1580c0bd5e2484eb8f3e6fa4b0325125fca4653b
- status: partial (Block 1/3, 6/24 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git pull --ff-only origin feat/phase2-timexer  (10ef7a6..1580c0b)
# uv sync
# HF_TOKEN unset. No prepare. Resume PID 15891 still alive. No new train.
```

## Outcome
- exit_code: in progress; 6/24 DONE; SWEEP_DONE absent; no traceback after RESUME SWEEP START
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 6 finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 15891
- log: outputs/paper-timexer-dual-active-ts.log
- resume_started_at: 2026-10-07T12:19:20Z
- current_job: Block 1 F2 [close, volume], n_features=2, e_layers=2, use_prototypes=false, seed=2 (started 2026-10-07T12:53:45Z)
- fix: job 4 reached proto kmeans++ init bank=(2560, 64) and finished
- splits: train 98116 / val 12048 / test 11904
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW.

### Finished jobs

| features | n_features | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7508 | 0.7641 | 0.6012 |
| close,volume | 2 | 1 | false | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7462 | 0.7692 | 0.6026 |
| close,volume | 2 | 1 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7775 | 0.7633 | 0.5998 |
| close,volume | 2 | 1 | true | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 126921 | 0.7705 | 0.7548 | 0.5967 |
| close,volume | 2 | 2 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 239051 | 0.7634 | 0.7818 | 0.6098 |
| close,volume | 2 | 2 | false | 1 | 8 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 239051 | 0.7612 | 0.7669 | 0.6020 |

Means n=3, sample std. Prototype rows are seed=1 only. F2 e=2 no-prototype is 2/3 seeds, so no mean yet.

| features | n_features | e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 122121 | 0.7582±0.0169 | 0.7655±0.0032 | 0.6012±0.0014 |

- oom: no
- traceback_summary: n/a after resume. The 11:17:16Z ValueError is the pre-fix crash.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-dual-active-ts.log

## Conclusions for Dev
1. Block 1 is 6/8 jobs done. 6/24 overall. Current job is F2, e=2, no prototypes, seed=2. One prototype seed remains in this block after it (e=2, seed=1).
2. The fix held. Job 4, F2 e=1 prototypes seed=1, finished: test 0.7548, mae 0.5967, shapes x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25). That single seed is 0.0001 above DLinear 0.7547 and 0.0013 above 0.7535. It is not a 3-seed mean.
3. F2 e=1 without prototypes stays at test 0.7655±0.0032. The two finished e=2 no-prototype seeds are 0.7818 and 0.7669, both worse than the e=1 no-prototype mean.
