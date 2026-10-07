# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T14:20:37Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@1580c0bd5e2484eb8f3e6fa4b0325125fca4653b
- status: partial (Block 2/3, 14/24 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Timer harvest while PID 15891 is alive. No new train.
```

## Outcome
- exit_code: in progress; Block 1 FINISHED at 2026-10-07T13:22:05Z; 14/24 DONE; SWEEP_DONE absent; no traceback after RESUME SWEEP START
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 14 finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 15891
- log: outputs/paper-timexer-dual-active-ts.log
- resume_started_at: 2026-10-07T12:19:20Z
- current_job: Block 2 F1 [close], n_features=1, e_layers=2, use_prototypes=false, seed=2 (started 2026-10-07T14:19:04Z)
- splits: train 98116 / val 12048 / test 11904
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Prototype rows are seed=1 only.

### Finished jobs

| features | n_features | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7508 | 0.7641 | 0.6012 |
| close,volume | 2 | 1 | false | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7462 | 0.7692 | 0.6026 |
| close,volume | 2 | 1 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7775 | 0.7633 | 0.5998 |
| close,volume | 2 | 1 | true | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 126921 | 0.7705 | 0.7548 | 0.5967 |
| close,volume | 2 | 2 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 239051 | 0.7634 | 0.7818 | 0.6098 |
| close,volume | 2 | 2 | false | 1 | 8 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 239051 | 0.7612 | 0.7669 | 0.6020 |
| close,volume | 2 | 2 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 239051 | 0.7608 | 0.7726 | 0.6059 |
| close,volume | 2 | 2 | true | 1 | 11 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 243851 | 0.7151 | 0.7701 | 0.6024 |
| close | 1 | 1 | false | 0 | 10 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 121353 | 0.7478 | 0.7678 | 0.6028 |
| close | 1 | 1 | false | 1 | 8 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 121353 | 0.7455 | 0.7552 | 0.5966 |
| close | 1 | 1 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 121353 | 0.7334 | 0.7587 | 0.5967 |
| close | 1 | 1 | true | 1 | 9 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 126153 | 0.7196 | 0.7565 | 0.5947 |
| close | 1 | 2 | false | 0 | 8 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 238283 | 0.7761 | 0.7621 | 0.5991 |
| close | 1 | 2 | false | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 238283 | 0.7154 | 0.7545 | 0.5954 |

Means n=3, sample std. Incomplete cells omitted (F1 e=2 no prototypes: 2/3 seeds). Prototype lines are one seed.

| features | n_features | e_layers | use_proto | n | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 3 | 122121 | 0.7582±0.0169 | 0.7655±0.0032 | 0.6012±0.0014 |
| close,volume | 2 | 1 | true | 1 | 126921 | 0.7705 | 0.7548 | 0.5967 |
| close,volume | 2 | 2 | false | 3 | 239051 | 0.7618±0.0014 | 0.7738±0.0075 | 0.6059±0.0039 |
| close,volume | 2 | 2 | true | 1 | 243851 | 0.7151 | 0.7701 | 0.6024 |
| close | 1 | 1 | false | 3 | 121353 | 0.7422±0.0077 | 0.7606±0.0065 | 0.5987±0.0036 |
| close | 1 | 1 | true | 1 | 126153 | 0.7196 | 0.7565 | 0.5947 |

- oom: no
- traceback_summary: n/a after resume

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-dual-active-ts.log

## Conclusions for Dev
1. Block 1 is complete. 14/24 overall. Block 2 (F1) has 6/8 jobs done. Current job is F1, e=2, no prototypes, seed=2. F5 has not started.
2. Best complete mean so far is F1 e=1 without prototypes: test 0.7606±0.0065, mae 0.5987±0.0036. That beats F2 e=1 no prototypes (0.7655±0.0032) and still sits above DLinear 0.7535 and 0.7547.
3. F2 e=2 no prototypes is worse than F2 e=1: test 0.7738±0.0075. The F2 e=2 prototype seed (0.7701) lowers best_val to 0.7151 and does not beat the e=1 no-prototype mean.
4. Closest single seeds to DLinear: F1 e=2 no prototypes seed=1 test 0.7545 (0.0002 under 0.7547, 0.0010 over 0.7535; the pair is incomplete, seed=0 is 0.7621), F2 e=1 prototypes seed=1 test 0.7548, F1 e=1 no prototypes seed=1 test 0.7552. No complete cell beats 0.7535.
