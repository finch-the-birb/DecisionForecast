# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T15:52:02Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@1580c0bd5e2484eb8f3e6fa4b0325125fca4653b
- status: partial (Block 3/3, 22/24 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Status harvest while PID 15891 is alive. No new train.
```

## Outcome
- exit_code: in progress; Block 1 FINISHED at 2026-10-07T13:22:05Z; Block 2 DONE at 2026-10-07T14:46:41Z; 22/24 DONE; SWEEP_DONE absent; no traceback after RESUME SWEEP START
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 22 finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 15891
- log: outputs/paper-timexer-dual-active-ts.log
- resume_started_at: 2026-10-07T12:19:20Z
- current_job: Block 3 F5 [close, volume, open, high, low], n_features=5, e_layers=2, use_prototypes=false, seed=2 (started 2026-10-07T15:44:34Z)
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
| close | 1 | 2 | false | 2 | 10 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 238283 | 0.7568 | 0.7787 | 0.6049 |
| close | 1 | 2 | true | 1 | 8 | 98116 | 12048 | 11904 | (32, 60, 1) | (32, 60, 15) | (32, 60, 25) | 243083 | 0.7037 | 0.7725 | 0.6052 |
| close,volume,open,high,low | 5 | 1 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 124425 | 0.7567 | 0.7842 | 0.6116 |
| close,volume,open,high,low | 5 | 1 | false | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 124425 | 0.7890 | 0.7555 | 0.5951 |
| close,volume,open,high,low | 5 | 1 | false | 2 | 10 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 124425 | 0.7450 | 0.7673 | 0.6060 |
| close,volume,open,high,low | 5 | 1 | true | 1 | 8 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 129225 | 0.7384 | 0.7895 | 0.6155 |
| close,volume,open,high,low | 5 | 2 | false | 0 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 241355 | 0.7606 | 0.7593 | 0.5978 |
| close,volume,open,high,low | 5 | 2 | false | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 241355 | 0.7863 | 0.7801 | 0.6103 |

Means n=3, sample std. Incomplete cells omitted (F5 e=2 no prototypes: 2/3 seeds). Prototype lines are one seed.

| features | n_features | e_layers | use_proto | n | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 3 | 122121 | 0.7582±0.0169 | 0.7655±0.0032 | 0.6012±0.0014 |
| close,volume | 2 | 1 | true | 1 | 126921 | 0.7705 | 0.7548 | 0.5967 |
| close,volume | 2 | 2 | false | 3 | 239051 | 0.7618±0.0014 | 0.7738±0.0075 | 0.6059±0.0039 |
| close,volume | 2 | 2 | true | 1 | 243851 | 0.7151 | 0.7701 | 0.6024 |
| close | 1 | 1 | false | 3 | 121353 | 0.7422±0.0077 | 0.7606±0.0065 | 0.5987±0.0036 |
| close | 1 | 1 | true | 1 | 126153 | 0.7196 | 0.7565 | 0.5947 |
| close | 1 | 2 | false | 3 | 238283 | 0.7494±0.0310 | 0.7651±0.0124 | 0.5998±0.0048 |
| close | 1 | 2 | true | 1 | 243083 | 0.7037 | 0.7725 | 0.6052 |
| close,volume,open,high,low | 5 | 1 | false | 3 | 124425 | 0.7636±0.0228 | 0.7690±0.0144 | 0.6042±0.0084 |
| close,volume,open,high,low | 5 | 1 | true | 1 | 129225 | 0.7384 | 0.7895 | 0.6155 |

- oom: no
- traceback_summary: n/a after resume

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-dual-active-ts.log

## Conclusions for Dev
1. Blocks 1 and 2 are complete. 22/24 overall. Block 3 (F5) has 6/8 jobs done. Current job is F5, e=2, no prototypes, seed=2. One prototype seed remains after it.
2. Complete no-prototype test means, e=1: F1 0.7606±0.0065, F2 0.7655±0.0032, F5 0.7690±0.0144. F1 is the best of the three and still above DLinear 0.7535 and 0.7547. F5 is the worst of the e=1 cells.
3. e=2 does not beat e=1. F1 e=2 no prototypes is 0.7651±0.0124: seed=1 was 0.7545, seed=2 is 0.7787. F2 e=2 is 0.7738±0.0075. F5 e=2 is 2/3 seeds (0.7593 and 0.7801); the mean waits on seed=2.
4. Prototype seed=1 lowers best_val and does not lower test versus the matching no-prototype mean, except F2 e=1 where the single prototype seed is 0.7548 against a no-prototype mean of 0.7655. No complete cell beats DLinear. Best single seed remains F1 e=2 no prototypes seed=1 at 0.7545.
