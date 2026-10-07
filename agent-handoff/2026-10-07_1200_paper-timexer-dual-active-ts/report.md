# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T12:19:42Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@1580c0bd5e2484eb8f3e6fa4b0325125fca4653b
- status: partial (resumed at job 4/24)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git pull --ff-only origin feat/phase2-timexer  (10ef7a6..1580c0b)
# uv sync
# HF_TOKEN unset. No prepare. Resume appended to the same log. 21 jobs, jobs 4..24.
```

## Outcome
- exit_code: in progress; previous PID 4271 died on job 4; resume PID 15891 is alive; SWEEP_DONE absent
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: first 3 jobs train device cuda, DataLoader num_workers=4. Resume job 4 has not finished yet.
- pid: 15891
- log: outputs/paper-timexer-dual-active-ts.log
- first_sweep: 2026-10-07T10:51:23Z, died 11:17:16Z on prototypes
- resume_started_at: 2026-10-07T12:19:20Z
- current_job: Block 1 F2 [close, volume], n_features=2, e_layers=1, use_prototypes=true, seed=1
- fix: 1580c0b passes batch ts from collect_segment_bank into TimeXerDual
- splits: train 98116 / val 12048 / test 11904
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW.

### Finished jobs

| features | n_features | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7508 | 0.7641 | 0.6012 |
| close,volume | 2 | 1 | false | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7462 | 0.7692 | 0.6026 |
| close,volume | 2 | 1 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 122121 | 0.7775 | 0.7633 | 0.5998 |

Mean n=3, sample std. Only the F2 e=1 no-prototype cell is complete. Resume rows are not in yet.

| features | n_features | e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 1 | false | 122121 | 0.7582±0.0169 | 0.7655±0.0032 | 0.6012±0.0014 |

- oom: no
- traceback_summary: the 11:17:16Z ValueError is the pre-fix crash. Resume started after 1580c0b. No new traceback at launch.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-dual-active-ts.log

## Conclusions for Dev
1. Resume is running at job 4/24: F2, e=1, prototypes, seed=1. The first three F2 e=1 no-prototype jobs stay in the table. Their test mean is 0.7655±0.0032, above DLinear 0.7535 and 0.7547.
2. Remaining grid after job 4: F2 e=2 no-prototype seeds 0,1,2 and F2 e=2 prototype seed=1, then F1 (8 jobs), then F5 (8 jobs).
