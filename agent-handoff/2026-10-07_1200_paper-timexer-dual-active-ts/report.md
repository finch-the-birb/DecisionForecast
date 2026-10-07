# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T16:21:24Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@1580c0bd5e2484eb8f3e6fa4b0325125fca4653b
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Final harvest after SWEEP_DONE. No new train.
# First sweep PID 4271 died on job 4 before 1580c0b. Resume PID 15891 finished jobs 4..24.
```

## Outcome
- exit_code: 0; SWEEP_DONE at 2026-10-07T16:16:26Z; 24/24 DONE; no traceback after RESUME SWEEP START
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 24 jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 15891 (exited after SWEEP_DONE)
- log: outputs/paper-timexer-dual-active-ts.log
- first_sweep: 2026-10-07T10:51:23Z, died 11:17:16Z on the pre-fix prototype bank
- resume: 2026-10-07T12:19:20Z to 2026-10-07T16:16:26Z
- block_done: Block 1 13:22:05Z; Block 2 14:46:41Z; Block 3 16:16:26Z
- splits: train 98116 / val 12048 / test 11904
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. No-prototype means are n=3, sample std. Prototype rows are seed=1 only.

### Table 1. Per seed

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
| close,volume,open,high,low | 5 | 2 | false | 2 | 8 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 241355 | 0.7362 | 0.7682 | 0.6031 |
| close,volume,open,high,low | 5 | 2 | true | 1 | 16 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 246155 | 0.7607 | 0.7691 | 0.6049 |

### Table 2. Means

| features | n_features | e_layers | use_proto | n | n_params | best_val_mse | test_mse | test_mae |
| close | 1 | 1 | false | 3 | 121353 | 0.7422±0.0077 | 0.7606±0.0065 | 0.5987±0.0036 |
| close | 1 | 1 | true | 1 | 126153 | 0.7196 | 0.7565 | 0.5947 |
| close | 1 | 2 | false | 3 | 238283 | 0.7494±0.0310 | 0.7651±0.0124 | 0.5998±0.0048 |
| close | 1 | 2 | true | 1 | 243083 | 0.7037 | 0.7725 | 0.6052 |
| close,volume | 2 | 1 | false | 3 | 122121 | 0.7582±0.0169 | 0.7655±0.0032 | 0.6012±0.0014 |
| close,volume | 2 | 1 | true | 1 | 126921 | 0.7705 | 0.7548 | 0.5967 |
| close,volume | 2 | 2 | false | 3 | 239051 | 0.7618±0.0014 | 0.7738±0.0075 | 0.6059±0.0039 |
| close,volume | 2 | 2 | true | 1 | 243851 | 0.7151 | 0.7701 | 0.6024 |
| close,volume,open,high,low | 5 | 1 | false | 3 | 124425 | 0.7636±0.0228 | 0.7690±0.0144 | 0.6042±0.0084 |
| close,volume,open,high,low | 5 | 1 | true | 1 | 129225 | 0.7384 | 0.7895 | 0.6155 |
| close,volume,open,high,low | 5 | 2 | false | 3 | 241355 | 0.7610±0.0251 | 0.7692±0.0104 | 0.6037±0.0063 |
| close,volume,open,high,low | 5 | 2 | true | 1 | 246155 | 0.7607 | 0.7691 | 0.6049 |

- oom: no
- traceback_summary: n/a after resume. The 11:17:16Z ValueError is the pre-fix crash and is not part of this result.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-dual-active-ts.log

## Conclusions for Dev
1. 24/24 finished on CUDA after the bank fix. Shapes are x=(32, 60, F), text_seq=(32, 60, 15), ts=(32, 60, 25), with F=1, 2, or 5. No OOM and no traceback on the resume.
2. No complete cell beats DLinear test 0.7535 or 0.7547. Best no-prototype mean is F1 e=1: test 0.7606±0.0065, mae 0.5987±0.0036. Gap to 0.7535 is +0.0071.
3. No-prototype e=1 test: F1 0.7606±0.0065, F2 0.7655±0.0032, F5 0.7690±0.0144. MAE follows the same order: 0.5987, 0.6012, 0.6042. Adding volume, then the rest of OHLCV, does not help test.
4. e=2 does not beat e=1. F1 e=2 is 0.7651±0.0124, F2 e=2 is 0.7738±0.0075, F5 e=2 is 0.7692±0.0104. The F5 e=1 and e=2 means differ by 0.0002.
5. Prototype seed=1 lowers best_val in the F1 and F2 cells. The only prototype test under its no-prototype mean by a clear margin is F2 e=1 seed=1 at 0.7548 versus 0.7655. F1 e=1 prototype seed=1 is 0.7565. F5 e=1 prototype seed=1 is 0.7895. Best single seed in the grid is F1 e=2 no prototypes seed=1 at 0.7545; seeds 0 and 2 of that cell are 0.7621 and 0.7787.
