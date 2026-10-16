# Report: paper-c1-hierarchical

- authored_by: runner
- created_at: 2026-10-07T18:10:28Z
- updated_at: 2026-10-07T19:39:58Z
- request_folder: agent-handoff/2026-10-07_2100_paper-c1-hierarchical/
- tested_ref: feat/phase2-timexer@6dbbf0fb8a85b1b64c048aa475f949607ddbd45a
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Final harvest after SWEEP_DONE. No new train.
# paper-timexer-dual-active-ts was not restarted.
```

## Outcome
- exit_code: 0; SWEEP_DONE at 2026-10-07T19:29:22Z; 6/6 DONE; no Traceback
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 6 jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 96529 (exited after SWEEP_DONE)
- log: outputs/paper-c1-hierarchical.log
- wall_clock: 2026-10-07T18:10:12Z to 2026-10-07T19:29:22Z
- block_done: Block 1 18:41:55Z; Block 2 19:29:22Z
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25)
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Means are n=3, sample std.

### Table 1. Per seed

| features | n_features | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 2 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 172169 | 0.7442 | 0.7675 | 0.6016 |
| close,volume | 2 | 2 | false | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 172169 | 0.7686 | 0.7634 | 0.6003 |
| close,volume | 2 | 2 | false | 2 | 11 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 172169 | 0.7794 | 0.7657 | 0.6004 |
| close,volume | 2 | 2 | true | 0 | 16 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 176969 | 0.7238 | 0.7714 | 0.6050 |
| close,volume | 2 | 2 | true | 1 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 176969 | 0.7341 | 0.7995 | 0.6240 |
| close,volume | 2 | 2 | true | 2 | 12 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 176969 | 0.7196 | 0.8008 | 0.6266 |

### Table 2. Mean ± std

| use_proto | n | n_params | best_val_mse | test_mse | test_mae |
| false | 3 | 172169 | 0.7641±0.0180 | 0.7655±0.0021 | 0.6008±0.0007 |
| true | 3 | 176969 | 0.7258±0.0075 | 0.7906±0.0166 | 0.6185±0.0118 |

- oom: no
- traceback_summary: n/a
- n_params budget: no-prototype jobs are 172169, under 185000. c1_dual F2 e=2 was 239051.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-c1-hierarchical.log

## Conclusions for Dev
1. 6/6 finished on CUDA. Shapes are x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25). No OOM and no traceback. No-prototype n_params=172169 stays under the 185000 budget and under c1_dual F2 e=2 at 239051.
2. Hierarchical does not beat DLinear. No-prototype test is 0.7655±0.0021, mae 0.6008±0.0007. DLinear F1 is 0.7535±0.0009 and DLinear F5 is 0.7547±0.0028. Gap to F1 is +0.0120.
3. The no-prototype mean matches c1_dual F2 e=1 no prototypes, 0.7655±0.0032. Best hierarchical seed is 0.7634, next to that dual cell's best seed 0.7633. It does beat c1_dual F2 e=2 no prototypes, 0.7738±0.0075, by 0.0083.
4. Prototypes lower best_val (0.7258±0.0075 versus 0.7641±0.0180) and raise test to 0.7906±0.0166. Seeds 1 and 2 are 0.7995 and 0.8008. Prototypes do not help test.
