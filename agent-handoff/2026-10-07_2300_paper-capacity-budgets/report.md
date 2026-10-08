# Report: paper-capacity-budgets

- authored_by: runner
- created_at: 2026-10-07T22:12:16Z
- updated_at: 2026-10-08T02:13:51Z
- request_folder: agent-handoff/2026-10-07_2300_paper-capacity-budgets/
- tested_ref: feat/phase2-timexer@ddfa2bf89f4e632537cc92236851f57cc9326100
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Final harvest after SWEEP_DONE. No new train.
# Full-width models, prototypes, F1, F5, and d_model=64 were not rerun.
```

## Outcome
- exit_code: 0; SWEEP_DONE at 2026-10-08T01:12:39Z; 18/18 DONE; no Traceback
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 18 jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 123275 (exited after SWEEP_DONE)
- log: outputs/paper-capacity-budgets.log
- wall_clock: 2026-10-07T22:11:58Z to 2026-10-08T01:12:39Z
- block_done: Block 1 22:59:19Z; Block 2 00:15:34Z; Block 3 01:12:39Z
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25)
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Means are n=3, sample std. use_prototypes=false on every job. Features are [close, volume].

### Table 1. Per seed

| model | e_layers | d_model | n_heads | d_ff | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | expected | best_val_mse | test_mse | test_mae |
| c1_dual | 1 | 56 | 2 | 200 | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7719 | 0.7779 | 0.6086 |
| c1_dual | 1 | 56 | 2 | 200 | 1 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7262 | 0.7681 | 0.6004 |
| c1_dual | 1 | 56 | 2 | 200 | 2 | 8 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 91601 | 91601 | 0.7526 | 0.7545 | 0.5935 |
| c1_dual | 1 | 48 | 3 | 96 | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 60777 | 60777 | 0.7621 | 0.7616 | 0.5976 |
| c1_dual | 1 | 48 | 3 | 96 | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 60777 | 60777 | 0.7349 | 0.7539 | 0.5932 |
| c1_dual | 1 | 48 | 3 | 96 | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 60777 | 60777 | 0.7431 | 0.7738 | 0.6065 |
| c1_dual | 2 | 56 | 2 | 200 | 0 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 178659 | 178659 | 0.7970 | 0.7830 | 0.6118 |
| c1_dual | 2 | 56 | 2 | 200 | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 178659 | 178659 | 0.7414 | 0.7590 | 0.5974 |
| c1_dual | 2 | 56 | 2 | 200 | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 178659 | 178659 | 0.7483 | 0.7721 | 0.6052 |
| c1_dual | 2 | 48 | 3 | 104 | 0 | 11 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 119211 | 119211 | 0.7892 | 0.7767 | 0.6068 |
| c1_dual | 2 | 48 | 3 | 104 | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 119211 | 119211 | 0.7793 | 0.7593 | 0.5969 |
| c1_dual | 2 | 48 | 3 | 104 | 2 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 119211 | 119211 | 0.7829 | 0.7577 | 0.5974 |
| c1_hierarchical | 2 | 56 | 2 | 208 | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 129113 | 129113 | 0.7592 | 0.7729 | 0.6066 |
| c1_hierarchical | 2 | 56 | 2 | 208 | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 129113 | 129113 | 0.7470 | 0.7553 | 0.5946 |
| c1_hierarchical | 2 | 56 | 2 | 208 | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 129113 | 129113 | 0.7596 | 0.7739 | 0.6059 |
| c1_hierarchical | 2 | 48 | 3 | 128 | 0 | 11 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 85993 | 85993 | 0.7397 | 0.7786 | 0.6125 |
| c1_hierarchical | 2 | 48 | 3 | 128 | 1 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 85993 | 85993 | 0.7522 | 0.7576 | 0.5975 |
| c1_hierarchical | 2 | 48 | 3 | 128 | 2 | 7 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 85993 | 85993 | 0.7365 | 0.7636 | 0.6009 |

### Table 2. Mean ± std (n=3)

| model | e_layers | d_model | n_heads | d_ff | n_params | expected | best_val_mse | test_mse | test_mae |
| c1_dual | 1 | 56 | 2 | 200 | 91601 | 91601 | 0.7502±0.0229 | 0.7668±0.0118 | 0.6008±0.0076 |
| c1_dual | 1 | 48 | 3 | 96 | 60777 | 60777 | 0.7467±0.0140 | 0.7631±0.0100 | 0.5991±0.0068 |
| c1_dual | 2 | 56 | 2 | 200 | 178659 | 178659 | 0.7622±0.0303 | 0.7714±0.0120 | 0.6048±0.0072 |
| c1_dual | 2 | 48 | 3 | 104 | 119211 | 119211 | 0.7838±0.0050 | 0.7646±0.0105 | 0.6004±0.0056 |
| c1_hierarchical | 2 | 56 | 2 | 208 | 129113 | 129113 | 0.7553±0.0072 | 0.7674±0.0105 | 0.6024±0.0067 |
| c1_hierarchical | 2 | 48 | 3 | 128 | 85993 | 85993 | 0.7428±0.0083 | 0.7666±0.0108 | 0.6036±0.0079 |

Full-width references, not rerun: hierarchical 172169 test 0.7655±0.0021; dual F2 e=1 122121 test 0.7655±0.0032; dual F2 e=2 239051 test 0.7738±0.0075; DLinear F1 0.7535±0.0009.

- oom: no
- traceback_summary: n/a
- n_params: every finished cell matches its locked budget (91601, 60777, 178659, 119211, 129113, 85993).

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-capacity-budgets.log

## Conclusions for Dev
1. 18/18 finished on CUDA. Shapes are x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25). No OOM and no traceback. All six width cuts landed on the locked n_params.
2. No complete cell beats DLinear F1 0.7535±0.0009. The best mean is dual e=1 at 50% (60777 params), test 0.7631±0.0100, gap +0.0096. Closest single seeds are that cell's seed=1 at 0.7539 and dual e=1 75% seed=2 at 0.7545.
3. Against the full-width means, the cuts stay inside a band of about 0.008. Dual e=1 75% is 0.7668±0.0118 versus full dual e=1 0.7655±0.0032. Dual e=1 50% is 0.0024 under that full mean. Dual e=2 75% is 0.7714±0.0120 and dual e=2 50% is 0.7646±0.0105, both under full dual e=2 0.7738±0.0075. Hierarchical 75% is 0.7674±0.0105 and hierarchical 50% is 0.7666±0.0108, both slightly above full hierarchical 0.7655±0.0021. Seed std on the cuts is larger than on the full-width runs, so these gaps are inside the noise.
4. Cutting width did not recover the DLinear gap. Extra depth still does not help: both e=2 dual cuts remain at or above the e=1 50% mean.
