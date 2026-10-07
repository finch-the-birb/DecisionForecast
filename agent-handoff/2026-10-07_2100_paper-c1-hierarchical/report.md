# Report: paper-c1-hierarchical

- authored_by: runner
- created_at: 2026-10-07T18:10:28Z
- updated_at: 2026-10-07T19:16:25Z
- request_folder: agent-handoff/2026-10-07_2100_paper-c1-hierarchical/
- tested_ref: feat/phase2-timexer@6dbbf0fb8a85b1b64c048aa475f949607ddbd45a
- status: partial (Block 2/2, 5/6 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Status harvest while PID 96529 is alive. No new train.
# paper-timexer-dual-active-ts was not restarted.
```

## Outcome
- exit_code: in progress; Block 1 DONE at 2026-10-07T18:41:55Z; 5/6 DONE; SWEEP_DONE absent; no Traceback
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: all 5 finished jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 96529
- log: outputs/paper-c1-hierarchical.log
- started_at: 2026-10-07T18:10:12Z
- current_job: Block 2 F2 [close, volume], e_layers=2, use_prototypes=true, seed=2 (started 2026-10-07T19:13:35Z)
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 2), text_seq=(32, 60, 15), ts=(32, 60, 25)
- skipped price files: UNH, VZ

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW.

### Finished jobs

| features | n_features | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| close,volume | 2 | 2 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 172169 | 0.7442 | 0.7675 | 0.6016 |
| close,volume | 2 | 2 | false | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 172169 | 0.7686 | 0.7634 | 0.6003 |
| close,volume | 2 | 2 | false | 2 | 11 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 172169 | 0.7794 | 0.7657 | 0.6004 |
| close,volume | 2 | 2 | true | 0 | 16 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 176969 | 0.7238 | 0.7714 | 0.6050 |
| close,volume | 2 | 2 | true | 1 | 9 | 98116 | 12048 | 11904 | (32, 60, 2) | (32, 60, 15) | (32, 60, 25) | 176969 | 0.7341 | 0.7995 | 0.6240 |

Means n=3, sample std. Prototypes are 2/3 seeds, so no mean yet.

| use_proto | n | n_params | best_val_mse | test_mse | test_mae |
| false | 3 | 172169 | 0.7641±0.0180 | 0.7655±0.0021 | 0.6008±0.0007 |

- oom: no
- traceback_summary: n/a
- n_params budget: no-prototype jobs are 172169, under 185000. c1_dual F2 e=2 was 239051.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-c1-hierarchical.log

## Conclusions for Dev
1. Block 1 is complete. 5/6 overall. Current job is the last prototype seed. No-prototype n_params=172169, inside the 185000 budget and below c1_dual e=2 at 239051.
2. No-prototype test is 0.7655±0.0021, mae 0.6008±0.0007. That matches c1_dual F2 e=1 no prototypes (0.7655±0.0032, best seed 0.7633). The hierarchical best seed is 0.7634. It beats c1_dual F2 e=2 no prototypes (0.7738±0.0075) and does not beat DLinear F1 0.7535±0.0009 or DLinear F5 0.7547±0.0028.
3. The two finished prototype seeds are 0.7714 and 0.7995. Both are worse than the no-prototype mean. The mean waits on seed=2.
