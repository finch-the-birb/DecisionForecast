# Report: paper-timexer-5ts-compact

- authored_by: runner
- created_at: 2026-10-06T01:20:11Z
- request_folder: agent-handoff/2026-10-06_1500_paper-timexer-5ts-compact/
- tested_ref: feat/phase2-timexer@a38f2f33f5006990d3633d45a23a2504a20a622b (contains a9006d2; harvest from 361753f started report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. No FinBERT/FinLang re-encode. No dlinear / plain / a / b / c0 / c1 / timexer_selected / ticker_set=dev.
# Human ping: PID 142068 dead, log has SWEEP_DONE 12/12. Harvest only — no new train.
```

## Outcome
- exit_code: 12/12 DONE; log ends with `=== SWEEP_DONE 2026-10-06T01:07:38Z ===`; nohup PID 142068 gone; no Traceback
- duration: 2026-10-05T23:15:59Z → 2026-10-06T01:07:38Z (~1h 52m)
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 12**. NVIDIA L4. No OOM.
- Every job: `x=(32, 60, 5)`, `text_seq=(32, 60, 15)`. Splits **98236 / 12048 / 11904**.
- Skipped price files: **UNH**, **VZ**. No missing compact-text parquet in the log.

`best_val_mse` = min Epoch `val_mse`. `stop_ep` = `Early stopping at epoch N`, else the last logged epoch. `test_mse` / `test_mae` = `METRICS_ROW` (best.pt). Means n=3, sample std.

n_params: e=1 no-proto 88967; e=1 proto=10 93767; e=2 no-proto 172487; e=2 proto=10 177287.

### Table 1 — per-seed

| e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
| 1 | false | 0 | 13 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 88967 | 0.7033 | 0.7583 | 0.5957 |
| 1 | false | 1 | 11 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 88967 | 0.7074 | 0.7654 | 0.6001 |
| 1 | false | 2 | 20 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 88967 | 0.7021 | 0.7679 | 0.6014 |
| 2 | false | 0 | 7 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 172487 | 0.7056 | 0.7679 | 0.6046 |
| 2 | false | 1 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 172487 | 0.7044 | 0.7734 | 0.6043 |
| 2 | false | 2 | 15 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 172487 | 0.7075 | 0.7620 | 0.5962 |
| 1 | true | 0 | 6 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 93767 | 0.7015 | 0.7618 | 0.5983 |
| 1 | true | 1 | 7 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 93767 | 0.6885 | 0.7815 | 0.6134 |
| 1 | true | 2 | 11 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 93767 | 0.6961 | 0.7581 | 0.5963 |
| 2 | true | 0 | 7 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 177287 | 0.7009 | 0.7611 | 0.5974 |
| 2 | true | 1 | 7 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 177287 | 0.6983 | 0.7728 | 0.6055 |
| 2 | true | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 177287 | 0.7219 | 0.7696 | 0.6033 |

### Table 2 — means n=3

| e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| 1 | false | 88967 | 0.7043±0.0028 | 0.7639±0.0050 | 0.5991±0.0030 |
| 1 | true | 93767 | 0.6954±0.0065 | 0.7671±0.0126 | 0.6027±0.0093 |
| 2 | false | 172487 | 0.7058±0.0016 | 0.7678±0.0057 | 0.6017±0.0048 |
| 2 | true | 177287 | 0.7070±0.0129 | 0.7678±0.0060 | 0.6021±0.0042 |

### Table 3 — vs prior paper test_mse

| label | test_mse | source |
| DLinear F1 | 0.7535 | paper-baselines-c1 |
| DLinear F5 Huber | 0.7547 | paper-selected-40d-run |
| C1.1 768D e=1 Huber F5 | 0.7543 | paper-baselines-c1 |
| TimeXer 26TS selected e=1 | 0.7902 | paper-timexer-selected-26ts |
| **c1_compact e=1 proto=false** | **0.7639±0.0050** | this |
| c1_compact e=1 proto=true | 0.7671±0.0126 | this |
| c1_compact e=2 proto=false | 0.7678±0.0057 | this |
| c1_compact e=2 proto=true | 0.7678±0.0060 | this |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-5ts-compact.log`

## Conclusions for Dev
1. **No config beat test 0.7535.** Best mean test is e=1 without prototypes, **0.7639** (+0.0104 vs DLinear F1, +0.0092 vs F5 0.7547, +0.0096 vs C1.1 768D 0.7543). Best single seed is e=1 proto seed=2 at **0.7581**, still above 0.7535.
2. **TimeXL prototypes did not help test.** At e=1 they lower best_val (0.6954 vs 0.7043) but raise test (0.7671 vs 0.7639): seed 1 has best_val 0.6885 and test 0.7815. At e=2, proto and no-proto share test 0.7678, and proto best_val is worse.
3. **e=2 is not better than e=1.** No-proto test 0.7678 vs 0.7639. Proto test 0.7678 vs 0.7671. The extra layer roughly doubles n_params and does not lower test.
4. Versus 26TS selected e=1 (0.7902), compact 5+15D is better by about 0.026. Every batch has `x` last dim 5 and `text_seq` last dim 15. CUDA, `num_workers=4`. UNH/VZ skipped. Prepare was not run. Did not commit `outputs/`, `mlflow.db`, or `Data/`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
