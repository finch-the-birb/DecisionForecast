# Report: paper-timexer-selected-26ts

- authored_by: runner
- created_at: 2026-10-05T22:23:46Z
- request_folder: agent-handoff/2026-10-06_1200_paper-timexer-selected-26ts/
- tested_ref: feat/phase2-timexer@8e35271fa7e223e311cff3c78a7e28339535824d (contains 9bf6a81; harvest from a60f073 started report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. No dlinear / plain / a / b / c0 / c1 / ticker_set=dev.
# Human ping: PID 95285 dead, log has SWEEP_DONE 6/6. Harvest only — no new train.
# pytest before the grid: 111 passed (100.21s).
```

## Outcome
- exit_code: prepare exit 0 (`PREPARE_DONE`); 6/6 trains DONE; log ends with `=== SWEEP_DONE 2026-10-05T22:16:38Z ===`; nohup PID 95285 gone; no Traceback
- duration: prepare 21:19:14Z → 21:31:52Z; trains 21:31:52Z → 22:16:38Z
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 6**. NVIDIA L4. No OOM.
- Every job: `x=(32, 60, 26)`, `text_seq=(32, 60, 15)`.

Prepare (verbatim):
```
INFO Text state /workspace/DecisionForecast/Data/FNSPID/cache/selected_signatures/paper_fold1_text_state.npz train_articles=150683
INFO Signature /workspace/DecisionForecast/Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json rows=75264 label_dates=2015-01-12..2019-12-10 columns=close, month_cos_delta, month_cos, month_sin_lag3, gk_n40_lag3, gk_n5, bb_width_n60_delta, log_ret_k5, parkinson_n10_delta, gk_n60_delta, month_cos_lag2, bb_pctb_n5, gk_n20_delta, day_sin_delta, natr_n10_lag3, log_ret_k5_lag3, log_ret_k60, bb_width_n5_lag1, log_ret_k10_lag1, macd_hist_norm_lag3, bb_width_n60, natr_n5_delta, bb_width_n5, bb_width_n40_delta, vwap_spread_n60_lag1, bb_width_n5_lag2
```
`rows=75264` (no separate `n_train_rows` token). First column is `close`. 26 names.

Skipped price files: **UNH**, **VZ**. News parquet missing for **ISRG**.

`best_val_mse` = min Epoch `val_mse`. `test_mse` / `test_mae` = `METRICS_ROW` (best.pt). Splits on all 6: train 98116 / val 12048 / test 11904.

n_params: e_layers=1 **105095**; e_layers=2 **188615**.

### Per-seed

| e_layers | seed | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
| 1 | 0 | 98116 | 12048 | 11904 | (32, 60, 26) | (32, 60, 15) | 105095 | 0.7515 | 0.7853 | 0.6153 |
| 1 | 1 | 98116 | 12048 | 11904 | (32, 60, 26) | (32, 60, 15) | 105095 | 0.7645 | 0.7983 | 0.6192 |
| 1 | 2 | 98116 | 12048 | 11904 | (32, 60, 26) | (32, 60, 15) | 105095 | 0.7631 | 0.7870 | 0.6119 |
| 2 | 0 | 98116 | 12048 | 11904 | (32, 60, 26) | (32, 60, 15) | 188615 | 0.7307 | 0.7901 | 0.6175 |
| 2 | 1 | 98116 | 12048 | 11904 | (32, 60, 26) | (32, 60, 15) | 188615 | 0.7683 | 0.7810 | 0.6110 |
| 2 | 2 | 98116 | 12048 | 11904 | (32, 60, 26) | (32, 60, 15) | 188615 | 0.7680 | 0.7983 | 0.6206 |

### Summary vs DLinear test 0.7547

Means n=3, sample std. DLinear number is the prior Huber F5 paper run (`paper-selected-40d-run`), not re-trained here.

| model | e_layers | n_params | best_val_mse | test_mse | test_mae | vs DLinear test 0.7547 |
| timexer_selected | 1 | 105095 | 0.7597±0.0071 | 0.7902±0.0071 | 0.6155±0.0037 | +0.0355 |
| timexer_selected | 2 | 188615 | 0.7557±0.0216 | 0.7898±0.0087 | 0.6164±0.0049 | +0.0351 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-selected-26ts.log`
- signature: `Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json`

## Conclusions for Dev
1. Prepare rewrote the fold-1 signature. `ts` block starts with `close` and has 26 names. All 6 jobs saw `x` last dim 26 and `text_seq` last dim 15.
2. **e=2 does not improve on e=1.** Mean best_val is 0.7557 vs 0.7597, but the edge is seed 0 (0.7307) against a larger std (0.0216 vs 0.0071); seeds 1 and 2 are worse than every e=1 seed. Mean test_mse is 0.7898 vs 0.7902 (gap 0.0004, inside the seed std). Mean test_mae is slightly worse (0.6164 vs 0.6155).
3. **Neither depth beats DLinear test 0.7547.** Closest single seed is e=2 seed 1 at test 0.7810, still +0.0263. Pinning `close` did move selected test from the prior 25D run (0.8830) down to ~0.790.
4. CUDA, `num_workers=4`. Did not run DLinear / plain / A / B / C0 / C1 / `ticker_set=dev`. Did not commit `outputs/`, `mlflow.db`, `Data/`, or `.ssh`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
