# Report: paper-selected-40d-run

- authored_by: runner
- created_at: 2026-10-05T20:23:57Z
- request_folder: agent-handoff/2026-10-06_1000_paper-selected-40d-run/
- tested_ref: feat/phase2-timexer@863942091f86d7c50d908e883314f36e9eb4c3ef (contains 074ff3c; harvest from bf1a572 started report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. No model=a/b/c0/c1. No ticker_set=dev.
# Human ping: PID 22447 dead, log has SWEEP_DONE 9/9. Harvest only — no new train.
```

## Outcome
- exit_code: prepare exit 0 (`PREPARE_DONE`); 9/9 trains DONE; log ends with `=== SWEEP_DONE 2026-10-05T20:15:29Z ===`; nohup PID 22447 gone; no Traceback
- duration: prepare 18:17:02Z → 19:07:55Z (~51 min); trains 19:07:55Z → 20:15:29Z (~1h 8m)
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 9**. `First batch tensors on x=cuda:0 text=cuda:0`. NVIDIA L4. No OOM.
- `timexer_plain` and `timexer_selected` first-batch shapes are `x=(32, 60, 25)` and `text_seq=(32, 60, 15)` on every seed. DLinear on `data=fnspid` stays `x=(32, 60, 5)` `text_seq=(32, 60, 768)`.

Prepare (verbatim):
```
INFO Text state /workspace/DecisionForecast/Data/FNSPID/cache/selected_signatures/paper_fold1_text_state.npz train_articles=150683
INFO Signature /workspace/DecisionForecast/Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json rows=75264 label_dates=2015-01-12..2019-12-10 columns=month_cos_delta, month_cos, month_sin_lag3, gk_n40_lag3, gk_n5, bb_width_n60_delta, log_ret_k5, parkinson_n10_delta, gk_n60_delta, month_cos_lag2, bb_pctb_n5, gk_n20_delta, day_sin_delta, natr_n10_lag3, log_ret_k5_lag3, log_ret_k60, bb_width_n5_lag1, log_ret_k10_lag1, macd_hist_norm_lag3, bb_width_n60, natr_n5_delta, bb_width_n5, bb_width_n40_delta, vwap_spread_n60_lag1, bb_width_n5_lag2
```
The log field is `rows=75264` (no separate `n_train_rows` token). 25 names, in that order.

Skipped price files: **UNH**, **VZ**. News parquet missing for **ISRG** (prepare and each DLinear job scanned `nasdaq_exteral_data.csv`).

`best_val_mse` = min Epoch `val_mse`. `test_mse` / `test_mae` = `METRICS_ROW` (best.pt). This sweep only.

n_params: DLinear 854; timexer_plain 86535; timexer_selected 104327.

| model | seed | data | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
| dlinear | 0 | fnspid | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 768) | 854 | 0.7026 | 0.7520 | 0.5907 |
| dlinear | 1 | fnspid | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 768) | 854 | 0.7019 | 0.7575 | 0.5936 |
| dlinear | 2 | fnspid | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 768) | 854 | 0.6999 | 0.7547 | 0.5923 |
| timexer_plain | 0 | fnspid_selected | 98116 | 12048 | 11904 | (32, 60, 25) | (32, 60, 15) | 86535 | 0.8281 | 0.8982 | 0.6634 |
| timexer_plain | 1 | fnspid_selected | 98116 | 12048 | 11904 | (32, 60, 25) | (32, 60, 15) | 86535 | 0.8336 | 0.9221 | 0.6749 |
| timexer_plain | 2 | fnspid_selected | 98116 | 12048 | 11904 | (32, 60, 25) | (32, 60, 15) | 86535 | 0.8271 | 0.9121 | 0.6706 |
| timexer_selected | 0 | fnspid_selected | 98116 | 12048 | 11904 | (32, 60, 25) | (32, 60, 15) | 104327 | 0.8545 | 0.8923 | 0.6630 |
| timexer_selected | 1 | fnspid_selected | 98116 | 12048 | 11904 | (32, 60, 25) | (32, 60, 15) | 104327 | 0.8633 | 0.8744 | 0.6531 |
| timexer_selected | 2 | fnspid_selected | 98116 | 12048 | 11904 | (32, 60, 25) | (32, 60, 15) | 104327 | 0.8218 | 0.8824 | 0.6586 |

Means n=3, sample std:

| model | best_val_mse | test_mse | test_mae |
| dlinear | 0.7015±0.0014 | 0.7547±0.0028 | 0.5922±0.0015 |
| timexer_plain | 0.8296±0.0035 | 0.9108±0.0120 | 0.6696±0.0058 |
| timexer_selected | 0.8465±0.0219 | 0.8830±0.0090 | 0.6582±0.0050 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-selected-40d-run.log`
- signature: `Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json`
- text_state: `Data/FNSPID/cache/selected_signatures/paper_fold1_text_state.npz`

## Conclusions for Dev
1. Prepare froze fold-1 train only: `rows=75264`, `label_dates=2015-01-12..2019-12-10`, 25 columns listed above. Paper forecast splits stayed through the existing cut: DLinear train/val/test **98236 / 12048 / 11904**. Selected jobs are **98116 / 12048 / 11904** (120 fewer train windows; val and test match).
2. `timexer_selected` saw `x` last dim 25 and `text_seq` last dim 15 on all 3 seeds and did not raise. Compact text did not beat the plain 25D head on best_val (0.8465 vs 0.8296); its mean test_mse is lower (0.8830 vs 0.9108).
3. Neither TimeXer beats this run's Huber DLinear (best_val 0.7015, test_mse 0.7547). That DLinear test is close to the prior MSE-trained paper DLinear test 0.7535, but the training loss here is Huber δ=0.5 and the table uses min val_mse, not last-epoch val.
4. CUDA, `num_workers=4`, no product-code edits. Did not run A/B/C0/C1 or `ticker_set=dev`. Did not point DLinear at `fnspid_selected`. Did not commit `outputs/`, `mlflow.db`, `Data/`, or `.ssh`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
