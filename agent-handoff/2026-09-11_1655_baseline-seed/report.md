# Report: baseline-seed

- authored_by: runner
- created_at: 2026-09-11T17:29:30Z
- request_folder: agent-handoff/2026-09-11_1655_baseline-seed/
- tested_ref: feat/phase2-timexer@349b0a09e9ff215a4b9e42962670f6d67fd99d81 (contains dc19237, 03f123b; harvest HEAD 0a7ff87 is the started-report commit, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not B / C0 / C1 / ablation_table / seed=42 / lr=0.001 / extra proto / other T/H / decay_lambda / d_model.
# Sweep already running from prior nohup (PID 159375). Human ping: PID dead, log has SWEEP_DONE. Harvest only — no new train.
```

## Outcome
- exit_code: 9/9 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 159375 gone; no Traceback
- duration: ~22m 21s (16:56:37Z → 17:18:58Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 9). NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seeds 0,1,2 (not 42). T=60 H=7, lr=0.0003, epochs=20, patience=5. yaml `d_model=64` `d_ff=256` (not overridden). `num_workers=4`. `n_prototypes=10` only on `a`.
- Winner among these three = **mean last-epoch val_mse** (n=3): **dlinear** (0.7272±0.0035). Test recorded, not used to pick.

GPU / loader (verbatim, first job dlinear seed=0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

TEXT_COVERAGE split (verbatim, first job; same sizes on the rest):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

Early stop: dlinear s0 ep13 / s1 ep20 (hit max) / s2 ep12; timexer_plain s0 ep10 / s1 ep13 / s2 ep8; a s0 ep7 / s1 ep6 / s2 ep7.

## Key signals
This sweep only. `best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `stop_ep` = `Early stopping at epoch N` (else last Epoch N); test = `METRICS_ROW` (best.pt). Do not mix B / C0 / C1 into Table 1.

### Table 1 — per-seed

| model | seed | stop_ep | best_val | last_val | test_mse | test_mae | mae_denorm |
| dlinear | 0 | 13 | 0.7233 | 0.7260 | 0.6751 | 0.5763 | 5.2562 |
| dlinear | 1 | 20 | 0.7221 | 0.7244 | 0.6725 | 0.5739 | 5.2335 |
| dlinear | 2 | 12 | 0.7258 | 0.7311 | 0.6740 | 0.5761 | 5.2421 |
| timexer_plain | 0 | 10 | 0.7358 | 0.7706 | 0.6725 | 0.5805 | 5.3482 |
| timexer_plain | 1 | 13 | 0.7450 | 0.7747 | 0.6733 | 0.5813 | 5.3976 |
| timexer_plain | 2 | 8 | 0.7432 | 0.7585 | 0.7202 | 0.5997 | 5.5453 |
| a | 0 | 7 | 0.7838 | 0.8891 | 0.7303 | 0.6213 | 5.7553 |
| a | 1 | 6 | 0.7810 | 0.8816 | 0.7501 | 0.6181 | 5.6913 |
| a | 2 | 7 | 0.7531 | 0.8864 | 0.7223 | 0.6093 | 5.5229 |

### Table 2 — means n=3 (sample std)

| model | last_val mean±std | best_val mean±std | test_mse mean±std |
| **dlinear** | **0.7272±0.0035** | **0.7237±0.0019** | **0.6739±0.0013** |
| timexer_plain | 0.7679±0.0084 | 0.7413±0.0049 | 0.6887±0.0273 |
| a | 0.8857±0.0038 | 0.7726±0.0170 | 0.7342±0.0143 |

METRICS_ROW (verbatim, 9/9):
```
METRICS_ROW model=dlinear horizon=7 mse=0.6751 mae=0.5763 mae_denorm=5.2562
METRICS_ROW model=timexer_plain horizon=7 mse=0.6725 mae=0.5805 mae_denorm=5.3482
METRICS_ROW model=a horizon=7 mse=0.7303 mae=0.6213 mae_denorm=5.7553
METRICS_ROW model=dlinear horizon=7 mse=0.6725 mae=0.5739 mae_denorm=5.2335
METRICS_ROW model=timexer_plain horizon=7 mse=0.6733 mae=0.5813 mae_denorm=5.3976
METRICS_ROW model=a horizon=7 mse=0.7501 mae=0.6181 mae_denorm=5.6913
METRICS_ROW model=dlinear horizon=7 mse=0.6740 mae=0.5761 mae_denorm=5.2421
METRICS_ROW model=timexer_plain horizon=7 mse=0.7202 mae=0.5997 mae_denorm=5.5453
METRICS_ROW model=a horizon=7 mse=0.7223 mae=0.6093 mae_denorm=5.5229
```

n_params (verbatim): dlinear 854; timexer_plain 146695; a proto=10 144199.

### Footnote — prior `seed-shortlist` (copied from git; not re-run)

From `agent-handoff/2026-09-11_1448_seed-shortlist/report.md` means (n=3, last-epoch val / test):

| model | proto | last_val mean±std | test_mse mean±std |
| b | 10 | 0.8816±0.0153 | 0.7244±0.0275 |
| c0 | 5 | 0.8338±0.0473 | 0.7018±0.0067 |
| c1 | 5 | 0.9072±0.0639 | 0.7018±0.0176 |

This sweep's last_val winner **dlinear** beats locked **C0 proto=5** on both mean last_val (0.7272 vs 0.8338) and mean test_mse (0.6739 vs 0.7018).

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/baseline-seed.log`
- dlinear s0: `outputs/2026-09-11/16-57-18/`
- timexer_plain s0: `outputs/2026-09-11/16-59-40/`
- a s0: `outputs/2026-09-11/17-02-22/`
- dlinear s1: `outputs/2026-09-11/17-04-49/`
- timexer_plain s1: `outputs/2026-09-11/17-07-43/`
- a s1: `outputs/2026-09-11/17-10-53/`
- dlinear s2: `outputs/2026-09-11/17-13-15/`
- timexer_plain s2: `outputs/2026-09-11/17-15-20/`
- a s2: `outputs/2026-09-11/17-17-18/`

## Conclusions for Dev
1. Uncapped baseline 3-seed **pass**: 9/9 CUDA, `num_workers=4`, both tables. Mean-last_val winner among dlinear / timexer_plain / A is **dlinear** (0.7272±0.0035).
2. A proto=10 is the weakest of the three (last_val 0.8857±0.0038). timexer_plain is in between; seed=2 test 0.7202 pulls its test mean up. DLinear is stable (test 0.6739±0.0013).
3. Did not run B / C0 / C1 / paper / seed=42 / lr=0.001 / extra proto / other T/H / decay_lambda / caps / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
