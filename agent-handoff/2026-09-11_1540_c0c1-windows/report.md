# Report: c0c1-windows

- authored_by: runner
- created_at: 2026-09-11T16:24:45Z
- request_folder: agent-handoff/2026-09-11_1540_c0c1-windows/
- tested_ref: feat/phase2-timexer@729a3a8add4d643e9ffa33f63a9d34b47bb49b35 (contains 149c06b, 03f123b; harvest HEAD db501f3 is the started-report commit, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not A / B / dlinear / ablation_table / extra seeds / decay_lambda / window_agg / proto≠5 / T=60 H=7 / d_model.
# Sweep already running from prior nohup (PID 126197). Human ping: PID dead, log has SWEEP_DONE. Harvest only — no new train.
```

## Outcome
- exit_code: 8/8 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 126197 gone; no Traceback
- duration: ~32m 20s (15:39:44Z → 16:12:04Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 8). NVIDIA L4. No OOM. Not paper.
- Uncapped dev; seed=42; lr=0.0003; epochs=20; patience=5; yaml `d_model=64` `d_ff=256` (not overridden); `n_prototypes=5`; `num_workers=4`.
- Pairs only: (60,14), (60,30), (36,7), (96,7). No T=60 H=7. No global winner across H.

GPU / loader (verbatim, first job c0 T=60 H=14):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

TEXT_COVERAGE split + First batch shapes **per (T,H)** (verbatim, first job of the pair; C1 same sizes):

**(60,14)** train 23373 / val 2510 / test 2480
```
TEXT_COVERAGE split=train windows=23373 mean_has_news_frac=0.291 zero_windows=0.566 mean_text_l2=0.1454
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.537 zero_windows=0.424 mean_text_l2=0.2134
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.761 zero_windows=0.224 mean_text_l2=0.3039
First batch shapes x=(32, 60, 5) text=(32, 768) text_seq=(32, 60, 768)
```

**(60,30)** train 23133 / val 2510 / test 2480
```
TEXT_COVERAGE split=train windows=23133 mean_has_news_frac=0.289 zero_windows=0.568 mean_text_l2=0.1448
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.519 zero_windows=0.443 mean_text_l2=0.2065
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.754 zero_windows=0.231 mean_text_l2=0.3004
First batch shapes x=(32, 60, 5) text=(32, 768) text_seq=(32, 60, 768)
```

**(36,7)** train 23838 / val 2510 / test 2480
```
TEXT_COVERAGE split=train windows=23838 mean_has_news_frac=0.293 zero_windows=0.574 mean_text_l2=0.1516
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.560 zero_windows=0.415 mean_text_l2=0.2209
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.768 zero_windows=0.221 mean_text_l2=0.3084
First batch shapes x=(32, 36, 5) text=(32, 768) text_seq=(32, 36, 768)
```

**(96,7)** train 22938 / val 2510 / test 2480
```
TEXT_COVERAGE split=train windows=22938 mean_has_news_frac=0.291 zero_windows=0.552 mean_text_l2=0.1433
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.525 zero_windows=0.415 mean_text_l2=0.2131
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.756 zero_windows=0.221 mean_text_l2=0.3035
First batch shapes x=(32, 96, 5) text=(32, 768) text_seq=(32, 96, 768)
```

text_seq length matches T on all four pairs.

## Key signals
This sweep only. `best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `stop_ep` = `Early stopping at epoch N`; test = `METRICS_ROW` (best.pt). Do not mix seed-shortlist / R3 / T=60 H=7 / ablation_table.

### Table 1 — metrics

| model | lookback_T | horizon | stop_ep | best_val | last_val | test_mse | test_mae | mae_denorm |
| c0 | 60 | 14 | 11 | 1.5123 | 1.6547 | 1.4046 | 0.8519 | 7.9716 |
| c1 | 60 | 14 | 16 | 1.4868 | 1.6959 | 1.5057 | 0.8618 | 7.6367 |
| c0 | 60 | 30 | 6 | 3.3635 | 5.1311 | 2.8314 | 1.2269 | 11.8952 |
| c1 | 60 | 30 | 6 | 3.3895 | 7.7111 | 2.8738 | 1.2282 | 11.5364 |
| c0 | 36 | 7 | 11 | 1.2411 | 1.3046 | 1.2141 | 0.7795 | 5.4174 |
| c1 | 36 | 7 | 11 | 1.2652 | 1.3734 | 1.3509 | 0.8263 | 5.5984 |
| c0 | 96 | 7 | 10 | 0.4967 | 0.5308 | 0.4139 | 0.4610 | 5.4765 |
| c1 | 96 | 7 | 10 | 0.4888 | 0.5345 | 0.3995 | 0.4538 | 5.4179 |

C0 vs C1 **per (T,H)** (no global winner):

| T | H | best_val winner | test_mse winner |
| 60 | 14 | **c1** (1.4868 vs 1.5123) | **c0** (1.4046 vs 1.5057) |
| 60 | 30 | **c0** (3.3635 vs 3.3895) | **c0** (2.8314 vs 2.8738) |
| 36 | 7 | **c0** (1.2411 vs 1.2652) | **c0** (1.2141 vs 1.3509) |
| 96 | 7 | **c1** (0.4888 vs 0.4967) | **c1** (0.3995 vs 0.4139) |

### Table 2 — overfit

| model | T | H | stop_ep | train_loss_at_best | train_loss_last | last_val - best_val |
| c0 | 60 | 14 | 11 | 2.5049 | 1.9707 | 0.1424 |
| c1 | 60 | 14 | 16 | 1.9745 | 1.7898 | 0.2091 |
| c0 | 60 | 30 | 6 | 6.5150 | 4.1591 | 1.7676 |
| c1 | 60 | 30 | 6 | 6.5500 | 3.7781 | 4.3216 |
| c0 | 36 | 7 | 11 | 2.1541 | 1.8745 | 0.0635 |
| c1 | 36 | 7 | 11 | 2.0963 | 1.8117 | 0.1082 |
| c0 | 96 | 7 | 10 | 1.4028 | 1.0422 | 0.0341 |
| c1 | 96 | 7 | 10 | 1.4055 | 1.0443 | 0.0457 |

Overfit: on **H=30** best_val is epoch 1 and stop_ep=6 (patience=5); train_loss keeps falling (c0 6.5150→4.1591, c1 6.5500→3.7781) while val explodes (Δ +1.7676 / +4.3216) — early stop is catching overfit; test is best.pt from ep1. On **H=14** train_loss still falls after best and last_val is worse (Δ +0.14 / +0.21); C1 ran to ep16. On **H=7** (T=36 and T=96) stop_ep is 10–11 and last_val−best_val is small (0.03–0.11) while train_loss still eases down — closer to noise / mild drift than memorization.

METRICS_ROW (verbatim, 8/8):
```
METRICS_ROW model=c0 horizon=14 mse=1.4046 mae=0.8519 mae_denorm=7.9716
METRICS_ROW model=c1 horizon=14 mse=1.5057 mae=0.8618 mae_denorm=7.6367
METRICS_ROW model=c0 horizon=30 mse=2.8314 mae=1.2269 mae_denorm=11.8952
METRICS_ROW model=c1 horizon=30 mse=2.8738 mae=1.2282 mae_denorm=11.5364
METRICS_ROW model=c0 horizon=7 mse=1.2141 mae=0.7795 mae_denorm=5.4174
METRICS_ROW model=c1 horizon=7 mse=1.3509 mae=0.8263 mae_denorm=5.5984
METRICS_ROW model=c0 horizon=7 mse=0.4139 mae=0.4610 mae_denorm=5.4765
METRICS_ROW model=c1 horizon=7 mse=0.3995 mae=0.4538 mae_denorm=5.4179
```

n_params (verbatim): (60,14) 201294; (60,30) 203358; (36,7) and (96,7) 200391. Same C0/C1 within a pair.

H3_ROW text_zero (all 8; record, do not fail on Δ):
```
H3_ROW model=c0 variant=text_zero mse=1.4054 mae=0.8484 delta_mse=+0.0008 delta_mae=-0.0036
H3_ROW model=c1 variant=text_zero mse=1.7676 mae=0.9661 delta_mse=+0.2619 delta_mae=+0.1043
H3_ROW model=c0 variant=text_zero mse=2.8800 mae=1.2227 delta_mse=+0.0486 delta_mae=-0.0042
H3_ROW model=c1 variant=text_zero mse=2.8153 mae=1.2147 delta_mse=-0.0585 delta_mae=-0.0135
H3_ROW model=c0 variant=text_zero mse=1.2075 mae=0.7711 delta_mse=-0.0066 delta_mae=-0.0084
H3_ROW model=c1 variant=text_zero mse=1.2538 mae=0.7861 delta_mse=-0.0971 delta_mae=-0.0402
H3_ROW model=c0 variant=text_zero mse=0.4033 mae=0.4518 delta_mse=-0.0106 delta_mae=-0.0091
H3_ROW model=c1 variant=text_zero mse=0.4141 mae=0.4664 delta_mse=+0.0145 delta_mae=+0.0126
```

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/c0c1-windows.log`
- c0 T60 H14: `outputs/2026-09-11/15-40-26/`
- c1 T60 H14: `outputs/2026-09-11/15-44-32/`
- c0 T60 H30: `outputs/2026-09-11/15-50-17/`
- c1 T60 H30: `outputs/2026-09-11/15-53-07/`
- c0 T36 H7: `outputs/2026-09-11/15-56-18/`
- c1 T36 H7: `outputs/2026-09-11/16-00-11/`
- c0 T96 H7: `outputs/2026-09-11/16-04-28/`
- c1 T96 H7: `outputs/2026-09-11/16-08-23/`

## Conclusions for Dev
1. Uncapped C0 vs C1 windows **pass**: 8/8 CUDA, `num_workers=4`, both tables, C0 vs C1 named per pair. C1 does **not** consistently beat C0. C0 wins both metrics at (60,30) and (36,7). Split at (60,14): C1 best_val, C0 test. C1 wins both at (96,7).
2. H=30 overfits hard (best at ep1, val collapse, C1 worse). text_zero is not ~0 on C1 everywhere: (60,14) Δmse=+0.2619 (text used); (36,7) Δmse=−0.0971 (zeroing text *helps* test).
3. Did not run A / B / dlinear / paper / extra seeds / decay_lambda / window_agg / proto≠5 / caps / T=60 H=7 / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
