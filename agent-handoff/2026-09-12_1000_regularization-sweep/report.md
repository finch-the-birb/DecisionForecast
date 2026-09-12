# Report: regularization-sweep

- authored_by: runner
- created_at: 2026-09-12T12:31:00Z
- request_folder: agent-handoff/2026-09-12_1000_regularization-sweep/
- tested_ref: feat/phase2-timexer@a409b3b1171f33b6b4902c2e3ce475c7905dba2c (contains 7681745; harvest HEAD 1183cd1 is still-running report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
# Sweep already running from prior nohup (PID 375076). Human ping: PID dead, log has SWEEP_DONE. Harvest only — no new train.
```

## Outcome
- exit_code: 60/60 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 375076 gone; no Traceback
- duration: ~3h 26m 44s (2026-09-12T06:45:19Z → 2026-09-12T10:12:03Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 60**. `First batch tensors on x=cuda:0 text=cuda:0`. NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seeds 0,1,2. T=60 H=7, lr=0.0003, epochs=20. Default yaml 64/256 unless overridden (job 16: 32/128).
- DLinear last/test **match** prior: **0.7272±0.0035 / 0.6739±0.0013**. C1.1 MSE last/test **0.8530±0.0316 / 0.6863±0.0056** (matches `head-pooling-sweep` 0.8531 / 0.6862).

GPU / loader (verbatim, first job C1.1 MSE seed=0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

TEXT_COVERAGE split (verbatim, first job):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt). Means n=3, sample std. This sweep only.

n_params: C1.1 default 191623; e_layers=1 124871; d=32/ff=128 62023; plain 137927; DLinear 854.

### Table 1 — losses on C1.1 last

| label | last_val | best_val | test_mse | overfit_gap |
| C1.1 MSE | 0.8530±0.0316 | 0.7640±0.0311 | 0.6863±0.0056 | 0.0890±0.0260 |
| C1.1 Huber δ=1.0 | 0.8840±0.0623 | 0.7436±0.0070 | 0.6939±0.0122 | 0.1404±0.0669 |
| C1.1 Huber δ=0.5 | 0.8449±0.0802 | 0.7334±0.0095 | 0.6852±0.0077 | 0.1114±0.0751 |
| C1.1 dir γ=0.1 | 0.8537±0.0314 | 0.7637±0.0302 | 0.6864±0.0058 | 0.0900±0.0290 |
| C1.1 dir γ=0.5 | 0.8571±0.0424 | 0.7646±0.0306 | 0.6863±0.0051 | 0.0925±0.0455 |
| C1.1 corr α=0.3 | 0.8478±0.0330 | 0.7747±0.0300 | 0.6878±0.0086 | 0.0732±0.0181 |

### Table 2 — prototype λ

| label | last_val | best_val | test_mse | overfit_gap |
| C1.1 λ=0 | 0.8574±0.0647 | 0.7610±0.0245 | 0.6966±0.0179 | 0.0964±0.0711 |
| C1.1 λ=0.1 (job 1 MSE) | 0.8530±0.0316 | 0.7640±0.0311 | 0.6863±0.0056 | 0.0890±0.0260 |
| C1.1 λ=0.3 | 0.8988±0.0388 | 0.7811±0.0272 | 0.6946±0.0142 | 0.1177±0.0521 |
| C1.1 λ=0.5 | 0.9129±0.0406 | 0.7882±0.0276 | 0.7012±0.0196 | 0.1247±0.0524 |
| C0 mean λ=0 | 0.8577±0.0311 | 0.7628±0.0150 | 0.6777±0.0070 | 0.0949±0.0450 |
| C0 mean λ=0.3 | 0.8308±0.0375 | 0.7634±0.0082 | 0.7209±0.0392 | 0.0674±0.0367 |
| plain last (no proto) | 0.7666±0.0119 | 0.7320±0.0061 | 0.6936±0.0075 | 0.0346±0.0113 |

### Table 3 — capacity / decay / dropout

| label | last_val | best_val | test_mse | overfit_gap |
| plain last | 0.7666±0.0119 | 0.7320±0.0061 | 0.6936±0.0075 | 0.0346±0.0113 |
| C1.1 wd=0.01 | 0.8506±0.0292 | 0.7640±0.0309 | 0.6861±0.0057 | 0.0866±0.0230 |
| C1.1 wd=0.05 | 0.8468±0.0269 | 0.7637±0.0322 | 0.6860±0.0051 | 0.0831±0.0123 |
| C1.1 e_layers=1 | 0.7415±0.0198 | 0.7097±0.0032 | 0.6800±0.0046 | 0.0319±0.0228 |
| C1.1 d=32 / ff=128 | 0.7835±0.0098 | 0.7404±0.0096 | 0.6839±0.0163 | 0.0431±0.0173 |
| C1.1 dropout=0.2 | 0.8372±0.0486 | 0.7659±0.0421 | 0.6917±0.0156 | 0.0713±0.0592 |
| plain last wd=0.01 | 0.7659±0.0121 | 0.7320±0.0062 | 0.6936±0.0076 | 0.0339±0.0112 |

### Table 4 — combo vs DLinear

| label | last_val | best_val | test_mse | overfit_gap |
| C1.1 MSE (job 1) | 0.8530±0.0316 | 0.7640±0.0311 | 0.6863±0.0056 | 0.0890±0.0260 |
| C1.1 Huber δ=0.5 (job 3) | 0.8449±0.0802 | 0.7334±0.0095 | 0.6852±0.0077 | 0.1114±0.0751 |
| C1.1 Huber δ=0.5 + wd=0.01 | 0.8415±0.0787 | 0.7329±0.0083 | 0.6839±0.0065 | 0.1086±0.0736 |
| **DLinear** | **0.7272±0.0035** | **0.7237±0.0019** | **0.6739±0.0013** | **0.0034±0.0016** |

### Per-seed — configs with mean test < 0.6862 or mean last_val < 0.7272

None of the TimeXer configs beat DLinear last_val. Below: test beat vs C1.1 MSE 0.6862, plus DLinear (both).

| label | seed | stop_ep | last_val | best_val | overfit_gap | test_mse |
| C1.1 Huber δ=0.5 | 0 | 8 | 0.8042 | 0.7387 | 0.0655 | 0.6910 |
| C1.1 Huber δ=0.5 | 1 | 11 | 0.7932 | 0.7225 | 0.0707 | 0.6764 |
| C1.1 Huber δ=0.5 | 2 | 7 | 0.9372 | 0.7391 | 0.1981 | 0.6882 |
| C0 mean λ=0 | 0 | 12 | 0.8837 | 0.7568 | 0.1269 | 0.6741 |
| C0 mean λ=0 | 1 | 11 | 0.8661 | 0.7518 | 0.1143 | 0.6733 |
| C0 mean λ=0 | 2 | 9 | 0.8233 | 0.7799 | 0.0434 | 0.6858 |
| C1.1 wd=0.01 | 0 | 8 | 0.8648 | 0.7518 | 0.1130 | 0.6823 |
| C1.1 wd=0.01 | 1 | 7 | 0.8171 | 0.7411 | 0.0760 | 0.6833 |
| C1.1 wd=0.01 | 2 | 7 | 0.8700 | 0.7992 | 0.0708 | 0.6926 |
| C1.1 wd=0.05 | 0 | 8 | 0.8475 | 0.7508 | 0.0967 | 0.6825 |
| C1.1 wd=0.05 | 1 | 7 | 0.8196 | 0.7399 | 0.0797 | 0.6836 |
| C1.1 wd=0.05 | 2 | 7 | 0.8733 | 0.8004 | 0.0729 | 0.6919 |
| C1.1 e_layers=1 | 0 | 9 | 0.7642 | 0.7061 | 0.0581 | 0.6853 |
| C1.1 e_layers=1 | 1 | 9 | 0.7329 | 0.7122 | 0.0207 | 0.6768 |
| C1.1 e_layers=1 | 2 | 12 | 0.7275 | 0.7107 | 0.0168 | 0.6779 |
| C1.1 d=32/ff=128 | 0 | 7 | 0.7729 | 0.7496 | 0.0233 | 0.6704 |
| C1.1 d=32/ff=128 | 1 | 9 | 0.7853 | 0.7304 | 0.0549 | 0.6793 |
| C1.1 d=32/ff=128 | 2 | 6 | 0.7922 | 0.7411 | 0.0511 | 0.7020 |
| C1.1 Huber δ=0.5 + wd=0.01 | 0 | 8 | 0.8020 | 0.7368 | 0.0652 | 0.6873 |
| C1.1 Huber δ=0.5 + wd=0.01 | 1 | 11 | 0.7903 | 0.7233 | 0.0670 | 0.6764 |
| C1.1 Huber δ=0.5 + wd=0.01 | 2 | 7 | 0.9321 | 0.7385 | 0.1936 | 0.6880 |
| DLinear | 0 | 13 | 0.7260 | 0.7233 | 0.0027 | 0.6751 |
| DLinear | 1 | 20 | 0.7244 | 0.7221 | 0.0023 | 0.6725 |
| DLinear | 2 | 12 | 0.7311 | 0.7258 | 0.0053 | 0.6740 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/regularization-sweep.log`

## Conclusions for Dev
1. Regularization sweep **pass**: 60/60 CUDA, `num_workers=4`, four tables. DLinear still wins last_val (0.7272) and test (0.6739). Closest TimeXer last_val is **C1.1 e_layers=1** (0.7415, gap 0.0319, test 0.6800) — does not beat DLinear.
2. Q1 losses: no loss clearly beats C1.1 MSE on last_val. Huber δ=0.5 is slightly better last/test (0.8449 / 0.6852) but **worse gap** and seed-2 blowup. Correlation shrinks gap (0.0732 vs 0.0890) without wrecking test (0.6878). Directional ≈ MSE. Huber δ=1.0 worse.
3. Q2 proto λ: on C1.1, **λ=0.1 is best**; 0.3/0.5 raise last_val and gap. λ=0 does not help. On C0, λ=0.3 improves last_val (0.8308) but wrecks test (0.7209); λ=0 has strong test 0.6777 (best TimeXer test here) with last 0.8577. Plain last (no proto) has the smallest TimeXer-ish gap (0.0346) and last 0.7666.
4. Q3 capacity: **e_layers=1** and **d=32** close the gap most (0.0319 / 0.0431) and both beat C1.1 MSE on test. wd 0.01/0.05 is a tiny nudge. dropout 0.2 helps last a bit, test slightly worse. plain last wd=0.01 ≈ plain last.
5. Q4 Huber+wd combo does **not** beat DLinear on last_val (0.8415 vs 0.7272) and only slightly beats C1.1 MSE on test (0.6839 vs 0.6863), still worse than DLinear test 0.6739.
6. Did not run flatten / A / B / paper / extra T/H / decay_lambda / caps / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
