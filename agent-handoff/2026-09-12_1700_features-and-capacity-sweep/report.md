# Report: features-and-capacity-sweep

- authored_by: runner
- created_at: 2026-09-12T15:47:07Z
- request_folder: agent-handoff/2026-09-12_1700_features-and-capacity-sweep/
- tested_ref: feat/phase2-timexer@e5a49626d25042f1a1a41fc301d1a1fe0c8f7d5c (contains 798bb80; harvest from 73093de still-running, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
# Human ping after timer still-running: PID 524337 dead, log has SWEEP_DONE 36/36. Harvest only — no new train.
```

## Outcome
- exit_code: 36/36 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 524337 gone; no Traceback
- duration: ~2h 15m 36s (2026-09-12T13:22:39Z → 2026-09-12T15:38:15Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 36**. `First batch tensors on x=cuda:0 text=cuda:0`. NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seeds 0,1,2. T=60 H=7, lr=0.0003, epochs=20. yaml `d_model=64` `d_ff=256` (not overridden).
- First-batch `x` last dim: **F1=1 / F2=2 / F5=5** (`x=(32, 60, 1|2|5)`).
- DLinear F1 last/test **match** prior F5 DLinear: **0.7272±0.0035 / 0.6739±0.0013**.
- C1.1 MSE F5 last/test **match** regularization-sweep `e_layers=1`: **0.7415±0.0198 / 0.6800±0.0046**.

GPU / loader (verbatim, first job C1.1 MSE F1 seed=0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
First batch shapes x=(32, 60, 1) text=(32, 768) text_seq=(32, 60, 768)
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

TEXT_COVERAGE split (verbatim, first job):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt). Means n=3, sample std. This sweep only. Pick on mean last_val.

n_params: C1 F1 121799 / F2 122567 / F5 124871; plain F1 68103 / F2 68871 / F5 71175; DLinear F1 854.

### Table 1 — C1.1 MSE

| label | last_val | best_val | test_mse | overfit_gap | n_params |
| C1.1 MSE F1 | 0.7474±0.0529 | 0.7039±0.0075 | 0.6840±0.0178 | 0.0435±0.0573 | 121799 |
| C1.1 MSE F2 | 0.7255±0.0223 | 0.7033±0.0054 | 0.6905±0.0114 | 0.0221±0.0225 | 122567 |
| C1.1 MSE F5 | 0.7415±0.0198 | 0.7097±0.0032 | 0.6800±0.0046 | 0.0319±0.0228 | 124871 |

### Table 2 — C1.1 Huber δ=0.5

| label | last_val | best_val | test_mse | overfit_gap | n_params |
| C1.1 Huber δ=0.5 F1 | 0.7294±0.0228 | 0.7012±0.0037 | 0.6764±0.0125 | 0.0282±0.0265 | 121799 |
| C1.1 Huber δ=0.5 F2 | 0.7730±0.0634 | 0.7075±0.0108 | 0.6910±0.0141 | 0.0655±0.0526 | 122567 |
| C1.1 Huber δ=0.5 F5 | 0.7363±0.0259 | 0.7037±0.0048 | 0.6729±0.0108 | 0.0325±0.0259 | 124871 |

### Table 3 — plain last e=1

| label | last_val | best_val | test_mse | overfit_gap | n_params |
| plain last e=1 F1 | 0.7712±0.0264 | 0.7221±0.0057 | 0.6753±0.0026 | 0.0492±0.0304 | 68103 |
| plain last e=1 F2 | 0.7512±0.0114 | 0.7243±0.0040 | 0.6697±0.0063 | 0.0268±0.0103 | 68871 |
| plain last e=1 F5 | 0.7659±0.0296 | 0.7383±0.0068 | 0.6824±0.0032 | 0.0276±0.0230 | 71175 |

### Table 4 — ranking vs DLinear (sorted by last_val)

DLinear last **0.7272** / test **0.6739**. Bold = TimeXer that beats that last or that test.

| label | last_val | best_val | test_mse | overfit_gap | n_params |
| **C1.1 MSE F2** | **0.7255±0.0223** | 0.7033±0.0054 | 0.6905±0.0114 | 0.0221±0.0225 | 122567 |
| DLinear F1 | 0.7272±0.0035 | 0.7237±0.0019 | 0.6739±0.0013 | 0.0034±0.0016 | 854 |
| C1.1 Huber δ=0.5 F1 | 0.7294±0.0228 | 0.7012±0.0037 | 0.6764±0.0125 | 0.0282±0.0265 | 121799 |
| **C1.1 Huber δ=0.5 F5** | 0.7363±0.0259 | 0.7037±0.0048 | **0.6729±0.0108** | 0.0325±0.0259 | 124871 |
| C1.1 MSE F5 | 0.7415±0.0198 | 0.7097±0.0032 | 0.6800±0.0046 | 0.0319±0.0228 | 124871 |
| C1.1 MSE F1 | 0.7474±0.0529 | 0.7039±0.0075 | 0.6840±0.0178 | 0.0435±0.0573 | 121799 |
| **plain last e=1 F2** | 0.7512±0.0114 | 0.7243±0.0040 | **0.6697±0.0063** | 0.0268±0.0103 | 68871 |
| plain last e=1 F5 | 0.7659±0.0296 | 0.7383±0.0068 | 0.6824±0.0032 | 0.0276±0.0230 | 71175 |
| plain last e=1 F1 | 0.7712±0.0264 | 0.7221±0.0057 | 0.6753±0.0026 | 0.0492±0.0304 | 68103 |
| C1.1 Huber δ=0.5 F2 | 0.7730±0.0634 | 0.7075±0.0108 | 0.6910±0.0141 | 0.0655±0.0526 | 122567 |
| C0 mean e=1 λ=0 F1 | 0.7999±0.0306 | 0.7487±0.0109 | 0.6777±0.0086 | 0.0512±0.0358 | 121799 |
| C0 mean e=1 λ=0 F2 | 0.8177±0.0154 | 0.7414±0.0093 | 0.6881±0.0097 | 0.0762±0.0212 | 122567 |

### Per-seed — configs that beat DLinear last 0.7272 or test 0.6739, plus DLinear

| label | seed | stop_ep | last_val | best_val | overfit_gap | test_mse |
| C1.1 MSE F2 | 0 | 14 | 0.7031 | 0.6999 | 0.0032 | 0.6999 |
| C1.1 MSE F2 | 1 | 10 | 0.7257 | 0.7095 | 0.0162 | 0.6938 |
| C1.1 MSE F2 | 2 | 10 | 0.7476 | 0.7006 | 0.0470 | 0.6779 |
| C1.1 Huber δ=0.5 F5 | 0 | 9 | 0.7348 | 0.6982 | 0.0366 | 0.6842 |
| C1.1 Huber δ=0.5 F5 | 1 | 12 | 0.7629 | 0.7067 | 0.0562 | 0.6627 |
| C1.1 Huber δ=0.5 F5 | 2 | 12 | 0.7111 | 0.7063 | 0.0048 | 0.6719 |
| plain last e=1 F2 | 0 | 9 | 0.7384 | 0.7234 | 0.0150 | 0.6756 |
| plain last e=1 F2 | 1 | 11 | 0.7602 | 0.7287 | 0.0315 | 0.6704 |
| plain last e=1 F2 | 2 | 12 | 0.7549 | 0.7209 | 0.0340 | 0.6630 |
| DLinear F1 | 0 | 13 | 0.7260 | 0.7233 | 0.0027 | 0.6751 |
| DLinear F1 | 1 | 20 | 0.7244 | 0.7221 | 0.0023 | 0.6725 |
| DLinear F1 | 2 | 12 | 0.7311 | 0.7258 | 0.0053 | 0.6740 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/features-and-capacity-sweep.log`

## Conclusions for Dev
1. Features-and-capacity sweep **pass**: 36/36 CUDA, `num_workers=4`, four tables. Feature last-dim check OK (1/2/5). DLinear F1 and C1.1 MSE F5 reproduce priors — not a feature-load bug.
2. By pick rule (mean last_val), **C1.1 MSE F2** is the only TimeXer that beats DLinear last (0.7255 vs 0.7272). The edge is inside F2 seed noise (std 0.0223 vs DLinear 0.0035) and **test is worse** (0.6905 vs 0.6739).
3. Two TimeXers beat DLinear **test**: **plain last e=1 F2** (0.6697) and **C1.1 Huber δ=0.5 F5** (0.6729). Neither beats DLinear last_val. Huber F5 test win is seed-1 driven (0.6627) with std 0.0108.
4. Hypothesis (drop O/H/L → close C1.1→DLinear gap): mixed. On C1.1 MSE, F2 improves last_val vs F5 (0.7255 vs 0.7415) but hurts test (0.6905 vs 0.6800). F1 last_val is worse than F5 and noisier. C0 F1/F2 stay well behind on last_val.
5. Did not run flatten / A / B / paper / extra T/H / decay_lambda / caps / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
