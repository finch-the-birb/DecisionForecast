# Report: head-pooling-sweep

- authored_by: runner
- created_at: 2026-09-12T00:17:30Z
- request_folder: agent-handoff/2026-09-12_0200_head-pooling-sweep/
- tested_ref: feat/phase2-timexer@02d47475575ca85d8a40d93dc62681ea326680cd (contains 93437e7; harvest HEAD f4d6be4 is started-report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
# Sweep already running from prior nohup (PID 294266). Timer fired: PID dead, log has SWEEP_DONE. Harvest only — no new train.
```

## Outcome
- exit_code: 24/24 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 294266 gone; no Traceback
- duration: ~1h 14m 20s (2026-09-11T22:47:14Z → 2026-09-12T00:01:34Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 24**. `First batch tensors on x=cuda:0 text=cuda:0`. NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seeds 0,1,2. T=60 H=7, lr=0.0003, epochs=20, patience=5. yaml `d_model=64` `d_ff=256`. `head.type=linear` `head.dropout=0.1`.
- C1.2 `pool=global`: 3/3 DONE, no crash (`g_en` passed).
- Winner among 8 configs = **mean last-epoch val_mse**: **DLinear** (0.7272±0.0035). Among TimeXer: **plain mean** (0.7652±0.0051).

GPU / loader (verbatim, first job plain mean seed=0):
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

`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; test = `METRICS_ROW` (best.pt). Means n=3, sample std. This sweep only.

n_params (verbatim, same across seeds of a config): plain 137927; C0/C1 191623; DLinear 854. Compact Linear(64→7) ≈ 455 weights, not Flatten Linear(576→7) ≈ 4039. Same n_params for mean/last/global on a given backbone (pool is a reduce, not extra matmul).

### Summary (mean ± sample std)

| label | pool | last_val | best_val | test_mse | n_params |
| plain mean | mean | 0.7652±0.0051 | 0.7386±0.0073 | 0.6953±0.0206 | 137927 |
| plain last | last | 0.7666±0.0119 | 0.7320±0.0061 | 0.6936±0.0075 | 137927 |
| C0 mean | mean | 0.7963±0.0141 | 0.7578±0.0202 | 0.7085±0.0017 | 191623 |
| C0 last | last | 0.8010±0.0131 | 0.7408±0.0023 | 0.7177±0.0244 | 191623 |
| C1 mean | mean | 0.8811±0.0677 | 0.7521±0.0040 | 0.7015±0.0053 | 191623 |
| C1.1 last | last | 0.8531±0.0316 | 0.7641±0.0310 | 0.6862±0.0056 | 191623 |
| C1.2 global | global | 0.8338±0.0589 | 0.7739±0.0210 | 0.7445±0.0470 | 191623 |
| **DLinear** | **—** | **0.7272±0.0035** | **0.7237±0.0019** | **0.6739±0.0013** | **854** |

### Per-seed (overfit = last_val − best_val)

| label | seed | stop_ep | last_val | best_val | last−best | test_mse |
| plain mean | 0 | 10 | 0.7636 | 0.7384 | 0.0252 | 0.7176 |
| plain mean | 1 | 15 | 0.7709 | 0.7460 | 0.0249 | 0.6913 |
| plain mean | 2 | 9 | 0.7611 | 0.7314 | 0.0297 | 0.6770 |
| plain last | 0 | 9 | 0.7779 | 0.7304 | 0.0475 | 0.6879 |
| plain last | 1 | 7 | 0.7676 | 0.7388 | 0.0288 | 0.7021 |
| plain last | 2 | 9 | 0.7542 | 0.7269 | 0.0273 | 0.6908 |
| C0 mean | 0 | 9 | 0.8123 | 0.7381 | 0.0742 | 0.7068 |
| C0 mean | 1 | 6 | 0.7905 | 0.7567 | 0.0338 | 0.7085 |
| C0 mean | 2 | 6 | 0.7860 | 0.7785 | 0.0075 | 0.7102 |
| C0 last | 0 | 9 | 0.8129 | 0.7382 | 0.0747 | 0.7237 |
| C0 last | 1 | 11 | 0.8030 | 0.7428 | 0.0602 | 0.7386 |
| C0 last | 2 | 12 | 0.7870 | 0.7413 | 0.0457 | 0.6909 |
| C1 mean | 0 | 8 | 0.8820 | 0.7544 | 0.1276 | 0.7039 |
| C1 mean | 1 | 6 | 0.8130 | 0.7544 | 0.0586 | 0.6954 |
| C1 mean | 2 | 6 | 0.9483 | 0.7475 | 0.2008 | 0.7052 |
| C1.1 last | 0 | 8 | 0.8708 | 0.7518 | 0.1190 | 0.6825 |
| C1.1 last | 1 | 7 | 0.8166 | 0.7412 | 0.0754 | 0.6834 |
| C1.1 last | 2 | 7 | 0.8719 | 0.7994 | 0.0725 | 0.6927 |
| C1.2 global | 0 | 9 | 0.7775 | 0.7501 | 0.0274 | 0.7776 |
| C1.2 global | 1 | 11 | 0.8289 | 0.7899 | 0.0390 | 0.7653 |
| C1.2 global | 2 | 6 | 0.8950 | 0.7817 | 0.1133 | 0.6907 |
| DLinear | 0 | 13 | 0.7260 | 0.7233 | 0.0027 | 0.6751 |
| DLinear | 1 | 20 | 0.7244 | 0.7221 | 0.0023 | 0.6725 |
| DLinear | 2 | 12 | 0.7311 | 0.7258 | 0.0053 | 0.6740 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/head-pooling-sweep.log`

## Conclusions for Dev
1. Compact pooling sweep **pass**: 24/24 CUDA, `num_workers=4`, both tables. DLinear still wins last_val (0.7272) and test (0.6739) — **exact match** to `baseline-seed`.
2. Q1: plain `mean` **reproduces last_val** (0.7652±0.0051 vs 0.7679±0.0084). Test 0.6953±0.0206 is a bit above 0.6887±0.0273 but overlapping.
3. Q2: compact `last` does **not** beat same-model `mean` on last_val (plain 0.7666 vs 0.7652; C0 0.8010 vs 0.7963). Test is mixed (plain last slightly better; C0 last worse).
4. Q3: among C1, last_val order is **global 0.8338 < last 0.8531 < mean 0.8811**. None beat C0 mean (0.7963) or DLinear (0.7272) on last_val. C1.1 last has the best TimeXer test (0.6862) — beats C0 mean test 0.7085, still loses to DLinear 0.6739. C1.2 global did not crash; test 0.7445 is the worst C1. C1 mean overfits (last−best up to 0.20).
5. Did not run flatten / A / B / paper / extra T/H / decay_lambda / caps / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
