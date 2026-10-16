# Report: paper-baselines-c1

- authored_by: runner
- created_at: 2026-09-12T19:10:11Z
- request_folder: agent-handoff/2026-09-12_2000_paper-baselines-c1/
- tested_ref: feat/phase2-timexer@7a891e970153a1f26d741b09a1dabcec7c237cfc (contains 089a1fe; harvest from f15bdc8 started report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. ticker_set=paper. No Model A / B / C0 / flatten / extra T/H / ticker_set=dev.
# Human ping: PID 590912 dead, log has SWEEP_DONE 12/12. Harvest only — no new train.
```

## Outcome
- exit_code: 12/12 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 590912 gone; no Traceback
- duration: ~2h 45m 20s wall (precache 16:14:57Z→16:42:57Z + train 16:42:57Z→19:00:17Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 12**. `First batch tensors on x=cuda:0 text=cuda:0`. NVIDIA L4. No OOM. Not `ticker_set=dev`.
- Precache: **ran** (not skip). `mu_paper_2015-01-01_2021-12-31.npy` written. Encoder public `FinLang/finance-embeddings-investopedia`.
- Paper splits (all 12 jobs identical): **train 98236 / val 12048 / test 11904**. First-batch `x` last dim: DLinear=1, plain=2, C1=5.
- Skipped missing price files on every job: **UNH**, **VZ**.

GPU / loader (verbatim, first job DLinear F1 seed=0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
First batch shapes x=(32, 60, 1) text=(32, 768) text_seq=(32, 60, 768)
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

Dataset sizes + TEXT_COVERAGE (verbatim, first job; same sizes on all 12):
```
Dataset sizes — train: 98236, val: 12048, test: 11904
TEXT_COVERAGE split=train windows=98236 mean_has_news_frac=0.585 zero_windows=0.223 mean_text_l2=0.2742
TEXT_COVERAGE split=val windows=12048 mean_has_news_frac=0.706 zero_windows=0.197 mean_text_l2=0.3047
TEXT_COVERAGE split=test windows=11904 mean_has_news_frac=0.764 zero_windows=0.173 mean_text_l2=0.3417
```

`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt). Means n=3, sample std. Pick on mean last_val. This sweep only.

n_params: DLinear 854; plain F2 68871; C1.1 e=1 124871; C1.1 e=2 191623.

### Table 1 — per-seed

| label | seed | stop_ep | last_val | best_val | overfit_gap | test_mse |
| DLinear F1 | 0 | 10 | 0.7015 | 0.6964 | 0.0051 | 0.7527 |
| DLinear F1 | 1 | 20 | 0.6991 | 0.6934 | 0.0057 | 0.7544 |
| DLinear F1 | 2 | 11 | 0.6950 | 0.6926 | 0.0024 | 0.7535 |
| plain last e=1 F2 | 0 | 8 | 0.7058 | 0.7031 | 0.0027 | 0.7608 |
| plain last e=1 F2 | 1 | 17 | 0.7217 | 0.7111 | 0.0106 | 0.7754 |
| plain last e=1 F2 | 2 | 9 | 0.7514 | 0.6996 | 0.0518 | 0.7588 |
| C1.1 Huber e=1 F5 | 0 | 16 | 0.7232 | 0.6900 | 0.0332 | 0.7533 |
| C1.1 Huber e=1 F5 | 1 | 10 | 0.7103 | 0.7018 | 0.0085 | 0.7524 |
| C1.1 Huber e=1 F5 | 2 | 10 | 0.7185 | 0.6967 | 0.0218 | 0.7573 |
| C1.1 Huber e=2 F5 | 0 | 6 | 0.7232 | 0.7118 | 0.0114 | 0.7680 |
| C1.1 Huber e=2 F5 | 1 | 9 | 0.7421 | 0.7090 | 0.0331 | 0.7807 |
| C1.1 Huber e=2 F5 | 2 | 9 | 0.7393 | 0.7261 | 0.0132 | 0.8382 |

### Table 2 — means n=3

| label | last_val | best_val | test_mse | overfit_gap | n_params |
| **DLinear F1** | **0.6985±0.0033** | **0.6941±0.0020** | **0.7535±0.0009** | **0.0044±0.0018** | 854 |
| C1.1 Huber e=1 F5 | 0.7173±0.0065 | 0.6962±0.0059 | 0.7543±0.0026 | 0.0212±0.0124 | 124871 |
| plain last e=1 F2 | 0.7263±0.0231 | 0.7046±0.0059 | 0.7650±0.0091 | 0.0217±0.0264 | 68871 |
| C1.1 Huber e=2 F5 | 0.7349±0.0102 | 0.7156±0.0092 | 0.7956±0.0374 | 0.0192±0.0120 | 191623 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-baselines-c1.log`
- mu_paper: `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_paper_2015-01-01_2021-12-31.npy`

## Conclusions for Dev
1. Paper DLinear F1 last/test = **0.6985±0.0033 / 0.7535±0.0009**. New scoreboard; **does not transfer** from dev 0.7272 / 0.6739. Paper test is harder than val (test > last_val). Winner by last_val is DLinear.
2. Plain last e=1 F2 does **not** keep the dev test edge. last_val 0.7263 (worse than DLinear 0.6985) and test 0.7650 (worse than 0.7535). Seed-2 last_val 0.7514 is noisy.
3. C1.1 e=2 does **not** beat e=1 at this scale: last 0.7349 vs 0.7173, test 0.7956 vs 0.7543 (e=2 seed-2 test 0.8382). One layer still better. Closest TimeXer to DLinear is C1.1 Huber e=1 (last gap 0.0188, test 0.7543 ≈ DLinear 0.7535).
4. CUDA, `num_workers=4`, precache **ran** (~28 min). Window counts logged. Missing price files UNH/VZ skipped. Did not run Model A / B / C0 / flatten / `ticker_set=dev`. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
