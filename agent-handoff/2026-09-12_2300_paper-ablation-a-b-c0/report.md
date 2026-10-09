# Report: paper-ablation-a-b-c0

- authored_by: runner
- created_at: 2026-09-12T22:04:27Z
- request_folder: agent-handoff/2026-09-12_2300_paper-ablation-a-b-c0/
- tested_ref: feat/phase2-timexer@98a3459d15e3c19cb0ba411948a8421740c4857f (contains 033996f; harvest from d99a4b1 started report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No precache. ticker_set=paper. No DLinear / plain / C1 / flatten / ticker_set=dev.
# Human ping: PID 689469 dead, log has SWEEP_DONE 9/9. Harvest only — no new train.
```

## Outcome
- exit_code: 9/9 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 689469 gone; no Traceback
- duration: ~1h 30m 46s (2026-09-12T20:01:14Z → 2026-09-12T21:32:00Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda` and `DataLoader num_workers=4` on **all 9**. `First batch tensors on x=cuda:0 text=cuda:0`. NVIDIA L4. No OOM. Not `ticker_set=dev`.
- Precache: **not run**. `mu_paper` already on disk.
- Paper splits (all 9 jobs identical): **train 98236 / val 12048 / test 11904**. First-batch `x=(32, 60, 5)`. UNH/VZ skipped (expected).

GPU / loader (verbatim, first job A seed=0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
First batch shapes x=(32, 60, 5) text=(32, 768) text_seq=(32, 60, 768)
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

Dataset sizes + TEXT_COVERAGE (verbatim, first job; same sizes on all 9):
```
Dataset sizes — train: 98236, val: 12048, test: 11904
TEXT_COVERAGE split=train windows=98236 mean_has_news_frac=0.585 zero_windows=0.223 mean_text_l2=0.2742
TEXT_COVERAGE split=val windows=12048 mean_has_news_frac=0.706 zero_windows=0.197 mean_text_l2=0.3047
TEXT_COVERAGE split=test windows=11904 mean_has_news_frac=0.764 zero_windows=0.173 mean_text_l2=0.3417
```

`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; `overfit_gap` = last_val − best_val; test = `METRICS_ROW` (best.pt). Means n=3, sample std. Pick on mean last_val. This sweep only.

n_params: A 127687; B 129799; C0 124871.

### Table 1 — per-seed

| label | seed | stop_ep | last_val | best_val | overfit_gap | test_mse |
| A late CNN proto=10 | 0 | 11 | 0.7856 | 0.7514 | 0.0342 | 0.8192 |
| A late CNN proto=10 | 1 | 12 | 0.7936 | 0.7360 | 0.0576 | 0.8058 |
| A late CNN proto=10 | 2 | 11 | 0.7817 | 0.7457 | 0.0360 | 0.7931 |
| B late e=1 last proto=10 | 0 | 7 | 0.7246 | 0.7212 | 0.0034 | 0.7706 |
| B late e=1 last proto=10 | 1 | 8 | 0.7350 | 0.7268 | 0.0082 | 0.7627 |
| B late e=1 last proto=10 | 2 | 6 | 0.7354 | 0.7208 | 0.0146 | 0.7729 |
| C0 mid-add e=1 last proto=5 | 0 | 12 | 0.7801 | 0.7170 | 0.0631 | 0.7699 |
| C0 mid-add e=1 last proto=5 | 1 | 7 | 0.7626 | 0.7137 | 0.0489 | 0.7604 |
| C0 mid-add e=1 last proto=5 | 2 | 8 | 0.7261 | 0.7185 | 0.0076 | 0.7762 |

### Table 2 — means n=3

| label | last_val | best_val | test_mse | overfit_gap | n_params |
| B late e=1 last proto=10 | 0.7317±0.0061 | 0.7229±0.0034 | 0.7687±0.0054 | 0.0087±0.0056 | 129799 |
| C0 mid-add e=1 last proto=5 | 0.7563±0.0276 | 0.7164±0.0025 | 0.7688±0.0080 | 0.0399±0.0288 | 124871 |
| A late CNN proto=10 | 0.7870±0.0061 | 0.7444±0.0078 | 0.8060±0.0131 | 0.0426±0.0130 | 127687 |

### Table 3 — paper scoreboard (61 tickers)

| label | last_val | test_mse | source |
| DLinear F1 | 0.6985±0.0033 | 0.7535±0.0009 | paper-baselines-c1 |
| TimeXer plain e=1 F2 | 0.7263±0.0231 | 0.7650±0.0091 | paper-baselines-c1 |
| Model A late CNN | 0.7870±0.0061 | 0.8060±0.0131 | this |
| Model B late e=1 | 0.7317±0.0061 | 0.7687±0.0054 | this |
| Model C0 mid-add e=1 | 0.7563±0.0276 | 0.7688±0.0080 | this |
| Model C1.1 Huber e=1 F5 | 0.7173±0.0065 | 0.7543±0.0026 | paper-baselines-c1 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-ablation-a-b-c0.log`

## Conclusions for Dev
1. **H1 paper last_val: B wins.** B 0.7317 vs A 0.7870 (and B test 0.7687 vs A 0.8060). TimeXer late beats TimeXL CNN on this protocol.
2. **H2 paper last_val: C1.1 still best among fusions** (locked 0.7173). B late 0.7317 beats C0 mid-add 0.7563. Mid-add does **not** beat late; cross-attn C1 beats both.
3. **None of A/B/C0 beat DLinear** last 0.6985 or test 0.7535. Closest this-run last_val is B (gap 0.0332). C1.1 remains the closest TimeXer overall (last 0.7173 / test 0.7543).
4. CUDA, `num_workers=4`, **no precache**. Splits match 98236/12048/11904. `x=(32, 60, 5)`. Did not run DLinear / plain / C1 / flatten / `ticker_set=dev`. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
