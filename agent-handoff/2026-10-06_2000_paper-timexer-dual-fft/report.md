# Report: paper-timexer-dual-fft

- authored_by: runner
- created_at: 2026-10-06T17:34:45Z
- updated_at: 2026-10-07T00:29:46Z
- request_folder: agent-handoff/2026-10-06_2000_paper-timexer-dual-fft/
- tested_ref: feat/phase2-timexer@33e4a41d2493381cbc72c88635a3a1ebf09641e1
- status: partial (Block 3/4, 32/48 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Timer harvest while PID 2777 is alive. No new train.
```

## Outcome
- exit_code: in progress; Block 1 DONE at 2026-10-06T20:19:01Z; Block 2 DONE at 2026-10-06T22:59:52Z; 32/48 DONE; `SWEEP_DONE` absent; no Traceback
- host: runpod pod `sw7txpqcwtk7tz` (hostname `86df0010ddba`)
- device: all 32 finished jobs `train device: cuda`, `DataLoader num_workers=4`. No OOM.
- pid: 2777
- log: `outputs/paper-timexer-dual-fft.log`
- current_job: Block 3, stride=12, d_model=64, e_layers=2, use_prototypes=false, seed=2 (started 2026-10-07T00:22:24Z)
- splits: train 98236 / val 12048 / test 11904
- first-batch shapes: `x=(32, 60, 5)`, `text_seq=(32, 60, 15)`
- skipped price files: UNH, VZ

`best_val_mse` = min Epoch `val_mse`. `stop_ep` = early-stop epoch, else last logged epoch. `test_*` = `METRICS_ROW`.

### Finished jobs

| stride | d_model | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | n_params | best_val_mse | test_mse | test_mae |
| 6 | 64 | 1 | false | 0 | 9 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7252 | 0.7636 | 0.6002 |
| 6 | 64 | 1 | false | 1 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7179 | 0.7718 | 0.6049 |
| 6 | 64 | 1 | false | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7322 | 0.7641 | 0.6008 |
| 6 | 64 | 1 | true | 0 | 15 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7037 | 0.7583 | 0.5977 |
| 6 | 64 | 1 | true | 1 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7134 | 0.7850 | 0.6167 |
| 6 | 64 | 1 | true | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7201 | 0.7715 | 0.6040 |
| 6 | 64 | 2 | false | 0 | 13 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7293 | 0.7768 | 0.6053 |
| 6 | 64 | 2 | false | 1 | 6 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7249 | 0.7661 | 0.6003 |
| 6 | 64 | 2 | false | 2 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7345 | 0.7779 | 0.6074 |
| 6 | 64 | 2 | true | 0 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 244491 | 0.7241 | 0.7642 | 0.6014 |
| 6 | 64 | 2 | true | 1 | 11 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 244491 | 0.7113 | 0.7825 | 0.6142 |
| 6 | 64 | 2 | true | 2 | 16 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 244491 | 0.7130 | 0.7728 | 0.6096 |
| 6 | 128 | 1 | false | 0 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 474889 | 0.7357 | 0.7680 | 0.6018 |
| 6 | 128 | 1 | false | 1 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 474889 | 0.7400 | 0.7665 | 0.6011 |
| 6 | 128 | 1 | false | 2 | 6 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 474889 | 0.7015 | 0.7705 | 0.6026 |
| 6 | 128 | 1 | true | 0 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 492681 | 0.7101 | 0.7967 | 0.6164 |
| 6 | 128 | 1 | true | 1 | 20 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 492681 | 0.7031 | 0.7555 | 0.5957 |
| 6 | 128 | 1 | true | 2 | 9 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 492681 | 0.7153 | 0.7773 | 0.6077 |
| 6 | 128 | 2 | false | 0 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 938123 | 0.7324 | 0.7859 | 0.6129 |
| 6 | 128 | 2 | false | 1 | 6 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 938123 | 0.7065 | 0.7882 | 0.6142 |
| 6 | 128 | 2 | false | 2 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 938123 | 0.7511 | 0.7773 | 0.6067 |
| 6 | 128 | 2 | true | 0 | 20 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 955915 | 0.7026 | 0.7622 | 0.5990 |
| 6 | 128 | 2 | true | 1 | 11 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 955915 | 0.7022 | 0.7962 | 0.6218 |
| 6 | 128 | 2 | true | 2 | 20 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 955915 | 0.7080 | 0.7723 | 0.6043 |
| 12 | 64 | 1 | false | 0 | 7 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7175 | 0.7615 | 0.5982 |
| 12 | 64 | 1 | false | 1 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7140 | 0.7667 | 0.6010 |
| 12 | 64 | 1 | false | 2 | 10 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 122761 | 0.7313 | 0.7658 | 0.6024 |
| 12 | 64 | 1 | true | 0 | 9 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7094 | 0.7878 | 0.6163 |
| 12 | 64 | 1 | true | 1 | 12 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7023 | 0.7632 | 0.6017 |
| 12 | 64 | 1 | true | 2 | 17 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 127561 | 0.7059 | 0.7600 | 0.5966 |
| 12 | 64 | 2 | false | 0 | 8 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7246 | 0.7634 | 0.5988 |
| 12 | 64 | 2 | false | 1 | 6 | 98236 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | 239691 | 0.7228 | 0.7648 | 0.6001 |

Means n=3, sample std. Incomplete configs are omitted (Block 3, e=2, no prototypes: 2/3 seeds).

| stride | d_model | e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| 6 | 64 | 1 | false | 122761 | 0.7251±0.0072 | 0.7665±0.0046 | 0.6020±0.0026 |
| 6 | 64 | 1 | true | 127561 | 0.7124±0.0082 | 0.7716±0.0134 | 0.6061±0.0097 |
| 6 | 64 | 2 | false | 239691 | 0.7296±0.0048 | 0.7736±0.0065 | 0.6043±0.0036 |
| 6 | 64 | 2 | true | 244491 | 0.7161±0.0070 | 0.7732±0.0092 | 0.6084±0.0065 |
| 6 | 128 | 1 | false | 474889 | 0.7257±0.0211 | 0.7683±0.0020 | 0.6018±0.0008 |
| 6 | 128 | 1 | true | 492681 | 0.7095±0.0061 | 0.7765±0.0206 | 0.6066±0.0104 |
| 6 | 128 | 2 | false | 938123 | 0.7300±0.0224 | 0.7838±0.0057 | 0.6113±0.0040 |
| 6 | 128 | 2 | true | 955915 | 0.7043±0.0032 | 0.7769±0.0175 | 0.6084±0.0119 |
| 12 | 64 | 1 | false | 122761 | 0.7209±0.0091 | 0.7647±0.0028 | 0.6005±0.0021 |
| 12 | 64 | 1 | true | 127561 | 0.7059±0.0036 | 0.7703±0.0152 | 0.6049±0.0102 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-dual-fft.log`

## Conclusions for Dev
1. Block 3 (stride=12, d_model=64) has 8/12 jobs done. 32/48 overall. Current job is e=2, no prototypes, seed=2. Seeds 0 and 1 of that config are already in the per-seed table (test 0.7634 and 0.7648); the mean waits on seed=2.
2. First complete stride=12 config, e=1 without prototypes: test 0.7647±0.0028. That is the best complete mean so far, slightly under stride=6 e=1 no prototypes (0.7665±0.0046). Still above DLinear 0.7535.
3. Stride=12, e=1 with prototypes: test 0.7703±0.0152. Prototypes lower best_val (0.7059 vs 0.7209) and do not lower test. Seed 0 is 0.7878; seeds 1 and 2 are 0.7632 and 0.7600.
4. No complete config beats DLinear test 0.7535. Best single seed remains stride=6, d_model=128, e=1, prototypes, seed=1: test 0.7555.
