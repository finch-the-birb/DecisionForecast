# Report: paper-timexer-3arch-sweep

- authored_by: runner
- created_at: 2026-10-08T12:11:32Z
- updated_at: 2026-10-08T17:46:53Z
- request_folder: agent-handoff/2026-10-08_1500_paper-timexer-3arch-sweep/
- tested_ref: feat/phase2-timexer@298dbf8fd45266ec90d2517bec000bc51395905f
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Final harvest after SWEEP_DONE. No new train.
# data=fnspid_dual_ts, features [close, volume, open, high, low], patch_len=12, patch_stride=6.
# c1_dual was not started. The blocked report in 2026-10-06_2200 was not rewritten.
# Product code is cd1f26edcc45f45d758914d1641ef4ff44caa887. HEAD at launch was the blocked-report commit.
```

## Outcome
- exit_code: 0; SWEEP_DONE at 2026-10-08T17:00:58Z; 36/36 DONE; no Traceback
- host: runpod pod 9hj7blo0fqus3p (hostname 279d25bcf0b8)
- device: all 36 jobs train device cuda, DataLoader num_workers=4. No OOM.
- pid: 43683 (exited after SWEEP_DONE)
- log: outputs/paper-timexer-3arch-sweep.log
- wall_clock: 2026-10-08T12:11:11Z to 2026-10-08T17:00:58Z
- block_done: Block 1 13:39:03Z; Block 2 15:37:16Z; Block 3 17:00:58Z
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 5), text_seq=(32, 60, 15), ts=(32, 60, 25)
- skipped price files: UNH, VZ
- patch_stride: 6

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Means are n=3, sample std.

### Table 1. Per seed

| model | e_layers | use_proto | seed | stop_ep | train | val | test | x shape | text_seq shape | ts shape | n_params | best_val_mse | test_mse | test_mae |
| c1_factored | 1 | false | 0 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 88895 | 0.7457 | 0.7628 | 0.6009 |
| c1_factored | 1 | false | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 88895 | 0.7264 | 0.7577 | 0.5975 |
| c1_factored | 1 | false | 2 | 12 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 88895 | 0.7440 | 0.7595 | 0.5972 |
| c1_factored | 1 | true | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 93695 | 0.7071 | 0.7687 | 0.6035 |
| c1_factored | 1 | true | 1 | 16 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 93695 | 0.7027 | 0.7851 | 0.6165 |
| c1_factored | 1 | true | 2 | 20 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 93695 | 0.7084 | 0.7629 | 0.6016 |
| c1_factored | 2 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 172415 | 0.7240 | 0.7692 | 0.6028 |
| c1_factored | 2 | false | 1 | 8 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 172415 | 0.7459 | 0.7575 | 0.5978 |
| c1_factored | 2 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 172415 | 0.7362 | 0.7587 | 0.5966 |
| c1_factored | 2 | true | 0 | 15 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 177215 | 0.7082 | 0.7511 | 0.5949 |
| c1_factored | 2 | true | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 177215 | 0.7235 | 0.7571 | 0.5981 |
| c1_factored | 2 | true | 2 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 177215 | 0.7521 | 0.8253 | 0.6353 |
| c1_inverted | 1 | false | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 93191 | 0.7340 | 0.7626 | 0.6010 |
| c1_inverted | 1 | false | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 93191 | 0.7338 | 0.7678 | 0.6014 |
| c1_inverted | 1 | false | 2 | 10 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 93191 | 0.7195 | 0.7615 | 0.6006 |
| c1_inverted | 1 | true | 0 | 16 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 97991 | 0.7136 | 0.7657 | 0.6022 |
| c1_inverted | 1 | true | 1 | 20 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 97991 | 0.7090 | 0.7520 | 0.5935 |
| c1_inverted | 1 | true | 2 | 11 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 97991 | 0.7213 | 0.7636 | 0.6013 |
| c1_inverted | 2 | false | 0 | 10 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 176711 | 0.7325 | 0.7693 | 0.6022 |
| c1_inverted | 2 | false | 1 | 10 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 176711 | 0.7385 | 0.7850 | 0.6173 |
| c1_inverted | 2 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 176711 | 0.7342 | 0.7841 | 0.6152 |
| c1_inverted | 2 | true | 0 | 17 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 181511 | 0.7265 | 0.7667 | 0.6012 |
| c1_inverted | 2 | true | 1 | 12 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 181511 | 0.7196 | 0.7720 | 0.6077 |
| c1_inverted | 2 | true | 2 | 16 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 181511 | 0.7090 | 0.7758 | 0.6081 |
| c1_late_fusion | 1 | false | 0 | 6 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 95303 | 0.7299 | 0.7766 | 0.6080 |
| c1_late_fusion | 1 | false | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 95303 | 0.7306 | 0.7685 | 0.6029 |
| c1_late_fusion | 1 | false | 2 | 6 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 95303 | 0.7197 | 0.7750 | 0.6073 |
| c1_late_fusion | 1 | true | 0 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 100103 | 0.7284 | 0.7657 | 0.5999 |
| c1_late_fusion | 1 | true | 1 | 17 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 100103 | 0.7093 | 0.7805 | 0.6115 |
| c1_late_fusion | 1 | true | 2 | 16 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 100103 | 0.7079 | 0.7555 | 0.5953 |
| c1_late_fusion | 2 | false | 0 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 178823 | 0.7279 | 0.7599 | 0.5977 |
| c1_late_fusion | 2 | false | 1 | 9 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 178823 | 0.7298 | 0.7566 | 0.5959 |
| c1_late_fusion | 2 | false | 2 | 8 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 178823 | 0.7259 | 0.7567 | 0.5953 |
| c1_late_fusion | 2 | true | 0 | 9 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 183623 | 0.7176 | 0.7702 | 0.6031 |
| c1_late_fusion | 2 | true | 1 | 7 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 183623 | 0.7165 | 0.7753 | 0.6055 |
| c1_late_fusion | 2 | true | 2 | 12 | 98116 | 12048 | 11904 | (32, 60, 5) | (32, 60, 15) | (32, 60, 25) | 183623 | 0.7169 | 0.7516 | 0.5937 |

### Table 2. Mean ± std (n=3)

| model | e_layers | use_proto | n_params | best_val_mse | test_mse | test_mae |
| c1_factored | 1 | false | 88895 | 0.7387±0.0107 | 0.7600±0.0026 | 0.5985±0.0021 |
| c1_factored | 1 | true | 93695 | 0.7061±0.0030 | 0.7722±0.0115 | 0.6072±0.0081 |
| c1_factored | 2 | false | 172415 | 0.7354±0.0110 | 0.7618±0.0064 | 0.5991±0.0033 |
| c1_factored | 2 | true | 177215 | 0.7279±0.0223 | 0.7778±0.0412 | 0.6094±0.0225 |
| c1_inverted | 1 | false | 93191 | 0.7291±0.0083 | 0.7640±0.0034 | 0.6010±0.0004 |
| c1_inverted | 1 | true | 97991 | 0.7146±0.0062 | 0.7604±0.0074 | 0.5990±0.0048 |
| c1_inverted | 2 | false | 176711 | 0.7351±0.0031 | 0.7795±0.0088 | 0.6116±0.0082 |
| c1_inverted | 2 | true | 181511 | 0.7184±0.0088 | 0.7715±0.0046 | 0.6057±0.0039 |
| c1_late_fusion | 1 | false | 95303 | 0.7267±0.0061 | 0.7734±0.0043 | 0.6061±0.0028 |
| c1_late_fusion | 1 | true | 100103 | 0.7152±0.0115 | 0.7672±0.0126 | 0.6022±0.0083 |
| c1_late_fusion | 2 | false | 178823 | 0.7279±0.0020 | 0.7577±0.0019 | 0.5963±0.0012 |
| c1_late_fusion | 2 | true | 183623 | 0.7170±0.0006 | 0.7657±0.0125 | 0.6008±0.0062 |

Stride-12 references, not rerun: DLinear F1 0.7535±0.0009; dual F2 e=1 no prototypes 0.7655±0.0032; best dual-fft mean (stride 12, d_model 64, e=1, no prototypes) 0.7647±0.0028; active-ts F5 e=1 no prototypes 0.7690±0.0144. This sweep uses patch_stride=6.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-3arch-sweep.log

## Conclusions for Dev
1. 36/36 finished on CUDA. Shapes are x=(32, 60, 5), text_seq=(32, 60, 15), ts=(32, 60, 25). No OOM and no traceback. patch_stride is 6. The stride-12 references below were not rerun.
2. No complete cell beats DLinear F1 0.7535±0.0009. The best mean is late fusion, e_layers=2, no prototypes: test 0.7577±0.0019, mae 0.5963±0.0012, n_params 178823. Gap to DLinear is +0.0042. That cell is under dual F2 e=1 0.7655±0.0032 by 0.0078, under the best dual-fft mean 0.7647±0.0028 by 0.0070, and under active-ts F5 e=1 0.7690±0.0144 by 0.0113.
3. Next tight cells are factored e=1 without prototypes, 0.7600±0.0026 at 88895 params, and inverted e=1 with prototypes, 0.7604±0.0074 at 97991 params. Extra depth helps late fusion (e=1 no proto 0.7734±0.0043 versus e=2 0.7577±0.0019) and does not help factored or inverted without prototypes.
4. Prototypes lower best_val in every pair and do not lower the 3-seed test mean. The closest single seeds are factored e=2 proto seed=0 at 0.7511 and late fusion e=2 proto seed=2 at 0.7516. The factored cell's other seed is 0.8253, so its mean is 0.7778±0.0412.
