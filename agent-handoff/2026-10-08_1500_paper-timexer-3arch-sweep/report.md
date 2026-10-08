# Report: paper-timexer-3arch-sweep

- authored_by: runner
- created_at: 2026-10-08T12:11:32Z
- updated_at: 2026-10-08T14:59:08Z
- request_folder: agent-handoff/2026-10-08_1500_paper-timexer-3arch-sweep/
- tested_ref: feat/phase2-timexer@298dbf8fd45266ec90d2517bec000bc51395905f
- status: partial (Block 2/3, 21/36 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No prepare. Status harvest while PID 43683 is alive. No new train.
# data=fnspid_dual_ts, features [close, volume, open, high, low], patch_len=12, patch_stride=6.
# c1_dual was not started. The blocked report in 2026-10-06_2200 was not rewritten.
```

## Outcome
- exit_code: in progress; Block 1 DONE at 2026-10-08T13:39:03Z; 21/36 DONE; SWEEP_DONE absent; no Traceback
- host: runpod pod 9hj7blo0fqus3p (hostname 279d25bcf0b8)
- device: all 21 finished jobs train device cuda, DataLoader num_workers=4. No OOM. The running job is also on cuda.
- pid: 43683
- log: outputs/paper-timexer-3arch-sweep.log
- started_at: 2026-10-08T12:11:11Z
- current_job: Block 2 c1_inverted e_layers=2 use_prototypes=true n_prototypes=10 seed=0 (started 2026-10-08T14:50:04Z)
- splits: train 98116 / val 12048 / test 11904
- first-batch shapes: x=(32, 60, 5), text_seq=(32, 60, 15), ts=(32, 60, 25)
- skipped price files: UNH, VZ
- patch_stride: 6

best_val_mse = min Epoch val_mse. stop_ep = early-stop epoch, else last logged epoch. test_* = METRICS_ROW. Means are n=3, sample std. Incomplete cells are omitted.

### Finished jobs

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

### Means

| model | e_layers | use_proto | n | n_params | best_val_mse | test_mse | test_mae |
| c1_factored | 1 | false | 3 | 88895 | 0.7387±0.0107 | 0.7600±0.0026 | 0.5985±0.0021 |
| c1_factored | 1 | true | 3 | 93695 | 0.7061±0.0030 | 0.7722±0.0115 | 0.6072±0.0081 |
| c1_factored | 2 | false | 3 | 172415 | 0.7354±0.0110 | 0.7618±0.0064 | 0.5991±0.0033 |
| c1_factored | 2 | true | 3 | 177215 | 0.7279±0.0223 | 0.7778±0.0412 | 0.6094±0.0225 |
| c1_inverted | 1 | false | 3 | 93191 | 0.7291±0.0083 | 0.7640±0.0034 | 0.6010±0.0004 |
| c1_inverted | 1 | true | 3 | 97991 | 0.7146±0.0062 | 0.7604±0.0074 | 0.5990±0.0048 |
| c1_inverted | 2 | false | 3 | 176711 | 0.7351±0.0031 | 0.7795±0.0088 | 0.6116±0.0082 |

Stride-12 references, not rerun: DLinear F1 0.7535±0.0009; dual F2 e=1 no prototypes 0.7655±0.0032; best dual-fft mean (stride 12, d_model 64, e=1, no prototypes) 0.7647±0.0028; active-ts F5 e=1 no prototypes 0.7690±0.0144. This sweep uses patch_stride=6.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-3arch-sweep.log

## Conclusions for Dev
1. Block 1 is complete. 21/36 overall. Inverted e=1 is complete both with and without prototypes. Inverted e=2 without prototypes is complete. Inverted e=2 with prototypes has seed 0 running. First-batch shapes stay x=(32, 60, 5), text_seq=(32, 60, 15), ts=(32, 60, 25).
2. Best complete mean is still factored e=1 without prototypes: test 0.7600±0.0026, mae 0.5985±0.0021, n_params 88895. Gap to DLinear F1 0.7535±0.0009 is +0.0065. Inverted e=1 with prototypes is next at 0.7604±0.0074 (97991 params). Its seed 1 is 0.7520, the closest single seed so far.
3. Inverted e=2 without prototypes is worse: test 0.7795±0.0088 at 176711 params. Factored prototypes still raise mean test. Factored e=2 with prototypes remains 0.7778±0.0412 because seed 2 is 0.8253.
