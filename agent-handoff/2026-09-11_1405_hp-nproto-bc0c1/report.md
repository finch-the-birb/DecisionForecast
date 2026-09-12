# Report: hp-nproto-bc0c1

- authored_by: runner
- created_at: 2026-09-11T14:43:20Z
- request_folder: agent-handoff/2026-09-11_1405_hp-nproto-bc0c1/
- tested_ref: feat/phase2-timexer@4ca9a07c0e2e68b658c83a529a4f8d4e7350d194 (contains 02e91e0, 03f123b)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not A / dlinear / ablation_table / lr=0.001 / d_model / d_ff.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=4ca9a07 (contains 02e91e0 and 03f123b)
uv sync
# MU already present; skipped precache
nohup bash -lc '... b,c0,c1 × n_prototypes=5,10,20; lr=0.0003; yaml 64/256; num_workers=4 ...' > outputs/hp-nproto-bc0c1.log 2>&1 &
# PID=79492 started 2026-09-11T14:07:15Z; harvest on human ping after SWEEP_DONE
```

## Outcome
- exit_code: uv sync=0; 9/9 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 79492 gone; no Traceback
- duration: ~30m 54s (14:07:15Z → 14:38:09Z)
- host: runpod pod `fprbi35eab7xe4`
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 9). NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seed=42, H=7, lr=0.0003, epochs=20, patience=5. yaml `d_model=64` `d_ff=256` (not overridden). `num_workers=4`.

GPU / loader (verbatim, first job b/n_prototypes=5):
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

Early stop: b/5 ep7; b/10 ep8; b/20 ep6; c0/5 ep12; c0/10 ep15; c0/20 ep8; c1/5 ep12; c1/10 ep6; c1/20 ep10.

## Key signals
METRICS_ROW (verbatim, 9/9):
```
METRICS_ROW model=b horizon=7 mse=0.7508 mae=0.6213 mae_denorm=5.5426
METRICS_ROW model=b horizon=7 mse=0.7111 mae=0.6005 mae_denorm=5.4745
METRICS_ROW model=b horizon=7 mse=0.6910 mae=0.5867 mae_denorm=5.3531
METRICS_ROW model=c0 horizon=7 mse=0.7225 mae=0.6067 mae_denorm=5.5707
METRICS_ROW model=c0 horizon=7 mse=0.6969 mae=0.5989 mae_denorm=5.6667
METRICS_ROW model=c0 horizon=7 mse=0.7374 mae=0.6157 mae_denorm=5.6681
METRICS_ROW model=c1 horizon=7 mse=0.7263 mae=0.6050 mae_denorm=5.5329
METRICS_ROW model=c1 horizon=7 mse=0.7711 mae=0.6352 mae_denorm=5.8956
METRICS_ROW model=c1 horizon=7 mse=0.7041 mae=0.5954 mae_denorm=5.4252
```

n_params (verbatim): b 5/10/20 = 212743 / 213063 / 213703; c0 5/10/20 = 200391 / 200711 / 201351; c1 5/10/20 = 200391 / 200711 / 201351.

Last-epoch val_mse from logs (not json). Winner = **best last-epoch val_mse**: **c1 / n_prototypes=5** (0.7815). Test recorded, not used to pick HP. This sweep only — do not mix hp-width-nproto / hp-uncapped-dev / ablation_table.

| model | n_prototypes | val_mse | test_mse | test_mae | mae_denorm | n_params |
| b | 5 | 0.8348 | 0.7508 | 0.6213 | 5.5426 | 212743 |
| b | 10 | 0.8206 | 0.7111 | 0.6005 | 5.4745 | 213063 |
| b | 20 | 0.8254 | 0.6910 | 0.5867 | 5.3531 | 213703 |
| c0 | 5 | 0.8182 | 0.7225 | 0.6067 | 5.5707 | 200391 |
| c0 | 10 | 0.8631 | 0.6969 | 0.5989 | 5.6667 | 200711 |
| c0 | 20 | 0.8730 | 0.7374 | 0.6157 | 5.6681 | 201351 |
| **c1** | **5** | **0.7815** | **0.7263** | **0.6050** | **5.5329** | **200391** |
| c1 | 10 | 0.9009 | 0.7711 | 0.6352 | 5.8956 | 200711 |
| c1 | 20 | 0.8473 | 0.7041 | 0.5954 | 5.4252 | 201351 |

json (test): b/5 mse=0.7508; b/10 mse=0.7111; b/20 mse=0.6910; c0/5 mse=0.7225; c0/10 mse=0.6969; c0/20 mse=0.7374; c1/5 mse=0.7263; c1/10 mse=0.7711; c1/20 mse=0.7041.

Val vs test same order on all 9. Lowest **test_mse** is b/20 (0.6910) — not the val winner. Within this log: B best val is proto=10; C0 best val is proto=5; C1 best val is proto=5.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-nproto-bc0c1.log`
- b/5: `outputs/2026-09-11/14-07-53/`
- b/10: `outputs/2026-09-11/14-10-52/`
- b/20: `outputs/2026-09-11/14-14-07/`
- c0/5: `outputs/2026-09-11/14-17-04/`
- c0/10: `outputs/2026-09-11/14-20-59/`
- c0/20: `outputs/2026-09-11/14-25-20/`
- c1/5: `outputs/2026-09-11/14-28-38/`
- c1/10: `outputs/2026-09-11/14-32-46/`
- c1/20: `outputs/2026-09-11/14-35-34/`

## Conclusions for Dev
1. Uncapped n_prototypes sweep **pass**: 9/9 CUDA, `num_workers=4`, coverage printed, full table. Best **val_mse** is **C1, n_prototypes=5** (0.7815).
2. Proto=20 is not the val winner on any of B/C0/C1. Proto=10 wins val only on B in this log.
3. Did not run A / dlinear / paper / lr=0.001 / d_model / d_ff / caps / ablation_table. Did not start another timer.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
