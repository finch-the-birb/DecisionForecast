# Report: seed-shortlist

- authored_by: runner
- created_at: 2026-09-11T15:26:30Z
- request_folder: agent-handoff/2026-09-11_1448_seed-shortlist/
- tested_ref: feat/phase2-timexer@87ed084697c4bf9c779b41bbc875853b15161ff5 (contains a93af33, 03f123b; harvest HEAD 197c783 is the started-report commit, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not A / dlinear / ablation_table / seed=42 / lr=0.001 / d_model.
# Sweep was already running from prior nohup (PID 102496). Human ping: PID dead, log has SWEEP_DONE. Harvest only — no new train.
```

## Outcome
- exit_code: 9/9 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 102496 gone; no Traceback
- duration: ~29m 29s (14:50:08Z → 15:19:37Z)
- host: runpod pod `fprbi35eab7xe4`
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 9). NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seeds 0,1,2 (not 42). H=7, lr=0.0003, epochs=20, patience=5. yaml `d_model=64` `d_ff=256` (not overridden). `num_workers=4`.
- Winner = **mean last-epoch val_mse** (n=3): **c0 / n_prototypes=5** (0.8338±0.0473). Test recorded, not used to pick.

GPU / loader (verbatim, first job b/n_prototypes=10/seed=0):
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

Early stop: b/10 s0 ep12; b/10 s1 ep9; b/10 s2 ep8; c0/5 s0 ep7; c0/5 s1 ep13; c0/5 s2 ep9; c1/5 s0 ep7; c1/5 s1 ep7; c1/5 s2 ep9.

## Key signals
This sweep only — last-epoch `val_mse` from the log; test from `METRICS_ROW`. Do not mix hp-nproto-bc0c1 / R3 / seed=42 / ablation_table.

Per-seed:

| model | n_prototypes | seed | val_mse | test_mse | test_mae | mae_denorm |
| b | 10 | 0 | 0.8650 | 0.7491 | 0.6190 | 5.6472 |
| b | 10 | 1 | 0.8846 | 0.7293 | 0.6122 | 5.5836 |
| b | 10 | 2 | 0.8951 | 0.6947 | 0.5947 | 5.4727 |
| c0 | 5 | 0 | 0.8592 | 0.7046 | 0.6001 | 5.5430 |
| c0 | 5 | 1 | 0.8630 | 0.7066 | 0.5987 | 5.5364 |
| c0 | 5 | 2 | 0.7792 | 0.6941 | 0.5896 | 5.4873 |
| c1 | 5 | 0 | 0.9057 | 0.6829 | 0.5876 | 5.3916 |
| c1 | 5 | 1 | 0.9718 | 0.7050 | 0.5980 | 5.4463 |
| c1 | 5 | 2 | 0.8440 | 0.7176 | 0.6000 | 5.4810 |

Means (n=3, sample std; last-epoch val; test from `METRICS_ROW`):

| model | n_prototypes | val_mse mean±std | test_mse mean±std |
| b | 10 | 0.8816±0.0153 | 0.7244±0.0275 |
| **c0** | **5** | **0.8338±0.0473** | **0.7018±0.0067** |
| c1 | 5 | 0.9072±0.0639 | 0.7018±0.0176 |

METRICS_ROW (verbatim, 9/9):
```
METRICS_ROW model=b horizon=7 mse=0.7491 mae=0.6190 mae_denorm=5.6472
METRICS_ROW model=b horizon=7 mse=0.7293 mae=0.6122 mae_denorm=5.5836
METRICS_ROW model=b horizon=7 mse=0.6947 mae=0.5947 mae_denorm=5.4727
METRICS_ROW model=c0 horizon=7 mse=0.7046 mae=0.6001 mae_denorm=5.5430
METRICS_ROW model=c0 horizon=7 mse=0.7066 mae=0.5987 mae_denorm=5.5364
METRICS_ROW model=c0 horizon=7 mse=0.6941 mae=0.5896 mae_denorm=5.4873
METRICS_ROW model=c1 horizon=7 mse=0.6829 mae=0.5876 mae_denorm=5.3916
METRICS_ROW model=c1 horizon=7 mse=0.7050 mae=0.5980 mae_denorm=5.4463
METRICS_ROW model=c1 horizon=7 mse=0.7176 mae=0.6000 mae_denorm=5.4810
```

n_params (verbatim): b/10 = 213063; c0/5 = 200391; c1/5 = 200391.

H3_ROW text_zero (C1 only; record, do not fail on Δ):
```
H3_ROW model=c1 variant=text_zero mse=0.6858 mae=0.5885 delta_mse=+0.0029 delta_mae=+0.0009
H3_ROW model=c1 variant=text_zero mse=0.7056 mae=0.5992 delta_mse=+0.0005 delta_mae=+0.0013
H3_ROW model=c1 variant=text_zero mse=0.7083 mae=0.5947 delta_mse=-0.0093 delta_mae=-0.0054
```

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/seed-shortlist.log`
- b/10 s0: `outputs/2026-09-11/14-50-49/`
- b/10 s1: `outputs/2026-09-11/14-53-51/`
- b/10 s2: `outputs/2026-09-11/14-56-29/`
- c0/5 s0: `outputs/2026-09-11/14-58-58/`
- c0/5 s1: `outputs/2026-09-11/15-01-22/`
- c0/5 s2: `outputs/2026-09-11/15-05-23/`
- c1/5 s0: `outputs/2026-09-11/15-09-03/`
- c1/5 s1: `outputs/2026-09-11/15-12-32/`
- c1/5 s2: `outputs/2026-09-11/15-16-12/`

## Conclusions for Dev
1. Uncapped 3-seed shortlist **pass**: 9/9 CUDA, `num_workers=4`, coverage printed, both tables. Mean-val winner is **C0, n_prototypes=5** (0.8338±0.0473).
2. C1/5 (seed=42 HP val winner) is worst mean val here (0.9072±0.0639). B/10 is more stable (0.8816±0.0153) but higher mean val than C0. Test_mse means: C0 and C1 both 0.7018; B 0.7244. Winner is mean val, not test.
3. Did not run A / dlinear / paper / seed=42 / lr=0.001 / d_model / extra proto / caps / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
