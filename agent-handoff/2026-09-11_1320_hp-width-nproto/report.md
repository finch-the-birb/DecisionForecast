# Report: hp-width-nproto

- authored_by: runner
- created_at: 2026-09-11T13:52:40Z
- request_folder: agent-handoff/2026-09-11_1320_hp-width-nproto/
- tested_ref: feat/phase2-timexer@069922e757685e5695bffdc9271106d49c2fc49c (contains 03f123b)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not ablation_table / dlinear / lr=0.001 / cartesian d_model×d_ff.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=069922e (contains 03f123b)
uv sync
# MU already present; skipped precache
nohup bash -lc '... a n_prototypes=5,10; b,c0,c1 × (64,256)|(128,512); lr=0.0003 num_workers=4 ...' > outputs/hp-width-nproto.log 2>&1 &
# PID=55728 started 2026-09-11T13:19:20Z; harvest on human ping after SWEEP_DONE
```

## Outcome
- exit_code: uv sync=0; 8/8 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 55728 gone; no Traceback
- duration: ~27m 27s (13:19:20Z → 13:46:47Z)
- host: runpod pod `fprbi35eab7xe4`
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 8). NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seed=42, H=7, lr=0.0003, epochs=20, patience=5. `normalize=per_window`, `train_start=2015-01-01`. `num_workers=4`.

GPU / loader (verbatim, first job a/n_prototypes=5):
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

Early stop: a/5 ep14; a/10 ep11; b 64/256 ep8; b 128/512 ep6; c0 64/256 ep15; c0 128/512 ep8; c1 64/256 ep6; c1 128/512 ep8.

## Key signals
METRICS_ROW (verbatim, 8/8):
```
METRICS_ROW model=a horizon=7 mse=0.8446 mae=0.6642 mae_denorm=6.4001
METRICS_ROW model=a horizon=7 mse=0.7148 mae=0.6079 mae_denorm=5.5304
METRICS_ROW model=b horizon=7 mse=0.7116 mae=0.6005 mae_denorm=5.4619
METRICS_ROW model=b horizon=7 mse=0.6931 mae=0.5972 mae_denorm=5.3913
METRICS_ROW model=c0 horizon=7 mse=0.6988 mae=0.5997 mae_denorm=5.6953
METRICS_ROW model=c0 horizon=7 mse=0.7170 mae=0.6036 mae_denorm=5.6071
METRICS_ROW model=c1 horizon=7 mse=0.7640 mae=0.6311 mae_denorm=5.8638
METRICS_ROW model=c1 horizon=7 mse=0.7169 mae=0.6021 mae_denorm=5.5036
```

n_params (verbatim): a/5=143879; a/10=144199; b 64=213063; b 128=646215; c0 64=200711; c0 128=670727; c1 64=200711; c1 128=670727.

Last-epoch val_mse from logs (not json). Winner = **best last-epoch val_mse**: **b / d_model=64 / d_ff=256 / n_prototypes=10** (0.8063). Test recorded, not used to pick HP. This sweep only — do not mix hp-uncapped-dev / cap-sweep / ablation_table (`num_workers=4` changes shuffle).

| model | d_model | d_ff | n_prototypes | val_mse | test_mse | test_mae | mae_denorm | n_params |
| a | 64 | n/a | 5 | 1.1576 | 0.8446 | 0.6642 | 6.4001 | 143879 |
| a | 64 | n/a | 10 | 0.9292 | 0.7148 | 0.6079 | 5.5304 | 144199 |
| **b** | **64** | **256** | **10** | **0.8063** | **0.7116** | **0.6005** | **5.4619** | **213063** |
| b | 128 | 512 | 10 | 0.8503 | 0.6931 | 0.5972 | 5.3913 | 646215 |
| c0 | 64 | 256 | 10 | 0.8592 | 0.6988 | 0.5997 | 5.6953 | 200711 |
| c0 | 128 | 512 | 10 | 0.8809 | 0.7170 | 0.6036 | 5.6071 | 670727 |
| c1 | 64 | 256 | 10 | 0.9021 | 0.7640 | 0.6311 | 5.8638 | 200711 |
| c1 | 128 | 512 | 10 | 0.9902 | 0.7169 | 0.6021 | 5.5036 | 670727 |

json (test): a/5 mse=0.8446; a/10 mse=0.7148; b 64 mse=0.7116; b 128 mse=0.6931; c0 64 mse=0.6988; c0 128 mse=0.7170; c1 64 mse=0.7640; c1 128 mse=0.7169.

Val vs test same order on all 8. Lowest **test_mse** is b 128/512 (0.6931) — not the val winner. Wider `(128, 512)` is worse last-epoch val on b/c0/c1 vs `(64, 256)`. A n_prototypes=5 is worse val than 10.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-width-nproto.log`
- a n=5: `outputs/2026-09-11/13-20-03/`
- a n=10: `outputs/2026-09-11/13-23-27/`
- b 64/256: `outputs/2026-09-11/13-26-21/`
- b 128/512: `outputs/2026-09-11/13-29-36/`
- c0 64/256: `outputs/2026-09-11/13-32-29/`
- c0 128/512: `outputs/2026-09-11/13-37-11/`
- c1 64/256: `outputs/2026-09-11/13-40-37/`
- c1 128/512: `outputs/2026-09-11/13-43-47/`

## Conclusions for Dev
1. Uncapped width/n_proto sweep **pass**: 8/8 CUDA, `num_workers=4`, coverage printed, full table. Best **val_mse** is **B, d_model=64, d_ff=256** (0.8063).
2. Paired width 128/512 did not beat 64/256 on last-epoch val. A n_prototypes=10 beats 5.
3. Did not run ablation_table / dlinear / paper / lr=0.001 / unpaired d_model×d_ff / caps. Did not start another timer.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
