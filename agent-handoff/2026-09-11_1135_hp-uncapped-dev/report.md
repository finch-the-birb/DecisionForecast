# Report: hp-uncapped-dev

- authored_by: runner
- created_at: 2026-09-11T13:05:00Z
- request_folder: agent-handoff/2026-09-11_1135_hp-uncapped-dev/
- tested_ref: feat/phase2-timexer@b082360d67c87434852e1ff06dd3a54fa09d3f28 (contains 79d2ff7)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not ablation_table / dlinear / extra seeds / d_model.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=b082360
uv sync
# MU already present; skipped precache
nohup bash -lc '... a,b,c0,c1 × lr=0.0003,0.001 ...' > outputs/hp-uncapped-dev.log 2>&1 &
# PID=26353 started 2026-09-11T11:47:07Z; harvest on human ping after SWEEP_DONE
```

## Outcome
- exit_code: uv sync=0; 8/8 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 26353 gone; no Traceback
- duration: ~56m 36s (11:47:07Z → 12:43:43Z)
- host: runpod pod `fprbi35eab7xe4`
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 8). NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seed=42, H=7, epochs=20, patience=5. `normalize=per_window`, `train_start=2015-01-01`.

GPU check (verbatim, first job a/0.0003):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
```

Early stop (patience=5): a/0.0003 ep8; a/0.001 ep7; b/0.0003 ep7; b/0.001 ep7; c0/0.0003 ep20 (full budget); c0/0.001 ep10; c1/0.0003 ep6; c1/0.001 ep12.

## Key signals
METRICS_ROW (verbatim, 8/8):
```
METRICS_ROW model=a horizon=7 mse=0.6940 mae=0.5960 mae_denorm=5.3979
METRICS_ROW model=a horizon=7 mse=0.7181 mae=0.6096 mae_denorm=5.6766
METRICS_ROW model=b horizon=7 mse=0.7200 mae=0.6062 mae_denorm=5.6011
METRICS_ROW model=b horizon=7 mse=0.8523 mae=0.6772 mae_denorm=6.2076
METRICS_ROW model=c0 horizon=7 mse=0.7156 mae=0.6123 mae_denorm=5.7182
METRICS_ROW model=c0 horizon=7 mse=0.7160 mae=0.6126 mae_denorm=5.6780
METRICS_ROW model=c1 horizon=7 mse=0.7664 mae=0.6326 mae_denorm=5.8723
METRICS_ROW model=c1 horizon=7 mse=0.7099 mae=0.5961 mae_denorm=5.3423
```

n_params (verbatim): a=144199; b=213063; c0=200711; c1=200711.

Last-epoch val_mse from logs (not json). Winner = **best last-epoch val_mse**: **c0 / lr=0.0003** (0.7426). Test recorded, not used to pick HP. This sweep only — do not mix R3 / cap-sweep / ablation_table.

| model | lr | val_mse | test_mse | test_mae | mae_denorm | n_params |
| a | 0.0003 | 0.8071 | 0.6940 | 0.5960 | 5.3979 | 144199 |
| a | 0.001 | 0.8219 | 0.7181 | 0.6096 | 5.6766 | 144199 |
| b | 0.0003 | 0.7900 | 0.7200 | 0.6062 | 5.6011 | 213063 |
| b | 0.001 | 0.9236 | 0.8523 | 0.6772 | 6.2076 | 213063 |
| **c0** | **0.0003** | **0.7426** | **0.7156** | **0.6123** | **5.7182** | **200711** |
| c0 | 0.001 | 0.9346 | 0.7160 | 0.6126 | 5.6780 | 200711 |
| c1 | 0.0003 | 0.8159 | 0.7664 | 0.6326 | 5.8723 | 200711 |
| c1 | 0.001 | 1.0319 | 0.7099 | 0.5961 | 5.3423 | 200711 |

json (test): a/0.0003 mse=0.6940266; a/0.001 mse=0.7180544; b/0.0003 mse=0.7199954; b/0.001 mse=0.8522627; c0/0.0003 mse=0.7155820; c0/0.001 mse=0.7160112; c1/0.0003 mse=0.7663580; c1/0.001 mse=0.7099327.

Val vs test same order on all 8 (no 5–18 test). Lowest **test_mse** is a/0.0003 (0.6940) — not the val winner.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-uncapped-dev.log`
- a/0.0003: `outputs/2026-09-11/11-47-48/`
- a/0.001: `outputs/2026-09-11/11-53-29/`
- b/0.0003: `outputs/2026-09-11/11-58-49/`
- b/0.001: `outputs/2026-09-11/12-04-32/`
- c0/0.0003: `outputs/2026-09-11/12-10-20/`
- c0/0.001: `outputs/2026-09-11/12-22-58/`
- c1/0.0003: `outputs/2026-09-11/12-30-07/`
- c1/0.001: `outputs/2026-09-11/12-35-29/`

## Conclusions for Dev
1. Uncapped HP sweep **pass**: 8/8 CUDA, full table. Best **val_mse** is **C0, lr=0.0003** (0.7426, full 20 epochs). Capped 256-window sweep is not this decision.
2. lr=0.001 is worse on last-epoch val for every model in this grid.
3. Did not run ablation_table / dlinear / paper / extra seeds / d_model / caps. Did not start another timer.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
