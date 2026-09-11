# Report: hp-sweep-cap

- authored_by: runner
- created_at: 2026-09-11T00:02:20Z
- request_folder: agent-handoff/2026-09-10_2346_hp-sweep-cap/
- tested_ref: feat/phase2-timexer@c43648231a81b627baeb1ceb9e9ea3d844351880 (contains 2c733ae)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not c0/dlinear/timexer_plain. Not paper. Not ablation_table.
# Sweep started with nohup; this turn does not wait on the PID.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
uv sync
# mu_dev_2015-01-01_2021-12-31.npy present
nohup bash -lc 'uv run python -m src.training.train -m model=a,b,c1 train.lr=0.0003,0.001 model.d_model=64,128 … caps 256/64/64'
```

## Outcome
- exit_code: uv sync=0; sweep **started** (not finished)
- duration: n/a (running)
- host: runpod pod `i6lv3n0222en43`
- device: CUDA expected (`train.device=cuda`); GPU check deferred until harvest
- pid: **90833**
- log: `/workspace/DecisionForecast/outputs/hp-sweep-cap.log`
- hydra sweep dir (when Hydra starts): `outputs/multirun/<date>/<time>/`

First push = started. Harvest `METRICS_ROW` / 12-row table after `SWEEP_DONE`. Timer 90 min (`delaySeconds=5400`).

## Key signals
- metrics: n/a (sweep in progress)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-sweep-cap.log`
- mlflow_run_id: n/a

## Conclusions for Dev
1. HP sweep (a/b/c1 × lr × d_model, 256/64/64, seed 42) launched under nohup PID 90833. Not paper.
2. Did not run c0 / dlinear / timexer_plain / ablation_table / paper.

## Suggested next command (optional)
```bash
# harvest after SWEEP_DONE — runner timer, do not start paper
tail -n 40 /workspace/DecisionForecast/outputs/hp-sweep-cap.log
```
