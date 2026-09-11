# Report: c0c1-windows

- authored_by: runner
- created_at: 2026-09-11T15:40:10Z
- request_folder: agent-handoff/2026-09-11_1540_c0c1-windows/
- tested_ref: feat/phase2-timexer@729a3a8add4d643e9ffa33f63a9d34b47bb49b35 (contains 149c06b, 03f123b)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not A / B / dlinear / ablation_table / extra seeds / decay_lambda / window_agg / proto≠5 / T=60 H=7 / d_model.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=729a3a8 (contains 149c06b and 03f123b)
uv sync
# MU already present; skipped precache
# nohup 8-job: c0,c1 × (T,H)=(60,14),(60,30),(36,7),(96,7); n_prototypes=5; lr=0.0003; yaml 64/256; seed=42; num_workers=4
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 126197
- log: `outputs/c0c1-windows.log`
- started_at: 2026-09-11T15:39:44Z
- first_job: `model=c0 data.lookback_T=60 data.horizon=14`
- timer: one-shot 5400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/c0c1-windows.log`

## Conclusions for Dev
1. Sweep launched; 8 sequential jobs (c0/c1 × four (T,H) pairs). Did not wait on PID. Did not run A / B / dlinear / paper / extra seeds / decay_lambda / window_agg / proto≠5 / caps / T=60 H=7 / ablation_table.
2. Per-pair C0 vs C1 + overfit tables will replace this report when the 90 min timer fires (or on human ping). No global winner across H.
