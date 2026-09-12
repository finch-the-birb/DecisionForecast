# Report: features-and-capacity-sweep

- authored_by: runner
- created_at: 2026-09-12T13:22:50Z
- request_folder: agent-handoff/2026-09-12_1700_features-and-capacity-sweep/
- tested_ref: feat/phase2-timexer@e5a49626d25042f1a1a41fc301d1a1fe0c8f7d5c (contains 798bb80 C1 last-patch one-layer default)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=e5a4962 (contains 798bb80)
uv sync
# MU already present; skipped precache
# nohup 36-job: C1.1 MSE/Huber × F1/F2/F5, plain last e=1 × F1/F2/F5, C0 mean e=1 λ=0 × F1/F2, DLinear F1 × seeds 0,1,2
# bash noglob (set -f) so Hydra data.features=[close] is not glob-expanded
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 524337
- log: `outputs/features-and-capacity-sweep.log`
- started_at: 2026-09-12T13:22:39Z
- first_job: `model=c1 ... pool=last data.features=[close] train.seed=0`
- timer: one-shot 7200s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/features-and-capacity-sweep.log`

## Conclusions for Dev
1. Sweep launched; 36 sequential jobs (12 configs × seeds 0/1/2). Did not wait on PID. Did not run flatten / A / B / paper / extra T/H.
2. Four tables (C1.1 MSE / Huber / plain / ranking vs DLinear) will replace this report when the 120 min timer fires (or on human ping).
