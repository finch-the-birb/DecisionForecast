# Report: head-pooling-sweep

- authored_by: runner
- created_at: 2026-09-11T22:47:20Z
- request_folder: agent-handoff/2026-09-12_0200_head-pooling-sweep/
- tested_ref: feat/phase2-timexer@02d47475575ca85d8a40d93dc62681ea326680cd (contains 93437e7 modular forecast head)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=02d4747 (contains 93437e7)
uv sync
# MU already present; skipped precache
# nohup 24-job: plain mean/last, C0 mean/last, C1 mean/last/global, dlinear × seeds 0,1,2
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 294266
- log: `outputs/head-pooling-sweep.log`
- started_at: 2026-09-11T22:47:14Z
- first_job: `model=timexer_plain model.head.pool=mean train.seed=0`
- timer: one-shot 5400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/head-pooling-sweep.log`

## Conclusions for Dev
1. Sweep launched; 24 sequential jobs (8 configs × seeds 0/1/2). Did not wait on PID. Did not run flatten / A / B / paper / extra T/H.
2. Summary table + per-seed overfit + answers to the four questions will replace this report when the 90 min timer fires (or on human ping).
