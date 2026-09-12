# Report: regularization-sweep

- authored_by: runner
- created_at: 2026-09-12T06:45:30Z
- request_folder: agent-handoff/2026-09-12_1000_regularization-sweep/
- tested_ref: feat/phase2-timexer@a409b3b1171f33b6b4902c2e3ce475c7905dba2c (contains 7681745 financial losses + regularization)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No flatten / A / B / extra T/H / decay_lambda / caps / ablation_table.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=a409b3b (contains 7681745)
uv sync
# MU already present; skipped precache
# nohup 60-job: C1.1 last losses / proto λ / wd+capacity+dropout / Huber+wd combo + DLinear × seeds 0,1,2
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 375076
- log: `outputs/regularization-sweep.log`
- started_at: 2026-09-12T06:45:19Z
- first_job: `model=c1 model.n_prototypes=5 model.head.pool=last model.loss.kind=mse train.seed=0`
- timer: one-shot 9000s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/regularization-sweep.log`

## Conclusions for Dev
1. Sweep launched; 60 sequential jobs (20 configs × seeds 0/1/2). Did not wait on PID. Did not run flatten / A / B / paper / extra T/H.
2. Four tables + answers (loss / proto λ / wd-capacity / Huber+wd vs DLinear) will replace this report when the 150 min timer fires (or on human ping).
