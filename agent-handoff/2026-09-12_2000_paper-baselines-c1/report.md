# Report: paper-baselines-c1

- authored_by: runner
- created_at: 2026-09-12T16:15:10Z
- request_folder: agent-handoff/2026-09-12_2000_paper-baselines-c1/
- tested_ref: feat/phase2-timexer@7a891e970153a1f26d741b09a1dabcec7c237cfc (contains 089a1fe)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID raw dump download. ticker_set=paper (not dev). No Model A / B / C0 / flatten / extra T/H.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=7a891e9 (contains 089a1fe)
uv sync
# mu_paper missing → precache news + embeddings runs inside nohup (first step)
# nohup 12-job: DLinear F1, plain last e=1 F2, C1.1 Huber e=1 F5, C1.1 Huber e=2 F5 × seeds 0,1,2
# bash noglob (set -f) so Hydra data.features=[close] is not glob-expanded
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 590912
- log: `outputs/paper-baselines-c1.log`
- started_at: 2026-09-12T16:14:57Z
- first_step: `=== PRECACHE paper news + embeddings` (`mu_paper_2015-01-01_2021-12-31.npy` missing)
- timer: one-shot 12600s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-baselines-c1.log`

## Conclusions for Dev
1. Sweep launched; 12 sequential paper jobs after precache. Did not wait on PID. Did not run Model A / B / C0 / flatten / ticker_set=dev.
2. Tables replace this report when the 210 min timer fires (or on human ping).
