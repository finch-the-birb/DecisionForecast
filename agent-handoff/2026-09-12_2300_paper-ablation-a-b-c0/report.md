# Report: paper-ablation-a-b-c0

- authored_by: runner
- created_at: 2026-09-12T20:01:25Z
- request_folder: agent-handoff/2026-09-12_2300_paper-ablation-a-b-c0/
- tested_ref: feat/phase2-timexer@98a3459d15e3c19cb0ba411948a8421740c4857f (contains 033996f)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No precache. No FNSPID download. ticker_set=paper (not dev). No DLinear / plain / C1 / flatten.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=98a3459 (contains 033996f)
uv sync
# mu_paper present — skipped precache
# nohup 9-job: A proto=10 Huber, B e=1 last proto=10 Huber, C0 e=1 last proto=5 Huber × seeds 0,1,2; F5
# bash noglob (set -f) so Hydra data.features=[close,volume,open,high,low] is not glob-expanded
```

## Outcome
- exit_code: uv sync=0; mu_paper present; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 689469
- log: `outputs/paper-ablation-a-b-c0.log`
- started_at: 2026-09-12T20:01:14Z
- first_job: `model=a ... proto=10 Huber δ=0.5 data.features=[close,volume,open,high,low] train.seed=0`
- timer: one-shot 9000s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-ablation-a-b-c0.log`

## Conclusions for Dev
1. Sweep launched; 9 sequential paper jobs (A / B / C0). Did not wait on PID. Did not run DLinear / plain / C1 / precache / ticker_set=dev.
2. Tables replace this report when the 150 min timer fires (or on human ping).
