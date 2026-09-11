# Report: seed-shortlist

- authored_by: runner
- created_at: 2026-09-11T14:50:20Z
- request_folder: agent-handoff/2026-09-11_1448_seed-shortlist/
- tested_ref: feat/phase2-timexer@87ed084697c4bf9c779b41bbc875853b15161ff5 (contains a93af33, 03f123b)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not A / dlinear / ablation_table / seed=42 / lr=0.001 / d_model.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=87ed084 (contains a93af33 and 03f123b)
uv sync
# MU already present; skipped precache
# nohup 9-job 3-seed confirmation: b proto=10, c0 proto=5, c1 proto=5 × seeds 0,1,2; lr=0.0003; yaml 64/256; num_workers=4
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4`
- device: cuda requested; not harvested yet
- pid: 102496
- log: `outputs/seed-shortlist.log`
- started_at: 2026-09-11T14:50:08Z
- timer: one-shot 5400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/seed-shortlist.log`

## Conclusions for Dev
1. Sweep launched; 9 sequential jobs (b/10, c0/5, c1/5 × seeds 0/1/2). Did not wait on PID. Did not run A / dlinear / paper / seed=42 / lr=0.001 / d_model / extra proto / caps / ablation_table.
2. Table + mean±std winner will replace this report when the 90 min timer fires (or on human ping).
