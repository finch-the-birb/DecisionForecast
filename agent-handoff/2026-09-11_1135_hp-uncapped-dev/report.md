# Report: hp-uncapped-dev

- authored_by: runner
- created_at: 2026-09-11T11:47:20Z
- request_folder: agent-handoff/2026-09-11_1135_hp-uncapped-dev/
- tested_ref: feat/phase2-timexer@b082360d67c87434852e1ff06dd3a54fa09d3f28 (contains 79d2ff7)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not ablation_table / dlinear / extra seeds / d_model.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=b082360
uv sync
# MU already present; skipped precache
# nohup 8-job grid a,b,c0,c1 × lr=0.0003,0.001; epochs=20; seed=42; H=7; ticker_set=dev
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4`
- device: cuda requested; not harvested yet
- pid: 26353
- log: `outputs/hp-uncapped-dev.log`
- started_at: 2026-09-11T11:47:07Z
- timer: one-shot 5400s harvest (cursor-subscriptions MCP not in this session; one-shot wake armed)

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-uncapped-dev.log`

## Conclusions for Dev
1. Sweep launched; 8 sequential jobs. Did not wait on PID. Did not run ablation_table / dlinear / caps / paper / extra seeds / d_model.
2. Table + winner will replace this report when the 90 min timer fires (or on human ping).
