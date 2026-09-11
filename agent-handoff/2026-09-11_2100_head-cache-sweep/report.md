# Report: head-cache-sweep

- authored_by: runner
- created_at: 2026-09-11T18:05:10Z
- request_folder: agent-handoff/2026-09-11_2100_head-cache-sweep/
- tested_ref: feat/phase2-timexer@c839e4512e337aa44ba0ffa887e9357f6bb09eff (contains 7f61ab9 FlattenHead + text cache key)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not seed=42 / extra T/H / ablation_table.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=c839e45 (contains 7f61ab9)
uv sync
# MU already present; skipped precache
# nohup 72-job: Block1 dlinear lr×seed (6); Block2 plain head.type×dropout×seed (18);
# Block3 plain/c0/c1/b × dropout × seed (24); Block4 c0/c1/b/a × lam × seed (24)
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 187648
- log: `outputs/head-cache-sweep.log`
- started_at: 2026-09-11T18:04:54Z
- first_job: `model=dlinear train.lr=0.0003 train.seed=0`
- timer: one-shot 10800s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/head-cache-sweep.log`

## Conclusions for Dev
1. Sweep launched; 72 sequential jobs in 4 blocks. Did not wait on PID. Did not run paper / seed=42 / extra T/H / caps / ablation_table.
2. Four block mean±std tables + answers (FlattenHead vs mean-pool, vs DLinear, decay_lambda) will replace this report when the 180 min timer fires (or on human ping).
