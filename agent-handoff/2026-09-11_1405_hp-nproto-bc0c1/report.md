# Report: hp-nproto-bc0c1

- authored_by: runner
- created_at: 2026-09-11T14:07:25Z
- request_folder: agent-handoff/2026-09-11_1405_hp-nproto-bc0c1/
- tested_ref: feat/phase2-timexer@4ca9a07c0e2e68b658c83a529a4f8d4e7350d194 (contains 02e91e0, 03f123b)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not A / dlinear / ablation_table / lr=0.001 / d_model / d_ff.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=4ca9a07 (contains 02e91e0 and 03f123b)
uv sync
# MU already present; skipped precache
# nohup 9-job grid: b,c0,c1 × n_prototypes=5,10,20; lr=0.0003; yaml width 64/256; num_workers=4
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4`
- device: cuda requested; not harvested yet
- pid: 79492
- log: `outputs/hp-nproto-bc0c1.log`
- started_at: 2026-09-11T14:07:15Z
- timer: one-shot 5400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-nproto-bc0c1.log`

## Conclusions for Dev
1. Sweep launched; 9 sequential jobs (b/c0/c1 × proto 5/10/20). Did not wait on PID. Did not run A / dlinear / paper / lr=0.001 / d_model / d_ff / caps / ablation_table.
2. Table + winner will replace this report when the 90 min timer fires (or on human ping).
