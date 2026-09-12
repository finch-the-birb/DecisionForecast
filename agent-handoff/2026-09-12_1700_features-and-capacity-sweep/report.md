# Report: features-and-capacity-sweep

- authored_by: runner
- created_at: 2026-09-12T13:22:50Z
- request_folder: agent-handoff/2026-09-12_1700_features-and-capacity-sweep/
- tested_ref: feat/phase2-timexer@e5a49626d25042f1a1a41fc301d1a1fe0c8f7d5c (contains 798bb80 C1 last-patch one-layer default)
- status: partial (still running 31/36)

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
- still running 31/36 (PID 524337 alive; no `SWEEP_DONE`; job 32 in progress: C0 mean e=1 λ=0 F2 seed=1). Remaining: C0 F2 seed=2 + DLinear F1 ×3. Timer fired; no second timer.

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/features-and-capacity-sweep.log`

## Conclusions for Dev
1. still running 31/36. Harvest when human pings (or PID dead + `SWEEP_DONE`).
