# Report: hp-width-nproto

- authored_by: runner
- created_at: 2026-09-11T13:19:30Z
- request_folder: agent-handoff/2026-09-11_1320_hp-width-nproto/
- tested_ref: feat/phase2-timexer@069922e757685e5695bffdc9271106d49c2fc49c (contains 03f123b)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not ablation_table / dlinear / lr=0.001 / cartesian d_model×d_ff.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=069922e (contains 03f123b Coverage + DataLoader)
uv sync
# MU already present; skipped precache
# nohup 8-job grid: a n_prototypes=5,10; b,c0,c1 × (64,256)|(128,512); lr=0.0003; num_workers=4
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4`
- device: cuda requested; not harvested yet
- pid: 55728
- log: `outputs/hp-width-nproto.log`
- started_at: 2026-09-11T13:19:20Z
- timer: one-shot 5400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/hp-width-nproto.log`

## Conclusions for Dev
1. Sweep launched; 8 sequential jobs (a n_proto 5/10; b/c0/c1 paired width). Did not wait on PID. Did not run ablation_table / dlinear / paper / lr=0.001 / unpaired d_model×d_ff / caps.
2. Table + winner will replace this report when the 90 min timer fires (or on human ping).
