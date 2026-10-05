# Report: paper-timexer-5ts-compact

- authored_by: runner
- created_at: 2026-10-05T23:16:10Z
- request_folder: agent-handoff/2026-10-06_1500_paper-timexer-5ts-compact/
- tested_ref: feat/phase2-timexer@a38f2f33f5006990d3633d45a23a2504a20a622b (contains a9006d2)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. No prepare_selected_data. No FinBERT/FinLang re-encode.
# No model=a/b/c0/c1/dlinear/timexer_plain/timexer_selected. No ticker_set=dev.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
git merge-base --is-ancestor a9006d25f91bd102ac5d8f1a0f958921b4dced46 HEAD
uv sync
# nohup 12-job: c1_compact + fnspid_compact, Huber δ=0.5, pool=last
# e_layers 1 then 2 × proto false (λ=0) × seeds 0,1,2
# then e_layers 1 then 2 × proto true n=10 λ_c=0.1 λ_e=0.1 λ_d=0.01 × seeds 0,1,2
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: cuda requested; not harvested yet
- pid: 142068
- log: `outputs/paper-timexer-5ts-compact.log`
- started_at: 2026-10-05T23:15:59Z
- first_job: `c1_compact e_layers=1 use_prototypes=false seed=0`
- timer: one-shot 14400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-5ts-compact.log`

## Conclusions for Dev
1. Sweep launched. Compact text cache is reused; prepare was not run. Did not wait on PID.
2. Per-seed and mean tables, plus the comparison to DLinear 0.7535 / 0.7547, C1.1 768D 0.7543, and TimeXer 26TS 0.7902, replace this report when the 4 h timer fires (or on human ping).
