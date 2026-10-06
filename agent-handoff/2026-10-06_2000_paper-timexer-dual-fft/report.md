# Report: paper-timexer-dual-fft

- authored_by: runner
- created_at: 2026-10-06T17:34:45Z
- request_folder: agent-handoff/2026-10-06_2000_paper-timexer-dual-fft/
- tested_ref: feat/phase2-timexer@33e4a41d2493381cbc72c88635a3a1ebf09641e1
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. No prepare_selected_data. No FinBERT/FinLang re-encode.
# No old baselines (dlinear / plain / a / b / c0 / c1 / timexer_selected / ticker_set=dev).
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=33e4a41 feat: implement selective revin, patch rfft, and dual-token timexer
uv sync
# compact text parquet count on disk: 61
# nohup 48 jobs: c1_dual + fnspid_selective, Huber δ=0.5, pool=last
# blocks: stride 6/12 × d_model 64/128 × e_layers 1/2 × proto false/true × seeds 0,1,2
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `sw7txpqcwtk7tz` (hostname `86df0010ddba`)
- device: cuda requested; not harvested yet
- pid: 2777
- log: `outputs/paper-timexer-dual-fft.log`
- started_at: 2026-10-06T17:34:33Z
- first_job: stride=6 d_model=64 e_layers=1 use_prototypes=false seed=0
- timer: one-shot 10000s progressive harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-dual-fft.log`

## Conclusions for Dev
1. Sweep launched on `33e4a41`. Block 1 (stride=6, d_model=64) is in progress. Did not wait on PID. Did not run prepare.
2. Completed `METRICS_ROW` rows will be appended on the next timer (or human ping). Full 16-config means replace this report at `SWEEP_DONE`.
