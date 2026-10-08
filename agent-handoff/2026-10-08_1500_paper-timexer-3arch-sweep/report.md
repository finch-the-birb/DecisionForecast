# Report: paper-timexer-3arch-sweep

- authored_by: runner
- created_at: 2026-10-08T12:11:32Z
- updated_at: 2026-10-08T12:11:32Z
- request_folder: agent-handoff/2026-10-08_1500_paper-timexer-3arch-sweep/
- tested_ref: feat/phase2-timexer@298dbf8fd45266ec90d2517bec000bc51395905f
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. git pull --ff-only was already up to date at 298dbf8.
# uv sync. c1_dual was not started.
# Inner sweep sets `set -f` so data.features=[close,volume,open,high,low] is not globbed.
# data=fnspid_dual_ts. patch_len=12 patch_stride=6.
# Log: outputs/paper-timexer-3arch-sweep.log
```

## Outcome
- exit_code: in progress; 0/36 DONE; SWEEP_DONE absent
- host: runpod pod 9hj7blo0fqus3p (hostname 279d25bcf0b8)
- device: pending first batch
- pid: 43683
- log: outputs/paper-timexer-3arch-sweep.log
- started_at: 2026-10-08T12:11:11Z
- current_job: Block 1 c1_factored e_layers=1 use_prototypes=false seed=0
- product_code: cd1f26edcc45f45d758914d1641ef4ff44caa887 (HEAD 298dbf8 is the blocked-report commit; no product diff after it)
- prior_blocked: agent-handoff/2026-10-06_2200_paper-timexer-3arch-sweep was not rewritten

The previous attempt used `data=fnspid_selective` and was blocked in `298dbf8`. This launch uses `data=fnspid_dual_ts` with features `[close, volume, open, high, low]`, `model.n_features=5`, `model.n_ts_features=25`. Expected first-batch shapes are `x=(32, 60, 5)`, `text_seq=(32, 60, 15)`, `ts=(32, 60, 25)`. Stride is 6, separate from the stride-12 dual-fft and active-ts references.

- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-timexer-3arch-sweep.log

## Conclusions for Dev
1. Sweep is running. PID 43683. Block 1 of 3, job 1 of 36: c1_factored, e_layers=1, no prototypes, seed=0. Started 2026-10-08T12:11:11Z.
2. c1_dual was not launched. The blocked report in `2026-10-06_2200_paper-timexer-3arch-sweep` was left as-is.
