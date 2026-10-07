# Report: paper-capacity-budgets

- authored_by: runner
- created_at: 2026-10-07T22:12:16Z
- updated_at: 2026-10-07T22:12:16Z
- request_folder: agent-handoff/2026-10-07_2300_paper-capacity-budgets/
- tested_ref: feat/phase2-timexer@ddfa2bf89f4e632537cc92236851f57cc9326100
- status: partial (Block 1/3, 0/18 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git pull --ff-only origin feat/phase2-timexer  (767eeee..ddfa2bf)
# uv sync
# HF_TOKEN unset. No prepare. nohup 18 jobs, ticker_set=paper, prototypes off.
# Full-width runs were not restarted. Previous 0/18 fail was the missing d_ff key.
```

## Outcome
- exit_code: in progress; SWEEP_DONE absent
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: sweep started; first job not finished yet
- pid: 123275
- log: outputs/paper-capacity-budgets.log
- started_at: 2026-10-07T22:11:58Z
- current_job: Block 1 c1_dual e_layers=1, d_model=56, n_heads=2, d_ff=200, seed=0
- fix: ddfa2bf declares d_ff on c1_dual so the Hydra override is in struct
- grid: 18 jobs, all F2 [close, volume], n_ts_features=25, no prototypes. Block 1 dual e=1 at 75% then 50%. Block 2 dual e=2 at 75% then 50%. Block 3 hierarchical e=2 at 75% then 50%. Three seeds each.
- expected n_params: dual e=1 91601 / 60777; dual e=2 178659 / 119211; hierarchical 129113 / 85993. A mismatch is recorded, not used to stop the block.

- oom: n/a
- traceback_summary: n/a at launch. The 20:04:21Z Hydra error is the previous attempt.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-capacity-budgets.log

## Conclusions for Dev
1. Sweep is running again after ddfa2bf. 0/18 finished. No ranking yet.
2. Full-width references, not re-run: hierarchical 172169 params, test 0.7655±0.0021; dual F2 e=1 122121 params, test 0.7655±0.0032; dual F2 e=2 239051 params, test 0.7738±0.0075; DLinear F1 0.7535±0.0009.
