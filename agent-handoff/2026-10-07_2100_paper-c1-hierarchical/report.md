# Report: paper-c1-hierarchical

- authored_by: runner
- created_at: 2026-10-07T18:10:28Z
- updated_at: 2026-10-07T18:10:28Z
- request_folder: agent-handoff/2026-10-07_2100_paper-c1-hierarchical/
- tested_ref: feat/phase2-timexer@6dbbf0fb8a85b1b64c048aa475f949607ddbd45a
- status: partial (Block 1/2, 0/6 jobs completed)

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git pull --ff-only origin feat/phase2-timexer  (190c8d8..6dbbf0f)
# uv sync
# HF_TOKEN unset. No prepare. nohup 6 jobs, ticker_set=paper.
# paper-timexer-dual-active-ts was already SWEEP_DONE. It was not restarted.
```

## Outcome
- exit_code: in progress; SWEEP_DONE absent
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: sweep started; first job not finished yet
- pid: 96529
- log: outputs/paper-c1-hierarchical.log
- started_at: 2026-10-07T18:10:12Z
- current_job: Block 1 F2 [close, volume], e_layers=2, use_prototypes=false, seed=0
- grid: 6 jobs. Block 1 three seeds without prototypes, then Block 2 three seeds with n_prototypes=10. Shared: c1_hierarchical, fnspid_dual_ts, n_ts_features=25, d_model=64, n_heads=4, d_ff=256, Huber delta=0.5, pool=last, patch 12/12, paper, epochs=20, patience=5, lr=0.0003, T=60, H=7, workers=4.
- n_params budget: flag in conclusions if a no-prototype job is above 185000. Do not kill the sweep for that.

- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-c1-hierarchical.log

## Conclusions for Dev
1. Sweep is running. 0/6 finished. No ranking yet.
2. Comparison numbers already on disk, not re-run: DLinear F1 0.7535±0.0009, DLinear F5 0.7547±0.0028, c1_dual F2 e=1 no prototypes 0.7655±0.0032 (best seed 0.7633), c1_dual F2 e=2 no prototypes 0.7738±0.0075 from paper-timexer-dual-active-ts.
