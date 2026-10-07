# Report: paper-timexer-dual-active-ts

- authored_by: runner
- created_at: 2026-10-07T10:51:37Z
- updated_at: 2026-10-07T10:51:37Z
- request_folder: agent-handoff/2026-10-07_1200_paper-timexer-dual-active-ts/
- tested_ref: feat/phase2-timexer@39ee9f0d58e9bf62155e17723b2209b4aa7fb7f8
- status: partial (Block 1/3, 0/24 jobs completed)

## Commands executed


## Outcome
- exit_code: in progress;  absent
- host: runpod pod  (hostname )
- device: sweep started; first job not finished yet
- pid: 4271
- log: 
- started_at: 2026-10-07T10:51:23Z
- current_job: Block 1 F2 [close, volume], e_layers=1, use_prototypes=false, seed=0
- grid: 24 jobs. Each block is e=1 then e=2: three seeds without prototypes, then prototypes on seed=1 only. Block 1 F2 n_features=2, Block 2 F1 n_features=1, Block 3 F5 n_features=5. Shared: c1_dual, fnspid_dual_ts, d_model=64, n_ts_features=25, patch_len=12, patch_stride=12, Huber δ=0.5, pool=last, paper, epochs=20, patience=5, lr=0.0003, T=60, H=7, workers=4.
- signature:  (26 columns, channel 0 is close, 25 indicators)

- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: 

## Conclusions for Dev
1. Sweep is running. 0/24 finished. No ranking yet against DLinear test 0.7535 / 0.7547.
