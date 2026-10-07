# Report: paper-capacity-budgets

- authored_by: runner
- created_at: 2026-10-07T20:03:41Z
- updated_at: 2026-10-07T21:05:41Z
- request_folder: agent-handoff/2026-10-07_2300_paper-capacity-budgets/
- tested_ref: feat/phase2-timexer@136a2d3694df0184801a26368fa4c7e8877890f8
- status: fail

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git pull --ff-only origin feat/phase2-timexer  (d196008..136a2d3)
# uv sync
# HF_TOKEN unset. No prepare. nohup 18 jobs. Sweep aborted by set -e on job 1. No restart.
```

## Outcome
- exit_code: sweep PID 110457 is dead; SWEEP_DONE absent; 0/18 DONE
- host: runpod pod sw7txpqcwtk7tz (hostname f095b9ee3fba)
- device: no job reached training
- pid: 110457 (dead)
- log: outputs/paper-capacity-budgets.log
- started_at: 2026-10-07T20:03:24Z
- died_at: log mtime 2026-10-07T20:04:21Z, during the first job
- failed_job: Block 1, model=c1_dual, e_layers=1, d_model=56, n_heads=2, d_ff=200, seed=0
- other sweeps: paper-timexer-dual-active-ts is already 24/24 pass; paper-c1-hierarchical is already 6/6 pass. Neither was restarted.

- oom: no
- traceback_summary: Hydra refused the override before training. configs/model/c1_dual.yaml has no d_ff key, and the config struct rejects a new key. configs/model/c1_hierarchical.yaml does have d_ff: 256. The log is:

```
Could not override 'model.d_ff'.
To append to your config use +model.d_ff=200
Key 'd_ff' is not in struct
    full_key: model.d_ff
    object_type=dict
```

src/training/train.py reads d_ff with cfg.model.get("d_ff", 4 * d_model), so the constructor can take the value once Hydra accepts the key. set -e stopped the sweep on job 1, so the dual e=2 and hierarchical blocks never started. Runner did not edit product code and did not relaunch.

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/paper-capacity-budgets.log

## Conclusions for Dev
1. The 18-job sweep is dead at 0/18. No n_params, test_mse, or test_mae were produced.
2. The first override, model.d_ff=200 on c1_dual, is not in the config struct. Hierarchical would have accepted d_ff, but it never ran.
3. Full-width references stay the ones already reported: hierarchical 172169 params, test 0.7655±0.0021; dual F2 e=1 122121 params, test 0.7655±0.0032; dual F2 e=2 239051 params, test 0.7738±0.0075; DLinear F1 0.7535±0.0009.
