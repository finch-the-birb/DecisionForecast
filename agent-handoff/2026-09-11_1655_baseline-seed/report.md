# Report: baseline-seed

- authored_by: runner
- created_at: 2026-09-11T16:56:50Z
- request_folder: agent-handoff/2026-09-11_1655_baseline-seed/
- tested_ref: feat/phase2-timexer@349b0a09e9ff215a4b9e42962670f6d67fd99d81 (contains dc19237, 03f123b)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not B / C0 / C1 / ablation_table / seed=42 / lr=0.001 / extra proto / other T/H / decay_lambda / d_model.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
# HEAD=349b0a0 (contains dc19237 and 03f123b)
uv sync
# MU already present; skipped precache
# nohup 9-job baselines: dlinear, timexer_plain, a/proto=10 × seeds 0,1,2; lr=0.0003; T=60 H=7; yaml 64/256; num_workers=4
# n_prototypes passed only on model=a
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: cuda requested; not harvested yet
- pid: 159375
- log: `outputs/baseline-seed.log`
- started_at: 2026-09-11T16:56:37Z
- first_job: `model=dlinear train.seed=0`
- timer: one-shot 5400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/baseline-seed.log`

## Conclusions for Dev
1. Sweep launched; 9 sequential jobs (dlinear / timexer_plain / a proto=10 × seeds 0/1/2). Did not wait on PID. Did not run B / C0 / C1 / paper / seed=42 / lr=0.001 / extra proto / other T/H / decay_lambda / caps / ablation_table.
2. 9-row table + mean±std last_val winner among these three will replace this report when the 90 min timer fires (or on human ping). seed-shortlist C0/B/C1 footnote copied at harvest, not re-run.
