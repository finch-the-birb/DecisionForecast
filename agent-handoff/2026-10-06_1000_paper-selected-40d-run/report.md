# Report: paper-selected-40d-run

- authored_by: runner
- created_at: 2026-10-05T18:17:10Z
- request_folder: agent-handoff/2026-10-06_1000_paper-selected-40d-run/
- tested_ref: feat/phase2-timexer@863942091f86d7c50d908e883314f36e9eb4c3ef (contains 074ff3c)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# git fetch succeeded with the existing deploy key. Private key not printed, not committed.
# HF_TOKEN unset. No FNSPID download. No model=a/b/c0/c1. No ticker_set=dev.
# dlinear uses data=fnspid F5, not fnspid_selected. Signature is fold 1 train only (prepare --fold 1).
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
git merge-base --is-ancestor 074ff3c9c396b6a394e16f625bf48dfb1d297204 HEAD
uv sync
# nohup: prepare_selected_data.py --ticker-set paper --fold 1 --device cuda
# then 9 trains: dlinear F5, timexer_plain fnspid_selected, timexer_selected fnspid_selected × seeds 0,1,2
```

## Outcome
- exit_code: uv sync=0; sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: cuda requested; not harvested yet
- pid: 22447
- log: `outputs/paper-selected-40d-run.log`
- started_at: 2026-10-05T18:17:02Z
- first_step: `=== PREPARE` (`prepare_selected_data.py --ticker-set paper --fold 1 --device cuda`)
- timer: one-shot 18000s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-selected-40d-run.log`

## Conclusions for Dev
1. Sweep launched. Prepare is running; 9 sequential paper jobs follow. Did not wait on PID. Did not run A / B / C0 / C1 / `ticker_set=dev`.
2. The metrics table replaces this report when the 5 h timer fires (or on human ping).
