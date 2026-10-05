# Report: paper-timexer-selected-26ts

- authored_by: runner
- created_at: 2026-10-05T21:19:15Z
- request_folder: agent-handoff/2026-10-06_1200_paper-timexer-selected-26ts/
- tested_ref: feat/phase2-timexer@8e35271fa7e223e311cff3c78a7e28339535824d (contains 9bf6a81)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. No dlinear / plain / a / b / c0 / c1 / ticker_set=dev.
# Signature is recomputed by prepare --fold 1 (close pinned + Top-25). Old 25-column signature is not reused.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
git merge-base --is-ancestor 9bf6a81aa29f5ec12b57685fe3079b02ead74c9b HEAD
uv sync
uv run pytest tests -q
# nohup: prepare_selected_data.py --ticker-set paper --fold 1 --device cuda
# then timexer_selected fnspid_selected Huber δ=0.5 × e_layers=1,2 × seeds 0,1,2
```

## Outcome
- exit_code: uv sync=0; pytest=0 (111 passed, 5 warnings, 100.21s); sweep started (nohup); harvest pending
- duration: n/a (in progress)
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: cuda requested; not harvested yet
- pid: 95285
- log: `outputs/paper-timexer-selected-26ts.log`
- started_at: 2026-10-05T21:19:14Z
- first_step: `=== PREPARE` (`prepare_selected_data.py --ticker-set paper --fold 1 --device cuda`)
- timer: one-shot 14400s harvest

## Key signals
- metrics: n/a (started)
- oom: n/a
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/paper-timexer-selected-26ts.log`

## Conclusions for Dev
1. Sweep launched. Prepare is rewriting the fold-1 signature (`close` + Top-25). Six sequential `timexer_selected` jobs follow. Did not wait on PID.
2. The per-seed table, the 26 column names, and the e=1 vs e=2 comparison against DLinear test 0.7547 replace this report when the 4 h timer fires (or on human ping).
