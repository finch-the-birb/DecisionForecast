# Agent handoff format

Ephemeral communication between **Dev** (local) and **Runner** (remote / Runpod).
All files live under `agent-handoff/` and are meant to be **purged from git history** before a final / paper submission (see `scripts/purge_agent_handoff_history.*`).

## Directory layout

```text
agent-handoff/
  <YYYY-MM-DD>_<HHMM>_<slug>/
    request.md    # Dev → Runner (what to run)
    report.md     # Runner → Dev (results + conclusions)
```

- `<slug>`: short kebab-case label (`phase0-smoke`, `fix-oom`, `model-b`).
- One folder = one handoff round. Do not overwrite old rounds; create a new folder.
- Commit **only** these markdown files. Never commit `outputs/`, checkpoints, raw logs, datasets, or MLflow DBs into `agent-handoff/`.

## `request.md` (Dev writes)

```markdown
# Request: <slug>

- authored_by: dev
- created_at: <ISO-8601>
- target_ref: <branch>@<full-or-short-sha>
- phase: <0|1|2|…|adhoc>
- priority: <normal|high>

## Goal
<one paragraph: what success looks like>

## Commands
Run exactly these (or the nearest equivalent on the Runner host). Prefer Hydra overrides over editing configs.

```bash
# example — replace per phase
uv run python -m src.training.train \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1 \
  train.ticker_set=dev
```

## Environment notes
- expected_cwd: repo root
- data_ready: <yes|no|unknown>
- gpu: <required|optional|cpu-ok>
- approx_ram_gb: <number or unknown>

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
```

## `report.md` (Runner writes)

```markdown
# Report: <slug>

- authored_by: runner
- created_at: <ISO-8601>
- request_folder: agent-handoff/<YYYY-MM-DD>_<HHMM>_<slug>/
- tested_ref: <branch>@<sha actually tested>
- status: <pass|fail|blocked|partial>

## Commands executed
```bash
<exact commands>
```

## Outcome
- exit_code: <int or n/a>
- duration: <e.g. 12m>
- host: <runpod pod id / hostname>
- device: <cuda:0|cpu|…>

## Key signals
- metrics: <mse/mae/… or n/a>
- oom: <yes|no>
- traceback_summary: <one short paragraph or n/a>

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: outputs/<run>/train.log
- explain: outputs/<run>/explain/…
- mlflow_run_id: <id or n/a>

## Conclusions for Dev
1. <actionable finding>
2. <…>

## Suggested next command (optional)
```bash
<follow-up smoke or debug command>
```
```

## Status values

| status   | meaning |
|----------|---------|
| `pass`   | Commands finished; smoke goal met |
| `fail`   | Ran, but errors / regressions |
| `blocked`| Could not run (missing data, auth, OOM before start, bad ref) |
| `partial`| Some steps ok, others failed; explain in Outcome |

## Git rules for handoff files

1. Only touch files under `agent-handoff/<this-round>/`.
2. Message style: `handoff: <slug> request` / `handoff: <slug> report (<status>)`.
3. Push to the branch named in `target_ref` (usually the Dev feature branch).
4. Before paper / archival freeze, run `scripts/purge_agent_handoff_history` so `agent-handoff/` disappears from history.
