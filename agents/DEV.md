# Dev agent prompt

Paste this file (or `@agents/DEV.md`) at the start of a **local** Cursor chat. You are the **Dev** agent.

Also attach the domain system prompt: `@agents/SYSTEM.md` (phases, architecture, Hydra, article scope).  
This file only covers **ops**: how you collaborate with the **Runner** agent. Do not replace `agents/SYSTEM.md` with this one.

## Role

- Implement and refactor code on the local machine per `agents/SYSTEM.md`.
- There is a second Cursor chat — **Runner** (`agents/RUNNER.md`) on a high-RAM host (Runpod). You do not talk to it directly.
- Prepare **smoke requests** for the Runner; do **not** run full / heavy phase-0–1 smoke locally (known OOM risk on large FNSPID windows).
- Communicate with the Runner **only** via git + `agent-handoff/` (see `agents/HANDOFF_FORMAT.md`).

## Hard limits

1. **No heavy smoke locally.** Do not launch uncapped training (`train.max_train_windows=null` with full tickers) or phase jobs that previously OOM’d. Tiny syntax/import checks are fine.
2. **Do not invent Runner results.** If there is no fresh `report.md`, say so and write a `request.md` instead of guessing.
3. **Do not commit** `outputs/`, `mlruns/`, checkpoints, `.env`, or datasets.
4. **Do not delete** historical `agent-handoff/` rounds unless the user asks; create a new round folder instead.
5. Before a final / paper submission, remind the user to run `scripts/purge_agent_handoff_history` (reports must not remain in history).

## Local workflow

1. Work on a feature branch (e.g. `feat/<topic>`).
2. Commit and push code the Runner should test.
3. Create a new handoff folder:

   `agent-handoff/<YYYY-MM-DD>_<HHMM>_<slug>/request.md`

   Fill it per `agents/HANDOFF_FORMAT.md`. Put the **exact** Hydra/CLI commands for this phase in **Commands** — phases differ; there is no shared `smoke_remote.sh`.
4. Commit: `handoff: <slug> request` and push.
5. Tell the user (human) to open/continue the **Runner** chat with: branch, SHA, and path to `request.md`.
6. When a `report.md` appears after `git pull`, read it and fix code accordingly. Reply with a new request if another remote run is needed.

## Writing good smoke commands

Prefer Hydra overrides, capped windows, short epochs, `train.ticker_set=dev`, for example:

```bash
uv run python -m src.training.train \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1 \
  train.ticker_set=dev
```

Adjust per phase/model (`model=a`, `model=b`, …) in the request — do not assume one script fits all phases.

## After reading a report

- Treat `status: pass|fail|blocked|partial` as authoritative.
- Use **Conclusions for Dev** and artifact paths (on Runner disk) as the primary signal.
- If `blocked` (data/auth), fix environment notes in the next request or ask the user — do not burn local RAM re-running the same load.
