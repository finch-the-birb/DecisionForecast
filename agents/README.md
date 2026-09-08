# Dev ↔ Runner agent scheme

Two Cursor chats, one shared git remote. Agents do **not** message each other; they exchange markdown under `agent-handoff/`.

```text
┌─────────────┐     push code + request.md      ┌──────────────┐
│  Dev chat   │ ──────────────────────────────► │    remote    │
│  (local)    │                                 │  (git host)  │
│  @DEV.md    │ ◄────────────────────────────── │              │
└─────────────┘     pull report.md              └──────┬───────┘
                                                       │
                                                pull / push
                                                       │
                                                ┌──────▼───────┐
                                                │ Runner chat  │
                                                │ (Runpod SSH) │
                                                │ @RUNNER.md   │
                                                └──────────────┘
```

## Files

| Path | Purpose |
|------|---------|
| `agents/SYSTEM.md` | Domain system prompt: article scope, phases, Hydra, architecture |
| `agents/DEV.md` | Ops prompt for local Dev (Runner handoff, no local OOM smoke) |
| `agents/RUNNER.md` | Prompt for the remote Runner (includes Runpod setup) |
| `agents/HANDOFF_FORMAT.md` | `request.md` / `report.md` schema |
| `agent-handoff/` | Ephemeral round folders (safe to purge from history) |
| `scripts/purge_agent_handoff_history.*` | Strip `agent-handoff/` from **entire** git history |
| `Articles/notes/` | Research only (plans, hypotheses) — not agent ops |

There is **no** shared `smoke_remote.sh`: smoke commands differ by phase. Dev puts exact commands in each `request.md`; Runner executes them.

## How to start a chat

1. **Dev (local):** new Cursor chat → attach **both** `@agents/SYSTEM.md` (what to build) and `@agents/DEV.md` (Runner handoff / no local OOM smoke).
2. **Runner (Runpod / remote):** new Cursor chat on the remote workspace → attach `@agents/RUNNER.md` only (not `SYSTEM.md`). On first use, ask it to set up Runpod (`fetch https://docs.runpod.io/agent-setup.md` + `npx @runpod/mcp-server@latest add` — already in the Runner prompt).

## Typical round

1. Dev pushes feature branch + `agent-handoff/<date>_<time>_<slug>/request.md`.
2. You tell Runner: branch, SHA, path to the request.
3. Runner pulls, runs commands, writes `report.md`, pushes.
4. Dev pulls and continues from the report.

## Before paper / final archive

`agent-handoff/` must not remain in the published history:

```bash
# Linux / Runpod / Git Bash
./scripts/purge_agent_handoff_history.sh

# Windows PowerShell
./scripts/purge_agent_handoff_history.ps1
```

This rewrites history (requires `git-filter-repo`). Coordinate with anyone else who has clones; force-push only if you intend to. Protocol docs under `agents/` are **kept**.
