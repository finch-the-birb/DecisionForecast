#!/usr/bin/env bash
# Remove agent-handoff/ from the entire git history (for final / paper archives).
# Rewrites history. Requires: git-filter-repo (https://github.com/newren/git-filter-repo)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a git repository" >&2
  exit 1
}
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not a git work tree" >&2
  exit 1
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "error: git-filter-repo not found on PATH." >&2
  echo "Install:  pip install git-filter-repo" >&2
  echo "Docs:     https://github.com/newren/git-filter-repo" >&2
  exit 1
fi

echo "This will REWRITE git history and delete path: agent-handoff/"
echo "Repo: $ROOT"
echo "Protocol docs under agents/ are kept."
if [[ "${PURGE_AGENT_HANDOFF_I_UNDERSTAND:-}" != "yes" ]]; then
  read -r -p "Type yes to continue: " confirm
  if [[ "$confirm" != "yes" ]]; then
    echo "aborted"
    exit 1
  fi
fi

# Full-history backup outside the rewrite (a normal branch would be rewritten too)
PARENT="$(dirname "$ROOT")"
BUNDLE="$PARENT/$(basename "$ROOT")-pre-purge-agent-handoff-$(date -u +%Y%m%dT%H%M%SZ).bundle"
git bundle create "$BUNDLE" --all
echo "Created recovery bundle: $BUNDLE"
echo "  Restore example: git clone \"$BUNDLE\" recover-pre-purge"

git filter-repo --force --invert-paths --path agent-handoff/

echo
echo "Done. agent-handoff/ removed from history."
echo "Next steps:"
echo "  1. Inspect: git log --all -- agent-handoff/   (should be empty)"
echo "  2. Re-add remote if filter-repo cleared it, then force-push intentionally:"
echo "       git remote add origin <url>"
echo "       git push --force-with-lease origin --all"
echo "       git push --force-with-lease origin --tags"
echo "  3. Ask collaborators to re-clone or hard-reset to the new history."
echo "Recovery bundle (unchanged old history): $BUNDLE"
