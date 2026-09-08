# Remove agent-handoff/ from the entire git history (for final / paper archives).
# Rewrites history. Requires: git-filter-repo (pip install git-filter-repo)
[CmdletBinding()]
param(
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

try {
    $Root = git rev-parse --show-toplevel 2>$null
    if (-not $Root) { throw "not inside a git repository" }
} catch {
    Write-Error "error: not inside a git repository"
    exit 1
}

Set-Location $Root

$filterRepo = Get-Command git-filter-repo -ErrorAction SilentlyContinue
if (-not $filterRepo) {
    Write-Error @"
error: git-filter-repo not found on PATH.
Install:  pip install git-filter-repo
Docs:     https://github.com/newren/git-filter-repo
"@
    exit 1
}

Write-Host "This will REWRITE git history and delete path: agent-handoff/"
Write-Host "Repo: $Root"
Write-Host "Protocol docs under agents/ are kept."

if (-not $Yes -and $env:PURGE_AGENT_HANDOFF_I_UNDERSTAND -ne "yes") {
    $confirm = Read-Host "Type yes to continue"
    if ($confirm -ne "yes") {
        Write-Host "aborted"
        exit 1
    }
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$parent = Split-Path -Parent $Root
$repoName = Split-Path -Leaf $Root
$bundle = Join-Path $parent "$repoName-pre-purge-agent-handoff-$stamp.bundle"
git bundle create $bundle --all
Write-Host "Created recovery bundle: $bundle"
Write-Host "  Restore example: git clone `"$bundle`" recover-pre-purge"

git filter-repo --force --invert-paths --path agent-handoff/

Write-Host ""
Write-Host "Done. agent-handoff/ removed from history."
Write-Host "Next steps:"
Write-Host "  1. Inspect: git log --all -- agent-handoff/   (should be empty)"
Write-Host "  2. Re-add remote if filter-repo cleared it, then force-push intentionally:"
Write-Host "       git remote add origin <url>"
Write-Host "       git push --force-with-lease origin --all"
Write-Host "       git push --force-with-lease origin --tags"
Write-Host "  3. Ask collaborators to re-clone or hard-reset to the new history."
Write-Host "Recovery bundle (unchanged old history): $bundle"
