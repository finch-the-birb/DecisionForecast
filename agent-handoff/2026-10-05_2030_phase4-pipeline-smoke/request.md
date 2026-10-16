# Request: phase4-pipeline-smoke

- authored_by: dev
- created_at: 2026-10-05T17:30:00Z
- target_ref: feat/phase2-timexer@bf4853502cb598670b7aff53101160c0fa669039
- phase: 4
- priority: high

## Goal
Confirm the Phase 4 branch runs on the pod. Copy the GitHub deploy key that is already on the volume, fast-forward `feat/phase2-timexer` to `bf48535` (40D TimeXer + walk-forward; parent commits add causal features, compact text, and the train-only selector), then pass `pytest` and one capped CUDA train of the existing pipeline.

Success: key authenticates to GitHub; `HEAD` contains `bf48535`; `pytest` exits 0; `model=timexer_plain` smoke exits 0 on CUDA with a `METRICS_ROW`; no OOM. Status `partial` if tests pass but the train fails.

`model=timexer_selected` is not trainable on FNSPID yet. The dataset still emits 5 OHLCV channels and 768D text. The 40D adapter is covered by `tests/test_timexer_selected.py` only.

## Commands
Deploy key on this pod (see the workspace listing): `/workspace/.ssh/id_ed25519_github` and `/workspace/.ssh/id_ed25519_github.pub`. Copy it into `~/.ssh` before any `git fetch`. Do not print the private key. Do not commit it.

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cp /workspace/.ssh/id_ed25519_github ~/.ssh/id_ed25519_github
cp /workspace/.ssh/id_ed25519_github.pub ~/.ssh/id_ed25519_github.pub
chmod 600 ~/.ssh/id_ed25519_github
chmod 644 ~/.ssh/id_ed25519_github.pub
touch ~/.ssh/config
chmod 600 ~/.ssh/config
grep -q "IdentityFile ~/.ssh/id_ed25519_github" ~/.ssh/config || cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
ssh -o BatchMode=yes -T git@github.com || true
```

Repo root is `/workspace/DecisionForecast`. If that directory is missing, clone it with the key just installed: `git clone git@github.com:finch-the-birb/DecisionForecast.git /workspace/DecisionForecast`. If `origin` is HTTPS, point it at SSH: `git remote set-url origin git@github.com:finch-the-birb/DecisionForecast.git`.

```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
git merge-base --is-ancestor bf4853502cb598670b7aff53101160c0fa669039 HEAD

uv sync
uv run pytest tests -q

uv run python -m src.training.train \
  model=timexer_plain \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=32 \
  train.max_val_windows=16 \
  train.max_test_windows=16 \
  train.epochs=1 \
  data.horizon=7
```

FNSPID and the text cache are already on the volume. Do not re-download. Do not set `HF_TOKEN`. Grep `METRICS_ROW` from the train log.

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID + `cache/text_series/` from earlier rounds; `timexer_plain` does not read text)
- gpu: required (`train.device=cuda`)
- approx_ram_gb: 16+
- git: `feat/phase2-timexer`, descendant of `bf4853502cb598670b7aff53101160c0fa669039`
- ssh: deploy key only from `/workspace/.ssh/id_ed25519_github`

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock; prefer reporting the blocker.
- Do not run `model=timexer_selected`, `model=a`, `model=b`, `model=c0`, `model=c1`, or `ticker_set=paper`.
- Do not clear `max_train_windows` or raise the caps above the command.
- Do not commit `outputs/`, `mlruns/`, checkpoints, `.env`, datasets, or anything under `.ssh/`.

## After the run
Write `report.md` in this same folder, commit on `feat/phase2-timexer`, push.
