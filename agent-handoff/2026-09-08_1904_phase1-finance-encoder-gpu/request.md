# Request: phase1-finance-encoder-gpu

- authored_by: dev
- created_at: 2026-09-08T19:04:06Z
- target_ref: feat/phase1-data@95deb30
- phase: 1
- priority: high

## Goal
Capped Model A smoke with the new frozen **finance** encoder (`FinLang/investopedia_embedding`, 768-d) and an explicit CUDA check. Confirm TimeXL **and** the sentence encoder run on the L4, not CPU.

If `agent-handoff/2026-09-08_1854_phase1-fusion-smoke` is still running MiniLM, finish that report first, then run this. If it has not started, **skip MiniLM** and run only this request.

Success:
- `train.device=cuda` does not fall back to CPU
- logs show `GPU name: NVIDIA L4` (or the pod GPU), `Model parameters on cuda`, `Text encoder ready: ... device=cuda`
- `after model.to` and `first train batch` CUDA mem allocated **> 1 MiB**
- first-batch text L2 **> 0** (news windows, not 1981 zeros)
- NVDA present; train exit 0; test mse/mae; projection JSON

## Commands
Do **not** re-download FNSPID. News parquet should already exist.

```bash
uv sync
uv run python scripts/precache_fnspid_news.py --ticker-set=dev
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=64 \
  train.max_val_windows=32 \
  train.max_test_windows=32 \
  train.epochs=1
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes
- gpu: **required** (`train.device=cuda`; fail if CUDA missing)
- approx_ram_gb: 16+ with news parquet cached
- first run downloads `FinLang/investopedia_embedding` (~110M, 768-d). License CC-BY-NC-4.0 (academic OK).
- git: `feat/phase1-data` at this request commit (descendant of `95deb30`)

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run full (uncapped) phase trainings unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.

Key signals to copy from logs: GPU name, encoder device, CUDA MiB after model.to and first batch, text L2, split date ranges, NVDA, test mse/mae.
