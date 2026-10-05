# Report: phase4-pipeline-smoke

- authored_by: runner
- created_at: 2026-10-05T17:31:29Z
- request_folder: agent-handoff/2026-10-05_2030_phase4-pipeline-smoke/
- tested_ref: feat/phase2-timexer@acbb3c9e0e7941cb9097345b70d1789ba13c7709 (contains bf4853502cb598670b7aff53101160c0fa669039)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# Deploy key copied from /workspace/.ssh into ~/.ssh. Private key not printed, not committed.
# origin already git@github.com:finch-the-birb/DecisionForecast.git (no HTTPS rewrite, no clone).
# HF_TOKEN unset. No FNSPID download. No timexer_selected / a / b / c0 / c1 / ticker_set=paper.
ssh -o BatchMode=yes -T git@github.com
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
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

## Outcome
- exit_code: ssh auth ok; uv sync=0; pytest=0 (106 passed, 3 warnings, 97.30s); train=0
- duration: pytest ~1m 37s; train 2026-10-05T17:29:23Z → 2026-10-05T17:31:29Z
- host: runpod pod `tmjd4ckzfsit8j` (hostname `a3011c4bbbe5`)
- device: `train device: cuda`, GPU NVIDIA L4, `First batch tensors on x=cuda:0 text=cuda:0`. No OOM.

SSH (verbatim, BatchMode):
```
Hi finch-the-birb/DecisionForecast! You've successfully authenticated, but GitHub does not provide shell access.
```

Caps held: train 32 / val 16 / test 16. Experiment `timexer_plain_T60_H7_s42`. `n_params=137927`. First-batch `x=(32, 60, 5)` and `text=(32, 768)` — existing pipeline, not the 40D adapter.

GPU / loader (verbatim):
```
train device: cuda
GPU name: NVIDIA L4
Dataset sizes — train: 32, val: 16, test: 16
DataLoader num_workers=4 pin_memory=True persistent_workers=True
Model parameters on cuda:0 n_params=137927
First batch shapes x=(32, 60, 5) text=(32, 768) text_seq=(32, 60, 768)
First batch tensors on x=cuda:0 text=cuda:0
Epoch 1 — train_loss=1.7206 l_pred=1.7206 l_c=0.0000 l_e=0.0000 l_d=0.0000 val_mse=2.8360 val_mae=1.2883 proto_nn_dist_mean=n/a proto_min_pairwise_dist=n/a
METRICS_ROW model=timexer_plain horizon=7 mse=1.9992 mae=1.2447 mae_denorm=14.4797
```

## Key signals
- metrics: METRICS_ROW mse=1.9992 mae=1.2447 mae_denorm=14.4797 (capped 1-epoch smoke, not a scoreboard)
- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: runner session stdout (marker file `outputs/phase4-pipeline-smoke.log` has only the start/exit lines; Hydra stdout was not redirected into it)
- mlflow_run_id: n/a (not parsed)

## Conclusions for Dev
1. Deploy key authenticates as `finch-the-birb/DecisionForecast`. `HEAD` `acbb3c9` contains `bf48535`. `uv sync` installed `lightgbm==4.7.0`.
2. `pytest tests -q`: **106 passed**. The 40D adapter was not trained; coverage is the new unit tests only.
3. `model=timexer_plain` capped CUDA smoke exited 0 with `METRICS_ROW`. Dataset still emits 5 OHLCV channels and 768D text (`x=(32, 60, 5)`, text dim 768), and this smoke still loaded `mu_dev_2015-01-01_2021-12-31.npy`.
4. Did not run `timexer_selected` / A / B / C0 / C1 / `ticker_set=paper`. Did not re-download FNSPID. Did not commit `outputs/`, `mlflow.db`, or `.ssh/`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
