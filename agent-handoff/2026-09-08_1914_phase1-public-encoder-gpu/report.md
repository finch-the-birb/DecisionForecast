# Report: phase1-public-encoder-gpu

- authored_by: runner
- created_at: 2026-09-08T19:19:46Z
- request_folder: agent-handoff/2026-09-08_1914_phase1-public-encoder-gpu/
- tested_ref: feat/phase1-data@c9cd99a67a9944ac872254971db67d2b3e4d4c87
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: c9cd99a (code 1553671). HF_TOKEN unset. No FNSPID re-download.
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

## Outcome
- exit_code: uv sync=0; precache=0; train=0
- duration: uv sync ~1s; precache ~1m 35s; train ~1m 50s
- host: runpod pod `p5cnr4o8bnd4c4`
- device: cuda:0 (NVIDIA L4, capability 8.9). No CPU fallback. No HF_TOKEN.

Public encoder Hub GETs returned 200 (rate-limit warning only, no 401). NVDA resolved (`NVDA.csv -> nvda.csv`). Splits: train 64 / 2021-12-16..22; val 32 / 2022-12-16..21; test 32 / 2023-12-13..18; all include NVDA.

## Key signals
- GPU name: NVIDIA L4
- encoder device: **cuda:0** — `Text encoder ready: FinLang/finance-embeddings-investopedia device=cuda:0`
- CUDA MiB after model.to: allocated=**0.6** reserved=**2.0** (warning < 1 MiB; encoder not loaded yet)
- CUDA MiB first train batch: allocated=**427.5** reserved=**498.0** (encoder on GPU)
- first-batch text L2 mean=**0.6250** (> 0)
- first-batch tensors: x=cuda:0 text=cuda:0
- epoch1: train_loss=10.3847 val_mse=5.5947 val_mae=2.1391
- test mse=**18.1757** mae=**3.2777**
- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- hydra run: `outputs/2026-09-08/19-18-32/`
- train_log: `outputs/2026-09-08/19-18-32/train.log`
- checkpoint: `outputs/2026-09-08/19-18-32/checkpoints/best.pt`
- explain: `outputs/2026-09-08/19-18-32/explain/projection_examples_a.json`
- mlflow_run_id: `e5c5be45c2904e7c84b74c821b614c29`

## Conclusions for Dev
1. Public id `FinLang/finance-embeddings-investopedia` unblocks the previous 401. Encoder and TimeXL both on CUDA; first-batch VRAM jumped 0.6 → 427.5 MiB; text L2 0.625.
2. `after model.to` still < 1 MiB because MiniLM-class weights load lazily on first encode. Use **first-batch** CUDA mem as the GPU assert, not `model.to`.
3. Capped 1-epoch test mse 18.18 / mae 3.28 is a smoke number (recent windows, z-scored close, tiny N), not a paper metric.

## Suggested next command (optional)
```bash
# optional: slightly larger cap now that encoder+CUDA path is green
uv run python -m src.training.train \
  model=a \
  train.device=cuda \
  train.ticker_set=dev \
  train.max_train_windows=256 \
  train.max_val_windows=64 \
  train.max_test_windows=64 \
  train.epochs=1
```
