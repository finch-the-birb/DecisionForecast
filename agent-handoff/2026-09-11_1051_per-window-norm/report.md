# Report: per-window-norm

- authored_by: runner
- created_at: 2026-09-11T11:18:10Z
- request_folder: agent-handoff/2026-09-11_1051_per-window-norm/
- tested_ref: feat/phase2-timexer@276123ecd901f0f86805b095e2cbee0487097d3a (contains 9da44d1)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No window caps. Not b/c0/c1.
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
# HEAD=276123e (contains 9da44d1)
uv sync
MU="Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy"
# $MU already present (3.2K, 2026-09-10); skipped precache
SHARED="train.device=cuda train.ticker_set=dev train.epochs=10 data.horizon=7"
uv run python -m src.training.train model=dlinear $SHARED
uv run python -m src.training.train model=a $SHARED
uv run python -m src.training.train model=timexer_plain $SHARED
```

## Outcome
- exit_code: uv sync=0; dlinear=0; a=0; timexer_plain=0
- duration: three trains ~14m 17s (856767 ms wall from first START). Early stopping (patience=5): dlinear ep8, a ep7, timexer_plain ep6 (not full 10). Precache skipped.
- host: runpod pod `fprbi35eab7xe4` (hostname `c658b923edf4`)
- device: cuda:0, NVIDIA L4. First batch `x=cuda:0 text=cuda:0`. No OOM. Not paper. Uncapped: train 23478 / val 2510 / test 2480.

config_resolved (all three): `data.normalize: per_window`, `data.split.train_start: '2015-01-01'`, `train.max_*_windows: null`, `train.ticker_set: dev`, `train.epochs: 10`. metrics.json also has `"normalize": "per_window"`.

Mu reused: `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy` L2=0.4390 dim=768.

## Key signals
TEXT_COVERAGE split (verbatim, dlinear load; same sizes on a/plain):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

METRICS_ROW (verbatim):
```
METRICS_ROW model=dlinear horizon=7 mse=0.6733 mae=0.5762 mae_denorm=5.2572
METRICS_ROW model=a horizon=7 mse=0.7300 mae=0.6182 mae_denorm=5.7072
METRICS_ROW model=timexer_plain horizon=7 mse=0.7159 mae=0.5976 mae_denorm=5.4825
```

n_params (verbatim):
```
Model parameters on cuda:0 n_params=854        # dlinear
Model parameters on cuda:0 n_params=144199     # a
Model parameters on cuda:0 n_params=146695     # timexer_plain
```

Last-epoch val_mse (logs, not json): dlinear ep8 val_mse=0.7241; a ep7 val_mse=0.8349; timexer_plain ep6 val_mse=0.7762.

Table from the three `metrics.json` + last-epoch val from logs (n_params from logs):

| model | val_mse (last epoch) | test_mse | test_mae | test_mae_denorm | n_params |
| dlinear | 0.7241 | 0.6733 | 0.5762 | 5.2572 | 854 |
| a | 0.8349 | 0.7300 | 0.6182 | 5.7072 | 144199 |
| timexer_plain | 0.7762 | 0.7159 | 0.5976 | 5.4825 | 146695 |

json: dlinear mse=0.6733139 mae=0.5761713 mae_denorm=5.2572427; a mse=0.7300007 mae=0.6182327 mae_denorm=5.7071896; timexer_plain mse=0.7158975 mae=0.5975943 mae_denorm=5.4824972.

All three `test_mse` in **0.1–1.5**. Val-MSE and test-MSE same order of magnitude (no 5–18 test). CUDA, METRICS_ROW, `test_mae_denorm` present.

Vs protocol-norm (20-ep budget, same protocol): dlinear test_mse identical 0.6733; plain 0.7159 vs 0.7174; a 0.7300 vs 0.9835 (this round early-stopped at ep7 instead of ep11).

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- dlinear: `outputs/2026-09-11/11-04-00/` (`metrics.json`, `train.log`, `config_resolved.yaml`)
- a: `outputs/2026-09-11/11-08-52/` (`explain/`)
- timexer_plain: `outputs/2026-09-11/11-13-57/`
- mu: `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev_2015-01-01_2021-12-31.npy`

## Conclusions for Dev
1. Confirmation on HEAD `276123e` (includes `tests/test_per_window_norm.py`): uncapped dlinear / a / timexer_plain, H=7, 10 epochs, all exit 0, CUDA, `normalize=per_window` + `train_start=2015-01-01`.
2. Status **pass**: test_mse 0.67 / 0.73 / 0.72, last-epoch val 0.72 / 0.83 / 0.78 — same order, none in 5–18.
3. Did not run b/c0/c1, paper, window caps, or lr tune. Did not change product code. Did not rebuild μ.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
