# Report: text-protocol

- authored_by: runner
- created_at: 2026-09-10T19:18:30Z
- request_folder: agent-handoff/2026-09-10_1735_text-protocol/
- tested_ref: feat/phase1-data@4443bbea8d147f4fb1cc4a2d1b3df79563bff9fe (contains d97b1fa)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: 4443bbe (handoff request; code d97b1fa). HF_TOKEN unset. No FNSPID download. Not B/C0/C1. Not paper.
uv sync
uv run python -m src.training.train --cfg job >/dev/null
uv run python scripts/precache_fnspid_news.py --ticker-set=dev
uv run python scripts/precache_text_embeddings.py --ticker-set=dev --device=cuda
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
- exit_code: uv sync=0; --cfg job=0; news parquet precache=0; text_precache=0; train a=0
- duration: text_precache ~7m 19s (438852 ms); train a ~1m 36s (95479 ms)
- host: runpod pod `i6lv3n0222en43` (container hostname `650fb48e9ad4`)
- device: cuda:0, NVIDIA L4. Encoder on cuda. First batch tensors `x=cuda:0 text=cuda:0`. No OOM. No HF 401. Not paper.

## Key signals
TEXT_COVERAGE (verbatim from train log):
```
TEXT_COVERAGE ticker=AAPL days_with_news=0.041
TEXT_COVERAGE ticker=MSFT days_with_news=0.043
TEXT_COVERAGE ticker=GOOGL days_with_news=0.097
TEXT_COVERAGE ticker=AMZN days_with_news=0.033
TEXT_COVERAGE ticker=FB days_with_news=0.036
TEXT_COVERAGE ticker=NVDA days_with_news=0.277
TEXT_COVERAGE ticker=TSLA days_with_news=0.188
TEXT_COVERAGE ticker=JPM days_with_news=0.039
TEXT_COVERAGE ticker=BAC days_with_news=0.072
TEXT_COVERAGE ticker=WMT days_with_news=0.097
TEXT_COVERAGE ticker=JNJ days_with_news=0.084
TEXT_COVERAGE ticker=PG days_with_news=0.049
TEXT_COVERAGE ticker=XOM days_with_news=0.089
TEXT_COVERAGE ticker=DIS days_with_news=0.073
TEXT_COVERAGE ticker=NFLX days_with_news=0.178
TEXT_COVERAGE split=train windows=64 mean_has_news_frac=0.517 zero_windows=0.422 mean_text_l2=0.1926
TEXT_COVERAGE split=val windows=32 mean_has_news_frac=0.709 zero_windows=0.281 mean_text_l2=0.2748
TEXT_COVERAGE split=test windows=32 mean_has_news_frac=0.803 zero_windows=0.188 mean_text_l2=0.3304
```

First-batch shapes (verbatim):
```
First batch shapes x=(32, 60, 5) text=(32, 768) text_seq=(32, 60, 768)
```
Also: `First batch text L2 mean=0.2048`. `Text mu ticker_set=dev L2=0.4391 dim=768`.

METRICS_ROW (verbatim):
```
METRICS_ROW model=a horizon=7 mse=18.1787 mae=3.2782
```

Saved projection examples:
`/workspace/DecisionForecast/outputs/2026-09-10/19-16-43/explain/projection_examples_a.json`

Cache files confirmed on disk:
- `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev.npy` (3.2K, L2=0.4391)
- `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/articles/{AAPL,AMZN,BAC,DIS,FB,GOOGL,JNJ,JPM,MSFT,NFLX,NVDA,PG,TSLA,WMT,XOM}.npz`
- `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/{AAPL,AMZN,BAC,DIS,FB,GOOGL,JNJ,JPM,MSFT,NFLX,NVDA,PG,TSLA,WMT,XOM}.npz`

- oom: no
- traceback_summary: n/a
- CUDA warning (non-fatal): `after model.to` / `first train batch` allocated < 1 MiB; tensors still reported on cuda:0

## Artifacts (paths on Runner disk — do not commit binaries)
- train: `outputs/2026-09-10/19-16-43/` (`metrics.json`, `checkpoints/best.pt`, `explain/projection_examples_a.json`, `explain/faithfulness_a.json`)
- mu: `/workspace/DecisionForecast/Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/mu_dev.npy`
- daily series npz: `/workspace/DecisionForecast/Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/*.npz`
- article npz: `/workspace/DecisionForecast/Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/articles/*.npz`

## Conclusions for Dev
1. New text protocol smoke **pass**: `text=(32, 768)` and `text_seq=(32, 60, 768)`; train-only mu L2=0.4391; non-zero first-batch text L2=0.2048.
2. `% windows with has_news_frac==0` is **not** 100%: train `zero_windows=0.422`, val `0.281`, test `0.188`.
3. Ticker `days_with_news` is calendar-day density: NVDA 0.277 / TSLA 0.188 / NFLX 0.178 highest; AAPL 0.041 and MSFT 0.043 are **not** high (same ballpark as FB 0.036). FB/BAC lower-than-mega-cap-window-coverage still holds at window level (`mean_has_news_frac` train 0.517).
4. Model A capped MSE/MAE 18.1787 / 3.2782 is in line with prior endo-OHLCV A smoke (18.1757 / 3.2777). Not a B/C0/C1 comparison.

## Suggested next command (optional)
```bash
# only if Dev wants B/C0/C1 on this protocol — new request.md; do not reuse inverted-exo 2001 numbers
```
