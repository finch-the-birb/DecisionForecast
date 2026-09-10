# Experiment configs (Hydra)

Forecasting article protocol: **TimeXL × TimeXer** on FNSPID.

## Models (ablations)

| Config | ID | Backend | Fusion |
|--------|----|---------|--------|
| `model=a` | A | 1D-CNN (TimeXL) | Late |
| `model=b` | B | TimeXer | Late |
| `model=c0` | C0 | TimeXer | Mid, no cross-attn |
| `model=c1` | C1 | TimeXer | Mid + G_en cross-attn |
| `model=timexer_plain` | — | TimeXer | none (sanity) |
| `model=dlinear` | — | DLinear | none (sanity) |

## Fixed protocol

- Task: LTSF regression (MSE/MAE), horizons `H ∈ {7, 14, 30}`.
- Look-back `T=60`, patch `P=12`, stride `6`.
- Temporal split (no leakage): train ≤ 2021-12-31, val ≤ 2022-12-31, test > 2022-12-31.
- Dev tickers: 15 liquid symbols in `configs/data/fnspid.yaml`.
- Look-backs: core `lookback_T=60`; sensitivity grid `lookbacks: [36, 60, 96]`.
- Text protocol (per article, then daily series — **not** concat+`max_chars`):
  encoder [`FinLang/finance-embeddings-investopedia`](https://huggingface.co/FinLang/finance-embeddings-investopedia)
  (768-d, public, no `HF_TOKEN`). Article text = `Lsa_summary`, else `Article_title`.
  Daily `E[d]` = mean of articles in `[prev_trading_day, d)` minus train-only `mu`,
  else carry with `missing_policy=decay` (`λ=0.03`). Window pool: `recency_weighted`.
  OHLCV is endogenous (all channels patched). Exogenous for C1 is **text only**.
  A/B: late fusion. C0: mid add to patches (no \(G_{en}\)→text). C1: text as the sole exo token via \(G_{en}\).

## Hypotheses

- **H1:** A vs B (backbone effect at fixed late fusion). Smoke: `model=a` then `model=b`.
- **H2:** B vs C0 vs C1 (fusion ablation). Smoke: `model=a,b,c0,c1` with the same caps.
- **H3:** Explanatory preservation (projection + faithfulness; Phase 4).
  Smoke: `model=a,b,c1` capped train (writes `explain/`) or `python -m src.explain.run` on a `best.pt`.

## Commands

```bash
uv sync
uv run python -m src.training.train --cfg job
uv run python scripts/precache_text_embeddings.py --ticker-set=dev
# lookback sensitivity (Model A)
uv run python -m src.training.train -m data.lookback_T=36,60,96 model=a train.ticker_set=dev
# Phase 2.1 sanity (no text in the model)
for m in dlinear timexer_plain; do
  uv run python -m src.training.train model=$m train.device=cuda train.ticker_set=dev train.epochs=10
done
# H2
for m in a b c0 c1; do
  uv run python -m src.training.train model=$m train.device=cuda train.ticker_set=dev train.epochs=1
done
# H3 (train writes explain/projection + faithfulness for a/b/c1)
for m in a b c1; do
  uv run python -m src.training.train model=$m train.device=cuda train.ticker_set=dev train.epochs=1
done
# H3 only, existing checkpoint:
uv run python -m src.explain.run model=a explain.checkpoint=outputs/<run>/checkpoints/best.pt \
  train.device=cuda train.ticker_set=dev
```
