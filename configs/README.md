# Experiment configs (Hydra)

Forecasting article protocol: **TimeXL × TimeXer** on FNSPID.

## Models (ablations)

| Config | ID | Backend | Fusion |
|--------|----|---------|--------|
| `model=a` | A | 1D-CNN (TimeXL) | Late |
| `model=b` | B | TimeXer | Late |
| `model=c0` | C0 | TimeXer | Mid, no cross-attn |
| `model=c1` | C1 | TimeXer | Mid + G_en cross-attn |

## Fixed protocol

- Task: LTSF regression (MSE/MAE), horizons `H ∈ {7, 14, 30}`.
- Look-back `T=60`, patch `P=12`, stride `6`.
- Temporal split (no leakage): train ≤ 2021-12-31, val ≤ 2022-12-31, test > 2022-12-31.
- Dev tickers: 15 liquid symbols in `data/fnspid.yaml`.
- Text: frozen **finance** sentence embeddings (`FinLang/finance-embeddings-investopedia`, 768-d).
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
