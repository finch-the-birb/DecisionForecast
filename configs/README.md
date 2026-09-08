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
- Text: frozen sentence embeddings (offline cache), late fusion in A/B.

## Hypotheses

- **H1:** A vs B (backbone effect at fixed late fusion).
- **H2:** B vs C0 vs C1 (fusion ablation).
- **H3:** Explanatory preservation (projection + faithfulness; Phase 4).

## Commands

```bash
uv sync
uv run python -m src.training.train --cfg job
uv run python -m src.training.train model=a data.horizon=7 train.epochs=2
```
