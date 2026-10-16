# Request: paper-economic-eval

- authored_by: planner
- created_at: 2026-10-14T01:00:00Z
- request_folder: agent-handoff/2026-10-14_1400_paper-economic-eval/
- target_branch: feat/phase2-timexer
- status: pending

---

## 1. Goal & Context
Implement `scripts/paper_economic_eval.py` to evaluate the economic utility of the predicted quantile corridors across the 3 Walk-Forward folds on the 61-ticker `paper` set.
This fulfills Step 3 of Stage 2 for the academic paper: Quantile-Driven Portfolio Evaluation.

## 2. Guardrails & Architecture Rules
- Strictly NO trading execution logic in `src/`. All simulation logic lives in `scripts/paper_economic_eval.py`.
- No heavy retraining locally. Offline evaluation operates directly on saved tensors (`test_predictions.pt` and `calibrated_predictions.pt` in `outputs/`).
- Python invocation strictly via `uv run python`.

## 3. Signal Formulation & Portfolio Logic
For each ticker $i$ on day $t$ for horizon $H=7$:
1. **Corridor-Adjusted Signal**:
   $$S_{i, t} = \frac{q_{0.50, i, t+H} - P_{i, t}}{q_{0.90, i, t+H} - q_{0.10, i, t+H} + \epsilon}$$
   (Expected return scaled by local forecast corridor width / uncertainty).
2. **Portfolio Weights**:
   - Long-Short Decile / Top-K ($K=10$): Long top $K$ signals, Short bottom $K$ signals (dollar-neutral).
   - Long-Only Top-K ($K=10$): Long top $K$ signals with positive drift ($q_{0.50} > P_{i, t}$).
   - Transaction costs: Fixed 10 bps ($0.0010$) turnover penalty.
3. **Benchmarks for Comparison**:
   - Baseline 1: Equal-Weight Buy & Hold (all 61 stocks).
   - Baseline 2: GARCH(1,1) Volatility-Scaled Signal: $(r_{\text{roll}} / \sigma_{\text{garch}})$.
   - Baseline 3: Directional-Only TimeXer ($\text{sign}(q_{0.50} - P_{i, t})$ without corridor scaling).
   - Baseline 4: TimeXer Raw Quantiles vs TimeXer ACI Conformal Quantiles.
4. **Report Metrics**:
   - Annualized Return (CAGR)
   - Annualized Volatility
   - Sharpe Ratio ($R_f = 0$ or historical risk-free rate)
   - Sortino Ratio (downside deviation)
   - Maximum Drawdown (MDD)
   - Turnover & Net Sharpe (after 10 bps fees)

## 4. Deliverables
1. `scripts/paper_economic_eval.py`:
   - CLI flags: `--fold {1,2,3,all}`, `--ticker-set paper`, `--cost-bps 10.0`, `--top-k 10`, `--rebalance-stride 7`.
   - Aggregates metrics across seeds 42, 43, 44 and reports Mean ± Std.
   - Outputs a markdown table and LaTeX table for the paper.
2. Verification:
   - Run a dry-run test on Fold 1/Seed 42 locally.
   - Ensure all existing unit tests pass: `uv run pytest`.

---

## 5. Instructions for Runner (Execution on RunPod)
After completing training of models on Folds 1, 2, 3:
```bash
# Run economic evaluation across all 3 Walk-Forward folds
uv run python scripts/paper_economic_eval.py \
  --fold all \
  --ticker-set paper \
  --cost-bps 10.0 \
  --top-k 10 \
  --rebalance-stride 7
```
Output Markdown (`outputs/economic_eval/economic_eval_fold*.md`) and LaTeX (`outputs/economic_eval/economic_eval_fold*.tex`) tables will be automatically generated. Include these tables in the final paper report `report.md`.

