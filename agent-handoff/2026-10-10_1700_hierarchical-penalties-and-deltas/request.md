# Request: hierarchical-penalties-and-deltas

- authored_by: dev
- created_at: 2026-10-10T17:00:00Z
- target_ref: feat/phase2-timexer@f86327f
- phase: 6
- priority: high

## Goal

Борьба с линейным вырождением ("линейкой под углом" с наклоном $\approx 15^\circ–20^\circ$) в канонической модели `c1_hierarchical` ($L=3$, `fft_mode: bridge`). Проверка двух независимых исследовательских гипотез:

1. **Эксперимент 1 (Loss Penalties Grid Search, 4 конфига $\times$ 3 сида = 12 прогонов)**:
   - Добавление аддитивных штрафов за направление ($\gamma_{\text{dir}} \in \{0.1, 0.2\}$) и за форму/корреляцию траектории ($\alpha_{\text{corr}} \in \{0.1, 0.3\}$).
   - Конфигурации:
     - `c1_hierarchical_penalties_g01_a01` ($\gamma_{\text{dir}}=0.1, \alpha_{\text{corr}}=0.1$)
     - `c1_hierarchical_penalties_g01_a03` ($\gamma_{\text{dir}}=0.1, \alpha_{\text{corr}}=0.3$)
     - `c1_hierarchical_penalties_g02_a01` ($\gamma_{\text{dir}}=0.2, \alpha_{\text{corr}}=0.1$)
     - `c1_hierarchical_penalties_g02_a03` ($\gamma_{\text{dir}}=0.2, \alpha_{\text{corr}}=0.3$)

2. **Эксперимент 2 (Delta / Return Forecasting, 1 конфиг $\times$ 3 сида = 3 прогона)**:
   - Прогнозирование относительных приращений (доходностей) $y_{\text{delta}} = \frac{P_{t+h} - P_t}{P_t} \in \mathbb{R}^H$.
   - Конфигурация: `c1_hierarchical_delta` (`data.target_mode: delta`).
   - Логирование двух сетов метрик:
     - **Сет А (delta space)**: `delta_mse`, `delta_mae`, `delta_da`.
     - **Сет Б (price space)**: $\hat{P}_{t+h} = P_t \cdot (1 + \hat{y}_{\text{delta}, h}) \rightarrow$ `test_mae_denorm` (долларовая ошибка \$), `test_da` (Directional Accuracy), `test_mse_price` (нормализованный ценовой MSE).
   - Построение траекторий инференса 2023 года для проверки устранения монотонного наклона.

## Commands

```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
uv sync

# 0. Smoke-проверка на GPU (1 эпоха, capped)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_penalties_g01_a01 \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_delta \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

# ==============================================================================
# СВИП 1: Grid Search штрафов за направление и форму (paper set, 61 тикер, 3 сида)
# ==============================================================================

# 1. c1_hierarchical_penalties_g01_a01
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_penalties_g01_a01 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 2. c1_hierarchical_penalties_g01_a03
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_penalties_g01_a03 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 3. c1_hierarchical_penalties_g02_a01
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_penalties_g02_a01 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 4. c1_hierarchical_penalties_g02_a03
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_penalties_g02_a03 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# ==============================================================================
# СВИП 2: Прогнозирование приращений (Delta Forecasting) (paper set, 3 сида)
# ==============================================================================

for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_delta \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# ==============================================================================
# ВИЗУАЛИЗАЦИЯ ИНФЕРЕНСА 2023 ГОДА
# ==============================================================================

# Найти лучший чекпоинт среди штрафов
BEST_PENALTIES_CKPT=$(find outputs -path "*penalties*" -name "best.pt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "Plotting inference for best penalties ckpt: $BEST_PENALTIES_CKPT"
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_PENALTIES_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots/penalties \
  --stride 7 \
  --device cuda

# Найти лучший чекпоинт среди delta
BEST_DELTA_CKPT=$(find outputs -path "*delta*" -name "best.pt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "Plotting inference for best delta ckpt: $BEST_DELTA_CKPT"
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_DELTA_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots/delta \
  --stride 7 \
  --device cuda
```

## Environment notes

- expected_cwd: `/workspace/DecisionForecast`
- data_ready: yes (кэш `technical` и `text_compact` готовы)
- gpu: required (CUDA)
- approx_ram_gb: >= 32
- **ВАЖНО**: строго использовать `train.ticker_set=paper` (61 тикер, 98k окон).
- Все метрики логируются в MLflow и в `metrics.json`.

## After the run

1. Собрать сводную таблицу по 3 сидам (mean ± std):
   - **Эксперимент 1 (Штрафы)**: Test MSE, Test MAE, Directional Accuracy (DA), MAE denorm (\$).
   - **Эксперимент 2 (Дельты)**: delta_mse, delta_mae, delta_da, test_mae_denorm (\$), test_da, test_mse_price.
2. Проверить графики инференса в `outputs/inference_plots/penalties` и `outputs/inference_plots/delta` на предмет исчезновения эффекта "линейки под углом".
3. Оформить отчёт `agent-handoff/2026-10-10_1700_hierarchical-penalties-and-deltas/report.md`, закоммитить и запушить в `feat/phase2-timexer`.
