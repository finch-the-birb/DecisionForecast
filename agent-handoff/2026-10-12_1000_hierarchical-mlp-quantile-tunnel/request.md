# Request: hierarchical-mlp-quantile-tunnel

- authored_by: dev
- created_at: 2026-10-12T10:00:00Z
- target_ref: feat/phase2-timexer
- phase: 7
- priority: high

## Goal
1. **Блок 2: Distributional LTSF (Квантильный коридор неопределенности)**:
   - Точечная регрессия цен на зашумленных финансовых рядах при минимизации MSE неизбежно вырождается в безусловное матожидание («линейку под углом»).
   - Реализовано квантильное вероятностное прогнозирование (Distributional LTSF) на горизонте $H=7$:
     - Нижняя граница: 10-й квантиль $q_{0.10}$ (Value-at-Risk / пессимистичный сценарий)
     - Медиана: 50-й квантиль $q_{0.50}$ (наиболее вероятная траектория)
     - Верхняя граница: 90-й квантиль $q_{0.90}$ (оптимистичный сценарий)
   - Архитектура головы `QuantileHead` проецирует латентные токены $G_{\text{ts}}, G_{\text{text}}$ и ценовые патчи через 2-слойный MLP в тензор формы `[B, horizon, 3]`.
   - Обучение проводится строго по Pinball Loss:
     $$\mathcal{L}_{\tau}(y, \hat{q}_{\tau}) = \max\left(\tau (y - \hat{q}_{\tau}), \; (\tau - 1)(y - \hat{q}_{\tau})\right)$$
     с регуляризацией от перехлеста квантилей (Non-crossing penalty, $\lambda_{\text{cross}}=1.0$):
     $$\mathcal{L}_{\text{non\_crossing}} = \text{mean}\left(\text{ReLU}(q_{0.10} - q_{0.50}) + \text{ReLU}(q_{0.50} - q_{0.90})\right)$$
   - Обучение в изолированном режиме (без Multi-Task смешивания с MSE), гарантирующее строгую калибровку доверительного интервала.

2. **Боевой прогон на Paper Set**:
   - Модель `c1_hierarchical_mlp_quantile` обучается на полной выборке `paper` (61 тикер, 98k окон) на 3 сидах (0, 1, 2) на Runpod GPU.

3. **Построение графиков инференса 2023 года**:
   - Визуализация фактических цен 2023 года и прогнозных 80%-х доверительных коридоров $[P_t(1+q_{0.10}), P_t(1+q_{0.90})]$ (заливка) с медианой $P_t(1+q_{0.50})$ для ключевых акций: AAPL, NVDA, MSFT, JPM.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
uv sync

# 1. Быстрый smoke-тест (GPU, проверка сборки QuantileHead и Pinball Loss)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

# 2. Боевые прогоны на Paper Set (61 тикер, 98k окон, сиды 0, 1, 2)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 3. Построение графиков инференса с 80%-м квантильным коридором для лучшего чекпоинта
# (Подставьте путь к лучшему чекпоинту best.pt из outputs/)
BEST_CKPT="outputs/.../checkpoints/best.pt"
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_quantile \
  --stride 7 \
  --device cuda
```

## Environment notes
- expected_cwd: repo root (`/workspace/DecisionForecast`)
- data_ready: yes (FNSPID 61 paper tickers)
- gpu: required (NVIDIA L4 / A100 / RTX 4090)
- approx_ram_gb: 32+ GB system RAM, 16+ GB VRAM

## Expected Metrics & Output
- **Distributional Quantile Metrics**:
  - `quantile_coverage`: фактическое покрытие 80%-го коридора (номинал 0.80)
  - `quantile_sharpness`: средняя ширина тоннеля $\text{mean}(q_{0.90} - q_{0.10})$
  - `quantile_winkler`: Winkler score (интервальный штрафной скор при $\alpha=0.20$)
- **Point Median Metrics ($q_{0.50}$)**:
  - `delta_mse`, `delta_mae`, `delta_da`
  - `test_mse_price`, `test_mae_denorm` ($), `test_da`
- **Inference Plots**:
  - PNG-графики в `outputs/inference_plots_quantile/` с отображением коридоров неопределенности для AAPL, NVDA, MSFT, JPM.
