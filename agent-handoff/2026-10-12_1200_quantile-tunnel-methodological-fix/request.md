# Request: quantile-tunnel-methodological-fix

- authored_by: dev
- created_at: 2026-10-12T12:00:00Z
- target_ref: feat/phase2-timexer
- phase: 7
- priority: high

## Goal
1. **Методологическое исправление пайплайна Distributional LTSF**:
   - В предыдущем эксперименте чекпоинтинг `best.pt` шёл по MSE медианы, что противоречило задаче калибровки вероятностного коридора. Теперь чекпоинтинг и early stopping настраиваемы и по умолчанию для квантильной модели оптимизируют **Winkler Score** (`val_metric: winkler`, patience=7).
   - В `QuantileHead` добавлена строгая архитектурная гарантия монотонности квантилей через `torch.sort(out, dim=-1)`, исключающая пересечение $q_{0.10} \le q_{0.50} \le q_{0.90}$ на 100% входных данных.
   - Внедрен расчёт бейзлайна наивного исторического коридора волатильности окна (Naive Historical Volatility Corridor, $\pm 1.28 \cdot \sigma \cdot \sqrt{h}$) для честного сравнения на тесте.
   - Интервальные метрики (`cov80`, `sharp`, `winkler`) выведены на первое место в отчётности и логах.

2. **Боевой перезапуск на Paper Set**:
   - Обучение `c1_hierarchical_mlp_quantile` на полной выборке `paper` (61 тикер, 98k окон) на 3 сидах (0, 1, 2) на Runpod GPU.

3. **Визуализация инференса 2023 года со скользящим шагом**:
   - Построение коридора с шагом `stride=1` (гладкий непрерывный коридор неопределенности вдоль всего 2023 года) и `stride=7` (кусочно-непрерывные 7-дневные прогнозы) для акций AAPL, NVDA, MSFT, JPM.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
uv sync

# 1. Быстрый GPU smoke-тест (проверка отбора по winkler и монотонности)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda train.val_metric=winkler

# 2. Боевые прогоны на Paper Set (61 тикер, 98k окон, сиды 0, 1, 2)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 3. Визуализация коридора инференса 2023 года для лучшего чекпоинта
BEST_CKPT=$(ls -t outputs/c1_hierarchical_mlp_quantile_*/checkpoints/best.pt | head -n 1)

# А) Скользящий непрерывный тоннель (stride=1)
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_stride1 \
  --stride 1 \
  --device cuda

# Б) Кусочный 7-дневный веер (stride=7)
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_stride7 \
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
  - `quantile_coverage`: покрытие 80%-го коридора (целевой номинал 0.80)
  - `quantile_sharpness`: средняя ширина тоннеля
  - `quantile_winkler`: интервальный скор Винклера (критерий отбора чекпоинта)
- **Naive Historical Corridor Baseline**:
  - `naive_coverage`, `naive_sharpness`, `naive_winkler`
- **Median Point Metrics ($q_{0.50}$)**:
  - `delta_mse`, `delta_mae`, `delta_da`
  - `test_mse_price`, `test_mae_denorm` ($), `test_da`
- **Inference Plots**:
  - `outputs/inference_plots_stride1/`: непрерывный коридор неопределенности
  - `outputs/inference_plots_stride7/`: дискретные 7-дневные прогнозы
