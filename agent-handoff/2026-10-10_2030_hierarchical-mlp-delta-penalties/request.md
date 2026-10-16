# Request: hierarchical-mlp-delta-penalties

- authored_by: dev
- created_at: 2026-10-10T20:30:00Z
- target_ref: feat/phase2-timexer
- phase: 6
- priority: high

## Goal
1. **TimeXerHierarchicalMLP (Non-linear MLP Head)**:
   - Переход с линейной головы `FlattenHead = nn.Linear` на 2-слойный MLP (`Flatten -> Linear(head_nf, 128) -> GELU -> Dropout(0.1) -> Linear(128, 7)`). Это устраняет физическую склонность модели вырождаться в жесткую прямую линию и позволяет генерировать нелинейную форму прогноза (вогнутость, выпуклость, локальные экстремумы) внутри горизонта H=7.
2. **Интеграция rFFT гармоник в пул признаков и расширение до Top-30**:
   - Реализован каузальный расчет 10 гармоник rFFT (`fft_harm_1`..`fft_harm_10`, периоды от 6 до 60 дней) по скользящему окну Close $t-59..t$.
   - Расширен отбор TreeSHAP до Top-30 (`top_k=30`), где гармоники конкурируют на равных с индикаторами волатильности, момента, объема и Беккера при пороге Спирмена $|r| < 0.85$.
3. **Дельта-свип с усиленными штрафами (5 конфигураций)**:
   - Запуск свипа по 3 сидам (0, 1, 2) на полной выборке `paper` (61 тикер, ~98k окон) для 5 комбинаций штрафов:
     1. `c1_hierarchical_mlp_delta_g02_a03`: $\gamma_{\text{dir}}=0.2, \alpha_{\text{corr}}=0.3$ (контрольная точка)
     2. `c1_hierarchical_mlp_delta_g04_a03`: $\gamma_{\text{dir}}=0.4, \alpha_{\text{corr}}=0.3$
     3. `c1_hierarchical_mlp_delta_g04_a05`: $\gamma_{\text{dir}}=0.4, \alpha_{\text{corr}}=0.5$
     4. `c1_hierarchical_mlp_delta_g06_a03`: $\gamma_{\text{dir}}=0.6, \alpha_{\text{corr}}=0.3$
     5. `c1_hierarchical_mlp_delta_g06_a05`: $\gamma_{\text{dir}}=0.6, \alpha_{\text{corr}}=0.5$
4. **Двойной набор метрик и визуализация инференса 2023 года**:
   - Логирование метрик как в пространстве приращений (`delta_mse`, `delta_mae`, `delta_da`), так и в пространстве развернутой цены $\hat{P}_t = P_t(1 + \hat{r}_t)$ (`test_mse_price`, `test_mae_denorm` ($), `test_da`).
   - Построение траекторий инференса 2023 года (`src.evaluation.plot_inference`) для лучшего чекпоинта свипа.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
uv sync

# 1. Перегенерация технического кэша с rFFT гармониками и отбор сигнатуры Top-30
uv run python scripts/prepare_selected_data.py --ticker-set=paper --fold=1 --device=cuda --top-k=30

# 2. Быстрый smoke-тест (GPU, проверка сборки TimeXerHierarchicalMLP и dual_ts 30 признаков)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g04_a03 \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

# 3. Боевой прогон Конфигурации 1: g02_a03 (gamma=0.2, alpha=0.3, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g02_a03 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 4. Боевой прогон Конфигурации 2: g04_a03 (gamma=0.4, alpha=0.3, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g04_a03 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 5. Боевой прогон Конфигурации 3: g04_a05 (gamma=0.4, alpha=0.5, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g04_a05 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 6. Боевой прогон Конфигурации 4: g06_a03 (gamma=0.6, alpha=0.3, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g06_a03 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 7. Боевой прогон Конфигурации 5: g06_a05 (gamma=0.6, alpha=0.5, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g06_a05 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 8. Построение графиков инференса 2023 года для лучшего чекпоинта свипа
# (Укажите путь к best.pt модели с минимальным test_mse_price)
BEST_CKPT="outputs/.../checkpoints/best.pt"
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots \
  --stride 7 \
  --device cuda
```

## Environment notes
- expected_cwd: repo root (`/workspace/DecisionForecast`)
- data_ready: yes (FNSPID 61 paper tickers)
- gpu: required (NVIDIA L4 / A100 / RTX 4090)
- approx_ram_gb: 32+ GB system RAM, 16+ GB VRAM

## Expected Metrics & Output
- **Delta Space**: `delta_mse`, `delta_mae`, `delta_da`
- **Unrolled Price Space**: `test_mse_price`, `test_mae_denorm` ($), `test_da`
- **Inference Plots**: PNG-графики в `outputs/inference_plots/` с визуализацией формы траекторий цен 2023 года для оценки нелинейности и устранения эффекта "линейки под углом".
