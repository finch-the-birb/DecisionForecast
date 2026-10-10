# Request: canonical-timexer-fft-ablation

- authored_by: dev
- created_at: 2026-10-10T13:30:00Z
- target_ref: feat/phase2-timexer@9b22ad2
- phase: 6
- priority: high

## Goal
1. **Канонический TimeXer (NeurIPS 2024 / TSLib)**:
   - Полный переход моделей `c1_dual` и `c1_hierarchical` на канонический TimeXer с `EnEmbeddingDual`, 2 глобальными токенами ($G_{ts}$ и $G_{text}$), инвертированным эмбеддингом технических индикаторов и `FlattenHead`.
   - Устранены устаревшие гейты `_GatedGlobalToPatch` и предварительные кросс-аттеншены до энкодера.
2. **Абляция интеграции частотной составляющей (FFT)**:
   - **Вариант А (`fft_mode=patch`)**: локальный аддитивный спектральный эмбеддинг патчей цен через `PatchFFT` + `Linear(6, d_model)`.
   - **Вариант Б (`fft_mode=bridge`)**: глобальный спектр 60-дневного окна Close (первые 6 гармоник) опрашивается мостом $G_{ts}$ в Кросс-Аттеншене.
3. **Боевой запуск на полной выборке `paper` (61 тикер, ~98k окон, горизонт H=7)**:
   - Свип по 3 сидам (0, 1, 2) на `train.ticker_set=paper` для всех 4 конфигураций:
     1. `c1_dual_patch_fft`
     2. `c1_dual_bridge_fft`
     3. `c1_hierarchical_patch_fft`
     4. `c1_hierarchical_bridge_fft`
4. **Метрики**:
   - Расчет Test MSE, Test MAE, Directional Accuracy (DA) и денормализованного MAE.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
uv sync

# 1. Быстрый smoke-тест (проверка сборки и логирования)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_dual_patch_fft \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

# 2. c1_dual_patch_fft (paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual_patch_fft \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 3. c1_dual_bridge_fft (paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual_bridge_fft \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 4. c1_hierarchical_patch_fft (paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_patch_fft \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 5. c1_hierarchical_bridge_fft (paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_bridge_fft \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 6. Построение графиков инференса 2023 года для лучшего чекпоинта
BEST_CKPT=$(find outputs -name "best.pt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "Plotting inference for best checkpoint: $BEST_CKPT"
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots \
  --stride 7 \
  --device cuda
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (кэш technical и сигнатура Fold 1 для paper готовы)
- gpu: required (CUDA)
- approx_ram_gb: >= 32
- **ВАЖНО**: флаг `train.ticker_set=paper` обязателен (61 тикер, ~98k окон), не использовать дефолтный dev-сет (15 тикеров).

## Out of scope
- Не перезапускать DLinear (бейзлайн 0.7535 зафиксирован).
- Не изменять продуктовый код на Runner хосте.

## After the run
1. Собрать метрики (test_mse, test_mae, test_da, mae_denorm) по всем 3 сидам для каждой из 4 моделей.
2. Составить сводную таблицу сравнения двух способов внедрения FFT (Patch vs Bridge) в Dual и Hierarchical архитектурах.
3. Записать отчёт в `agent-handoff/2026-10-10_1330_canonical-timexer-fft-ablation/report.md`, закоммитить и запушить в `feat/phase2-timexer`.
