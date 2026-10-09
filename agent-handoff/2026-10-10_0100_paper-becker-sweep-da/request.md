# Request: paper-becker-sweep-da

- authored_by: dev
- created_at: 2026-10-10T01:00:00Z
- target_ref: feat/phase2-timexer@b5cb333
- phase: 6
- priority: high

## Goal
1. **Боевой прогон на полной выборке `paper` (61 тикер, ~98k окон, горизонт H=7)**:
   - Чистый боевой прогон трёх сильнейших архитектур со стабилизированным CFI Бекера и обучаемыми Query-токенами на полной выборке `paper` по 3 сидам (0, 1, 2).
2. **Метрика Directional Accuracy (DA)**:
   - Автоматически рассчитывать и логировать метрику направления ценового движения на 7 дней вперед ($\Delta y = y_{[:, -1]} - Close_T, \Delta \hat{y} = \hat{y}_{[:, -1]} - Close_T$) для валидации и теста.
3. **Визуализация инференса (траектории 2023 года)**:
   - Сгенерировать графики инференса на тестовом периоде 2023 года для 4 ключевых тикеров (`AAPL,NVDA,MSFT,JPM`) с помощью скрипта `src.evaluation.plot_inference`.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer

# 1. Синхронизация окружения (включая matplotlib)
uv sync

# 2. Быстрый smoke-тест (проверка логирования DA и инференса)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

# 3. c1_late_fusion (F5 OHLCV, e=2, stride=6, paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_late_fusion \
    data.patch_stride=6 model.e_layers=2 model.use_prototypes=false \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 4. c1_inverted с обучаемым text_token (F5 OHLCV, e=1, stride=12, paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_inverted \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 5. c1_hierarchical с обучаемыми ts_token и text_token (F2: close+volume, e=2, stride=12, paper set, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=2 model.use_prototypes=false \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 6. Построение графиков инференса (траектории 2023 года) для лучшего чекпоинта раунда
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

## Out of scope
- Не перезапускать DLinear (его бейзлайн 0.7535 на paper уже известен и зафиксирован).
- Не изменять продуктовый код на Runner хосте.

## After the run
1. Собрать метрики (best_val_mse, test_mse, test_mae, test_da, mae_denorm) по всем 3 сидам для `c1_late_fusion`, `c1_inverted`, `c1_hierarchical`.
2. Сформировать сводную таблицу (Mean ± std) и сопоставить с базовым DLinear (0.7535).
3. Прикрепить ссылки на построенные графики инференса (`outputs/inference_plots/`).
4. Записать отчёт в `agent-handoff/2026-10-10_0100_paper-becker-sweep-da/report.md`, закоммитить и запушить.
