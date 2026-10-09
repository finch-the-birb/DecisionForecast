# Request: learnable-tokens-and-dlinear26

- authored_by: dev
- created_at: 2026-10-09T22:00:00Z
- target_ref: feat/phase2-timexer@69acb34
- phase: 6
- priority: high

## Goal
1. **Линейный контроль (DLinear 26TS)**: Запустить DLinear в абсолютно равных с TimeXer условиях — на 26 каналах (`close` + 25 индикаторов TreeSHAP fold-1) на 3 сидах (0, 1, 2) для строгого научного бенчмарка.
2. **Обучаемые токены в TimeXer**: Валидировать архитектурное обновление — замену 60-дневного среднего `mean(dim=1)` на обучаемые векторные параметры `self.text_token` и `self.ts_token` (`nn.Parameter(torch.randn(1, 1, d_model) * 0.02)`):
   - `c1_dual` (F2: close+volume, e=1, stride 12, 3 сида);
   - `c1_hierarchical` (F2: close+volume, e=2, stride 12, 3 сида);
   - `c1_inverted` (F5: OHLCV, e=1, stride 12, 3 сида) с обучаемым `text_token`.
3. Сопоставить качество прогноза `c1_hierarchical` и `c1_dual` против честного `DLinear 26TS` и чемпиона `c1_late_fusion` (0.6880).

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer

# 1. Синхронизация окружения
uv sync

# 2. Быстрый smoke-тест (DLinear 26TS и Hierarchical с обучаемыми токенами)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=dlinear "data.features=[close]" \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

# 3. Линейный контроль DLinear на 26 каналах (data=fnspid_dual_ts, close + 25 indicators, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=dlinear "data.features=[close]" \
    train.seed=$s train.device=cuda
done

# 4. Прогон c1_dual с обучаемыми токенами (e=1, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 5. Боевой прогон c1_hierarchical с обучаемыми токенами (e=2, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=2 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 6. Прогон c1_inverted с обучаемым text_token (e=1, F5: OHLCV, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_inverted \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (кэш и сигнатура Fold 1 стабилизированы)
- gpu: required (CUDA)
- approx_ram_gb: >= 32

## Out of scope
- Не изменять продуктовый код на Runner хосте.
- Не запускать сетку в интерактивном foreground режиме.

## After the run
Собрать метрики (best_val_mse, test_mse, test_mae, mae_denorm) по всем сидам для `DLinear 26TS`, `c1_dual`, `c1_hierarchical`, `c1_inverted`. Зафиксировать влияние честных обучаемых параметров на качество селекции информации, оформить сводную таблицу (Mean ± std) в `report.md`, закоммитить и запушить.
