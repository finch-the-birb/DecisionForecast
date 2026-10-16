# Request: becker-dual-hierarchical

- authored_by: dev
- created_at: 2026-10-09T20:00:00Z
- target_ref: feat/phase2-timexer@HEAD
- phase: 6
- priority: high

## Goal
1. Перегенерировать кэш технических признаков и сигнатуру Fold 1 с безопасной стабилизированной формулой Crash Fragility Index (`cfi`: знаменатель `rsv_plus + 1e-4`, клип `[0.0, 50.0]`) и жестким ограничением в селективной нормализации (`bounded` -> `[-1.0, 1.0]`).
2. Перезапустить `c1_dual` (F2: close+volume, e=1, stride 12, `n_ts_features=25`, 3 сида), устранив дивергенцию в NaN.
3. Запустить двухслойный `c1_hierarchical` (F2, e=2, stride 12, `n_ts_features=25`, 3 сида), где Слой 1 опрашивает 25 индикаторов режима, а Слой 2 модулирует патчи новостями.
4. Выполнить контрольный повтор рекордного `c1_late_fusion` (F5 OHLCV, e=2, stride 6, 3 сида) для валидации исторического рекорда Test MSE 0.6988 при стабилизированном кэше.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer

# 1. Синхронизация окружения
uv sync

# 2. Перегенерация технических признаков и новой сигнатуры TreeSHAP Top-25 со стабилизированным CFI
uv run python scripts/prepare_selected_data.py --ticker-set=paper --fold=1 --device=cuda

# 3. Контрольный smoke-тест (1 эпоха, урезанные окна)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

# 4. Перезапуск c1_dual (e=1, no proto, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 5. Боевой прогон c1_hierarchical (e=2, no proto, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=2 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 6. Контрольный повтор рекордного c1_late_fusion (e=2, no proto, F5 OHLCV, stride=6, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_late_fusion \
    data.patch_stride=6 model.e_layers=2 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done
```

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (FNSPID raw Parquet present)
- gpu: required (CUDA)
- approx_ram_gb: >= 32

## Out of scope
- Не изменять продуктовый код на Runner хосте.
- Не запускать сетку в интерактивном foreground режиме.

## After the run
Собрать метрики (best_val_mse, test_mse, test_mae) по всем 3 сидам для `c1_dual`, `c1_hierarchical` и `c1_late_fusion`. Сформировать сводную таблицу (Mean ± std), проверить сходимость `c1_dual` (отсутствие NaN), записать отчёт в `report.md` и запушить.
