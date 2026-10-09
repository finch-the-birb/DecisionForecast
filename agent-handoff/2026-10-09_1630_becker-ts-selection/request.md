# Request: becker-ts-selection

- authored_by: dev
- created_at: 2026-10-09T16:30:00Z
- target_ref: feat/phase2-timexer@HEAD
- phase: 6
- priority: high

## Goal
Пересчитать расширенный пул технических признаков (с новыми формулами Бекера 1991 и поведенческой микроструктуры: EII, Q, CFI, MRD, CGO), переобучить GBDT + TreeSHAP (Huber loss $\delta=0.5$) на train-сплите Fold 1 для отбора и фиксации новой сигнатуры 25 индикаторов, после чего запустить сравнительное обучение базовых моделей `c1_dual` (F2: close+volume, stride 12, e=1) и `c1_late_fusion` (F5 OHLCV, stride 6, e=2) на 3 сидах (0, 1, 2) на выборке paper (61 тикер).

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer

# 1. Синхронизация окружения
uv sync

# 2. Перегенерация технических признаков и новой сигнатуры TreeSHAP Top-25
uv run python scripts/prepare_selected_data.py --ticker-set=paper --fold=1 --device=cuda

# 3. Контрольный smoke-тест (1 эпоха, урезанные окна)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" model.n_features=2 \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

# 4. Боевой прогон c1_dual (e=1, no proto, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" model.n_features=2 \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 5. Боевой прогон чемпиона c1_late_fusion (e=2, no proto, stride=6, 3 seeds)
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
Собрать метрики (test_mse, test_mae, val_mse) по всем сидам, составить сравнительную таблицу с предыдущими бейзлайнами (DLinear: 0.7535, старый c1_late_fusion: 0.7577, старый c1_dual F2: 0.7655), зафиксировать, какие из новых фичей Бекера вошли в Top-25 TreeSHAP, записать отчёт в `report.md` и запушить.
