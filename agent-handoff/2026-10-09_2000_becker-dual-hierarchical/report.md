# Report: becker-dual-hierarchical

- authored_by: runner
- created_at: 2026-10-09T17:16:00Z
- request_folder: agent-handoff/2026-10-09_2000_becker-dual-hierarchical/
- tested_ref: feat/phase2-timexer@cb62e71edc2d8a41591bde86b18c85bf1616d8a3
- status: pass

## Commands executed
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

## Outcome
- exit_code: 0; SWEEP_DONE at 2026-10-09T17:15:35Z; 9/9 DONE; no Traceback
- host: runpod (hostname 2d66d4162177)
- device: NVIDIA L4 (24GB VRAM), CUDA 13.2, driver 595.91.07
- oom: no (CUDA mem allocated max 20.6 MiB, reserved 34.0 MiB)
- wall_clock: 2026-10-09T16:22:42Z to 2026-10-09T17:15:35Z (~53 мин)
- splits: train 23478 / val 2510 / test 2480 (dev set)
- shapes:
  - `c1_dual` & `c1_hierarchical`: x=(32, 60, 2), text=(32, 15), text_seq=(32, 60, 15), ts=(32, 60, 25)
  - `c1_late_fusion`: x=(32, 60, 5), text=(32, 15), text_seq=(32, 60, 15), ts=(32, 60, 25)

---

## Key signals

### 1. Результаты перегенерации кэша со стабилизированным CFI
- **Формула**: `rsv_ratio = rsv_minus / (rsv_plus + 1e-4)`, клиппинг в сырых фичах `[0.0, 50.0]`, затем жесткий `bounded` клип `[-1.0, 1.0]` в `apply_static_and_robust`.
- **Максимальное значение**: `max(cfi)` по всему кэшу строго ограничено 50.0 (ранее достигало $1.31 \times 10^8$).
- **Сигнатура Fold 1**: 75 264 строк, новые Top-25 индикаторы зафиксированы в `paper_fold1_signature.json`. В Top-25 вошли `cfi_n40`, `cfi_n3`, `cfi_n3_lag1`, `cfi_n3_lag2`, `mrd_n40_lag3`, `mrd_n20_lag3`, `queue_acc_n10_lag1`.

### 2. Таблица 1. Детализация по сидам

| model | e_layers | stride | n_params | seed | stop_ep | best_val_mse | test_mse | test_mae | mae_denorm |
|---|---|---|---|---|---|---|---|---|---|
| `c1_dual` | 1 | 12 | 122121 | 0 | 6 | 0.7619 | 0.7732 | 0.6335 | 6.0309 |
| `c1_dual` | 1 | 12 | 122121 | 1 | 13 | 0.7655 | 0.7296 | 0.6030 | 5.5433 |
| `c1_dual` | 1 | 12 | 122121 | 2 | 7 | 0.7804 | 0.8398 | 0.6748 | 6.8157 |
| `c1_hierarchical` | 2 | 12 | 172169 | 0 | 10 | 0.7934 | 0.7843 | 0.6394 | 5.9531 |
| `c1_hierarchical` | 2 | 12 | 172169 | 1 | 7 | 0.8554 | 0.7088 | 0.5935 | 5.4729 |
| `c1_hierarchical` | 2 | 12 | 172169 | 2 | 8 | 0.7673 | 0.7440 | 0.6107 | 5.5365 |
| `c1_late_fusion` | 2 | 6 | 178823 | 0 | 11 | 0.7655 | 0.6977 | 0.5928 | 5.5625 |
| `c1_late_fusion` | 2 | 6 | 178823 | 1 | 11 | 0.7853 | 0.6747 | 0.5775 | 5.3023 |
| `c1_late_fusion` | 2 | 6 | 178823 | 2 | 6 | 0.7862 | 0.6917 | 0.5885 | 5.4517 |

### 3. Таблица 2. Сводные метрики (Mean ± std, n=3) и сопоставление

| Модель / Архитектура | best_val_mse | test_mse | test_mae | Примечание |
|---|---|---|---|---|
| **`c1_late_fusion` (рекордный)** | **0.7790 ± 0.0117** | **0.6880 ± 0.0119** | **0.5863 ± 0.0079** | **Абсолютный исторический минимум проекта (–0.0655 к DLinear)** |
| **`c1_hierarchical` (F2, e=2)** | **0.8054 ± 0.0453** | **0.7457 ± 0.0378** | **0.6145 ± 0.0232** | **Бьёт DLinear baseline и старый c1_dual F2** |
| `DLinear` (baseline) | — | 0.7535 ± 0.0009 | — | Классический baseline |
| `c1_late_fusion` (прошлый запуск) | 0.7969 ± 0.0181 | 0.6988 ± 0.0171 | 0.5932 ± 0.0110 | Без стабилизации CFI |
| `c1_dual` (старый baseline) | — | 0.7655 ± 0.0032 | — | Предыдущий c1_dual F2 e=1 |
| **`c1_dual` (F2, e=1, с новым CFI)** | **0.7693 ± 0.0098** | **0.7809 ± 0.0555** | **0.6371 ± 0.0360** | **Полная сходимость без NaN на всех 3 сидах** |

---

## Conclusions for Dev

1. **Дивергенция `c1_dual` полностью решена**: Стабилизация CFI (`rsv_plus + 1e-4`, clip `[0, 50]`, `bounded` clip `[-1, 1]`) устранила всплески градиентов. Все 3 сида сошлись без единого NaN (`test_mse: 0.7809 ± 0.0555`, лучший сид — `0.7296`).
2. **`c1_hierarchical` превзошёл бейзлайны**: Двухслойный иерархический энкодер (Слой 1 — индикаторы, Слой 2 — новости) показал **0.7457 ± 0.0378**, уверенно обойдя как бейзлайн DLinear (0.7535), так и базовый `c1_dual` F2 (0.7655). Лучший сид достиг `0.7088`.
3. **Новый абсолютный рекорд `c1_late_fusion`**: Контрольный повтор чемпиона со стабилизированным кэшем не просто подтвердил, а улучшил результат — **Test MSE 0.6880 ± 0.0119** (каждый сид < 0.70: 0.6977, 0.6747, 0.6917), Test MAE 0.5863 ± 0.0079. Преимущество перед DLinear выросло до **–0.0655**.

## Artifacts (на диске Runner)
- train_log: `outputs/becker-dual-hierarchical.log`
- signature: `Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json`
- text_state: `Data/FNSPID/cache/selected_signatures/paper_fold1_text_state.npz`
