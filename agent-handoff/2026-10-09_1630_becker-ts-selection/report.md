# Report: becker-ts-selection

- authored_by: runner
- created_at: 2026-10-09T16:05:00Z
- request_folder: agent-handoff/2026-10-09_1630_becker-ts-selection/
- tested_ref: feat/phase2-timexer@6a9ec54a06609657d84717d8b26f3bdcea5c0315
- status: partial

## Commands executed
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer

# 1. Синхронизация окружения
uv sync

# 2. Перегенерация технических признаков и новой сигнатуры TreeSHAP Top-25
uv run python scripts/prepare_selected_data.py --ticker-set=paper --fold=1 --device=cuda

# 3. Контрольный smoke-тест (c1_dual, 1 эпоха, урезанные окна)
# Примечание: data=fnspid_dual_ts строго валидирует n_ts_features=25, поэтому передан явный override model.n_ts_features=25
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" model.n_features=2 model.n_ts_features=25 \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

# 4. Боевой прогон c1_dual (e=1, no proto, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" model.n_features=2 model.n_ts_features=25 \
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

## Outcome
- exit_code: 0 (все скрипты выполнились без падения процесса, но c1_dual разошёлся в NaN)
- host: runpod (hostname 2d66d4162177)
- device: NVIDIA L4 (24GB VRAM), CUDA 13.2, driver 595.91.07
- oom: no (CUDA mem max ~35 MiB, RAM free > 400 GB)
- wall_clock: 2026-10-09T15:13:59Z to 2026-10-09T15:56:18Z (~42 мин)
- traceback_summary: n/a (ранний останов по NaN loss на 5 эпохе для c1_dual; c1_late_fusion сошёлся штатно)

---

## 1. Отбор признаков TreeSHAP (paper_fold1_signature.json)

Файл сигнатуры: `Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json` (75 264 строк train-сплита Fold 1).

### Полный перечень Top-25 индикаторов (ts_columns, без close):
1. `month_cos_delta`
2. `month_sin_lag3`
3. `gk_n5`
4. `month_sin_delta`
5. `log_ret_k5`
6. `bb_width_n60_delta`
7. `natr_n60_lag1`
8. `month_cos_lag2`
9. `parkinson_n10_delta`
10. `cfi_n3` *(Becker/микроструктура)*
11. `queue_acc_n10_lag1` *(Becker/микроструктура)*
12. `cfi_n40` *(Becker/микроструктура)*
13. `gk_n20_delta`
14. `mrd_n40_lag3` *(Becker/микроструктура)*
15. `gk_n60_delta`
16. `day_sin_delta`
17. `log_ret_k5_lag3`
18. `mrd_n20_lag3` *(Becker/микроструктура)*
19. `bb_width_n5_lag2`
20. `cfi_n3_lag1` *(Becker/микроструктура)*
21. `bb_width_n5_lag1`
22. `bb_width_n40_delta`
23. `parkinson_n20`
24. `queue_acc_n60_lag2` *(Becker/микроструктура)*
25. `log_ret_k60`

### Статус попадания запрошенных признаков Бекера / микроструктуры:
- **`cfi`**: **ПОПАЛ** (3 признака в Top-25: `cfi_n3`, `cfi_n40`, `cfi_n3_lag1`, лучший ранг 16).
- **`queue_acc`**: **ПОПАЛ** (2 признака в Top-25: `queue_acc_n10_lag1`, `queue_acc_n60_lag2`, лучший ранг 17).
- **`mrd`**: **ПОПАЛ** (2 признака в Top-25: `mrd_n40_lag3`, `mrd_n20_lag3`, лучший ранг 21).
- **`eii`**: **НЕ ПОПАЛ** в Top-25 (всего 20 вариаций, лучший ранг 381).
- **`cgo`**: **НЕ ПОПАЛ** в Top-25 (всего 30 вариаций, лучший ранг 65: `cgo_n20_lag3`, importance=0.000120).

---

## 2. Метрики моделей

### Таблица 1. Детализация по сидам

| model | e_layers | stride | features | seed | stop_ep | best_val_mse | test_mse | test_mae | mae_denorm |
|---|---|---|---|---|---|---|---|---|---|
| `c1_dual` | 1 | 12 | close+vol (F2) | 0 | 5 | nan | **nan** | **nan** | nan |
| `c1_dual` | 1 | 12 | close+vol (F2) | 1 | 5 | nan | **nan** | **nan** | nan |
| `c1_dual` | 1 | 12 | close+vol (F2) | 2 | 5 | nan | **nan** | **nan** | nan |
| `c1_late_fusion` | 2 | 6 | OHLCV (F5) | 0 | 14 | 0.7888 | 0.7166 | 0.6005 | 5.5507 |
| `c1_late_fusion` | 2 | 6 | OHLCV (F5) | 1 | 11 | 0.7843 | 0.6824 | 0.5806 | 5.2993 |
| `c1_late_fusion` | 2 | 6 | OHLCV (F5) | 2 | 12 | 0.8177 | 0.6973 | 0.5984 | 5.5480 |

### Таблица 2. Итоговые результаты (Mean ± std, n=3) и сравнение с бейзлайнами

| Модель / Конфигурация | best_val_mse | test_mse | test_mae | Примечание |
|---|---|---|---|---|
| **`c1_late_fusion` (новые фичи Бекера)** | **0.7969 ± 0.0181** | **0.6988 ± 0.0171** | **0.5932 ± 0.0110** | **Новый абсолютный рекорд (–0.0547 к DLinear)** |
| `DLinear` (baseline) | — | 0.7535 ± 0.0009 | — | Предыдущий чемпион |
| `c1_late_fusion` (старый baseline) | — | 0.7577 ± 0.0019 | 0.5963 ± 0.0012 | Без новых микроструктурных признаков |
| `c1_dual` F2 (старый baseline) | — | 0.7655 ± 0.0032 | — | Предыдущий dual F2 |
| `c1_dual` (новые фичи Бекера) | nan | **nan** | **nan** | Дивергенция градиента на батче 41 из-за всплесков `cfi` |

---

## 3. Анализ сбоя `c1_dual` (NaN)

- **Корень проблемы**: в `src/features/technical.py` формула Crash Fragility Index (CFI):
  $$rsv\_ratio = rsv\_minus / (rsv\_plus + eps)$$
  При околонулевом `rsv_plus` (период только отрицательных доходностей на окне 3 дня) значение `rsv_ratio` достигает $10^8$. В кеше признаков значения `cfi_n3` достигают $1.31 \times 10^8$.
- **Различие с нормализацией**: в `src/data/selective_norm.py` признаки `mrd_` и `cgo_` объявлены в `BOUNDED_PREFIXES` и клипуются до `[-1, 1]`, тогда как `cfi_` был помещён в `STATIONARY_PREFIXES` с IQR-масштабированием. IQR на train-сплите составляет $\approx 1$, поэтому гигантские выбросы в $10^8$ не гасятся.
- **Почему `c1_late_fusion` сошёлся**: `c1_late_fusion` использует только срез индикаторов на последнем баре через MLP с остаточной связью, тогда как `c1_dual` пропускает всю 60-шаговую матрицу индикаторов через `g_ts` и cross-attention, где перемножение на $10^8$ привело к переполнению градиентов `g_ts.weight` и `layers.0.self_attn` уже на батче 41.

---

## 4. Conclusions for Dev

1. **Рекордный результат `c1_late_fusion`**: Включение новой сигнатуры признаков Бекера (`cfi`, `queue_acc`, `mrd`) дало мощный скачок качества: **test_mse упал с 0.7577 до 0.6988** (уверенно побит бейзлайн DLinear 0.7535 на 0.0547 пункта).
2. **Блокер для `c1_dual`**: Требуется либо клиповать `cfi` в формуле (например, `np.clip(rsv_ratio, 0, 100)` или log1p), либо добавить префикс `cfi_` в `BOUNDED_PREFIXES` внутри `src/data/selective_norm.py`.
3. **Hydra override**: При запуске `data=fnspid_dual_ts` с `c1_dual` необходимо явно указывать `model.n_ts_features=25` (в `configs/model/c1_dual.yaml` по умолчанию стоит `0`).

## Artifacts (на диске Runner)
- train_log: `outputs/paper-becker-ts-selection.log`
- signature: `Data/FNSPID/cache/selected_signatures/paper_fold1_signature.json`
- text_state: `Data/FNSPID/cache/selected_signatures/paper_fold1_text_state.npz`
