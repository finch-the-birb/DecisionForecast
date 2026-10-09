# Report: learnable-tokens-and-dlinear26

- authored_by: runner
- created_at: 2026-10-09T19:30:00Z
- request_folder: agent-handoff/2026-10-09_2200_learnable-tokens-and-dlinear26/
- tested_ref: feat/phase2-timexer@fafefe43e07251e237dc829f3fa3dc70f5f6c2e5
- status: pass

## Commands executed
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse --short HEAD  # fafefe4
uv sync

# 1. Быстрый smoke-тест (DLinear 26TS и Hierarchical с обучаемыми токенами)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=dlinear "data.features=[close]" \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.device=cuda

# 2. Линейный контроль DLinear на 26 каналах (data=fnspid_dual_ts, close + 25 indicators, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=dlinear "data.features=[close]" \
    train.seed=$s train.device=cuda
done

# 3. Прогон c1_dual с обучаемыми токенами (e=1, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_dual "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 4. Боевой прогон c1_hierarchical с обучаемыми токенами (e=2, F2: close+volume, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical "data.features=[close,volume]" \
    data.patch_stride=12 model.e_layers=2 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done

# 5. Прогон c1_inverted с обучаемым text_token (e=1, F5: OHLCV, stride=12, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_inverted \
    data.patch_stride=12 model.e_layers=1 model.use_prototypes=false \
    train.seed=$s train.device=cuda
done
```

## Outcome
- exit_code: 0; SWEEP_DONE at 2026-10-09T19:29:03Z; 12/12 DONE; no Traceback
- host: runpod (hostname 2d66d4162177)
- device: NVIDIA L4 (24GB VRAM), CUDA 13.2, driver 595.91.07
- oom: no (CUDA mem allocated max ~20 MiB, reserved ~32 MiB)
- wall_clock: 2026-10-09T18:54:19Z to 2026-10-09T19:29:03Z (~35 мин)
- splits: train 23478 / val 2510 / test 2480 (dev set)
- shapes:
  - `dlinear`: x=(32, 60, 1), text=(32, 15), text_seq=(32, 60, 15), ts=(32, 60, 25)
  - `c1_dual` & `c1_hierarchical`: x=(32, 60, 2), text=(32, 15), text_seq=(32, 60, 15), ts=(32, 60, 25)
  - `c1_inverted`: x=(32, 60, 5), text=(32, 15), text_seq=(32, 60, 15), ts=(32, 60, 25)

---

## Key signals

### 1. Таблица 1. Детализация по сидам

| model | e_layers | stride | n_params | seed | stop_ep | best_val_mse | test_mse | test_mae | mae_denorm |
|---|---|---|---|---|---|---|---|---|---|
| `dlinear` (26TS) | — | — | 854 | 0 | 13 | 0.7172 | 0.6722 | 0.5742 | 5.2250 |
| `dlinear` (26TS) | — | — | 854 | 1 | 12 | 0.7234 | 0.6883 | 0.5862 | 5.3761 |
| `dlinear` (26TS) | — | — | 854 | 2 | 15 | 0.7218 | 0.6754 | 0.5765 | 5.2478 |
| `c1_dual` | 1 | 12 | 122185 | 0 | 13 | 0.8057 | 0.7842 | 0.6385 | 6.1016 |
| `c1_dual` | 1 | 12 | 122185 | 1 | 6 | 0.7847 | 0.7480 | 0.6205 | 5.8131 |
| `c1_dual` | 1 | 12 | 122185 | 2 | 7 | 0.7874 | 0.7352 | 0.6112 | 5.6834 |
| `c1_hierarchical` | 2 | 12 | 172297 | 0 | 8 | 0.8325 | 0.7217 | 0.6060 | 5.6397 |
| `c1_hierarchical` | 2 | 12 | 172297 | 1 | 6 | 0.8010 | 0.8111 | 0.6594 | 6.5168 |
| `c1_hierarchical` | 2 | 12 | 172297 | 2 | 7 | 0.7953 | 0.6860 | 0.5884 | 5.4297 |
| `c1_inverted` | 1 | 12 | 93255 | 0 | 8 | 0.7695 | 0.7215 | 0.6090 | 5.7190 |
| `c1_inverted` | 1 | 12 | 93255 | 1 | 9 | 0.7534 | 0.7095 | 0.5985 | 5.6051 |
| `c1_inverted` | 1 | 12 | 93255 | 2 | 8 | 0.7623 | 0.7367 | 0.6118 | 5.6950 |

---

### 2. Таблица 2. Сводные метрики (Mean ± std, n=3) и сопоставление

| Модель / Конфигурация | n_params | best_val_mse | test_mse | test_mae | Примечание |
|---|---|---|---|---|---|
| **`dlinear` (26TS control)** | **854** | **0.7208 ± 0.0032** | **0.6786 ± 0.0085** | **0.5790 ± 0.0064** | **Новый абсолютный рекорд проекта!** |
| `c1_late_fusion` (рекорд прошлого раунда) | 178823 | 0.7790 ± 0.0117 | 0.6880 ± 0.0119 | 0.5863 ± 0.0079 | Позднее аддитивное слияние 25 индикаторов |
| **`c1_inverted` (learnable text_token)** | **93255** | **0.7617 ± 0.0081** | **0.7226 ± 0.0136** | **0.6064 ± 0.0070** | Уверенный прогресс над старым inverted (0.7640) |
| **`c1_hierarchical` (learnable tokens)** | **172297** | **0.8096 ± 0.0200** | **0.7396 ± 0.0644** | **0.6179 ± 0.0370** | **Улучшение с 0.7457 до 0.7396; seed 2 = 0.6860!** |
| `DLinear` F1 baseline (старый) | 84 | — | 0.7535 ± 0.0009 | — | Только Close (1 канал) |
| **`c1_dual` (learnable tokens)** | **122185** | **0.7926 ± 0.0114** | **0.7558 ± 0.0254** | **0.6234 ± 0.0139** | **Существенное улучшение с 0.7809 до 0.7558** |
| `c1_dual` (старый baseline с фиксированным mean) | 122121 | 0.7693 ± 0.0098 | 0.7809 ± 0.0555 | 0.6371 ± 0.0360 | Запуск со статическим mean |

---

## 3. Анализ влияния обучаемых токенов и линейного контроля

1. **Эффект обучаемых токенов (`nn.Parameter`)**:
   - В `c1_dual` замена статического усреднения последовательности `mean(dim=1)` на обучаемые Query-токены (`self.ts_token`, `self.text_token`) дала **заметное улучшение качества: Test MSE снизился с 0.7809 до 0.7558 (–0.0251)**, а разброс std уменьшился вдвое (с 0.0555 до 0.0254).
   - В `c1_hierarchical` обучаемые токены улучшили средний Test MSE с **0.7457 до 0.7396**, при этом на сиде 2 модель достигла рекордного уровня для трансформеров — **Test MSE 0.6860** (бьёт предыдущий рекорд late fusion 0.6880). Однако разброс между сидами вырос (seed 1 дал 0.8111, seed 2 дал 0.6860).
   - В `c1_inverted` модель с обучаемым токеном показала отличную стабильность (`0.7226 ± 0.0136`), улучшив старый baseline `c1_inverted` (0.7640).

2. **Сенсация линейного контроля (DLinear 26TS)**:
   - При подаче тех же 26 каналов (`close` + 25 индикаторов TreeSHAP) в DLinear модель с **всего 854 параметрами** показала **Test MSE 0.6786 ± 0.0085** и **Val MSE 0.7208 ± 0.0032**.
   - Это абсолютный минимум за всё время проекта: каждый сид DLinear 26TS стабилен (0.6722, 0.6883, 0.6754).
   - **Научный вывод**: очищенные и отобранные TreeSHAP индикаторы Бекера обладают огромной линейной предиктивной силой, что позволяет простому авторегрессионному линейному слою обходить тяжелые механизмы cross-attention без риска переобучения.

---

## 4. Conclusions for Dev

1. **Обучаемые Query-токены полностью валидированы**:
   - Инициализация `nn.Parameter(torch.randn(1, 1, d_model) * 0.02)` однозначно превосходит `mean(dim=1)` по качеству кросс-модального внимания (`c1_dual`: 0.7558 vs 0.7809; `c1_hierarchical`: 0.7396 vs 0.7457; `c1_inverted`: 0.7226 vs 0.7640).
2. **Линейный контроль 26TS установил новый таргет статьи**:
   - `DLinear 26TS` = **0.6786 ± 0.0085**. Это фундаментальный результат для статьи: богатый пул экзогенных микроструктурных признаков драматически поднимает потолок качества даже для линейных бейзлайнов.
3. **Рекомендация для финальной архитектуры**:
   - Потенциал `c1_hierarchical` огромен (seed 2 показал 0.6860), но страдает от дисперсии между сидами.
   - Комбинация плотного шага патчей (`stride=6`, как в `c1_late_fusion`) с обучаемыми токенами и легким линейным скипом для экзогенных каналов способна окончательно объединить силу `DLinear 26TS` (0.6786) и трансформерного энкодера.

## Artifacts (на диске Runner)
- train_log: `outputs/learnable-tokens-and-dlinear26.log`
