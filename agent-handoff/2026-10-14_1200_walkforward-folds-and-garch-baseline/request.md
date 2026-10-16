# Request: walkforward-folds-and-garch-baseline

- authored_by: dev
- created_at: 2026-10-14T12:00:00Z
- target_ref: feat/phase2-timexer
- phase: 8
- priority: high

## 1. Goal & Research Context (Tier-1 Conference Framing)

В рамках Этапа 2 исследовательского плана (`Articles/notes/research/План_исследований_и_промты_Этап_1_и_2.md`) закрываются ключевые методологические требования рецензентов ведущих конференций (NeurIPS / ICLR / KDD):

1. **Expanding Walk-Forward Cross-Validation (Folds 1, 2, 3):**
   Оценка на единственном тестовом окне 2023 года подвержена риску смещения («бычий рынок AI-ралли»). В кодовую базу интегрирована расширяющаяся схема валидации с полным устранением утечек в будущее:
   - **Fold 1:** Train 2015–2020 | Val 2021-01–2021-06 | Test 2021-07–2021-12
   - **Fold 2:** Train 2015–2021 | Val 2022-01–2022-06 | Test 2022-07–2022-12 (Медвежий рынок, цикл подъема ставок)
   - **Fold 3:** Train 2015–2022 | Val 2023-01–2023-06 | Test 2023-07–2023-12 (Восстановление, технологический бум)
   - Строгий **Purge** ($H-1 = 6$ баров) и **Embargo** ($E = 14$ баров) между сплитами гарантируют отсутствие наложения таргетов и автокорреляционной утечки. Окна теста строго ограничены `last_target_date <= test_end`.

2. **Эконометрический бенчмарк GARCH(1,1):**
   Вместо примитивной скользящей исторической волатильности реализован классический эталон финансовой эконометрики (`src/evaluation/garch_baseline.py` на базе пакета `arch`):
   - Обучается на скользящей истории $T_{\text{garch}} \ge 252$ торговых дней для каждого тикера.
   - Аналитическая экстраполяция условной дисперсии на горизонт $H=7$:
     $$\mathbb{E}[\sigma_{t+h}^2 \mid \mathcal{F}_t] = \sigma^2 + (\alpha + \beta)^{h-1} (\sigma_{t+1}^2 - \sigma^2)$$
   - Кумулятивная дисперсия $\sigma_{\text{cum}}^2(h) = \sum_{i=1}^h \mathbb{E}[\sigma_{t+i}^2 \mid \mathcal{F}_t]$ формирует честный 80% квантильный коридор:
     $$[q_{0.10}, q_{0.90}] = [-z_{\alpha/2} \sigma_{\text{cum}}(h), +z_{\alpha/2} \sigma_{\text{cum}}(h)], \quad z_{0.10} = 1.28155$$

3. **Adaptive Conformal Inference (ACI) Calibration:**
   Для устранения смещения эмпирического покрытия при смене рыночных режимов и кластеризации волатильности реализован модуль постобработки (`src/evaluation/conformal.py`):
   - Динамическая калибровка доверительного уровня:
     $$\alpha_{t+1} = \alpha_t + \gamma (\alpha_{\text{target}} - \text{err}_t), \quad \text{err}_t = \mathbb{I}(y_t \notin [q_{\alpha_t/2}, q_{1 - \alpha_t/2}])$$
   - Гарантирует асимптотическую сходимость к заданному покрытию (80.0%) на тестовой выборке без переобучения базовой нейросети.

---

## 2. Экспериментальный протокол и инструкции для Runner

### Шаг 0. Синхронизация репозитория на RunPod GPU
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
uv sync
```

### Шаг 1. Расчет эконометрического GARCH(1,1) бейзлайна по всем 3 фолдам
Запустить независимый расчет бейзлайна на CPU/GPU для Folds 1, 2, 3:
```bash
for f in 1 2 3; do
  echo "=== Running GARCH(1,1) Baseline for Fold $f ==="
  uv run python -m src.evaluation.garch_baseline \
    --ticker-set paper \
    --fold $f \
    --device cpu
done
```
*Результаты зафиксировать в таблице метрик отчета.*

---

### Шаг 2. GPU Smoke-тест на Fold 3 (верификация пайплайна)
```bash
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  data.fold=3 data.signature_path=data/signatures/paper_fold3_signature_volatility_30ts.json \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda train.val_metric=pinball
```

---

### Шаг 3. Полные прогоны обучения `c1_hierarchical_mlp_quantile` (Folds 1, 2, 3 по 3 сида)
Для обеспечения статистической достоверности каждый фолд обучается на 3 случайных сидах ($s \in \{42, 43, 44\}$):

#### 3.1 Fold 1 (Тест: 2021 H2)
```bash
for s in 42 43 44; do
  echo "=== Fold 1 | Seed $s ==="
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.fold=1 data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
done
```

#### 3.2 Fold 2 (Тест: 2022 H2 — Bear Market / Rate Hikes)
```bash
for s in 42 43 44; do
  echo "=== Fold 2 | Seed $s ==="
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.fold=2 data.signature_path=data/signatures/paper_fold2_signature_volatility_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
done
```

#### 3.3 Fold 3 (Тест: 2023 H2 — Recovery / AI rally)
```bash
for s in 42 43 44; do
  echo "=== Fold 3 | Seed $s ==="
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.fold=3 data.signature_path=data/signatures/paper_fold3_signature_volatility_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
done
```

---

### Шаг 4. Постобработка: Конформная калибровка (ACI)
Для каждого обученного чекпоинта применить скрипт оценки с ACI:
```bash
for f in 1 2 3; do
  for s in 42 43 44; do
    echo "=== Evaluating & ACI Calibrating Fold $f Seed $s ==="
    uv run python scripts/eval_saved_predictions.py \
      --fold $f \
      --seed $s \
      --conformal-gamma 0.005
  done
done
```
*(Либо через вызов `apply_adaptive_conformal_inference` в аналитическом скрипте Runner).*

---

## 3. Требования к финальному отчету `report.md`

В отчет **должны быть включены**:

1. **Главная сводная таблица Walk-Forward (Fold 1, Fold 2, Fold 3 и Average):**
   - Строки:
     - `GARCH(1,1) Baseline`
     - `TimeXer Hierarchical MLP (Raw Quantiles)`
     - `TimeXer Hierarchical MLP (ACI Calibrated)`
   - Столбцы (Mean ± Std по 3 сидам):
     - `Coverage 80% (%)` (цель: 80.0%)
     - `Sharpness` (ширина коридора, меньше — лучше)
     - `Winkler Score (alpha=0.20)` (главная интегральная метрика неопределенности)
     - `Median DA (%)` (Directional Accuracy медианы $q_{0.50}$)
     - `Median Dollar MAE ($)`

2. **Анализ устойчивости к рыночным фазам**:
   - Как ведет себя GARCH(1,1) против нейросетевого квантильного коридора на медвежьем рынке 2022 года (Fold 2) vs бычьем рынке 2021 (Fold 1) и 2023 (Fold 3)?
   - Насколько ACI подтягивает реальное покрытие к 80% в периоды внезапных скачков волатильности?

3. **Визуализация инференса коридоров**:
   - Сравнение 80% коридоров (GARCH vs TimeXer Raw vs TimeXer ACI) для 3 ключевых тикеров:
     - `AAPL` (Big Tech)
     - `NVDA` (High Volatility)
     - `JPM` (Financial sector)
