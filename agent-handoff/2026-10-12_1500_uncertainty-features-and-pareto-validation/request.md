# Request: uncertainty-features-volatility-vs-quantile-comparison

- authored_by: dev
- created_at: 2026-10-12T15:30:00Z
- target_ref: feat/phase2-timexer
- phase: 7
- priority: high

## 1. Goal & Context
Сравнить две альтернативные стратегии отбора признаков под задачу распределительного прогнозирования коридора неопределенности (Distributional LTSF, 80% коридор риска $q_{0.10}, q_{0.90}$ и медиана $q_{0.50}$ на $H=7$):

1. **Сет 1: `selection_objective="volatility"`**:
   Обучение LightGBM на абсолютные лог-доходности $|r_{t, t+H}| = |\ln(P_{t+H}/P_t)|$ (прокси условного размаха волатильности).
2. **Сет 2: `selection_objective="quantile"`**:
   Прямое обучение двух моделей LightGBM с нативным Pinball loss (`objective="quantile"`) для $\alpha=0.10$ и $\alpha=0.90$ с усреднением важностей TreeSHAP:
   $$\text{Importance} = \frac{1}{2}\left(|\phi_{0.10}| + |\phi_{0.90}|\right)$$

Оба сета обучаются на модели `c1_hierarchical_mlp_quantile` на полной выборке `paper` (61 тикер, 98k обучающих окон) по 3 сидам ($s \in \{0, 1, 2\}$) с валидационным отбором по `val_metric=pinball`.

---

## 2. Обязательные требования к финальному отчету `report.md`
В отчет **должны быть включены**:
1. **Полные списки отобранных признаков (Top-30 TS columns)** для обоих сетов с их TreeSHAP-важностями.
2. **Сопоставительный анализ признаков**:
   - Сравнение топ-10 признаков Сета 1 (`volatility`) vs Сета 2 (`quantile`) vs старого сета (`huber`, регрессия среднего).
   - Какие индикаторы волатильности (`natr`, `gk`, `parkinson`, `bb_width`, `volume_z`) и новостной энтропии вошли в топ.
3. **Сводная таблица метрик качества (усреднение по 3 сидам ± std)**:
   - `Quantile Coverage` (ном. 80.0%)
   - `Quantile Sharpness` (ширина тоннеля)
   - `Quantile Winkler Score` ($\alpha=0.20$)
   - `Quantile Pinball Loss`
   - `Directional Accuracy` (DA медианы, %)
   - `Dollar MAE` ($)
   - Сравнение с Naive Historical Baseline (Coverage 82.08%, Sharpness 0.0843, Winkler 0.1190) и старым раном на Huber-сигнатуре (Sharpness 0.0793, Winkler 0.1208).
4. **Визуализация инференса 2023 года**:
   - Скользящий непрерывный коридор (`stride=1`) и 7-дневный веер (`stride=7`) для лучших чекпоинтов каждого сета.

---

## 3. Commands to Execute on RunPod GPU

```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
uv sync

# ==============================================================================
# ШАГ 1. Генерация двух сетов признаков (Paper Set, Fold 1)
# ==============================================================================

# 1.1 Генерация Сета 1: Volatility Signature (абсолютная доходность |r|)
uv run python scripts/prepare_selected_data.py \
  --ticker-set paper \
  --top-k 30 \
  --selection-objective volatility

# 1.2 Генерация Сета 2: Quantile Signature (Pinball loss alpha=0.10 и 0.90)
uv run python scripts/prepare_selected_data.py \
  --ticker-set paper \
  --top-k 30 \
  --selection-objective quantile

# Проверка наличия обоих JSON файлов
ls -lh data/signatures/paper_fold1_signature_volatility_30ts.json
ls -lh data/signatures/paper_fold1_signature_quantile_30ts.json


# ==============================================================================
# ШАГ 2. Быстрый GPU smoke-тест для обоих сетов (по 1 эпохе на 64 окнах)
# ==============================================================================
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda train.val_metric=pinball

uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  data.signature_path=data/signatures/paper_fold1_signature_quantile_30ts.json \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda train.val_metric=pinball


# ==============================================================================
# ШАГ 3. Эксперимент 1: Обучение на СЕТЕ 1 (volatility) — 3 сида (0, 1, 2)
# ==============================================================================
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
done


# ==============================================================================
# ШАГ 4. Эксперимент 2: Обучение на СЕТЕ 2 (quantile) — 3 сида (0, 1, 2)
# ==============================================================================
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.signature_path=data/signatures/paper_fold1_signature_quantile_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
done


# ==============================================================================
# ШАГ 5. Визуализация инференса 2023 года для лучших чекпоинтов Сета 1 и Сета 2
# ==============================================================================
# Определяем директории последних лучших чекпоинтов
# (или укажите точные пути к лучшим чекпоинтам Сета 1 и Сета 2 по наименьшему Pinball/Winkler)
BEST_CKPT_VOLATILITY=$(ls -td outputs/*/* | grep -v "quantile" | head -n 1)/checkpoints/best.pt
BEST_CKPT_QUANTILE=$(ls -td outputs/*/* | head -n 1)/checkpoints/best.pt

# Графики для Сета 1 (Volatility): stride=1 и stride=7
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_VOLATILITY" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_volatility_stride1 \
  --stride 1 \
  --device cuda

uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_VOLATILITY" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_volatility_stride7 \
  --stride 7 \
  --device cuda

# Графики для Сета 2 (Quantile): stride=1 и stride=7
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_QUANTILE" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_quantile_stride1 \
  --stride 1 \
  --device cuda

uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_QUANTILE" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots_quantile_stride7 \
  --stride 7 \
  --device cuda
```

## 4. Environment & Hardware
- cwd: `/workspace/DecisionForecast`
- GPU: NVIDIA L4 / A100 / RTX 4090 (24GB VRAM)
- RAM: 32+ GB
- dataset: FNSPID full paper set (61 tickers, 98,110 train windows, 12,048 val windows, 11,904 test windows).
