# Request: uncertainty-features-and-pareto-validation

- authored_by: dev
- created_at: 2026-10-12T15:00:00Z
- target_ref: feat/phase2-timexer
- phase: 7
- priority: high

## Goal
1. **Отбор признаков под неопределенность/волатильность (Uncertainty-Aware Feature Selection)**:
   - Внедрена поддержка `selection_objective: str = "volatility"` (обучение LightGBM на абсолютных лог-доходностях $|r_{t, t+H}| = |\ln(P_{t+H}/P_t)|$) и `selection_objective: str = "quantile"` (усреднение TreeSHAP важностей двух моделей $\alpha=0.10$ и $\alpha=0.90$).
   - Сгенерирована и сохранена сигнатура `data/signatures/paper_fold1_signature_volatility_30ts.json` для Fold 1 Paper set.
   - **Анализ сдвига признаков (Feature Importance Shift)**: в отличие от регрессии среднего (доминирование лагов доходности `log_ret_k*`), при волатильностном отборе в Top-10 вышли истинные драйверы ценовой дисперсии: `natr_n60`, `gk_n3` (Garman-Klass), `bb_width_n60_delta`, `parkinson_n10` (волатильность Паркинсона), `volume_z_n5` и спектральные гармоники `fft_harm_*`.

2. **Мульти-критериальный валидационный чекпоинтинг**:
   - `train.val_metric = "pinball"`: оптимизирует суммарный Pinball loss по всем 3 квантилям ($q_{0.10}, q_{0.50}, q_{0.90}$), обеспечивая калибровку коридора и точность медианы.
   - `train.val_metric = "winkler_da_pareto"`: минимизирует Winkler score с жестким штрафом за потерю направленности при $DA < 0.53$:
     $$\text{Score} = \text{val\_winkler} + \max(0, 0.53 - \text{val\_da}) \times 1.0$$

3. **Боевые прогоны на Paper Set (61 тикер, 98k окон) на Runpod GPU**:
   - Обучить модель `c1_hierarchical_mlp_quantile` на новой волатильностной сигнатуре по двум критериям отбора чекпоинта на 3 сидах ($s \in \{0, 1, 2\}$).

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
uv sync

# 1. Быстрый GPU smoke-тест (проверка отбора по pinball и winkler_da_pareto на новой сигнатуре)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda train.val_metric=pinball

uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
  data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda train.val_metric=winkler_da_pareto

# 2. Серия А: Волатильностная сигнатура + Pinball чекпоинтинг (3 сида: 0, 1, 2)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
done

# 3. Серия Б: Волатильностная сигнатура + Pareto Winkler-DA чекпоинтинг (3 сида: 0, 1, 2)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
    data.signature_path=data/signatures/paper_fold1_signature_volatility_30ts.json \
    train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=winkler_da_pareto \
    train.da_threshold=0.53
done

# 4. Визуализация инференса 2023 года для лучших чекпоинтов Серии А и Б
BEST_CKPT_PINBALL=$(ls -td outputs/c1_hierarchical_mlp_quantile_* | grep -v pareto | head -n 1)/checkpoints/best.pt
BEST_CKPT_PARETO=$(ls -td outputs/c1_hierarchical_mlp_quantile_* | head -n 1)/checkpoints/best.pt

# А) Непрерывный коридор со скользящим шагом stride=1
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_PINBALL" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_volatility_pinball_stride1 \
  --stride 1 \
  --device cuda

uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_PARETO" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_volatility_pareto_stride1 \
  --stride 1 \
  --device cuda

# Б) Дискретные 7-дневные прогнозы stride=7
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_PINBALL" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_volatility_pinball_stride7 \
  --stride 7 \
  --device cuda

uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT_PARETO" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_volatility_pareto_stride7 \
  --stride 7 \
  --device cuda
```

## Environment notes
- expected_cwd: repo root (`/workspace/DecisionForecast`)
- data_ready: yes (FNSPID 61 paper tickers)
- gpu: required (NVIDIA L4 / A100 / RTX 4090)
- approx_ram_gb: 32+ GB system RAM, 16+ GB VRAM

## Expected Metrics & Output
- **Distributional Quantile Metrics**:
  - `quantile_coverage`: номинал 0.80 (80%-й коридор)
  - `quantile_sharpness`: ширина тоннеля (сравнение против наивного бейзлайна 0.0843)
  - `quantile_winkler`: интервальный скор Винклера
  - `pinball`: Pinball loss по всем 3 квантилям
- **Directional & Median Metrics**:
  - `da`, `test_da`: направленная точность траектории медианы $q_{0.50}$ (цель $\ge 53\%$)
  - `test_mae_denorm` ($), `test_mse_price`
- **Inference Plots**:
  - `outputs/inference_volatility_pinball_stride1/`
  - `outputs/inference_volatility_pareto_stride1/`
  - `outputs/inference_volatility_pinball_stride7/`
  - `outputs/inference_volatility_pareto_stride7/`
