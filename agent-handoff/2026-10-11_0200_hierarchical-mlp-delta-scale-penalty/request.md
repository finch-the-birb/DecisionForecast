# Request: hierarchical-mlp-delta-scale-penalty

- authored_by: dev
- created_at: 2026-10-11T02:00:00Z
- target_ref: feat/phase2-timexer
- phase: 6
- priority: high

## Goal
1. **Регуляризация размаха амплитуды приращений (Scale Shrinkage Penalty)**:
   - В эксперименте P11.1 модель `TimeXerHierarchicalMLP` (`c1_hierarchical_mlp_delta_g06_a03`) успешно сломала артефакт «линейки под углом», повысив кривизну траекторий в 6–13 раз и зафиксировав DA 55.74% на 61 акции (и 65–67% на tech-лидерах). Однако возник овершутинг амплитуды (volatility overshooting), завышающий колебания в долларах (Test MAE $7.37 vs $6.18).
   - Внедрен штраф L2-сжатия амплитуды приращений:
     $$\mathcal{L}_{\text{scale}} = \beta_{\text{scale}} \cdot \frac{1}{H} \sum_{h=1}^H (\hat{y}_{\text{delta}, h})^2 = \beta_{\text{scale}} \cdot \text{mean}(\hat{y}^2)$$
     прижимающий избыточные выбросы размаха и возвращающий MAE к оптимальным уровням ($3.5–$6.0) при сохранении адаптивной кривизны и высокого DA.
2. **Свип по силе штрафа $\beta_{\text{scale}} \in [0.01, 0.05, 0.10]$**:
   - 3 конфигурации на базе лучшей точки ($\gamma_{\text{dir}}=0.6, \alpha_{\text{corr}}=0.3$):
     1. `c1_hierarchical_mlp_delta_g06_a03_b001` ($\beta=0.01$)
     2. `c1_hierarchical_mlp_delta_g06_a03_b005` ($\beta=0.05$)
     3. `c1_hierarchical_mlp_delta_g06_a03_b010` ($\beta=0.10$)
   - Каждая конфигурация обучается на полной выборке `paper` (61 тикер, 98k окон) по 3 сидам (0, 1, 2) на Runpod GPU (всего $3 \times 3 = 9$ прогонов).
3. **Визуализация инференса 2023 года**:
   - Построение траекторий цен (`src.evaluation.plot_inference`) для лучшего чекпоинта свипа по акциям AAPL, NVDA, MSFT, JPM.

## Commands
```bash
cd /workspace/DecisionForecast
git fetch origin
git checkout feat/phase2-timexer
git pull --ff-only origin feat/phase2-timexer
git rev-parse HEAD
uv sync

# 1. Быстрый smoke-тест (GPU, проверка сборки TimeXerHierarchicalMLP и scale penalty)
uv run python -m src.training.train \
  data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g06_a03_b005 \
  train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 \
  train.epochs=1 train.ticker_set=paper train.device=cuda

# 2. Боевой прогон Конфигурации 1: b001 (beta=0.01, gamma=0.6, alpha=0.3, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g06_a03_b001 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 3. Боевой прогон Конфигурации 2: b005 (beta=0.05, gamma=0.6, alpha=0.3, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g06_a03_b005 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 4. Боевой прогон Конфигурации 3: b010 (beta=0.10, gamma=0.6, alpha=0.3, 3 seeds)
for s in 0 1 2; do
  uv run python -m src.training.train \
    data=fnspid_dual_ts model=c1_hierarchical_mlp_delta_g06_a03_b010 \
    train.ticker_set=paper train.seed=$s train.device=cuda
done

# 5. Построение графиков инференса 2023 года для лучшего чекпоинта свипа
# (Укажите путь к best.pt модели с минимальным test_mse_price / лучшим балансом MAE и DA)
BEST_CKPT="outputs/.../checkpoints/best.pt"
uv run python -m src.evaluation.plot_inference \
  --checkpoint "$BEST_CKPT" \
  --tickers AAPL,NVDA,MSFT,JPM \
  --out-dir outputs/inference_plots \
  --stride 7 \
  --device cuda
```

## Environment notes
- expected_cwd: repo root (`/workspace/DecisionForecast`)
- data_ready: yes (FNSPID 61 paper tickers)
- gpu: required (NVIDIA L4 / A100 / RTX 4090)
- approx_ram_gb: 32+ GB system RAM, 16+ GB VRAM

## Expected Metrics & Output
- **Delta Space**: `delta_mse`, `delta_mae`, `delta_da`
- **Unrolled Price Space**: `test_mse_price`, `test_mae_denorm` ($), `test_da`
- **Inference Plots**: PNG-графики в `outputs/inference_plots/` с визуализацией динамики цен 2023 года для оценки обуздания овершутинга амплитуды.
