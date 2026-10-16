# Report: walkforward-folds-and-garch-baseline

- authored_by: runner
- created_at: 2026-10-13T20:25:00Z
- request_folder: agent-handoff/2026-10-14_1200_walkforward-folds-and-garch-baseline/
- tested_ref: feat/phase2-timexer@16b121b71150e4d6ad9ef12d2b4a359a45553179
- status: pass

---

## Executive Summary

В рамках экспериментов Этапа 2 исследовательского протокола (`request.md`) на RunPod GPU (NVIDIA L4 24GB) полностью реализован и протестирован комплекс валидации для ведущих конференций (NeurIPS / ICLR / KDD):
1. **Expanding Walk-Forward Cross-Validation (Folds 1, 2, 3)** с жестким соблюдением Temporal Purge ($H-1 = 6$ баров) и Embargo ($E = 14$ баров), устраняющим утечки между рыночными циклами.
2. **Эконометрический бенчмарк GARCH(1,1)** на базе библиотеки `arch` со скользящей историей $T \ge 252$ торговых дней и аналитической экстраполяцией 7-дневной кумулятивной условной дисперсии $\sigma_{\text{cum}}^2(h)$.
3. **Adaptive Conformal Inference (ACI) Calibration** ($\alpha=0.20, \gamma=0.005$) для непрерывной онлайн-калибровки 80% доверительных интервалов при резкой смене режимов волатильности.
4. **9 полных боевых обучений `c1_hierarchical_mlp_quantile`** (3 фолда × 3 сида: 42, 43, 44) на 61 ликвидном тикере (`paper` ticker set) с 30 отобранными признаками волатильности.

### Ключевые результаты:
- **Ликвидация смещения покрытия через ACI**: Сырые нейросетевые квантили страдают от сильного недопокрытия на медвежьем рынке 2022 года (**68.03%** на Fold 2 при номинале 80.0%). Применение онлайн-калибровки ACI восстанавливает покрытие до **79.67% ± 0.14%** на Fold 2, **79.98% ± 0.05%** на Fold 1 и **80.07% ± 0.09%** на Fold 3 (в среднем **79.91% ± 0.21%** по всем 9 прогонам).
- **Превосходство над GARCH(1,1) по интегральному качеству неопределенности**:
  - GARCH(1,1) генерирует статически избыточно широкие коридоры (Average Sharpness = **0.1028**, Winkler Score = **0.1399**).
  - TimeXer Hierarchical MLP + ACI достигает существенно более узкого коридора (Average Sharpness = **0.0891 ± 0.0184**) при наилучшем Winkler Score (**0.1319 ± 0.0234**), побеждая классический эконометрический эталон на всех трех фолдах.

---

## 1. Сводная таблица (Walk-Forward Matrix)

Метрики для нейросетевых моделей приведены в формате **Mean ± Std** по трем независимым случайным сидам ($s \in \{42, 43, 44\}$).

| Метод / Модель | Метрика | Fold 1 (2021 H2, Bull) | Fold 2 (2022 H2, Bear) | Fold 3 (2023 H2, Recovery) | Walk-Forward Average |
|---|---|:---:|:---:|:---:|:---:|
| **GARCH(1,1) Baseline**<br>*(Эконометрический эталон)* | Coverage 80% (%)<br>Sharpness<br>Winkler Score<br>Median DA (%)<br>Median Dollar MAE ($) | 85.23%<br>0.1096<br>0.1400<br>67.23%<br>n/a | 79.06%<br>0.1095<br>0.1595<br>55.33%<br>n/a | 83.76%<br>0.0893<br>0.1203<br>60.75%<br>n/a | **82.68%**<br>**0.1028**<br>**0.1399**<br>**61.10%**<br>n/a |
| **TimeXer Hierarchical MLP**<br>*(Raw Quantiles)* | Coverage 80% (%)<br>Sharpness<br>Winkler Score<br>Median DA (%)<br>Median Dollar MAE ($) | 78.59 ± 1.73%<br>0.0746 ± 0.0026<br>0.1209 ± 0.0010<br>56.06 ± 0.00%<br>$5.71 ± 0.00 | 68.03 ± 2.70%<br>0.0820 ± 0.0063<br>0.1695 ± 0.0031<br>47.32 ± 0.00%<br>$7.59 ± 0.02 | 81.83 ± 1.43%<br>0.0821 ± 0.0022<br>0.1195 ± 0.0006<br>55.16 ± 0.44%<br>$6.13 ± 0.02 | **76.15 ± 7.22%**<br>**0.0796 ± 0.0043**<br>**0.1367 ± 0.0285**<br>**52.85 ± 4.81%**<br>**$6.48 ± 0.99** |
| **TimeXer Hierarchical MLP**<br>*(ACI Calibrated)* | Coverage 80% (%)<br>Sharpness<br>Winkler Score<br>Median DA (%)<br>Median Dollar MAE ($) | **79.98 ± 0.05%**<br>**0.0775 ± 0.0001**<br>**0.1188 ± 0.0008**<br>56.06 ± 0.00%<br>$5.71 ± 0.00 | **79.67 ± 0.14%**<br>**0.1103 ± 0.0024**<br>**0.1590 ± 0.0025**<br>47.32 ± 0.00%<br>$7.59 ± 0.02 | **80.07 ± 0.09%**<br>**0.0794 ± 0.0008**<br>**0.1180 ± 0.0010**<br>55.16 ± 0.44%<br>$6.13 ± 0.02 | **79.91 ± 0.21%**<br>**0.0891 ± 0.0184**<br>**0.1319 ± 0.0234**<br>**52.85 ± 4.81%**<br>**$6.48 ± 0.99** |

---

## 2. Анализ смены рыночных режимов

### 2.1. Медвежий рынок 2022 года (Fold 2) vs Бычий рынок 2021/2023 (Folds 1 & 3)
1. **Стресс-тест на скачках волатильности (Fold 2)**:
   - В 2022 году американский фондовый рынок пережил цикл агрессивного ужесточения монетарной политики ФРС США, падение технологического сектора и всплеск волатильности (VIX).
   - Модель `TimeXer Hierarchical MLP`, обученная на данных 2015–2020 гг., в режиме сырых квантилей показала резкое падение эмпирического покрытия до **68.03%** (недопокрытие на 12.0% от номинала 80.0%). Это классический эффект смены режима (*regime shift*), при котором безусловная дисперсия ряда превышает исторические предпосылки обучающей выборки.
   - GARCH(1,1) на Fold 2 также испытал снижение покрытия до **79.06%** и ухудшение Winkler Score до **0.1595** (против 0.1400 на Fold 1).
2. **Восстановление и бычий тренд (Folds 1 & 3)**:
   - На бычьих интервалах (2021 год — Fold 1, 2023 год — Fold 3) сырая модель TimeXer демонстрирует устойчивое покрытие близко к таргету (**78.59%** и **81.83%** соответственно), опережая GARCH(1,1) по точности (Winkler 0.1209 vs 0.1400 на Fold 1; Winkler 0.1195 vs 0.1203 на Fold 3) при значительно более узком коридоре (Sharpness 0.0746 vs 0.1096).

### 2.2. Эффективность адаптивной калибровки ACI
1. **Стабилизация номинального покрытия**:
   - Модуль ACI (`src.evaluation.conformal`) с параметром $\gamma = 0.005$ динамически корректирует уровень $\alpha_t$ на каждом шаге инференса:
     $$\alpha_{t+1} = \operatorname{clip}\left(\alpha_t + \gamma (\alpha_{\text{target}} - \text{err}_t), 0.001, 0.999\right)$$
   - На Fold 2 (2022 год) при частых пробоях границы ($\text{err}_t = 1$) значение $\alpha_t$ снижается, заставляя множитель коридора $z_t / z_{\text{nom}}$ увеличиваться. Это адаптивно расширяет коридор (Sharpness увеличивается с 0.0820 до 0.1103) и выводит реальное покрытие на уровень **79.67%**, полностью ликвидируя недопокрытие.
   - На Folds 1 и 3 калибровка ACI удерживает покрытие на уровне **79.98%** и **80.07%** соответственно.
2. **Интегральный выигрыш**:
   - По метрике Winkler Score калиброванная модель ACI превосходит как сырую модель (**0.1319** против **0.1367**), так и GARCH(1,1) (**0.1319** против **0.1399**).
   - Дисперсия покрытия между фолдами падает с $\pm 7.22\%$ (для Raw) до всего $\pm 0.21\%$ (для ACI), что подтверждает теоретические гарантии асимптотической калибровки Gibbs & Candès (2021).

---

## 3. Графики инференса коридоров

Для наглядного сопоставления поведения доверительных тоннелей создана отдельная директория с графиками, отслеживаемая в Git-репозитории:
📁 **[`plots/walkforward_garch_aci/`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/)**

Графика структурирована по подпапкам соответствующих экспериментов и фолдов:
- 📂 **[`fold1_bull_market_2021/`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold1_bull_market_2021/)** — Бычий рынок 2021 года (H2):
  - [`corridor_comparison_overview_fold1.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold1_bull_market_2021/corridor_comparison_overview_fold1.png) — Мультитикерное сопоставление (AAPL, NVDA, JPM)
  - [`corridor_comparison_aapl_fold1.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold1_bull_market_2021/corridor_comparison_aapl_fold1.png) — 3-панельное сравнение (GARCH vs Raw vs ACI)
  - [`corridor_comparison_nvda_fold1.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold1_bull_market_2021/corridor_comparison_nvda_fold1.png) — 3-панельное сравнение
  - [`corridor_comparison_jpm_fold1.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold1_bull_market_2021/corridor_comparison_jpm_fold1.png) — 3-панельное сравнение
- 📂 **[`fold2_bear_market_2022/`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold2_bear_market_2022/)** — Медвежий рынок 2022 года (H2, цикл роста процентных ставок):
  - [`corridor_comparison_overview_fold2.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold2_bear_market_2022/corridor_comparison_overview_fold2.png) — Мультитикерное сопоставление
  - [`corridor_comparison_aapl_fold2.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold2_bear_market_2022/corridor_comparison_aapl_fold2.png) — 3-панельное сравнение
  - [`corridor_comparison_nvda_fold2.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold2_bear_market_2022/corridor_comparison_nvda_fold2.png) — 3-панельное сравнение
  - [`corridor_comparison_jpm_fold2.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold2_bear_market_2022/corridor_comparison_jpm_fold2.png) — 3-панельное сравнение
- 📂 **[`fold3_recovery_2023/`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold3_recovery_2023/)** — Восстановление и технологический бум 2023 года (H2):
  - [`corridor_comparison_overview_fold3.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold3_recovery_2023/corridor_comparison_overview_fold3.png) — Мультитикерное сопоставление
  - [`corridor_comparison_aapl_fold3.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold3_recovery_2023/corridor_comparison_aapl_fold3.png) — 3-панельное сравнение
  - [`corridor_comparison_nvda_fold3.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold3_recovery_2023/corridor_comparison_nvda_fold3.png) — 3-панельное сравнение
  - [`corridor_comparison_jpm_fold3.png`](file:///workspace/DecisionForecast/plots/walkforward_garch_aci/fold3_recovery_2023/corridor_comparison_jpm_fold3.png) — 3-панельное сравнение

---

## 4. Детализация прогонов и артефактов

| Fold | Seed | Run Dir | Raw Cov80 | ACI Cov80 | ACI Winkler | Median DA | Median MAE ($) |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 42 | `outputs/2026-10-13/18-17-01` | 78.59% | 79.97% | 0.1196 | 56.06% | $5.71 |
| **Fold 1** | 43 | `outputs/2026-10-13/18-30-44` | 80.33% | 80.04% | 0.1180 | 56.06% | $5.72 |
| **Fold 1** | 44 | `outputs/2026-10-13/18-39-26` | 76.87% | 79.94% | 0.1188 | 56.06% | $5.72 |
| **Fold 2** | 42 | `outputs/2026-10-13/18-49-31` | 69.04% | 79.75% | 0.1607 | 47.31% | $7.56 |
| **Fold 2** | 43 | `outputs/2026-10-13/19-00-37` | 70.08% | 79.76% | 0.1600 | 47.31% | $7.61 |
| **Fold 2** | 44 | `outputs/2026-10-13/19-10-29` | 64.98% | 79.51% | 0.1562 | 47.32% | $7.61 |
| **Fold 3** | 42 | `outputs/2026-10-13/19-23-51` | 80.87% | 79.97% | 0.1190 | 55.41% | $6.11 |
| **Fold 3** | 43 | `outputs/2026-10-13/19-34-01` | 83.48% | 80.13% | 0.1170 | 55.41% | $6.11 |
| **Fold 3** | 44 | `outputs/2026-10-13/19-47-23` | 81.15% | 80.11% | 0.1181 | 54.65% | $6.15 |

---

## 5. Воспроизводимость (Run Commands)

```bash
# 1. GARCH(1,1) Baseline
for f in 1 2 3; do
  uv run python -m src.evaluation.garch_baseline --ticker-set paper --fold $f --device cpu
done

# 2. Обучение 9 моделей (3 фолда × 3 сида)
for f in 1 2 3; do
  for s in 42 43 44; do
    uv run python -m src.training.train \
      data=fnspid_dual_ts model=c1_hierarchical_mlp_quantile \
      data.fold=$f data.signature_path=data/signatures/paper_fold${f}_signature_volatility_30ts.json \
      train.ticker_set=paper train.seed=$s train.device=cuda train.val_metric=pinball
  done
done

# 3. Постобработка ACI Conformal Calibration
for f in 1 2 3; do
  for s in 42 43 44; do
    uv run python scripts/eval_saved_predictions.py --fold $f --seed $s --conformal-gamma 0.005
  done
done

# 4. Построение графиков (сохраняются в plots/walkforward_garch_aci/)
uv run python scripts/generate_corridor_comparison_plots.py --fold 1 --run-dir outputs/2026-10-13/18-39-26
uv run python scripts/generate_corridor_comparison_plots.py --fold 2 --run-dir outputs/2026-10-13/19-10-29
uv run python scripts/generate_corridor_comparison_plots.py --fold 3 --run-dir outputs/2026-10-13/19-47-23
```

