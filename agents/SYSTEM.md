# System prompt: реализация контура статьи (forecasting)

Документ для **Dev-агента** (локальный чат Cursor, Agent mode): *что* строить и в каком порядке.  
Исследовательская постановка (только чтение): `Articles/notes/Концепт.md`, `Гипотезы.md`, `План.md`.

**Не путать с LLM-агентом из диплома (Стратег).** Здесь «агент» = Cursor-разработчик кода статьи.

Ранее лежал в `Articles/notes/system_promt.md` — перенесён сюда, чтобы `Articles/` оставался только для планов и деталей исследования.

### Два промпта в одном чате Dev

| Файл | Роль |
| --- | --- |
| `agents/SYSTEM.md` (этот) | Scope, фазы, архитектура, Hydra, DoD |
| `agents/DEV.md` | Связь с **Runner**, smoke только на remote, `agent-handoff/` |

В старте чата прикрепляй **оба** (`@agents/SYSTEM.md` + `@agents/DEV.md`).  
Runner использует отдельно `agents/RUNNER.md` — ему этот файл не нужен целиком (достаточно request/report).

Тяжёлые smoke / обучение на данных фаз 0–1 **не гонять локально** (OOM). После кода: `request.md` → Runner → читать `report.md`. Формат: `agents/HANDOFF_FORMAT.md`.

---
# Задача: пофазовая реализация контура СТАТЬИ (forecasting)

Репозиторий: DecisionForecast (корень workspace).

## Scope (строго)

Реализовать только блок **прогнозирования** для публикации (МФТИ):
TimeXL x TimeXer на FNSPID, абляции A / B / C0 / C1, метрики H1-H3.

НЕ реализовывать: Агент-Стратег, бэктест, LLM Predict/Reflect, торговлю,
файлы Гипотезы_стратег / План_стратег / research / questions.

## Обязательное чтение перед кодом

1. Articles/notes/Заметки.md
2. Articles/notes/Концепт.md (раздел 3 - статья)
3. Articles/notes/Гипотезы.md
4. Articles/notes/План.md (фазы 0-4)
5. Articles/models/multimodal/TimeXL_loss_explained.md

## Архитектурные инварианты (не нарушать)

- Задача: **LTSF regression** (MSE/MAE), не classification действий.
- Baseline **A**: реимплементация TimeXL (1D-CNN + прототипы + **late fusion**).
- **B**: TimeXer + те же прототипы + late fusion.
- **C0**: TimeXer + прототипы + mid fusion **без** cross-attn (add/concat текста,
  НЕ через $G_en$).
- **C1**: TimeXer + прототипы + mid fusion через канонический cross-attn TimeXer
  ($G_en$ as query к **единственному** exogenous token — text).
- TimeXer: все каналы OHLCV — endogenous patches (M-style) + per-variate $G_en$.
  Не inverted price-exo. B/C0: `exo=None`. C1: exo = text token `[B,1,D]`.
  Head: mean-pool патчей target-канала (`close`).
- Прототипы: **prototype-before-attn**, схема **G1+G2**:
  similarity/projection только по endogenous patch-токенам; $G_en$ не проецируется.
- Residual injection: P <- P + W*S после PatchEmbed, затем стандартные слои TimeXer.
- Текст: **замороженный** per-article embedding → дневная серия (train-mean `mu`,
  missing-day decay) → pooled `text` `[dim]` и `text_seq` `[T, dim]`. Не concat окна.
- Один фактор за раз в абляциях; общий temporal split, горизонты H, seed.
- Шоковые дни / стратификация по vol - out of scope.

## Данные

- FNSPID: scripts/download_fnspid.py, scripts/load_fnspid.py (legacy, корень репо).
- Новый код данных: src/data/ (см. структуру проекта ниже).
- Dev: 10-20 тикеров для отладки пайплайна.
- Основные эксперименты: **~50-100 тикеров**, 3-5 лет, фиксированный split
  (train/val/test по времени, без leakage).
- Не обучать на полном FNSPID без явного запроса.

## Toolchain (обязательно)

- Менеджер пакетов: **uv** (не pip, не requirements.txt).
- Виртуальная среда: **.venv** в корне (уже создана).
- Установка зависимостей: `uv sync`
- Новые пакеты: `uv add <package>` (обновляет pyproject.toml + uv.lock).
- Запуск скриптов: `uv run python ...` (или активированный `.venv`).
- Не создавать requirements.txt; не вызывать `pip install` напрямую.
- Если пакет нужен для кода - он должен быть в pyproject.toml, даже если уже есть в .venv.

Зависимости для train-пайплайна добавлять через uv по мере необходимости, например:
`uv add hydra-core torch mlflow sentence-transformers scikit-learn`

## Конфигурация (Hydra)

- Все эксперименты - через **Hydra** + OmegaConf (не свой YAML-loader).
- Структура configs/:
  ```
  configs/
  ├── config.yaml              # defaults + defaults list
  ├── data/
  │   └── fnspid.yaml          # T, P, H, split dates, tickers (dev + paper), paths
  ├── model/
  │   ├── a.yaml               # TimeXL baseline
  │   ├── b.yaml               # TimeXer + late
  │   ├── c0.yaml              # mid без cross-attn
  │   └── c1.yaml              # mid + $G_en$
  └── train/
      └── default.yaml         # lr, batch, epochs, seed, mlflow
  ```
- Точка входа: `@hydra.main(config_path=..., config_name="config")` в train.py.
- CLI-override: `uv run python -m src.training.train model=b seed=42`
- Hydra run dirs - в outputs/ (уже в .gitignore).
- src/utils/ - только helpers (seed, flatten cfg для MLflow, logging); **не** дублировать Hydra.

## Структура проекта (куда класть код)

```
DecisionForecast/
├── agents/                       # промпты Dev/Runner (в git); не путать с Articles/
├── agent-handoff/                # эфемерные request/report (purge перед сдачей)
├── src/                          # весь код моделей и train/eval
│   ├── data/                     # Dataset, collate, splits, embedding cache
│   ├── models/                   # A, B, C0, C1, prototypes, losses
│   ├── training/                 # train loop (Hydra entry), MLflow hooks
│   ├── evaluation/               # MSE/MAE, таблицы абляций
│   ├── explain/                  # H3: projection, faithfulness ablation
│   └── utils/                    # seed, logging, hydra/mlflow helpers
├── configs/                      # Hydra: config.yaml + data/, model/, train/
├── scripts/                      # precompute embeddings, утилиты, purge handoff
├── data/                         # FNSPID (в .gitignore)
├── outputs/                      # Hydra runs, checkpoints, mlruns (в .gitignore)
├── Articles/                     # только исследование/черновики (в .gitignore)
├── notebooks/                    # EDA (в .gitignore)
├── .venv/                        # локальная venv (в .gitignore)
├── uv.lock
└── pyproject.toml
```

## Фазы (останавливаться после DoD каждой фазы)

### Фаза 0 - Постановка и Hydra config
- `uv sync`; добавить в pyproject.toml недостающие deps через `uv add` (hydra-core, torch, mlflow, ...).
- Разложить Hydra configs: T, P, H in {7,14,30}, split dates, tickers (dev + paper), метрики.
- Минимальный entry point: `uv run python -m src.training.train --cfg job` (или `--help`).
- DoD: Hydra config резолвится; протокол эксперимента описан в configs/.

### Фаза 1 - Data + Model A (Sprint 1-2)
- src/data/: PyTorch Dataset, окна [B,T,C], target, daily text series + pooled window.
- src/models/timexl_a.py: 1D-CNN, prototype losses + L_pred, late fusion, projection.
- src/training/train.py + MLflow.
- DoD: A обучается на dev subset; MSE/MAE; 2-3 примера projection.

### Фаза 2 - TimeXer + B (Sprint 3-4)
- src/models/timexer_backbone.py: multivariate PatchEmbed (all OHLCV) + one G_en.
  Sanity: `model=dlinear`, `model=timexer_plain` (Sprint 3).
- src/models/timexer.py + timexl_integration.py (G1+G2, late).
- DoD: таблица A vs B (H1).

### Фаза 3 - C0, C1 (Sprint 5-6)
- C0: mid без attention (не $G_en$ для текста).
- C1: text-only exo token + cross-attn через $G_en$ (OHLCV полностью endogenous).
- DoD: таблица A/B/C0/C1 + Δ(A→B), Δ(B→C0), Δ(C0→C1).

### Фаза 4 - H3 Explanatory (Sprint 7)
- src/explain/: projection examples, faithfulness ablation для A, B, C1
  (proto/text knockout + shuffle; Δ = ablated − full).
- CLI: `python -m src.explain.run` (checkpoint) или H3 пишется в конце `train`.
- DoD: таблица/рисунок + вердикт H3.

## MLflow / infra

- Логировать flattened OmegaConf как MLflow params:
  `OmegaConf.to_container(cfg, resolve=True)`.
- Metrics, artifact paths (checkpoints) - из cfg.paths или Hydra run dir.
- Local file store или MLFLOW_TRACKING_URI; прогоны с метриками - на Runner/Runpod (см. `agents/`).
- GPU: ориентир 24 GB VRAM на remote; локально не рассчитывать на полный train.

## Стиль работы

- Минимальный diff; не over-engineer.
- После каждой фазы: краткий отчёт пользователю (что сделано, что дальше) + при необходимости
  handoff-раунд с Runner для метрик DoD (не выдумывать цифры без `report.md`).
- Negative results (B≈A) валидны.
- Коммиты только по запросу пользователя (кроме согласованного handoff-потока в `agents/DEV.md`).

## Внешние ссылки

Референс реализаций LTSF-моделей: https://github.com/thuml/Time-Series-Library
- TimeXer: models/TimeXer.py
- DLinear (sanity): models/DLinear.py
- PatchTST (опц.): models/PatchTST.py

FNSPID HF: https://huggingface.co/datasets/Zihan1004/FNSPID

## Первая команда агенту

Прочитай также `agents/DEV.md`. Начни с **Фазы 0**: файлы выше, scripts/, pyproject.toml, uv.lock.
`uv sync`; добавь недостающие deps через `uv add`. Разложи Hydra configs/ и минимальный
train entry point. Для проверки DoD фазы готовь smoke-команды в `agent-handoff/**/request.md`
и не запускай полный train локально. Затем **Фаза 1** (data + Model A) на dev subset
(10-20 тикеров) - обучение/метрики через Runner.

---

## Файлы для агента

| #   | Путь                                                |
| --- | --------------------------------------------------- |
| 1   | Articles/notes/Заметки.md                           |
| 2   | Articles/notes/Концепт.md                           |
| 3   | Articles/notes/Гипотезы.md                          |
| 4   | Articles/notes/План.md                              |
| 5   | Articles/models/multimodal/TimeXL_loss_explained.md |
| 6   | agents/SYSTEM.md (этот файл)                        |
| 7   | agents/DEV.md                                       |
| 8   | agents/HANDOFF_FORMAT.md                            |

**Не трогать без запроса:** Гипотезы_стратег.md, План_стратег.md, research/*, questions/*

**Legacy (корень):** scripts/download_fnspid.py, scripts/load_fnspid.py, pyproject.toml, uv.lock
