# Анализ архитектур и динамики обучения: c1_dual, c1_hierarchical, c1_late_fusion

В данном отчёте представлены детальные схемы архитектур трёх исследованных моделей мультимодального прогнозирования (`c1_dual`, `c1_hierarchical`, `c1_late_fusion`), а также визуализация и численный разбор динамики лосса (`Huber Loss`) и валидационной ошибки (`Val MSE`) по эпохам для всех сидов (0, 1, 2).

---

## 1. Графики обучения (Training Loss и Validation MSE)

Графики построены по логам боевого прогона на выборке FNSPID (`outputs/becker-dual-hierarchical.log`).

![Динамика Train Loss и Validation MSE](loss_curves.png)

### Наблюдения по динамике обучения:
1. **`c1_dual` (F2, e=1, stride 12)**:
   - В отличие от предыдущего прогона с взрывом градиента на 41-м батче, со стабилизированным CFI лосс монотонно и плавно снижается с $\sim 0.465$ до $\sim 0.377$.
   - Ранний останов срабатывает на 6, 13 и 7 эпохах при достижении минимума валидационной ошибки $\text{Val MSE} \approx 0.76 - 0.78$.
2. **`c1_hierarchical` (F2, e=2, stride 12)**:
   - Показывает наиболее стабильную траекторию убывания Train Loss (с $0.466$ до $0.375$).
   - Лучший сид (Seed 1) остановился на 7 эпохе с $\text{Test MSE} = 0.7088$.
3. **`c1_late_fusion` (F5 OHLCV, e=2, stride 6)**:
   - Быстрая и глубокая оптимизация: за счёт мелкого шага патчей (`stride=6`) модель быстрее находит устойчивое представление.
   - Минимальный $\text{Val MSE} \approx 0.765 - 0.786$. На тесте модель демонстрирует рекордную генерализацию: $\text{Test MSE} = 0.6880 \pm 0.0119$ (все сиды строго ниже 0.70).

---

## 2. Схемы архитектур

### 2.1. `c1_dual` — Dual-Token TimeXer с изолированными модальностями

В этой архитектуре временные ряды цен патчируются вместе с rFFT-спектром. Индикаторы ($G_{ts}$) и текст ($G_{text}$) представлены отдельными глобальными токенами в одном пространстве, но **взаимно изолированы** маской внимания.

```mermaid
flowchart TD
    subgraph Inputs["1. Входные модальности"]
        X["Цены: Close, Volume<br/>[B, T, 2]"]
        TS["25 Индикаторов режима<br/>[B, T, 25]"]
        TXT["Текст новостей FinBERT<br/>[B, T, 15]"]
    end

    subgraph Embedding["2. Эмбеддинг и токенизация"]
        Unfold["Unfold патчей (L=12, S=12) + rFFT(close)"]
        PatchProj["Linear Projection -> [B, K, 64]"]
        X --> Unfold --> PatchProj
        
        Gts_proj["Linear(25 -> 64) -> Mean"]
        TS --> Gts_proj
        Gts_tok["Токен G_ts [B, 1, 64]"]
        Gts_proj --> Gts_tok
        
        Gtxt_proj["Linear(15 -> 64) -> Mean"]
        TXT --> Gtxt_proj
        Gtxt_tok["Токен G_text [B, 1, 64]"]
        Gtxt_proj --> Gtxt_tok
    end

    subgraph DualAttention["3. Dual-Mask Self-Attention"]
        ConcatTokens["Конкатенация: [Patches, G_ts, G_text] -> 7 токенов"]
        PatchProj --> ConcatTokens
        Gts_tok --> ConcatTokens
        Gtxt_tok --> ConcatTokens
        
        MaskedAttn["Self-Attention с маской:<br/>- Патчи видят всё<br/>- G_ts <-> G_text заблокированы (-inf)"]
        ConcatTokens --> MaskedAttn
    end

    subgraph CrossAndBridge["4. Cross-Attention и Gated Bridge"]
        CrossTS["G_ts Cross-Attention на TS_exo [B, T, 64]"]
        CrossTXT["G_text Cross-Attention на Text_exo [B, T, 64]"]
        MaskedAttn --> CrossTS
        MaskedAttn --> CrossTXT
        
        Bridge["Gated Global-to-Patch Bridge:<br/>Patches += tanh(α_ts)*Δ_ts + tanh(α_text)*Δ_text"]
        CrossTS --> Bridge
        CrossTXT --> Bridge
    end

    subgraph Prediction["5. Прогноз"]
        Head["ForecastHead (pooling патчей)"]
        Bridge --> Head
        Out["Прогноз лог-доходности Y<br/>[B, H=7]"]
        Head --> Out
    end
```

---

### 2.2. `c1_hierarchical` — Двухслойный иерархический TimeXer

В этой архитектуре модальности разнесены по разным слоям энкодера: **Слой 1** интегрирует макро/рыночный режим из 25 индикаторов, а **Слой 2** модулирует полученные представления новостным сентиментом. Токены $G_{ts}$ и $G_{text}$ никогда не пересекаются в одной последовательности.

```mermaid
flowchart TD
    subgraph InputsH["1. Входы"]
        Xh["Цены: Close, Volume<br/>[B, T, 2]"]
        TSh["25 Индикаторов режима<br/>[B, T, 25]"]
        TXTh["15D Текст новостей<br/>[B, T, 15]"]
    end

    subgraph PatchingH["2. Патчирование цен"]
        UnfoldH["Unfold (L=12, S=12) + rFFT(close)"]
        ProjH["Patch Linear Projection"]
        Xh --> UnfoldH --> ProjH
        P0["Патчи Z_0 [B, K, 64]"]
        ProjH --> P0
    end

    subgraph Layer1["3. Слой 1: Режим рынка (Индикаторы)"]
        TS_exo["TS Linear -> TS_exo [B, T, 64]"]
        TSh --> TS_exo
        Gts_H["Токен G_ts [B, 1, 64]"]
        TS_exo --> Gts_H
        
        CrossL1["G_ts Cross-Attention на TS_exo"]
        Gts_H --> CrossL1
        
        SelfL1["Self-Attention: [Z_0, G_ts]"]
        P0 --> SelfL1
        CrossL1 --> SelfL1
        
        BridgeL1["Gated Token Bridge:<br/>Z_1 = Norm(Z_0 + tanh(α_ts) * Attn(Z_0, G_ts))"]
        SelfL1 --> BridgeL1
        P1["Патчи Z_1 (обогащены индикаторами)"]
        BridgeL1 --> P1
    end

    subgraph Layer2["4. Слой 2: Модуляция новостями (Текст)"]
        TXT_exo["Text Linear -> Text_exo [B, T, 64]"]
        TXTh --> TXT_exo
        Gtxt_H["Токен G_text [B, 1, 64]"]
        TXT_exo --> Gtxt_H
        
        CrossL2["G_text Cross-Attention на Text_exo"]
        Gtxt_H --> CrossL2
        
        SelfL2["Self-Attention: [Z_1, G_text]"]
        P1 --> SelfL2
        CrossL2 --> SelfL2
        
        BridgeL2["Gated Token Bridge:<br/>Z_2 = Norm(Z_1 + tanh(α_text) * Attn(Z_1, G_text))"]
        SelfL2 --> BridgeL2
        P2["Патчи Z_2 (обогащены новостями)"]
        BridgeL2 --> P2
    end

    subgraph HeadH["5. Прогноз"]
        HeadHier["ForecastHead (pooling Z_2)"]
        P2 --> HeadHier
        OutH["Прогноз лог-доходности Y<br/>[B, H=7]"]
        HeadHier --> OutH
    end
```

---

### 2.3. `c1_late_fusion` — Позднее слияние (Рекордная модель)

В архитектуре чемпиона энкодер фокусируется исключительно на плотном анализе динамики OHLCV (шаг 6 баров) и текстового потока, а 25 индикаторов подаются в самом конце через 2-слойный MLP как аддитивная поправка к пуленному векторному представлению.

```mermaid
flowchart TD
    subgraph InputsLF["1. Входы"]
        Xlf["OHLCV цены (5 каналов)<br/>[B, T, 5]"]
        TXTlf["15D Текст новостей<br/>[B, T, 15]"]
        TSlf["25 Индикаторов на последнем баре<br/>[B, 25] (Bar T)"]
    end

    subgraph DenseEncoder["2. Плотный TimeXer Энкодер (e_layers=2, Stride=6)"]
        UnfoldLF["Unfold (L=12, Stride=6) + rFFT(close)<br/>[B, 9 патчей, 64]"]
        Xlf --> UnfoldLF
        
        GlbTxt["Глобальный токен новостей G_text"]
        TXTlf --> GlbTxt
        
        EncBlock["2 Слоя TimeXerEncoder + GlobalToPatch Bridge:<br/>Патчи глубоко взаимодействуют с новостями"]
        UnfoldLF --> EncBlock
        GlbTxt --> EncBlock
        
        Pool["Пул патчей (Last bar / Mean)<br/>Z_patch [B, 64]"]
        EncBlock --> Pool
    end

    subgraph LateMLP["3. Late Residual Fusion"]
        TechMLP["Tech MLP:<br/>Linear(25 -> 64) -> GELU -> Dropout -> Linear(64 -> 64)"]
        TSlf --> TechMLP
        Ztech["Z_tech [B, 64]"]
        TechMLP --> Ztech
        
        ResidualAdd["Аддитивное слияние:<br/>Z_final = LayerNorm(Z_patch + Z_tech)"]
        Pool --> ResidualAdd
        Ztech --> ResidualAdd
    end

    subgraph PredLF["4. Прогноз"]
        HeadLF["Линейная голова ForecastHead"]
        ResidualAdd --> HeadLF
        OutLF["Прогноз лог-доходности Y<br/>[B, H=7]<br/>(Test MSE: 0.6880)"]
        HeadLF --> OutLF
    end
```

---

## 3. Сводное сопоставление архитектурных решений

| Характеристика | `c1_dual` | `c1_hierarchical` | `c1_late_fusion` |
|---|---|---|---|
| **Входные ценовые каналы** | 2 (Close, Volume) | 2 (Close, Volume) | **5 (OHLCV)** |
| **Шаг патча (Patch Stride)** | 12 (5 патчей) | 12 (5 патчей) | **6 (9 патчей — в 1.8 раза плотнее)** |
| **Число слоёв энкодера** | 1 | 2 (Слой TS + Слой Text) | 2 |
| **Интеграция индикаторов** | Cross-attention по всем 60 барам в единый токен $G_{ts}$ | Cross-attention + Self-attention в изолированном Слое 1 | **Late Fusion: MLP над срезом индикаторов на последнем баре** |
| **Изоляция модальностей** | Маска внимания $-\infty$ между $G_{ts}$ и $G_{text}$ | Разделение по разным слоям | Индикаторы исключены из внимания, подаются на выход энкодера |
| **Параметры ($N_{params}$)** | 122 121 | 172 169 | 178 823 |
| **Test MSE (Mean ± std)** | 0.7809 ± 0.0555 | 0.7457 ± 0.0378 | **0.6880 ± 0.0119** |
| **Ключевое преимущество** | Равномерное параллельное внимание к обеим модальностям | Чёткое разделение ролей: режим рынка $\to$ влияние новостей | **Минимум шума во внимании, максимальная плотность патчей цен** |

> [!TIP]
> **Почему `c1_late_fusion` побеждает с большим отрывом?**
> В финансовом прогнозировании внимание между 25 коррелированными техническими индикаторами на всех 60 временных шагах склонно переобучаться и зашумлять патчи цен. `c1_late_fusion` решает эту проблему элегантно: энкодер обучается извлекать динамику цен и новостей без помех, а актуальный рыночный режим на момент прогноза ($T$) добавляется аддитивно через компактный MLP, что дает оптимальную регуляризацию и рекордное качество.
