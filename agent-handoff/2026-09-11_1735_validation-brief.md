# Validation brief — DecisionForecast phase-2 (dev)

For an agent that **validates the current solution and finds bottlenecks**.
Do not treat this file as a request to train or to open a new HP grid.
Authoritative metrics live in `agent-handoff/*/report.md` (Runner logs), not in chat.

- **Repo:** `finch-the-birb/DecisionForecast`
- **Branch:** `feat/phase2-timexer` (base `feat/phase1-data`)
- **PR:** https://github.com/finch-the-birb/DecisionForecast/pull/2
- **HEAD at brief:** `9b0636b` (`handoff: baseline-seed report (pass)`)
- **Code that all recent uncapped runs share:** `03f123b` (Coverage + DataLoader prefetch)
- **Handoff protocol:** `agents/HANDOFF_FORMAT.md` — Dev ↔ Runner via `agent-handoff/` only

---

## Mission

1. Check that the **locked protocol** in code matches what was actually trained.
2. Recompute / sanity-check the **scoreboard** below from the cited reports (do not mix tables).
3. Find **narrow places**: bugs, metric protocol mismatches, fusion that does not use text, cache keys, evaluation scripts, overfit.
4. Propose the **smallest** next code or experiment step. Do **not** launch `ticker_set=paper` or a cartesian T×H×model grid unless the user explicitly asks.

---

## What the system is

Article contour: **TimeXL × TimeXer on FNSPID** (H1 backbone, H2 fusion, H3 faithfulness).

| id | class | fusion |
|----|--------|--------|
| A | 1D-CNN TimeXL | late concat pooled text at head |
| B | TimeXer + prototypes | late concat; encoder `exo=None` |
| C0 | TimeXer + prototypes | mid **add** of text onto patches after proto; no \(G_{en}\)→text |
| C1 | TimeXer + prototypes | mid **cross-attn**: query = single \(G_{en}\), key/value = text exo (`per_day`) |
| `timexer_plain` | TimeXer | no proto, no text |
| `dlinear` | DLinear | no text (data still built with text; model ignores it) |

OHLCV = multivariate PatchEmbed (endogenous). C1 exo slot is **text only**. Not C2 (text in self-attn).

Key files: `src/models/timexer_backbone.py`, `timexl_a.py`, `timexer_b.py`, `timexer_c0.py`, `timexer_c1.py`, `src/data/dataset.py`, `src/data/text_series.py`, `src/training/train.py`.

---

## Locked data protocol (do not change unless asked)

From `configs/data/fnspid.yaml` + `configs/README.md`:

- `ticker_set=dev` (15 names). **Not paper.**
- `lookback_T=60`, `patch_len=12`, `patch_stride=6`, core `horizon=7`
- Split: drop prices `< 2015-01-01`; train last-target ≤ `2021-12-31`; val ≤ `2022-12-31`; test after
- `normalize: per_window` (RevIN-style on each lookback; MSE/MAE in **z-space**; also `mae_denorm`)
- Uncapped windows on this split: **train 23478 / val 2510 / test 2480** (T=60 H=7)
- Text encoder: public `FinLang/finance-embeddings-investopedia` (768-d). **Never** gated `FinLang/investopedia_embedding`. **No `HF_TOKEN`.**
- μ file: `mu_dev_2015-01-01_2021-12-31.npy`
- Day-0 news interval `[d0−1 calendar day, d0)` (not `int64.min`)
- Daily `E[d]`: mean articles in `[prev_trading_day, d)` minus train μ; missing days `decay` with `λ=0.03`
- Window pool: `recency_weighted` (same λ)
- Toolchain: **uv**, Hydra. Train: `uv run python -m src.training.train`

Coverage log (uncapped T=60 H=7, stable across jobs):

```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

Train `zero_windows=0.565` is **expected** on full 2015–2021 (not a protocol bug). Old 0.42 was a capped late-2021 slice.

---

## Locked training HP (after sweeps)

Use this as “current solution” for A/B/C* comparisons:

| knob | value | why |
|------|--------|-----|
| `train.lr` | **0.0003** | `hp-uncapped-dev`: 0.001 worse last-epoch val on A/B/C0/C1 |
| `d_model` / `d_ff` | **64 / 256** | paired; 128/512 lost val on B/C0/C1 |
| C0 `n_prototypes` | **5** | 3-seed val winner among B/C0/C1 shortlist |
| B `n_prototypes` | 10 | best val among B in proto sweep |
| C1 `n_prototypes` | 5 | seed=42 proto-sweep val winner; **lost** 3-seed mean val |
| A `n_prototypes` | 10 | 5 lost to 10 on A only |
| `train.num_workers` | **4** | `pin_memory` on CUDA, `persistent_workers` |
| `epochs` / `patience` | 20 / 5 | early stop; **test loads `best.pt`** |
| seed for HP screens | 42 | confirmation seeds **0,1,2** |

**Do not mix tables** across: capped 256-window HP, R3 (`lr=0.001`, proto=10), seed=42 vs 0/1/2, `num_workers=0` vs 4 (shuffle changes).

`ablation_table` (`src/evaluation/ablation_table.py`) is **unsafe**: latest-per-(model,H,seed) mixed seed=42 **capped** smokes into R3 → n=4. Do not trust that markdown.

---

## Authoritative scoreboard (T=60 H=7, lr=3e-4, workers=4, uncapped)

### A. 3-seed baselines — `agent-handoff/2026-09-11_1655_baseline-seed/report.md`

n=3 seeds 0/1/2. `last_val` = last logged epoch val_mse (early-stop epoch). `test` = `METRICS_ROW` = **best.pt**.

| model | last_val mean±std | best_val mean±std | test_mse mean±std | n_params |
|--------|-------------------|-------------------|-------------------|----------|
| **dlinear** | **0.7272±0.0035** | **0.7237±0.0019** | **0.6739±0.0013** | 854 |
| timexer_plain | 0.7679±0.0084 | 0.7413±0.0049 | 0.6887±0.0273 | 146695 |
| A proto=10 | 0.8857±0.0038 | 0.7726±0.0170 | 0.7342±0.0143 | 144199 |

DLinear also most stable. Plain test pulled by seed=2 (0.720 vs ~0.673 on 0/1).

### B. 3-seed shortlist — `agent-handoff/2026-09-11_1448_seed-shortlist/report.md`

Same HP. **Not re-run** in baseline-seed; compare as prior log.

| model | proto | last_val mean±std | test_mse mean±std | n_params |
|--------|------|-------------------|-------------------|----------|
| C0 | 5 | 0.8338±0.0473 | 0.7018±0.0067 | 200391 |
| B | 10 | 0.8816±0.0153 | 0.7244±0.0275 | 213063 |
| C1 | 5 | 0.9072±0.0639 | 0.7018±0.0176 | 200391 |

**Finding:** DLinear beats locked C0 on mean last_val **and** mean test. Prototypes + news **hurt** vs DLinear / plain on this protocol.

C1 `text_zero` Δmse on those three seeds: `+0.0029`, `+0.0005`, `−0.0093` ≈ **text unused**.

### C. Combined ranking (test_mse n=3, T=60 H=7)

DLinear 0.674 < plain 0.689 < C0 ≈ C1 0.702 < B 0.724 < A 0.734.

H1 (A vs B): B better than A on this HP, both lose to DLinear.  
H2 (fusion): C0 ≥ C1 on val; C1 does not win as cross-attn.  
H3: C1 text knockout is ~0 at core window.

---

## Other sweeps (do not fold into the n=3 table)

| folder | what | takeaway |
|--------|------|----------|
| `2026-09-11_1051_per-window-norm` | protocol confirm | DLinear test **0.673** (seed 42, **lr=0.001**, 10–20 ep) already in-band with later 3-seed |
| `2026-09-10_2150_dev-full-ablation` | A/B/C0/C1 × seeds 0/1/2 **lr=0.001 proto=10** | C1 best **test** among four; C0 best **val**; `ablation_table` n=4 contaminated |
| `2026-09-11_1135_hp-uncapped-dev` | A/B/C0/C1 × lr 3e-4 vs 1e-3, seed 42 | **0.001 loses val** on all four; val winner C0 @ 3e-4 (one seed) |
| `2026-09-11_1320_hp-width-nproto` | B/C0/C1 width pair + A proto 5/10 | **128/512 loses val**; A proto 5 loses; val winner B 64/256 (seed 42, shuffle≠shortlist) |
| `2026-09-11_1405_hp-nproto-bc0c1` | B/C0/C1 × proto 5/10/20 seed 42 | proto **20 never wins val**; C1 proto=5 val winner **this log**; C1 proto=10 worst |
| `2026-09-11_1540_c0c1-windows` | C0 vs C1 proto=5, seed 42, other (T,H) | C1 **not consistent**. Only **T=96 H=7** C1 wins val+test (tiny). H=30 **overfit** (best ep **1**, val explodes). C1 `text_zero` Δmse **+0.26** at T=60 H=14 (text used) but C0 still wins **test**. At T=36 H=7 zeroing C1 text **helps**. |
| `2026-09-10_2346_hp-sweep-cap` | 256/64/64 windows | **invalid HP** (val≠test scale/order vs uncapped) |

z-MSE is **not comparable across H** (H=14 ~1.4, H=30 ~2.8, H=7 T=96 ~0.40). `mae_denorm` on H=7 is ~5.4 for both T=36 and T=96 — T=96 “tiny mse” is mostly per-window z-scale.

---

## Metric protocol pitfalls (validate these)

1. **Test is `best.pt`**, but HP was often picked on **last-epoch `val_mse`** (the patience-expired epoch, usually worse than min val). See `src/training/train.py` (save on val improve; `load_state_dict(best.pt)` before test).
2. **A last_val ≫ best_val** (0.886 vs 0.773): early stop fires; ranking on last_val **penalizes A extra**. DLinear last≈best (stable). Re-rank on `best_val` if you audit HP.
3. `METRICS_ROW` has **horizon but not lookback** — parse `=== START` for T.
4. Per-window z-score: longer T → different mse magnitude. Prefer `mae_denorm` when comparing T.

---

## Known bottlenecks / bugs to inspect

### 1. Daily text cache ignores `decay_lambda` (blocks λ sweep)

`load_or_build_daily_series` in `src/data/text_series.py` keys `{ticker}.npz` only by dates + dim, **not** `lam` / `missing_policy`. Override `data.text.decay_lambda` **reuses E built at 0.03**. Window `pool_window` *would* use the new λ. A λ sweep today is invalid unless caches are deleted or the key is fixed.

### 2. Fusion may not use text (core result)

C1 `G_en` is **one** token attending `per_day` exo `[B,T,d]`. At T=60 H=7, `text_zero` Δ≈0. C0 is a single per-patch add of pooled-in-patch text. Both lose to a model with **no text**. Bottleneck may be: (a) news not predictive on this target (close, 7d, z-space); (b) too much missingness (`zero_windows` 0.57 train); (c) C1 capacity (1 query); (d) recency pool + decay washing signal.

### 3. Overfit at long H

H=30: best val at **epoch 1**, stop at 6; train_loss falls, val +1.8 (C0) / **+4.3** (C1). Test = ep1 checkpoint. Do not interpret H=30 mse as a trained TimeXer.

### 4. Compute / GPU (not a train-on-CPU bug)

- Model ~150–200k params (DLinear 854). Batch `[32, 60, 5]` + `text_seq` `[32, 60, 768]` ≈ **6 MiB**. L4 VRAM ~300 MiB (CUDA context). Util ~5–10% is **expected**.
- Encoder is **not** on GPU at train time (precache → disk).
- Coverage used to call `__getitem__` on all ~28k windows (RevIN + pool). Fixed in `split_text_coverage` (`src/data/dataset.py`). Log format unchanged.
- DataLoader: `make_forecast_loader` in `src/data/collate.py`; default `num_workers: 4`. Workers must stay CPU.

### 5. `ablation_table` mixing

Filter by `max_train_windows`, seed set, lr, proto, `num_workers` — or do not use MLflow sqlite across sweeps.

### 6. DLinear still runs full text pipeline

Fair for “same dataset”, wasteful for wall time. Not a correctness bug.

### 7. HP / shuffle

`num_workers>0` changes shuffle vs older seed=42 runs. Never splice `hp-uncapped-dev` numbers into `seed-shortlist`.

---

## What was **not** done

- `ticker_set=paper`
- Honest `decay_lambda` / `window_agg` sweep
- 3-seed at T=96 H=7 (C1’s only dual win, likely noise)
- Dropout / loss λ_c,e,d / C1 `exo_tokens=per_patch`
- Re-ranking HP on `best_val` instead of last epoch
- Fixing cache key
- Purging `agent-handoff/` from git history (pre-paper only)

---

## Suggested validation checks (read-only first)

- [ ] `FNSPIDForecastDataset` per-window stats: x lookback mean~0 std~1; `y` denorms with `y_mean`/`y_std` (`tests/test_per_window_norm.py`)
- [ ] C1 forward: `exo` shape `[B,T,d]` or per_patch; query is `[B,1,d]` (`src/models/timexer_c1.py`, `tests/test_timexer_c1.py`)
- [ ] Cache path for daily series: confirm λ not in filename
- [ ] `evaluate()` uses same collate/device as train; test after `best.pt`
- [ ] Recompute Table 2 means from per-seed rows in the two reports (sample std n=3)
- [ ] Confirm DLinear does not consume `text` in `forward`

---

## Smallest next steps (recommend, don’t run unless asked)

1. **Code:** key daily `.npz` by `lam` + `missing_policy` (+ maybe `train_start`/`mu`). Then λ ∈ {0.01, 0.03, 0.1} on **C0 and DLinear** at T=60 H=7 (does news decay matter, and vs linear).
2. **Eval:** rank models on **mean best_val** and **mae_denorm**, not only last_val z-MSE.
3. **Science:** if text still unused after (1), treat H2 as a **negative result on this target/protocol**, not “need more width”.
4. **Do not** go to paper tickers until something beats DLinear on dev at H=7.

Primary sources to read: `configs/README.md`, `configs/data/fnspid.yaml`, `src/data/text_series.py`, `src/training/train.py`, and the two reports named in the scoreboard.
