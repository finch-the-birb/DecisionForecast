# Report: dev-full-ablation

- authored_by: runner
- created_at: 2026-09-10T23:00:50Z
- request_folder: agent-handoff/2026-09-10_2150_dev-full-ablation/
- tested_ref: feat/phase2-timexer@10f83dad68d356ec846e4f8fe2f08262625db2e5 (contains 2c733ae)
- status: partial

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not dlinear/timexer_plain. Not paper. Uncapped.
# mu_dev_2015-01-01_2021-12-31.npy present — precache skipped.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
uv sync
SHARED="train.device=cuda train.ticker_set=dev train.epochs=20 data.horizon=7 train.max_train_windows=null train.max_val_windows=null train.max_test_windows=null"
for m in a b c0 c1; do
  for s in 0 1 2; do
    uv run python -m src.training.train model=$m train.seed=$s $SHARED
  done
done
uv run python -m src.evaluation.ablation_table \
  --tracking-uri sqlite:///mlflow.db \
  --experiment decision-forecast \
  --out-dir outputs/tables \
  --ticker-set dev \
  --lookback 60 \
  --normalize per_window \
  --train-start 2015-01-01
```

## Outcome
- exit_code: 12/12 trains=0; ablation_table=0
- duration: grid ~59m 16s (3556303 ms); ablation_table ~41s
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4. No OOM. Uncapped train/val/test=23478/2510/2480. Early stopping (patience=5), not full 20 epochs. Did not run dlinear/timexer_plain/paper.

`partial` because `ablation_table` reports **n=4**, not n=3: it also keeps seed=42 capped smokes from `c1-rewrite` (latest-per-(model,H,seed) still includes that extra seed). 12/12 uncapped jobs are green; C1 text_zero mean Δmse from the table is −0.0031 ≠ 0.

## Key signals
TEXT_COVERAGE (uncapped, same on all jobs):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

12-row scratch from logs / `metrics.json` (seeds 0/1/2 only; last-epoch val_mse from train.log):

| model | seed | val_mse | test_mse | test_mae | mae_denorm |
| a | 0 | 1.0751 | 0.8772 | 0.6830 | 6.4417 |
| a | 1 | 0.8643 | 0.7106 | 0.5963 | 5.3928 |
| a | 2 | 0.9391 | 0.7395 | 0.6154 | 5.7487 |
| b | 0 | 0.9016 | 0.7113 | 0.6046 | 5.4507 |
| b | 1 | 0.9751 | 0.7778 | 0.6435 | 5.8306 |
| b | 2 | 0.9197 | 0.7225 | 0.6035 | 5.5332 |
| c0 | 0 | 0.8362 | 0.7325 | 0.6165 | 5.7779 |
| c0 | 1 | 0.8144 | 0.7367 | 0.6114 | 5.7284 |
| c0 | 2 | 0.7926 | 0.7257 | 0.6130 | 5.7116 |
| c1 | 0 | 1.0847 | 0.6998 | 0.5915 | 5.4805 |
| c1 | 1 | 0.9395 | 0.7257 | 0.6110 | 5.6198 |
| c1 | 2 | 0.8125 | 0.6846 | 0.5861 | 5.3709 |

Val vs test same order on every job (no ~0.4 vs ~5). METRICS_ROW matches json to 4 decimals.

C1 H3_ROW text_zero (verbatim, train-run; seeds 0/1/2):
```
H3_ROW model=c1 variant=text_zero mse=0.6970 mae=0.5900 delta_mse=-0.0028 delta_mae=-0.0014
H3_ROW model=c1 variant=text_zero mse=0.7104 mae=0.6015 delta_mse=-0.0153 delta_mae=-0.0095
H3_ROW model=c1 variant=text_zero mse=0.6894 mae=0.5893 delta_mse=+0.0048 delta_mae=+0.0031
```

`outputs/tables/ablation_a_b_c0_c1.md` verbatim (do not commit this file):

```
# Ablation A / B / C0 / C1 (mean ± std over seeds)

## H=7

| model | mse | mae | mae_denorm |
|-------|-----|-----|------------|
| A | 0.8277 ± 0.1268 (n=4) | 0.6540 ± 0.0584 (n=4) | 6.2023 ± 0.8096 (n=4) |
| B | 0.9131 ± 0.3530 (n=4) | 0.6897 ± 0.1462 (n=4) | 6.7225 ± 2.2413 (n=4) |
| C0 | 0.7942 ± 0.1252 (n=4) | 0.6564 ± 0.0855 (n=4) | 6.3246 ± 1.1709 (n=4) |
| C1 | 0.7663 ± 0.1270 (n=4) | 0.6436 ± 0.0953 (n=4) | 6.2550 ± 1.5327 (n=4) |
| Δ(A→B) | +0.0854 ± 0.2660 (n=4) | +0.0356 ± 0.1124 (n=4) | |
| Δ(B→C0) | -0.1189 ± 0.2282 (n=4) | -0.0333 ± 0.0628 (n=4) | |
| Δ(C0→C1) | -0.0279 ± 0.0127 (n=4) | -0.0128 ± 0.0152 (n=4) | |

# H3 faithfulness (mean ± std over seeds)

| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |
|-------|-----|-----------------|--------------------|----------------|-------------------|
| A | 0.8277 ± 0.1268 (n=4) | +0.3284 ± 0.1593 (n=4) | +0.3557 ± 0.1947 (n=4) | -0.0386 ± 0.1306 (n=4) | +0.0073 ± 0.0060 (n=4) |
| B | 0.9131 ± 0.3530 (n=4) | +0.1907 ± 0.8479 (n=4) | +0.2994 ± 0.6009 (n=4) | +0.0356 ± 0.0224 (n=4) | +0.0033 ± 0.0015 (n=4) |
| C0 | 0.7942 ± 0.1252 (n=4) | +0.1430 ± 0.2857 (n=4) | +0.2262 ± 0.4662 (n=4) | -0.0263 ± 0.0300 (n=4) | +0.0268 ± 0.0418 (n=4) |
| C1 | 0.7663 ± 0.1270 (n=4) | -0.0255 ± 0.2382 (n=4) | +0.2748 ± 0.5716 (n=4) | -0.0031 ± 0.0087 (n=4) | +0.0033 ± 0.0028 (n=4) |
```

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- A s0/s1/s2: `outputs/2026-09-10/21-57-04/` `22-03-13/` `22-07-53/`
- B s0/s1/s2: `22-12-17/` `22-16-39/` `22-20-46/`
- C0 s0/s1/s2: `22-25-16/` `22-31-19/` `22-35-37/`
- C1 s0/s1/s2: `22-41-16/` `22-45-48/` `22-50-37/`
- ablation md: `outputs/tables/ablation_a_b_c0_c1.md` (not committed)

## Conclusions for Dev
1. Uncapped 12/12 CUDA **green**; val/test same order. Last job on this ladder — no paper run started.
2. `ablation_table` **n=4** mixes seed=42 capped `c1-rewrite` into mean±std. Pass wanted n=3 (seeds 0/1/2 only). Filter seed=42 in the script, or drop those MLflow runs, before treating the pasted table as H2.
3. C1 text_zero mean Δmse from the n=4 table is **−0.0031 ≠ 0**. Seeds 0/1/2 alone: −0.0028 / −0.0153 / +0.0048.
4. Did not run dlinear / timexer_plain / paper. Did not commit `outputs/` or the table markdown.

## Suggested next command (optional)
```bash
# Dev-side: exclude train.seed=42 from ablation_table aggregation
```
