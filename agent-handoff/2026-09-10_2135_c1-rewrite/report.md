# Report: c1-rewrite

- authored_by: runner
- created_at: 2026-09-10T21:44:10Z
- request_folder: agent-handoff/2026-09-10_2135_c1-rewrite/
- tested_ref: feat/phase2-timexer@72df2fc17b1afe65f329bb60dd850133f6203731 (contains 2c733ae)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not a/dlinear/timexer_plain. Not paper.
# mu_dev_2015-01-01_2021-12-31.npy present — precache skipped.
git fetch origin && git checkout feat/phase2-timexer && git pull --ff-only origin feat/phase2-timexer
uv sync
SHARED="train.device=cuda train.ticker_set=dev train.max_train_windows=64 train.max_val_windows=32 train.max_test_windows=32 train.epochs=10 data.horizon=7"
for m in b c0 c1; do uv run python -m src.training.train model=$m $SHARED; done
```

## Outcome
- exit_code: uv sync=0; b=0; c0=0; c1=0
- duration: three trains ~3m 50s (230382 ms)
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4. No OOM. Not paper. Caps 64/32/32. Did not re-run protocol-norm models.

C1 fusion (`outputs/2026-09-10/21-42-53/config_resolved.yaml` + MLflow run `29981d5c261248c09560cf6f959b80c3`):
```
fusion.kind=mid_cross_attn
fusion.text_at_head=False
fusion.text_to_patches=False
fusion.text_as_exogenous=True
fusion.exo_tokens=per_day
```

n_params_backbone_g12 **equal** (verbatim):
```
n_params_backbone_g12=142272 (backbone=137472 g12=4800)   # b, c0, and c1
```

proto kmeans++ (verbatim):
```
proto kmeans++ init bank=(576, 64) nn_dist_mean=0.0013 min_pairwise=3.9096   # b
proto kmeans++ init bank=(576, 64) nn_dist_mean=0.0014 min_pairwise=3.5587   # c0
proto kmeans++ init bank=(576, 64) nn_dist_mean=0.0014 min_pairwise=3.5587   # c1
```

## Key signals
TEXT_COVERAGE (same on all three; record only, not a fail bar):
```
TEXT_COVERAGE split=train windows=64 mean_has_news_frac=0.517 zero_windows=0.422 mean_text_l2=0.1917
TEXT_COVERAGE split=val windows=32 mean_has_news_frac=0.709 zero_windows=0.281 mean_text_l2=0.2713
TEXT_COVERAGE split=test windows=32 mean_has_news_frac=0.803 zero_windows=0.188 mean_text_l2=0.3268
```

METRICS_ROW (verbatim):
```
METRICS_ROW model=b horizon=7 mse=1.4408 mae=0.9071 mae_denorm=10.0756
METRICS_ROW model=c0 horizon=7 mse=0.9819 mae=0.7846 mae_denorm=8.0804
METRICS_ROW model=c1 horizon=7 mse=0.9552 mae=0.7857 mae_denorm=8.5489
```

Last-epoch val (verbatim Epoch 10):
```
b  val_mse=0.8346  train_l_pred=0.6610 proto_nn_dist_mean=0.0978 proto_min_pairwise_dist=3.7918
c0 val_mse=0.9042  train_l_pred=0.4925 proto_nn_dist_mean=0.0988 proto_min_pairwise_dist=3.5063
c1 val_mse=0.7363  train_l_pred=0.5148 proto_nn_dist_mean=0.0978 proto_min_pairwise_dist=3.5188
```

n_params: b=213063; c0=200711; c1=200711. n_params_backbone_g12=142272 all three.

Table from the three `metrics.json` + logs. Δ = later − earlier:

| model | val_mse (last epoch) | test_mse | test_mae | test_mae_denorm | n_params | n_params_backbone_g12 |
| b | 0.8346 | 1.4408 | 0.9071 | 10.0756 | 213063 | 142272 |
| c0 | 0.9042 | 0.9819 | 0.7846 | 8.0804 | 200711 | 142272 |
| c1 | 0.7363 | 0.9552 | 0.7857 | 8.5489 | 200711 | 142272 |
| Δ(B→C0) mse | −0.4590 |  |  |  |  |  |
| Δ(C0→C1) mse | −0.0267 |  |  |  |  |  |

json: B mse=1.4408439 mae=0.9071414 mae_denorm=10.0755949; C0 mse=0.9818578 mae=0.7845932 mae_denorm=8.0803976; C1 mse=0.9551526 mae=0.7856601 mae_denorm=8.5489426.
Δmse B→C0=−0.4589862; C0→C1=−0.0267051.

Val vs test same order for each (B 0.83 vs 1.44; C0 0.90 vs 0.98; C1 0.74 vs 0.96). Not ~0.4 vs ~5.

n_distinct_nearest=10 for b, c0, c1. Projection JSON:
- `outputs/2026-09-10/21-40-16/explain/projection_examples_b.json`
- `outputs/2026-09-10/21-41-48/explain/projection_examples_c0.json`
- `outputs/2026-09-10/21-42-53/explain/projection_examples_c1.json`

H3_ROW text_zero (verbatim; train-run faithfulness JSON, not explain.run):
```
H3_ROW model=b variant=text_zero mse=1.4542 mae=0.9109 delta_mse=+0.0134 delta_mae=+0.0038
H3_ROW model=c0 variant=text_zero mse=0.9117 mae=0.7704 delta_mse=-0.0701 delta_mae=-0.0142
H3_ROW model=c1 variant=text_zero mse=0.9560 mae=0.7864 delta_mse=+0.0008 delta_mae=+0.0007
```
C1 text_zero Δmse = +0.0008 **≠ 0.0000**.

| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |
| b | 1.4408 | −0.9316 | +1.1997 | +0.0134 | +0.0034 |
| c0 | 0.9819 | −0.2330 | +0.9253 | −0.0701 | +0.0894 |
| c1 | 0.9552 | −0.3763 | +1.1321 | +0.0008 | +0.0024 |

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- B: `outputs/2026-09-10/21-40-16/` (`metrics.json`, `explain/projection_examples_b.json`, `explain/faithfulness_b.json`)
- C0: `outputs/2026-09-10/21-41-48/`
- C1: `outputs/2026-09-10/21-42-53/`
- mlflow_run_id: B `d7e981a4b49e452380073ac9da00fac6`; C0 `f65235700e7842b2861da32f71d3f6e1`; C1 `29981d5c261248c09560cf6f959b80c3`

## Conclusions for Dev
1. Rewritten C1 smoke **pass**: fusion flags match; backbone+g12 parity 142272; proto kmeans++ logged; C1 text_zero Δmse +0.0008 (not identically zero). Magnitude is small on 64 windows, as the request allowed.
2. Δmse B→C0 = −0.46, C0→C1 = −0.027 (C0/C1 better than B on this cap). Valid negatives.
3. Did not re-run a/dlinear/timexer_plain. Did not chase zero_windows.

## Suggested next command (optional)
```bash
# only if Dev wants uncapped H2 — new request.md
```
