# Report: head-cache-sweep

- authored_by: runner
- created_at: 2026-09-11T22:20:39Z
- request_folder: agent-handoff/2026-09-11_2100_head-cache-sweep/
- tested_ref: feat/phase2-timexer@c839e4512e337aa44ba0ffa887e9357f6bb09eff (contains 7f61ab9 FlattenHead + text cache key; harvest HEAD 5120f56 is still-running report, same code)
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. No FNSPID download. Not paper. No caps. Not seed=42 / extra T/H / ablation_table.
# Sweep already running from prior nohup (PID 187648). Human ping: PID dead, log has SWEEP_DONE. Harvest only — no new train.
```

## Outcome
- exit_code: 72/72 jobs DONE; log ends with `SWEEP_DONE`; nohup PID 187648 gone; no Traceback
- duration: ~4h 5m 38s (18:04:54Z → 22:10:32Z)
- host: runpod pod `fprbi35eab7xe4` (container `c658b923edf4`)
- device: `train device: cuda`; `First batch tensors on x=cuda:0 text=cuda:0` (all 72). NVIDIA L4. No OOM. Not paper.
- Uncapped: train 23478 / val 2510 / test 2480. seeds 0,1,2 (not 42). T=60 H=7, epochs=20, patience=5. yaml `d_model=64` `d_ff=256` (not overridden). `num_workers=4`.
- Default lr=0.0003 except Block 1 also lr=0.001.

GPU / loader (verbatim, first job dlinear lr=0.0003 seed=0):
```
train device: cuda
First batch tensors on x=cuda:0 text=cuda:0
DataLoader num_workers=4 pin_memory=True persistent_workers=True
```

TEXT_COVERAGE split (verbatim, first job; same window counts on T=60 H=7 jobs):
```
TEXT_COVERAGE split=train windows=23478 mean_has_news_frac=0.292 zero_windows=0.565 mean_text_l2=0.1457
TEXT_COVERAGE split=val windows=2510 mean_has_news_frac=0.545 zero_windows=0.415 mean_text_l2=0.2164
TEXT_COVERAGE split=test windows=2480 mean_has_news_frac=0.763 zero_windows=0.221 mean_text_l2=0.3055
```

Daily cache key fix: **yes**. 15/15 tickers have both files under `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/`:
- `*_decay_lam0.01.npz` (15)
- `*_decay_lam0.1.npz` (15; λ=0.10)
- prior `*_decay_lam0.03.npz` still present (not overwritten)

`best_val` = min Epoch `val_mse`; `last_val` = last Epoch `val_mse`; test = `METRICS_ROW` (best.pt). Means n=3, sample std. This sweep only.

### Table 1 — Block 1 DLinear lr

| config | last_val mean±std | best_val mean±std | test_mse mean±std |
| dlinear lr=0.0003 | 0.7272±0.0035 | 0.7237±0.0019 | 0.6739±0.0013 |
| dlinear lr=0.001 | 0.7323±0.0041 | 0.7208±0.0032 | 0.6786±0.0085 |

lr=0.0003 matches `baseline-seed` DLinear (last 0.7272 / test 0.6739). lr=0.001 does not help test (noisier).

### Table 2 — Block 2 plain FlattenHead

| config | last_val mean±std | best_val mean±std | test_mse mean±std |
| plain linear drop=0.0 | 0.8077±0.0361 | 0.7555±0.0078 | 0.7438±0.0382 |
| plain linear drop=0.1 | 0.7914±0.0285 | 0.7519±0.0064 | 0.7293±0.0267 |
| **plain linear drop=0.2** | **0.7884±0.0318** | **0.7485±0.0097** | **0.7241±0.0212** |
| plain mlp drop=0.0 | 0.8860±0.0136 | 0.7728±0.0218 | 0.7141±0.0082 |
| plain mlp drop=0.1 | 0.8571±0.0581 | 0.7772±0.0177 | 0.7276±0.0109 |
| plain mlp drop=0.2 | 0.8424±0.0560 | 0.7668±0.0099 | 0.7169±0.0112 |

Among heads, **linear drop=0.2** wins mean last_val. MLP has slightly lower test at drop=0.0 (0.7141) but worse last_val.

### Table 3 — Block 3 FlattenHead linear (plain / C0 / C1 / B)

| config | last_val mean±std | best_val mean±std | test_mse mean±std |
| timexer_plain drop=0.1 | 0.7914±0.0285 | 0.7519±0.0064 | 0.7293±0.0267 |
| timexer_plain drop=0.2 | 0.7884±0.0318 | 0.7485±0.0097 | 0.7241±0.0212 |
| c0 proto=5 drop=0.1 | 0.8318±0.0854 | 0.7562±0.0103 | 0.7000±0.0127 |
| c0 proto=5 drop=0.2 | 0.8799±0.0547 | 0.7504±0.0067 | 0.7105±0.0364 |
| c1 proto=5 drop=0.1 | 0.9983±0.1231 | 0.7732±0.0050 | 0.6918±0.0173 |
| c1 proto=5 drop=0.2 | 0.9800±0.0838 | 0.7637±0.0133 | 0.6955±0.0308 |
| b proto=10 drop=0.1 | 0.8020±0.0103 | 0.7630±0.0113 | 0.7191±0.0180 |
| b proto=10 drop=0.2 | 0.8635±0.0972 | 0.7687±0.0067 | 0.7183±0.0323 |

Plain drop 0.1/0.2 repeats Block 2 linear (same seeds; same means). Lowest test here is C1 drop=0.1 (0.6918) but last_val is worst (~1.00, overfit). Best last_val among these is plain drop=0.2 / B drop=0.1.

### Table 4 — Block 4 decay λ (linear drop=0.1)

| config | last_val mean±std | best_val mean±std | test_mse mean±std |
| c0 proto=5 lam=0.01 | 0.8893±0.0598 | 0.7558±0.0110 | 0.7148±0.0383 |
| c0 proto=5 lam=0.10 | 0.8270±0.0759 | 0.7557±0.0103 | 0.6994±0.0131 |
| c1 proto=5 lam=0.01 | 0.9118±0.0457 | 0.7762±0.0061 | 0.6969±0.0252 |
| c1 proto=5 lam=0.10 | 0.9842±0.0578 | 0.7591±0.0136 | 0.7019±0.0229 |
| b proto=10 lam=0.01 | 0.8275±0.0117 | 0.7760±0.0128 | 0.6982±0.0084 |
| b proto=10 lam=0.10 | 0.7808±0.0126 | 0.7561±0.0158 | 0.7239±0.0179 |
| a proto=10 lam=0.01 | 0.9124±0.0988 | 0.8071±0.0289 | 0.6957±0.0182 |
| a proto=10 lam=0.10 | 0.8951±0.1024 | 0.7911±0.0181 | 0.7310±0.0653 |

### Answers
1. **FlattenHead vs mean pooling — no, it does not beat it on plain.** Prior `baseline-seed` mean-pool plain: last 0.7679±0.0084 / test 0.6887±0.0273. Best FlattenHead plain here (linear drop=0.2): last 0.7884 / test 0.7241 — both worse. C0 drop=0.1 is a wash vs old C0 (last 0.8318 vs 0.8338, test 0.7000 vs 0.7018).
2. **Does any FlattenHead model beat DLinear? No.** DLinear lr=0.0003: last 0.7272 / best 0.7237 / test 0.6739. Best FlattenHead last_val is B lam=0.10 at 0.7808; best test is C1 drop=0.1 at 0.6918. Gap remains.
3. **Does text decay matter? Yes, now that cache is keyed.** λ=0.01 vs 0.10 moves last_val and test, but the sign is model-dependent (C0 last better at 0.10; C1 last better at 0.01; B last better at 0.10 but test better at 0.01). Not a uniform λ winner.

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: `outputs/head-cache-sweep.log`
- daily series: `Data/FNSPID/cache/text_series/FinLang__finance-embeddings-investopedia/*_decay_lam0.01.npz` and `*_decay_lam0.1.npz`

## Conclusions for Dev
1. 72-job FlattenHead + cache-key sweep **pass**: 72/72 CUDA, `num_workers=4`, four block tables. FlattenHead does not close the DLinear gap and does not beat old mean-pool plain.
2. linear head > mlp on last_val; extra dropout 0.2 slightly helps linear last_val. C1 FlattenHead overfits (last_val ~1.0). Honest λ sweep works (separate npz files).
3. Did not run paper / seed=42 / extra T/H / caps / ablation_table. No new train. Did not commit `outputs/` or `mlflow.db`.

## Suggested next command (optional)
```bash
# no extra jobs from this request
```
