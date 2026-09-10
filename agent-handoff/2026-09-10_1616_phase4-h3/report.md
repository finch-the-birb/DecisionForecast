# Report: phase4-h3

- authored_by: runner
- created_at: 2026-09-10T16:37:11Z
- request_folder: agent-handoff/2026-09-10_1616_phase4-h3/
- tested_ref: feat/phase1-data@10934ff41d0296a85e2ba4cb059805b5a9089a2c
- status: pass

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HEAD: 10934ff (contains 475d674). HF_TOKEN unset. No C0. No ticker_set=paper. No FNSPID download.
# uv rebuilt decision-forecast then:
uv run python -m src.training.train --cfg job >/dev/null   # exit 0
# then train a,b,c1 (64/32/32, epochs=1, cuda) and src.explain.run on each new best.pt
```

## Outcome
- exit_code: --cfg job=0; train a=0 b=0 c1=0; explain.run a=0 b=0 c1=0
- duration: trains+explain ~13m 25s (804559 ms); --cfg job ~2m (torch import)
- host: runpod pod `i6lv3n0222en43`
- device: cuda:0, NVIDIA L4. No CPU fallback. No OOM. No HF 401. Not C0. Not paper.

CKPT (newest metrics.json per model):
- A `/workspace/DecisionForecast/outputs/2026-09-10/16-23-34/checkpoints/best.pt`
- B `/workspace/DecisionForecast/outputs/2026-09-10/16-26-02/checkpoints/best.pt`
- C1 `/workspace/DecisionForecast/outputs/2026-09-10/16-28-24/checkpoints/best.pt`

## Key signals
Train-run H3_ROW (verbatim; table source is these Hydra dirs, not explain.run):
```
H3_ROW model=a variant=full mse=18.1757 mae=3.2777 delta_mse=n/a delta_mae=n/a
H3_ROW model=a variant=proto_zero mse=18.2539 mae=3.2878 delta_mse=+0.0781 delta_mae=+0.0101
H3_ROW model=a variant=proto_shuffle mse=18.1758 mae=3.2777 delta_mse=+0.0000 delta_mae=+0.0000
H3_ROW model=a variant=text_zero mse=18.2219 mae=3.2833 delta_mse=+0.0462 delta_mae=+0.0056
H3_ROW model=a variant=text_shuffle mse=18.1757 mae=3.2777 delta_mse=-0.0000 delta_mae=+0.0000
H3_ROW model=b variant=full mse=16.3450 mae=3.0208 delta_mse=n/a delta_mae=n/a
H3_ROW model=b variant=proto_zero mse=16.3357 mae=3.0185 delta_mse=-0.0093 delta_mae=-0.0023
H3_ROW model=b variant=proto_shuffle mse=16.3449 mae=3.0208 delta_mse=-0.0001 delta_mae=-0.0000
H3_ROW model=b variant=text_zero mse=16.4294 mae=3.0324 delta_mse=+0.0843 delta_mae=+0.0115
H3_ROW model=b variant=text_shuffle mse=16.3447 mae=3.0208 delta_mse=-0.0004 delta_mae=-0.0000
H3_ROW model=c1 variant=full mse=16.9914 mae=3.1174 delta_mse=n/a delta_mae=n/a
H3_ROW model=c1 variant=proto_zero mse=16.9830 mae=3.1142 delta_mse=-0.0084 delta_mae=-0.0032
H3_ROW model=c1 variant=proto_shuffle mse=16.9915 mae=3.1174 delta_mse=+0.0001 delta_mae=+0.0000
H3_ROW model=c1 variant=text_zero mse=16.9919 mae=3.1175 delta_mse=+0.0005 delta_mae=+0.0001
H3_ROW model=c1 variant=text_shuffle mse=16.9914 mae=3.1174 delta_mse=+0.0000 delta_mae=+0.0000
```

`format_h3_table` on train faithfulness JSON (`CKPT` parent `/explain/`):

| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |
|-------|-----|-----------------|--------------------|----------------|-------------------|
| A | 18.1757 | +0.0781 | +0.0000 | +0.0462 | -0.0000 |
| B | 16.3450 | -0.0093 | -0.0001 | +0.0843 | -0.0004 |
| C1 | 16.9914 | -0.0084 | +0.0001 | +0.0005 | +0.0000 |

JSON paths:
- `outputs/2026-09-10/16-23-34/explain/faithfulness_a.json`
- `outputs/2026-09-10/16-26-02/explain/faithfulness_b.json`
- `outputs/2026-09-10/16-28-24/explain/faithfulness_c1.json`

Projection nearest segments (train-run, 3 per model):
- A: AMZN 2021-12-21 dist=0.357; AMZN 2021-12-17 dist=0.319; AMZN 2021-12-21 dist=0.342
- B: BAC 2020-06-18 dist=27.628 / 28.313 / 27.972 (all same nearest index)
- C1: BAC 2020-06-18 dist=27.548 / 28.234 / 27.893 (all same nearest index)

- oom: no
- traceback_summary: n/a

## Artifacts (paths on Runner disk — do not commit binaries)
- A train: `outputs/2026-09-10/16-23-34/` (best.pt, metrics.json, explain/projection_examples_a.json, explain/faithfulness_a.json)
- B train: `outputs/2026-09-10/16-26-02/`
- C1 train: `outputs/2026-09-10/16-28-24/`
- explain.run (CLI confirm only): `outputs/2026-09-10/16-30-43/` (A), `16-32-48/` (B), `16-34-46/` (C1)

## Conclusions for Dev
1. Prototypes **used** (proto_zero Δmse > 0): **A yes** (+0.0781). **B no** (−0.0093). **C1 no** (−0.0084). proto_shuffle is ~0 on all three.
2. Text **used** (text_zero or text_shuffle Δmse > 0): **A yes** (text_zero +0.0462). **B yes** (text_zero +0.0843). **C1** text_zero +0.0005 (positive but near-zero on 1 epoch / 64 windows). Shuffle text Δ ~0 / tiny negative.
3. **H3 preservation (smoke):** proto_zero signs are **not** the same (A +, B −, C1 −). Opposite sign on B/C1 vs A → **preservation not supported on this slice**.
4. Negative proto Δ on B/C1 is valid (unfaithful / unused on this cap), not a failed job. Not paper H3.
5. Standalone `src.explain.run` matched the train-run H3_ROW numbers (same ckpt).

## Suggested next command (optional)
```bash
# only if Dev wants a less noisy H3 (still not paper); do not run C0
# raise epochs / windows in a new request.md
```
