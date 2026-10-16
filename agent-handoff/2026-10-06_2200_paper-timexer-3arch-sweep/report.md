# Report: paper-timexer-3arch-sweep

- authored_by: runner
- created_at: 2026-10-08T11:58:27Z
- updated_at: 2026-10-08T11:58:27Z
- request_folder: agent-handoff/2026-10-06_2200_paper-timexer-3arch-sweep/
- tested_ref: feat/phase2-timexer@cd1f26edcc45f45d758914d1641ef4ff44caa887
- status: blocked

## Commands executed
```bash
# cwd: /workspace/DecisionForecast
# HF_TOKEN unset. git fetch && git pull --ff-only origin feat/phase2-timexer.
# Sweep was not started. uv sync was not run. No train process was launched.
```

## Outcome
- exit_code: n/a; 0/36 jobs; no log
- host: runpod pod 9hj7blo0fqus3p (hostname 279d25bcf0b8)
- device: n/a
- pid: n/a
- log: n/a
- c1_dual: not running on this host after the restart. It was not started and not killed.

Pulled `cd1f26e` (`feat: implement 4 multimodal ts architectures and selective revin`). Configs `configs/model/c1_factored.yaml`, `c1_inverted.yaml`, and `c1_late_fusion.yaml` are present. `build_model` registers all three names. Each config sets `n_features: 5`, `n_ts_features: 25`, `text_dim: 15`, and `d_ff: null`.

The requested `BASE` uses `data=fnspid_selective`. That config inherits `features_mode: ohlcv_compact_text` from `fnspid_compact`. `build_datasets` attaches the 25 exogenous indicators only when `features_mode == dual_ts` (`src/data/dataset.py`, `_attach_exogenous_ts`). On the first train batch, `train.py` rejects these three models unless `ts` is `[B, 60, 25]`:

```text
RuntimeError: c1_factored batch x=... text_seq=... ts=None; expected x [B, 60, 5], text 15, ts [B, 60, 25]
```

The model forwards raise the same gap (`c1_factored requires ts [B, 60, 25], got None`, and the same for `c1_inverted` and `c1_late_fusion`). `set -euo pipefail` would then stop the remaining 35 jobs. No product code was edited.

- oom: n/a
- traceback_summary: not run; the first-batch check in `src/training/train.py` raises before a backward pass when `ts` is missing

## Artifacts (paths on Runner disk — do not commit binaries)
- train_log: n/a

## Conclusions for Dev
1. Do not rerun this script with `data=fnspid_selective`. The three architectures read 25 indicators from `batch["ts"]`, and selective/compact data does not build that tensor.
2. The data config that keeps 5 OHLCV and attaches the paper fold-1 indicators is `data=fnspid_dual_ts` (`features` default `[close, volume, open, high, low]`, `signature_path` paper fold-1, `n_ts_features` stays 25). `patch_stride` stays 6 via `fnspid_selective`.
3. `c1_dual` is not running on pod `9hj7blo0fqus3p`. After a restart it needs its own launch if that sweep is still required.
