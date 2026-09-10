# Request: phase4-h3

- authored_by: dev
- created_at: 2026-09-10T16:16:53Z
- target_ref: feat/phase1-data@475d674de92899ac75bec124d0cc38d8cce713e1
- phase: 4
- priority: normal

## Goal
Capped **H3** smoke for **A / B / C1** (not C0): projection examples + faithfulness knockouts on the same protocol as the last H2 smoke (dev tickers, H=7, 1 epoch, 64/32/32 windows, `train.device=cuda`).

Fresh checkpoints are required (new eval-time `proto_mode` / `text_mode` on the models). `src.training.train` writes `explain/` after test. Then re-run `python -m src.explain.run` on each new `best.pt` to confirm the standalone CLI.

Success: all trains and explain CLIs exit 0; no OOM; CUDA; each model has `projection_examples_{m}.json` and `faithfulness_{m}.json`; `H3_ROW` in logs; H3 table from **train-run** faithfulness JSON via `format_h3_table`; a short H3 verdict in Conclusions. Smoke only — not a paper H3 / `ticker_set=paper`.

## Commands
FNSPID, news parquet, encoder npy, `.venv` are on volume `/workspace` — **do not re-download**, **no HF_TOKEN**, **not** `ticker_set=paper`. `uv sync` only if import/CUDA breaks.

cwd: `/workspace/DecisionForecast`. Deploy key must be in `~/.ssh` before the final `git push` of `report.md` (volume does not store `~/.ssh`).

```bash
git fetch origin
git checkout feat/phase1-data
git pull --ff-only origin feat/phase1-data
git rev-parse HEAD   # must contain 475d674

uv run python -m src.training.train --cfg job >/dev/null

declare -A CKPT
for m in a b c1; do
  uv run python -m src.training.train \
    model=$m \
    train.device=cuda \
    train.ticker_set=dev \
    train.max_train_windows=64 \
    train.max_val_windows=32 \
    train.max_test_windows=32 \
    train.epochs=1
  CKPT[$m]=$(uv run python - <<PY
from pathlib import Path
import json
cands = []
for p in Path("outputs").glob("*/**/metrics.json"):
    d = json.loads(p.read_text())
    if d.get("model") == "$m":
        ckpt = p.parent / "checkpoints" / "best.pt"
        if ckpt.is_file():
            cands.append((p.stat().st_mtime, ckpt.resolve()))
print(str(max(cands)[1]) if cands else "")
PY
)
  echo "CKPT[$m]=${CKPT[$m]}"
done

for m in a b c1; do
  uv run python -m src.explain.run \
    model=$m \
    explain.checkpoint="${CKPT[$m]}" \
    train.device=cuda \
    train.ticker_set=dev \
    train.max_train_windows=64 \
    train.max_val_windows=32 \
    train.max_test_windows=32
done
```

Build the table from the **train** Hydra dirs (the three `CKPT[*]` parents), not the later explain.run dirs:

```python
import json
from pathlib import Path
from src.explain.faithfulness import format_h3_table
reports = {}
for m, ckpt in [("a", CKPT_A), ("b", CKPT_B), ("c1", CKPT_C1)]:
    reports[m] = json.loads((Path(ckpt).parent.parent / "explain" / f"faithfulness_{m}.json").read_text())
print(format_h3_table(reports))
```

Expected table shape (fill only from JSON; Δ = ablated − full, so **+** means the knockout hurt):

```text
| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |
| A | … | … | … | … | … |
| B | … | … | … | … | … |
| C1 | … | … | … | … | … |
```

Copy `H3_ROW` lines verbatim. Also paste 2–3 projection nearest-segment rows per model (ticker, end_date, distance).

## H3 verdict (Runner writes, do not invent)
Use **only** this run’s table:

1. Prototypes **used** if proto_zero Δmse > 0 (and/or proto_shuffle > 0) on a model.
2. Text **used** if text_zero or text_shuffle Δmse > 0.
3. **H3 preservation (smoke):** A, B, and C1 show the **same sign** on proto_zero (prototypes still a live path after TimeXer). Magnitude need not match. Opposite sign or ~0 on B/C1 vs A → preservation not supported on this slice.
4. Near-zero text Δ is allowed on 1 epoch / 64 windows; say so. Negative Δ is valid (unfaithful / unused), not a failed job.
5. This is **not** paper H3.

## Environment notes
- expected_cwd: /workspace/DecisionForecast
- data_ready: yes (volume `7hejn56qf4` → `/workspace`)
- gpu: **required** (`train.device=cuda`)
- approx_ram_gb: 16+
- git: checkout `feat/phase1-data` at this request commit (descendant of `475d674`)
- pod: `i6lv3n0222en43` is fine; hostname may differ

## Out of scope
- Do not change product code unless a trivial one-liner is required to unblock the run; prefer reporting the blocker.
- Do not run `model=c0` for H3.
- Do not run full (uncapped) or `ticker_set=paper` trains unless this request says so.
- Do not run `scripts/download_fnspid.py` unless `Data/FNSPID` is missing.
- Do not invent metrics. Do not commit `outputs/`, checkpoints, or JSON binaries.

## After the run
Write `report.md` in this same folder, commit on the same branch, push.
