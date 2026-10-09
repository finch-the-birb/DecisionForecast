"""Build A/B/C0/C1 × H markdown from MLflow runs (mean ± std over seeds)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.explain.faithfulness import format_h3_table


def _mean_std(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var**0.5, n


def _fmt(xs: list[float], signed: bool = False) -> str:
    mean, std, n = _mean_std(xs)
    core = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    if n <= 1:
        return core
    return f"{core} ± {std:.4f} (n={n})"


def _match(params: dict[str, str], key: str, want: str | None) -> bool:
    if want is None:
        return True
    got = params.get(key)
    return got is not None and str(got) == str(want)


def _collect_runs(
    tracking_uri: str,
    experiment_name: str,
    ticker_set: str,
    lookback: int,
    normalize: str | None,
    train_start: str | None,
) -> list:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")
    runs = []
    for run in client.search_runs([exp.experiment_id], max_results=5000):
        params = run.data.params
        name = params.get("model.name") or params.get("model")
        if name not in {"a", "b", "c0", "c1"}:
            continue
        if not _match(params, "train.ticker_set", ticker_set):
            continue
        if not _match(params, "data.lookback_T", str(lookback)):
            continue
        if normalize is not None and not _match(params, "data.normalize", normalize):
            continue
        if train_start is not None and not _match(params, "data.split.train_start", train_start):
            continue
        raw_h = params.get("data.horizon") or params.get("horizon")
        if raw_h is None:
            continue
        metrics = run.data.metrics
        if "test_mse" not in metrics or "test_mae" not in metrics:
            continue
        seed = params.get("train.seed") or params.get("seed") or "0"
        runs.append(
            {
                "name": str(name),
                "horizon": int(raw_h),
                "seed": str(seed),
                "start": int(run.info.start_time or 0),
                "mse": float(metrics["test_mse"]),
                "mae": float(metrics["test_mae"]),
                "mae_denorm": float(metrics["test_mae_denorm"])
                if "test_mae_denorm" in metrics
                else None,
                "h3": {
                    k: float(v)
                    for k, v in metrics.items()
                    if k.startswith("h3_")
                },
            }
        )
    return runs


def _latest_per_seed(runs: list[dict]) -> list[dict]:
    best: dict[tuple[str, int, str], dict] = {}
    for row in runs:
        key = (row["name"], row["horizon"], row["seed"])
        prev = best.get(key)
        if prev is None or row["start"] >= prev["start"]:
            best[key] = row
    return list(best.values())


def render_ablation(runs: list[dict]) -> str:
    by_h: dict[int, list[dict]] = defaultdict(list)
    for row in _latest_per_seed(runs):
        by_h[row["horizon"]].append(row)
    lines = ["# Ablation A / B / C0 / C1 (mean ± std over seeds)", ""]
    labels = {"a": "A", "b": "B", "c0": "C0", "c1": "C1"}
    pairs = (("a", "b", "A→B"), ("b", "c0", "B→C0"), ("c0", "c1", "C0→C1"))
    for horizon in sorted(by_h):
        lines.append(f"## H={horizon}")
        lines.append("")
        lines.append("| model | mse | mae | mae_denorm |")
        lines.append("|-------|-----|-----|------------|")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in by_h[horizon]:
            grouped[row["name"]].append(row)
        for name in ("a", "b", "c0", "c1"):
            if name not in grouped:
                continue
            rows = grouped[name]
            den = [r["mae_denorm"] for r in rows if r["mae_denorm"] is not None]
            den_s = _fmt(den) if den else "n/a"
            lines.append(
                f"| {labels[name]} | {_fmt([r['mse'] for r in rows])} | "
                f"{_fmt([r['mae'] for r in rows])} | {den_s} |"
            )
        for left, right, lab in pairs:
            if left not in grouped or right not in grouped:
                continue
            by_seed_l = {r["seed"]: r for r in grouped[left]}
            by_seed_r = {r["seed"]: r for r in grouped[right]}
            common = sorted(set(by_seed_l) & set(by_seed_r))
            d_mse = [by_seed_r[s]["mse"] - by_seed_l[s]["mse"] for s in common]
            d_mae = [by_seed_r[s]["mae"] - by_seed_l[s]["mae"] for s in common]
            lines.append(f"| Δ({lab}) | {_fmt(d_mse, True)} | {_fmt(d_mae, True)} | |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_h3_from_runs(runs: list[dict]) -> str:
    latest = _latest_per_seed(runs)
    reports: dict[str, list[dict]] = defaultdict(list)
    for row in latest:
        h3 = row["h3"]
        if "h3_full_mse" not in h3:
            continue
        variants = {
            "full": {"mse": h3["h3_full_mse"]},
            "proto_zero": {"delta_mse": h3.get("h3_proto_zero_delta_mse", 0.0)},
            "proto_shuffle": {"delta_mse": h3.get("h3_proto_shuffle_delta_mse", 0.0)},
            "text_zero": {"delta_mse": h3.get("h3_text_zero_delta_mse", 0.0)},
            "text_shuffle": {"delta_mse": h3.get("h3_text_shuffle_delta_mse", 0.0)},
        }
        reports[row["name"]].append({"variants": variants})
    if not reports:
        return ""
    return "# H3 faithfulness (mean ± std over seeds)\n\n" + format_h3_table(dict(reports))


def write_ablation_tables(
    tracking_uri: str,
    experiment_name: str,
    out_dir: Path,
    ticker_set: str = "dev",
    lookback: int = 60,
    normalize: str | None = "per_window",
    train_start: str | None = "2015-01-01",
) -> Path:
    runs = _collect_runs(
        tracking_uri, experiment_name, ticker_set, lookback, normalize, train_start
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_a_b_c0_c1.md"
    body = render_ablation(runs)
    h3 = render_h3_from_runs(runs)
    if h3:
        body = body + "\n" + h3
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="decision-forecast")
    parser.add_argument("--out-dir", default="outputs/tables")
    parser.add_argument("--ticker-set", default="dev", choices=["dev", "paper"])
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--normalize", default="per_window")
    parser.add_argument("--train-start", default="2015-01-01")
    args = parser.parse_args()
    path = write_ablation_tables(
        args.tracking_uri,
        args.experiment,
        Path(args.out_dir),
        ticker_set=args.ticker_set,
        lookback=args.lookback,
        normalize=args.normalize,
        train_start=args.train_start,
    )
    print(path)


if __name__ == "__main__":
    main()
