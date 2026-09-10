"""Build A/B/C0/C1 × H markdown from MLflow runs (test_mse / test_mae)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.evaluation.ablation import format_ablation_table


def _latest_by_model_horizon(tracking_uri: str, experiment_name: str) -> dict[int, dict[str, dict[str, float]]]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} not found")
    grouped: dict[tuple[str, int], tuple[int, dict[str, float]]] = {}
    for run in client.search_runs([exp.experiment_id], max_results=5000):
        params = run.data.params
        name = params.get("model.name") or params.get("model")
        if name not in {"a", "b", "c0", "c1"}:
            continue
        raw_h = params.get("data.horizon") or params.get("horizon")
        if raw_h is None:
            continue
        metrics = run.data.metrics
        if "test_mse" not in metrics or "test_mae" not in metrics:
            continue
        start = int(run.info.start_time or 0)
        key = (str(name), int(raw_h))
        prev = grouped.get(key)
        if prev is None or start >= prev[0]:
            grouped[key] = (
                start,
                {"mse": float(metrics["test_mse"]), "mae": float(metrics["test_mae"])},
            )
    by_h: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for (name, horizon), (_ts, row) in grouped.items():
        by_h[horizon][name] = row
    return dict(by_h)


def render_tables(by_horizon: dict[int, dict[str, dict[str, float]]]) -> str:
    lines: list[str] = ["# Ablation A / B / C0 / C1", ""]
    for horizon in sorted(by_horizon):
        lines.append(f"## H={horizon}")
        lines.append("")
        lines.append(format_ablation_table(by_horizon[horizon]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_ablation_tables(
    tracking_uri: str,
    experiment_name: str,
    out_dir: Path,
) -> Path:
    by_h = _latest_by_model_horizon(tracking_uri, experiment_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_a_b_c0_c1.md"
    path.write_text(render_tables(by_h), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="decision-forecast")
    parser.add_argument(
        "--out-dir",
        default="outputs/tables",
        help="Directory for markdown tables",
    )
    args = parser.parse_args()
    path = write_ablation_tables(args.tracking_uri, args.experiment, Path(args.out_dir))
    print(path)


if __name__ == "__main__":
    main()
