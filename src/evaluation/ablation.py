"""Ablation table helpers (H1 A vs B, H2 B/C0/C1)."""

from __future__ import annotations


def format_ab_table(a: dict[str, float], b: dict[str, float]) -> str:
    """Markdown table for H1. Pass only numbers from a real run."""
    return format_ablation_table({"a": a, "b": b})


def format_ablation_table(metrics: dict[str, dict[str, float]]) -> str:
    """Markdown A/B/C0/C1 table plus pairwise deltas. Keys are model ids."""
    order = [m for m in ("a", "b", "c0", "c1") if m in metrics]
    labels = {"a": "A", "b": "B", "c0": "C0", "c1": "C1"}
    lines = [
        "| model | mse | mae |",
        "|-------|-----|-----|",
    ]
    for key in order:
        row = metrics[key]
        lines.append(f"| {labels.get(key, key)} | {row['mse']:.4f} | {row['mae']:.4f} |")
    pairs = (("a", "b", "A→B"), ("b", "c0", "B→C0"), ("c0", "c1", "C0→C1"))
    for left, right, name in pairs:
        if left not in metrics or right not in metrics:
            continue
        d_mse = metrics[right]["mse"] - metrics[left]["mse"]
        d_mae = metrics[right]["mae"] - metrics[left]["mae"]
        lines.append(f"| Δ({name}) | {d_mse:+.4f} | {d_mae:+.4f} |")
    return "\n".join(lines) + "\n"
