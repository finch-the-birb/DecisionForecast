"""Ablation table helpers (H1 A vs B, later H2)."""

from __future__ import annotations


def format_ab_table(a: dict[str, float], b: dict[str, float]) -> str:
    """Markdown table for H1. Pass only numbers from a real run."""
    a_mse, a_mae = a["mse"], a["mae"]
    b_mse, b_mae = b["mse"], b["mae"]
    d_mse = b_mse - a_mse
    d_mae = b_mae - a_mae
    return (
        "| model | mse | mae |\n"
        "|-------|-----|-----|\n"
        f"| A | {a_mse:.4f} | {a_mae:.4f} |\n"
        f"| B | {b_mse:.4f} | {b_mae:.4f} |\n"
        f"| Δ(A→B) | {d_mse:+.4f} | {d_mae:+.4f} |\n"
    )
