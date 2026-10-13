"""Evaluate and apply Adaptive Conformal Inference (ACI) on trained fold/seed models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

from src.evaluation.conformal import apply_adaptive_conformal_inference
from src.evaluation.metrics import empirical_coverage, tunnel_sharpness, winkler_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_saved_predictions")


def find_run_dir(fold: int, seed: int, base_dir: Path = Path("outputs")) -> Path | None:
    """Find the most recent run output directory matching fold and seed."""
    candidates = []
    for cfg_path in base_dir.rglob("config_resolved.yaml"):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            data_fold = int(cfg.get("data", {}).get("fold", 1))
            run_seed = int(cfg.get("train", {}).get("seed", 42))
            model_name = str(cfg.get("model", {}).get("name", ""))
            if (
                data_fold == fold
                and run_seed == seed
                and "c1_hierarchical_mlp_quantile" in model_name
            ):
                # Check that metrics.json exists
                run_dir = cfg_path.parent
                if (run_dir / "metrics.json").exists():
                    candidates.append((cfg_path.stat().st_mtime, run_dir))
        except Exception:
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def evaluate_run(
    run_dir: Path,
    alpha: float = 0.20,
    gamma: float = 0.005,
) -> dict[str, float]:
    """Load test predictions from run_dir, apply ACI calibration, and record results."""
    preds_path = run_dir / "test_predictions.pt"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing test_predictions.pt in {run_dir}")

    data = torch.load(preds_path, weights_only=True)
    preds = data["preds"]      # [N, H, 3]
    targets = data["targets"]  # [N, H]

    raw_cov = empirical_coverage(preds, targets)
    raw_shp = tunnel_sharpness(preds)
    raw_wkl = winkler_score(preds, targets, alpha=alpha)

    aci_res = apply_adaptive_conformal_inference(preds, targets, alpha=alpha, gamma=gamma)
    cal_cov = aci_res["coverage"]
    cal_shp = aci_res["sharpness"]
    cal_wkl = aci_res["winkler"]

    # Save calibrated predictions
    torch.save(aci_res["calibrated_preds"], run_dir / "calibrated_predictions.pt")

    # Update metrics.json
    metrics_path = run_dir / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        try:
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    metrics_data.update(
        {
            "quantile_coverage": raw_cov,
            "quantile_sharpness": raw_shp,
            "quantile_winkler": raw_wkl,
            "aci_coverage": cal_cov,
            "aci_sharpness": cal_shp,
            "aci_winkler": cal_wkl,
            "aci_gamma": gamma,
            "aci_alpha": alpha,
        }
    )
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    log.info(
        "Run %s: Raw (cov80=%.4f, sharp=%.4f, winkler=%.4f) -> ACI (cov80=%.4f, sharp=%.4f, winkler=%.4f)",
        run_dir.name,
        raw_cov,
        raw_shp,
        raw_wkl,
        cal_cov,
        cal_shp,
        cal_wkl,
    )
    return {
        "raw_coverage": raw_cov,
        "raw_sharpness": raw_shp,
        "raw_winkler": raw_wkl,
        "aci_coverage": cal_cov,
        "aci_sharpness": cal_shp,
        "aci_winkler": cal_wkl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and apply ACI calibration on saved runs.")
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conformal-gamma", type=float, default=0.005)
    parser.add_argument("--conformal-alpha", type=float, default=0.20)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_run_dir(args.fold, args.seed)

    if not run_dir or not run_dir.exists():
        raise FileNotFoundError(f"Could not find run directory for fold {args.fold}, seed {args.seed}")

    res = evaluate_run(
        run_dir,
        alpha=args.conformal_alpha,
        gamma=args.conformal_gamma,
    )
    print("\n" + "=" * 60)
    print(f"Results for Fold {args.fold} Seed {args.seed} ({run_dir})")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

