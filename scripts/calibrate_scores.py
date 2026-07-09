#!/usr/bin/env python
"""Apply a fitted calibrator to raw model scores (probabilities).

Loads uncalibrated probability scores and a calibrator file, then
outputs calibrated probabilities.  Works with any calibrator method
(temperature, Platt, isotonic) — auto-detected from the file.

Supports both .npy (single array) and .npz (multi-array) inputs.

Usage::

    # .npy input
    uv run python scripts/calibrate_scores.py \
        --scores raw_probs.npy \
        --calibrator temperature_calibration.json \
        --output calibrated.npy

    # .npz input (reads "y_hat" key by default)
    uv run python scripts/calibrate_scores.py \
        --scores mcmc_20steps.npz \
        --calibrator temperature_calibration.json \
        --output calibrated.npz

    # .npz input with explicit key
    uv run python scripts/calibrate_scores.py \
        --scores mcmc_20steps.npz \
        --key y_hat \
        --calibrator temperature_calibration.json \
        --output calibrated.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mcmc.calibration import BaseCalibrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate raw model probability scores."
    )
    p.add_argument("--scores", required=True,
                   help="Path to .npy or .npz file with raw probability scores")
    p.add_argument("--key", default="y_hat",
                   help="Array key within .npz file (default: y_hat). Ignored for .npy.")
    p.add_argument("--calibrator", required=True,
                   help="Calibrator file (JSON or .npz, auto-detects method)")
    p.add_argument("--output", required=True,
                   help="Path to save calibrated probabilities (.npy or .npz)")
    return p.parse_args()


def load_scores(path: str, key: str) -> tuple[np.ndarray, dict | None]:
    """Load scores from .npy or .npz.

    Returns (scores_array, extra_dict) where extra_dict contains any
    additional arrays from a .npz file (or None for .npy).
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
        scores = data[key]
        # Collect extra arrays to preserve in output
        extra = {k: data[k] for k in data.files if k != key}
        return scores, extra
    else:
        return np.load(path), None


def prob_to_logit(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert probabilities to logits (inverse sigmoid)."""
    prob = np.clip(prob.astype(np.float64), eps, 1.0 - eps)
    return np.log(prob / (1.0 - prob))


def logit_to_prob(logit: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-logit))


def main():
    args = parse_args()

    # -- Load scores --------------------------------------------------------
    print(f"📂 Loading scores from {args.scores}")
    raw_probs, extra = load_scores(args.scores, args.key)
    print(f"   ✅ Loaded — shape={raw_probs.shape}, "
          f"range=[{raw_probs.min():.4f}, {raw_probs.max():.4f}]")
    if extra:
        print(f"   📦 Extra keys in .npz: {list(extra.keys())}")

    # -- Load calibrator ----------------------------------------------------
    print(f"📂 Loading calibrator from {args.calibrator}")
    calibrator = BaseCalibrator.load(args.calibrator)
    print(f"   ✅ Loaded — {calibrator}")

    # -- Calibrate ----------------------------------------------------------
    print("🔧 Calibrating...")

    # Convert probabilities → logits, apply calibrator, convert back.
    logits = prob_to_logit(raw_probs)
    cal_logits = calibrator.transform_logits(logits)
    cal_probs = logit_to_prob(cal_logits)

    print(f"   ✅ Calibrated — range=[{cal_probs.min():.4f}, {cal_probs.max():.4f}]")

    # -- Save ---------------------------------------------------------------
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".npz":
        out_data = {args.key: cal_probs}
        if extra:
            out_data.update(extra)
        np.savez(output, **out_data)
        print(f"💾 Saved calibrated scores (key='{args.key}') → {output}")
    else:
        np.save(output, cal_probs)
        print(f"💾 Saved calibrated scores → {output}")

    print("✅ Done!")


if __name__ == "__main__":
    main()
