#!/usr/bin/env python
"""Apply a fitted calibrator to raw model scores (probabilities).

Loads uncalibrated probability scores and a calibrator file, then
outputs calibrated probabilities.  Works with any calibrator method
(temperature, Platt, isotonic) — auto-detected from the file.

Usage::

    uv run python scripts/calibrate_scores.py \\
        --scores /path/to/raw_scores.npy \\
        --calibrator /path/to/calibrator.json \\
        --output calibrated_scores.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

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
                   help="Path to .npy file with raw probability scores (shape (N,))")
    p.add_argument("--calibrator", required=True,
                   help="Calibrator file (JSON or .npz, auto-detects method)")
    p.add_argument("--output", required=True,
                   help="Path to save calibrated probabilities (.npy)")
    return p.parse_args()


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
    raw_probs = np.load(args.scores)
    print(f"   ✅ Loaded — shape={raw_probs.shape}, "
          f"range=[{raw_probs.min():.4f}, {raw_probs.max():.4f}]")

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
    np.save(args.output, cal_probs)
    print(f"💾 Saved calibrated scores → {args.output}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
