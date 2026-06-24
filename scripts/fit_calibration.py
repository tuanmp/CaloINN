#!/usr/bin/env python
"""Fit temperature scaling calibrator for MCMC density ratio estimation.

Loads a trained MLP classifier checkpoint and the validation split from
its training data, collects raw predictions, and fits the optimal
temperature T by minimising negative log-likelihood on the validation set.
The fitted T is saved as JSON for use by ``IMHSampler``.

Usage::

    uv run python scripts/fit_calibration.py \\
        --classifier-ckpt /path/to/classifier.ckpt \\
        --config /path/to/config.yaml \\
        --output /path/to/T.json

The ``--config`` YAML must match the config used to train the classifier
(it defines the data module, data paths, and preprocessing settings).
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Ensure caloxtreme_clf is importable
_CALOXTREME_ROOT = "/global/cfs/cdirs/m3443/usr/pmtuan/caloxtreme_clf"
if _CALOXTREME_ROOT not in sys.path:
    sys.path.insert(0, _CALOXTREME_ROOT)

# Our ported modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_SRC = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _PROJECT_SRC not in sys.path:
    sys.path.insert(0, _PROJECT_SRC)

from data.sharded_datamodule import LargeHDF5MLPDataModule
from mcmc.calibration import (
    TemperatureCalibrator,
    compare_calibration_methods,
)
from mcmc.classifier import ClassifierWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit temperature calibrator for MCMC density ratio."
    )
    parser.add_argument(
        "--classifier-ckpt",
        required=True,
        help="Path to MLPClassifier .ckpt checkpoint",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config used to train the classifier",
    )
    parser.add_argument(
        "--output",
        default="temperature_calibration.json",
        help="Path to save the fitted calibrator (JSON)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap on validation samples (for quick tests)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_validation_predictions(
    model: ClassifierWrapper,
    dataloader,
    device: str,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run classifier on validation set and collect raw predictions + labels.

    The dataloader yields batches of ``(X_proc, cond_proc, X_hlf, y)``.
    We concatenate the first three fields to form the 772D input expected
    by the classifier.
    """
    all_preds = []
    all_labels = []
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting validation predictions"):
            X_proc, cond_proc, X_hlf, y = batch

            # Concatenate into 772D input (matches BaseModel.get_input_from_batch)
            x = torch.cat([X_proc, cond_proc, X_hlf], dim=-1).to(device)

            logits = model.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.append(probs)
            all_labels.append(y.cpu().numpy().flatten())

            total += len(y)
            if max_samples is not None and total >= max_samples:
                break

    yhat = np.concatenate(all_preds)
    y = np.concatenate(all_labels)
    print(f"Collected {len(yhat)} validation predictions")
    print(f"  Label distribution: real={y.sum():.0f} ({y.mean()*100:.1f}%)")
    return yhat, y


def main():
    args = parse_args()

    # -- Load config ---------------------------------------------------------
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # -- Load classifier -----------------------------------------------------
    print(f"Loading classifier: {args.classifier_ckpt}")
    device = args.device
    classifier = ClassifierWrapper.load_from_checkpoint(
        args.classifier_ckpt, device=device
    )

    # -- Build validation dataloader -----------------------------------------
    print("Building validation dataloader...")
    dm_kwargs = dict(config["data"]["init_args"])
    # If the cache doesn't exist, fall back to on-the-fly preprocessing
    # instead of rebuilding it (which would re-read all raw HDF5s).
    if dm_kwargs.get("cache_mode") == "memmap":
        cache_dir = dm_kwargs.get("cache_dir", "")
        lt = int(dm_kwargs.get("log_transform", False))
        vc = int(dm_kwargs.get("voxel_energy_cutoff") or 0)
        suffix = f"_log{lt}_cutoff{vc}"
        train_meta = Path(cache_dir) / f"train{suffix}_memmap.json"
        val_meta = Path(cache_dir) / f"val{suffix}_memmap.json"
        if not train_meta.exists() or not val_meta.exists():
            print(f"  Cache missing at {cache_dir}, falling back to cache_mode='none'")
            dm_kwargs["cache_mode"] = "none"
        else:
            print(f"  Using memmap cache at {cache_dir}")
    datamodule = LargeHDF5MLPDataModule(**dm_kwargs)
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    print(f"Validation batches: {len(val_loader)}")

    # -- Collect predictions -------------------------------------------------
    yhat, y = collect_validation_predictions(
        classifier, val_loader, device, args.max_val_samples
    )

    # -- Fit and compare calibration methods ---------------------------------
    print("\nFitting calibrators...")
    # Use 80/20 split for temperature fitting (fit on 80%, eval on 20%)
    n_fit = int(len(yhat) * 0.8)
    idx = np.random.RandomState(42).permutation(len(yhat))
    yhat_fit, y_fit = yhat[idx[:n_fit]], y[idx[:n_fit]]
    yhat_eval, y_eval = yhat[idx[n_fit:]], y[idx[n_fit:]]

    results = compare_calibration_methods(yhat_fit, y_fit, yhat_eval, y_eval)
    _print_comparison(results)

    # -- Save best temperature calibrator ------------------------------------
    calib = TemperatureCalibrator()
    calib.fit(yhat_fit, y_fit)
    calib.save(args.output)
    print(f"\nSaved temperature calibrator (T={calib.T:.4f}) to {args.output}")

    # -- Also save comparison to a sidecar file ------------------------------
    sidecar = Path(args.output).with_suffix(".comparison.json")
    import json
    comparison_summary = {
        method: {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
            if k != "scores"
        }
        for method, metrics in results.items()
    }
    with open(sidecar, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"Calibration comparison saved to {sidecar}")


def _print_comparison(results: dict) -> None:
    """Pretty-print calibration comparison."""
    header = f"\n{'─' * 70}\nCalibration Method Comparison\n{'─' * 70}"
    print(header)
    print(f"{'Method':<16} {'ECE':>10} {'Brier':>10}")
    print("-" * 38)
    for method, metrics in results.items():
        print(f"  {method:<14} {metrics['ece']:>10.4f} {metrics['brier']:>10.4f}")
    print("─" * 70)

    best_ece = min(results.keys(), key=lambda m: results[m]["ece"])
    best_brier = min(results.keys(), key=lambda m: results[m]["brier"])
    print(f"  Best ECE:   {best_ece}")
    print(f"  Best Brier: {best_brier}")

    if "temperature" in results and "T" in results["temperature"]:
        print(f"  Optimal T:  {results['temperature']['T']:.3f}")
    print("─" * 70)


if __name__ == "__main__":
    main()
