#!/usr/bin/env python
"""Run MCMC density ratio sampling at a single energy.

Loads a trained CINN model, classifier, and temperature calibrator, then
runs Independent Metropolis-Hastings chains to correct CINN-generated
showers toward the true Geant4 distribution.

Usage::

    uv run python scripts/run_mcmc.py \
        --cinn-ckpt /path/to/cinn.ckpt \
        --clf-ckpt /path/to/classifier.ckpt \
        --calibrator /path/to/T.json \
        --energy 16 \
        --config params/pions_odd.yaml \
        --output mcmc_samples.hdf5

If the CINN checkpoint's init data is unavailable, provide it explicitly::

    --init-data /path/to/init_sample.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure src/ is on sys.path (bare imports like ``import data_util``)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_util
import lightning as pl
from lightning_module import CaloINNLightningModule
from mcmc.calibration import TemperatureCalibrator
from mcmc.classifier import ClassifierWrapper
from mcmc.convert import cinn_sample_to_classifier_input
from mcmc.sampler import IMHSampler

torch.set_default_dtype(torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCMC density ratio sampling for CaloINN."
    )
    p.add_argument("--cinn-ckpt", required=True, help="CINN Lightning checkpoint")
    p.add_argument("--clf-ckpt", required=True, help="Classifier .ckpt checkpoint")
    p.add_argument("--calibrator", required=True, help="Temperature calibrator JSON")
    p.add_argument("--energy", type=float, default=16.0, help="log2 energy (e.g. 16)")
    p.add_argument("--n-chains", type=int, default=100)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--burn-in", type=int, default=10)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--output", default="mcmc_samples.hdf5", help="Output HDF5 path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-transform", action="store_true",
                   help="Apply log1p to energy-normalised cells")
    p.add_argument("--voxel-cutoff", type=float, default=None,
                   help="Zero cells below this MeV threshold")
    p.add_argument("--r-clip", type=float, default=100.0,
                   help="Clip density ratios to [1/r_clip, r_clip]")
    p.add_argument("--init-data", default=None,
                   help="Override init data path (if ckpt's path is stale)")
    return p.parse_args()


def load_cinn(ckpt_path: str, device: str, init_data_override: str | None = None):
    """Load CaloINNLightningModule from checkpoint.

    Uses Lightning's ``load_from_checkpoint`` which reads hyperparameters
    from the checkpoint and calls ``__init__``.  If the original init data
    path is unavailable, provide ``--init-data`` to override it.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})

    if init_data_override is not None:
        hp["setup_data_sample_path"] = init_data_override
        print(f"Overriding init data path → {init_data_override}")

    # Lightning load_from_checkpoint internally calls __init__(**hp),
    # then loads state_dict.  We need to pass the hp explicitly.
    model = CaloINNLightningModule(**hp)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model


def main():
    args = parse_args()
    device = args.device
    energy_gev = 2 ** args.energy / 1e3

    # -- Load CINN ---------------------------------------------------------
    print(f"Loading CINN from {args.cinn_ckpt}")
    cinn = load_cinn(args.cinn_ckpt, device, args.init_data)
    print(f"  num_dim={cinn.num_dim}, device={device}")

    # -- Load classifier ---------------------------------------------------
    print(f"Loading classifier from {args.clf_ckpt}")
    classifier = ClassifierWrapper.load_from_checkpoint(args.clf_ckpt, device=device)

    # -- Load calibrator ---------------------------------------------------
    print(f"Loading calibrator from {args.calibrator}")
    calibrator = TemperatureCalibrator.load(args.calibrator)
    print(f"  T = {calibrator.T:.4f}")

    # -- Build conversion closure ------------------------------------------
    from functools import partial

    convert_fn = partial(
        cinn_sample_to_classifier_input,
        layer_boundaries=cinn.layer_boundaries,
        q=cinn.q,
        width_noise=cinn.width_noise,
        xml_path=cinn.hparams.xml_path,
        particle=cinn.hparams.xml_ptype,
        log_transform=args.log_transform,
        voxel_energy_cutoff=args.voxel_cutoff,
    )

    # -- Run MCMC ----------------------------------------------------------
    print(f"\nRunning IMH: {args.n_chains} chains × {args.n_steps} steps")
    print(f"  energy={energy_gev:.2f} GeV (2^{args.energy} MeV)")
    print(f"  burn_in={args.burn_in}, thin={args.thin}")
    print(f"  log_transform={args.log_transform}, voxel_cutoff={args.voxel_cutoff}")

    sampler = IMHSampler(
        model=cinn.model,
        classifier=classifier,
        calibrator=calibrator,
        conversion_fn=convert_fn,
        device=device,
        r_clip=args.r_clip,
    )

    result = sampler.sample(
        energy_gev=energy_gev,
        n_chains=args.n_chains,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=args.seed,
    )

    print(f"\nMCMC complete:")
    print(f"  samples: {result['samples'].shape[0]}")
    print(f"  acceptance_rate: {result['acceptance_rate']:.4f}")
    print(f"  density_ratio range: [{result['density_ratios'].min():.3f}, "
          f"{result['density_ratios'].max():.3f}]")
    print(f"  density_ratio median: {np.median(result['density_ratios']):.3f}")

    # -- Postprocess -------------------------------------------------------
    samples = result["samples"]
    samples = samples - cinn.width_noise
    energies_np = np.full((samples.shape[0], 1), energy_gev, dtype=np.float32)

    data = data_util.postprocess(
        samples,
        energies_np,
        layer_boundaries=cinn.layer_boundaries,
        threshold=cinn.width_noise,
        quantiles=cinn.q.detach().cpu().numpy(),
    )

    # -- Save --------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_util.save_data(data, filename=str(output_path))
    print(f"\nSaved MCMC showers to {output_path}")

    # -- Save metadata -----------------------------------------------------
    import json
    meta = {
        "cinn_ckpt": args.cinn_ckpt,
        "clf_ckpt": args.clf_ckpt,
        "calibrator_T": calibrator.T,
        "energy_log2": args.energy,
        "energy_gev": energy_gev,
        "n_chains": args.n_chains,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "n_samples": int(samples.shape[0]),
        "acceptance_rate": float(result["acceptance_rate"]),
        "density_ratio_mean": float(result["density_ratios"].mean()),
        "density_ratio_median": float(np.median(result["density_ratios"])),
        "density_ratio_min": float(result["density_ratios"].min()),
        "density_ratio_max": float(result["density_ratios"].max()),
        "log_transform": args.log_transform,
        "voxel_cutoff": args.voxel_cutoff,
        "seed": args.seed,
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
