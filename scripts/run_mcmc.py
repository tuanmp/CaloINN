#!/usr/bin/env python
"""Run MCMC density ratio sampling at a single energy.

Loads a trained CINN model, classifier, and calibrator, then
runs Independent Metropolis-Hastings chains to correct CINN-generated
showers toward the true Geant4 distribution.

Usage::

    uv run python scripts/run_mcmc.py \
        --cinn-ckpt /path/to/cinn.ckpt \
        --clf-ckpt /path/to/classifier.ckpt \
        --calibrator /path/to/calibrator.json \
        --config params/pions_odd.yaml \
        --output mcmc_samples.hdf5

The calibrator file can be a JSON (temperature or Platt scaling)
or .npz (isotonic regression). The method is auto-detected.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import yaml

# Ensure src/ is on sys.path (bare imports like ``import data_util``)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_util
import torch_postprocess
import lightning as pl
from lightning_module import CaloINNLightningModule
from lightning_data import CaloINNDataModule
from mcmc.calibration import BaseCalibrator
from mcmc.classifier import ClassifierWrapper
from mcmc.convert import cinn_sample_to_classifier_input, cinn_sample_to_classifier_input_torch
from mcmc.sampler import IMHSampler

torch.set_default_dtype(torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCMC density ratio sampling for CaloINN."
    )
    p.add_argument("--cinn-ckpt", required=True, help="CINN Lightning checkpoint")
    p.add_argument("--clf-ckpt", required=True, help="Classifier .ckpt checkpoint")
    p.add_argument("--cinn-config", help="CINN config YAML")
    # p.add_argument("--clf-config", help="Classifier config YAML")
    p.add_argument("--calibrator", required=True,
                   help="Calibrator file (JSON or .npz, auto-detects method)")
    p.add_argument("--batch-size", type=int, default=10000, help="CINN batch size")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--burn-in", type=int, default=10)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--output", default="mcmc_samples.hdf5", help="Output HDF5 path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=-1, help="Number of MCMC samples to generate")
    p.add_argument("--log-transform", action="store_true",
                   help="Apply log1p to energy-normalised cells")
    p.add_argument("--voxel-cutoff", type=float, default=None,
                   help="Zero cells below this MeV threshold")
    p.add_argument("--r-clip", type=float, default=100.0,
                   help="Clip density ratios to [1/r_clip, r_clip]")
    p.add_argument("--profile", action="store_true",
                   help="Print per-component timing breakdown")
    return p.parse_args()


def load_cinn(ckpt_path: str, cinn_config: str, device: str, batch_size: int):
    """Load CaloINNLightningModule from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cinn_cfg = yaml.safe_load(open(cinn_config, "r"))
    hp = cinn_cfg["model"]

    model = CaloINNLightningModule(**hp)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)

    # load datamodule 
    dm_init_kw = cinn_cfg["data"]["init_args"]
    dm_init_kw["predict_batch_size"] = batch_size
    dm_init_kw["shuffle"] = True
    datamodule = CaloINNDataModule(**dm_init_kw)
    datamodule.setup()
    return model, datamodule


def main():
    args = parse_args()
    device = args.device

    # -- Load CINN ---------------------------------------------------------
    print(f"📂 Loading CINN from {args.cinn_ckpt}")
    cinn, cinn_dm = load_cinn(args.cinn_ckpt, args.cinn_config, device, args.batch_size)
    dataloaders = cinn_dm.predict_dataloader()
    print(f"   ✅ Loaded — num_dim={cinn.num_dim}, device={device}, dataloaders={len(dataloaders)}")

    # -- Load classifier ---------------------------------------------------
    print(f"📂 Loading classifier from {args.clf_ckpt}")
    classifier = ClassifierWrapper.load_from_checkpoint(args.clf_ckpt, device=device)
    print(f"   ✅ Loaded on {device}")

    # -- Load calibrator ---------------------------------------------------
    print(f"📂 Loading calibrator from {args.calibrator}")
    calibrator = BaseCalibrator.load(args.calibrator)
    print(f"   ✅ Loaded — {calibrator}")

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

    convert_fn_torch = partial(
        cinn_sample_to_classifier_input_torch,
        layer_boundaries=cinn.layer_boundaries,
        q=cinn.q,
        width_noise=cinn.width_noise,
        xml_path=cinn.hparams.xml_path,
        particle=cinn.hparams.xml_ptype,
        log_transform=args.log_transform,
        voxel_energy_cutoff=args.voxel_cutoff,
    )

    # -- Run MCMC ----------------------------------------------------------
    print(f"\n🔄 Running IMH: {args.n_steps} steps")
    print(f"   burn_in = {args.burn_in}, thin = {args.thin}")
    print(f"   log_transform = {args.log_transform}, voxel_cutoff = {args.voxel_cutoff}")

    sampler = IMHSampler(
        model=cinn.model,
        classifier=classifier,
        calibrator=calibrator,
        conversion_fn=convert_fn,
        conversion_fn_torch=convert_fn_torch,
        device=device,
        r_clip=args.r_clip,
    )

    for i, dataloader in enumerate(dataloaders):
        postprocessed_data = None
        n_samples = 0
        for _, c in tqdm(dataloader, desc=f"Processing dataloader {i}"):
            result = sampler.sample_multiple_energies(
                energy_gev=c,
                n_steps=args.n_steps,
                burn_in=args.burn_in,
                thin=args.thin,
                seed=args.seed,
                profile=args.profile,
            )

            if args.profile and "profile" in result:
                p = result["profile"]
                use_torch = p.get("use_torch_path", False)
                print(f"\n   ⏱️  Profile ({p['n_chains']} chains × {p['n_steps']} steps)"
                      f"{' [GPU path]' if use_torch else ' [NumPy path]'}:")
                print(f"   {'─' * 45}")
                print(f"   {'Component':<28} {'total (s)':>8} {'ms/step':>8}")
                print(f"   {'─' * 45}")
                if use_torch:
                    labels_keys = [
                        ("Propose (CINN sample)", "t_propose"),
                        ("Density ratio (total)", "t_density_ratio"),
                        ("  ├─ Convert (torch+HLF)", "  convert_np"),
                        ("  ├─ Classifier forward", "  clf_forward"),
                        ("  └─ Compute r(x)", "  compute_ratio"),
                        ("Accept / reject", "t_accept_reject"),
                    ]
                else:
                    labels_keys = [
                        ("Propose (CINN sample)", "t_propose"),
                        ("Density ratio (total)", "t_density_ratio"),
                        ("  ├─ GPU→CPU transfer", "  gpu_to_cpu"),
                        ("  ├─ Convert (numpy)", "  convert_np"),
                        ("  ├─ CPU→GPU transfer", "  cpu_to_gpu"),
                        ("  ├─ Classifier forward", "  clf_forward"),
                        ("  └─ Compute r(x)", "  compute_ratio"),
                        ("Accept / reject", "t_accept_reject"),
                    ]
                for label, key in labels_keys:
                    t = p[key]
                    ms = 1000 * t / p["n_steps"]
                    print(f"   {label:<28} {t:>8.3f} {ms:>8.3f}")
                print(f"   {'─' * 45}")
                print(f"   {'TOTAL':<28} {p['t_total']:>8.3f} {p['ms_per_step']:>8.3f}")

            # -- Postprocess (GPU-native) -------------------------------------------
            sample_np = result["samples"]
            n_samples += sample_np.shape[0]

            # Move to GPU, run post-processing, bring back to CPU dict.
            sample_t = torch.from_numpy(sample_np - cinn.width_noise).to(device)
            c_t = c.to(device)
            data = torch_postprocess.postprocess(
                sample_t,
                c_t,
                layer_boundaries=cinn.layer_boundaries,
                quantiles=cinn.q,
            )
            postprocessed = {k: v.detach().cpu().numpy() for k, v in data.items()}

            if postprocessed_data is None:
                postprocessed_data = postprocessed
            else:
                postprocessed_data = {
                    key : np.concatenate([postprocessed_data[key], postprocessed[key]], axis=0)
                    for key in postprocessed_data.keys()
                }
            
            if args.n_samples > 0 and n_samples >= args.n_samples:
                print(f"\n   ✅ Reached target of {args.n_samples} samples, stopping early.")
                break

        # -- Save --------------------------------------------------------------
        output_path = args.output.replace(".hdf5", f"_dataloader{i}.hdf5")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        truth_fmt = getattr(cinn_dm, "truth_format", None)
        if truth_fmt is not None:
            data_util.save_data_with_format(postprocessed_data, str(output_path), truth_fmt)
        else:
            data_util.save_data(postprocessed_data, filename=str(output_path))
        print(f"\n💾 Saved MCMC showers → {output_path}")

    # -- Save metadata -----------------------------------------------------
    import json
    meta = {
        "cinn_ckpt": args.cinn_ckpt,
        "clf_ckpt": args.clf_ckpt,
        "calibrator_path": args.calibrator,
        "calibrator_class": calibrator.__class__.__name__,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
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
    print(f"   📋 Metadata saved → {meta_path}")
    print("✅ Done!")

    print(f"\n   ✅ MCMC complete!")
    # print(f"   📊 samples:      {result['samples'].shape[0]}")
    # print(f"   📊 accept rate:  {result['acceptance_rate']:.4f}")
    # print(f"   📊 r(x) range:   [{result['density_ratios'].min():.3f}, "
    #       f"{result['density_ratios'].max():.3f}]")
    # print(f"   📊 r(x) median:  {np.median(result['density_ratios']):.3f}")



if __name__ == "__main__":
    main()
