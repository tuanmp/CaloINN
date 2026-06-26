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
from mcmc.calibration import TemperatureCalibrator
from mcmc.classifier import ClassifierWrapper
from mcmc.convert import cinn_sample_to_classifier_input, cinn_sample_to_classifier_input_torch
from mcmc.sampler import IMHSampler

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
#  Autocorrelation diagnostics (batched GPU FFT)
# ---------------------------------------------------------------------------


def batch_autocorrelation(ratios):
    """Compute ACF for all chains in parallel via Wiener-Khinchin (FFT).

    Zero-pads to 2L-1 to avoid circular wrap-around, then a single
    batched cuFFT call processes all C chains simultaneously.

    Args:
        ratios: shape (L, C) --- L=chain_length, C=n_chains.
    Returns:
        acf: shape (L, C) --- autocorrelation per chain, rho(k,c).
    """
    L = ratios.shape[0]
    x = ratios - ratios.mean(dim=0, keepdim=True)
    n_fft = 2 * L - 1
    X = torch.fft.rfft(x, n=n_fft, dim=0)
    power = X.real ** 2 + X.imag ** 2
    acf_raw = torch.fft.irfft(power, dim=0)[:L]
    acf = acf_raw / (acf_raw[0:1] + 1e-10)
    return acf


def integrated_autocorr_time(acf):
    """IACT via Geyer (1992) initial monotone sequence estimator.

    Pairs adjacent lags: Gamma_m = rho(2m) + rho(2m+1).
    Stops at first m where Gamma_m < 0.

        tau_hat = -1 + 2 * sum_{m=0}^{M} Gamma_m

    Args:
        acf: shape (L,) --- mean ACF averaged over all chains.
    Returns:
        tau: scalar tensor --- integrated autocorrelation time (steps).
    """
    L = acf.shape[0]
    tau = -torch.ones((), device=acf.device, dtype=acf.dtype)
    for m in range(L // 2):
        if m == 0:
            window = acf[0] + acf[1]
        else:
            j, j1 = 2 * m, 2 * m + 1
            w1 = acf[j]
            w2 = acf[j1] if j1 < L else torch.zeros_like(w1)
            window = w1 + w2
        tau = tau + 2.0 * window
        if window <= 0:
            break
    return tau


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCMC density ratio sampling for CaloINN."
    )
    p.add_argument("--cinn-ckpt", required=True, help="CINN Lightning checkpoint")
    p.add_argument("--clf-ckpt", required=True, help="Classifier .ckpt checkpoint")
    p.add_argument("--cinn-config", help="CINN config YAML")
    # p.add_argument("--clf-config", help="Classifier config YAML")
    p.add_argument("--calibrator", required=True, help="Temperature calibrator JSON")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--burn-in", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--output", default="mcmc_samples.hdf5", help="Output HDF5 path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=10000, help="Number of MCMC samples to generate")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size for CINN sampling")
    p.add_argument("--energy", type=float, default=10000, help="Energy in MeV for CINN sampling")
    p.add_argument("--log-transform", action="store_true",
                   help="Apply log1p to energy-normalised cells")
    p.add_argument("--voxel-cutoff", type=float, default=None,
                   help="Zero cells below this MeV threshold")
    p.add_argument("--r-clip", type=float, default=100.0,
                   help="Clip density ratios to [1/r_clip, r_clip]")
    p.add_argument("--init-data", default=None,
                   help="Override init data path (if ckpt's path is stale)")
    return p.parse_args()


def load_cinn(ckpt_path: str, cinn_config: str, device: str, batch_size: int):
    """Load CaloINNLightningModule from checkpoint.

    Uses Lightning's ``load_from_checkpoint`` which reads hyperparameters
    from the checkpoint and calls ``__init__``.  If the original init data
    path is unavailable, provide ``--init-data`` to override it.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cinn_cfg = yaml.safe_load(open(cinn_config, "r"))
    hp = cinn_cfg["model"]

    model = CaloINNLightningModule(**hp)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)

    return model


def main():
    args = parse_args()
    device = args.device
    c = torch.tensor(args.energy, dtype=torch.float32, device=device) / 1000.0  # MeV → GeV

    # -- Load CINN ---------------------------------------------------------
    print(f"📂 Loading CINN from {args.cinn_ckpt}")
    cinn = load_cinn(args.cinn_ckpt, args.cinn_config, device, args.batch_size)
    print(f"   ✅ Loaded — num_dim={cinn.num_dim}, device={device}")

    # -- Load classifier ---------------------------------------------------
    print(f"📂 Loading classifier from {args.clf_ckpt}")
    classifier = ClassifierWrapper.load_from_checkpoint(args.clf_ckpt, device=device)
    print(f"   ✅ Loaded on {device}")

    # -- Load calibrator ---------------------------------------------------
    print(f"📂 Loading calibrator from {args.calibrator}")
    calibrator = TemperatureCalibrator.load(args.calibrator)
    print(f"   ✅ T = {calibrator.T:.4f}")

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

    postprocessed_data = None
    density_ratios = []
    for _ in tqdm(range(0, args.n_samples, args.batch_size), desc="Batches"):

        result = sampler.sample_single_energy(
            energy_gev=c,
            n_steps=args.n_steps,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=args.seed,
            n_chains=args.batch_size,
        )

        # -- Postprocess (GPU-native) -------------------------------------------
        ratios = result["density_ratios"]  # shape (chain_length, batch_size)

        density_ratios.append(ratios)

    # -- Aggregate all chains ---------------------------------------------------
    density_ratios = torch.from_numpy(np.concatenate(density_ratios, axis=1))
    L, C = density_ratios.shape  # L = chain_length, C = total chains
    print(f"\n   📊 Collected {C} chains × {L} steps = {C * L} density-ratio samples")

    # -- Autocorrelation diagnostics (GPU FFT) ----------------------------------
    print("\n   🔬 Computing autocorrelation (batched GPU FFT) ...")
    density_ratios_gpu = density_ratios.to(device)

    acf_per_chain = batch_autocorrelation(density_ratios_gpu)     # (L, C)
    acf_mean = acf_per_chain.mean(dim=1).cpu()                    # (L,)
    tau = integrated_autocorr_time(acf_mean)                      # scalar

    # -- Summary statistics -----------------------------------------------------
    r_mean = density_ratios.float().mean()
    r_std = density_ratios.float().std()
    ess = C * L / tau

    print(f"\n   ╔{'═' * 50}╗")
    print(f"   ║  {'MCMC Autocorrelation Diagnostics':^48} ║")
    print(f"   ╠{'═' * 50}╣")
    print(f"   ║  {'Total chains':>24}: {C:>6}          ║")
    print(f"   ║  {'Steps per chain':>24}: {L:>6}          ║")
    print(f"   ║  {'─' * 40}  ║")
    print(f"   ║  {'r(x) mean':>24}: {r_mean:>12.4f} ║")
    print(f"   ║  {'r(x) std':>24}: {r_std:>12.4f} ║")
    print(f"   ║  {'─' * 40}  ║")
    print(f"   ║  {'IACT (Geyer 1992)':>24}: {tau:>8.1f} steps   ║")
    print(f"   ║  {'Eff. sample size':>24}: {ess:>12.0f} ║")
    print(f"   ║  {'ESS per chain':>24}: {(ess / C):>12.2f} ║")
    print(f"   ╚{'═' * 50}╝")

    # -- Save ACF to file for plotting ------------------------------------------
    output_base = args.output.replace(".hdf5", "")
    acf_path = f"{output_base}_acf.npy"
    np.save(acf_path, acf_mean.numpy())
    print(f"\n   💾 Mean ACF saved → {acf_path}")
    print("✅ Done!")



if __name__ == "__main__":
    main()
