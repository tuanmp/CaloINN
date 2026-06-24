"""CINN internal representation → classifier input conversion.

Replicates the preprocessing pipeline from ``caloxtreme_clf`` so that
CINN-generated showers can be fed to the ported classifier for density
ratio estimation during MCMC.

The pipeline replicates:
1. ``data_util.postprocess`` — CINN internal (730D) → physical layers (GeV)
2. Convert to MeV for HLF computation
3. ``scale_shower`` — energy-normalised cells + energy in GeV
4. ``get_high_level_features`` — 51 physics-motivated features
5. Concatenate into 772D classifier input
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Ensure src/ is on sys.path so bare imports like ``import data_util`` work
# (matches main.py lines 12-15).
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import data_util
from ._hlf import HighLevelFeatures


def _compute_hlf(
    X_mev: np.ndarray,
    Einc_mev: np.ndarray,
    xml_path: str,
    particle: str,
) -> np.ndarray:
    """Compute high-level features from raw showers in MeV.

    Replicates ``data.utils.get_high_level_features`` from caloxtreme_clf.
    The 51 features are (in order):
      - E_tot (1)
      - Per-layer energies (10)
      - Eta centroids per layer (10)
      - Phi centroids per layer (10)
      - Eta widths per layer (10)
      - Phi widths per layer (10)

    Sparsity is intentionally excluded — it is commented out in the
    classifier training pipeline and would change the input dimension.

    Parameters
    ----------
    X_mev : np.ndarray  shape (N, 720)
        Raw shower cell energies in MeV.
    Einc_mev : np.ndarray  shape (N, 1)
        Incident energies in MeV.
    xml_path : str
        Path to the XML binning file.
    particle : str
        Particle name (e.g. "pion").

    Returns
    -------
    np.ndarray  shape (N, 51)
        Stacked high-level features.
    """
    hlf = HighLevelFeatures(particle, filename=xml_path)
    # setattr(hlf, "Einc", Einc_mev)  — not used by CalculateFeatures, set for reference
    hlf.CalculateFeatures(X_mev)

    # Order must match caloxtreme_clf data/utils.py exactly
    features = [
        hlf.GetEtot(),
        *hlf.GetElayers().values(),
        *hlf.GetECEtas().values(),
        *hlf.GetECPhis().values(),
        *hlf.GetWidthEtas().values(),
        *hlf.GetWidthPhis().values(),
    ]
    return np.stack(features, axis=1, dtype=np.float32)


def _scale_shower(X_mev: np.ndarray, Einc_mev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Energy-normalise shower cells and convert energy to GeV.

    Replicates ``data.utils.scale_shower`` from caloxtreme_clf.

    Parameters
    ----------
    X_mev : np.ndarray  shape (N, 720)
        Raw shower cells in MeV.
    Einc_mev : np.ndarray  shape (N, 1)
        Incident energies in MeV.

    Returns
    -------
    X_proc : np.ndarray  shape (N, 720)
        Energy-normalised cells (X_mev / Einc_mev).
    cond_proc : np.ndarray  shape (N, 1)
        Incident energy in GeV (Einc_mev / 1000).
    """
    cond = Einc_mev.reshape(-1, 1).astype(np.float64)
    X = X_mev.astype(np.float64) / cond
    cond_gev = cond / 1000.0
    return X.astype(np.float32), cond_gev.astype(np.float32)


def _apply_voxel_cutoff(X_mev: np.ndarray, cutoff_mev: float | None) -> np.ndarray:
    """Zero out shower cells below an energy threshold (in MeV).

    Replicates the ``voxel_energy_cutoff`` step from the classifier's
    ``_preprocess_batch``: ``X[X <= cutoff] = 0.0``.
    """
    if cutoff_mev is None or cutoff_mev <= 0:
        return X_mev
    return np.where(X_mev > cutoff_mev, X_mev, 0.0)


def cinn_sample_to_classifier_input(
    x_internal: np.ndarray | torch.Tensor,
    c: np.ndarray | torch.Tensor,
    layer_boundaries: list[int],
    q: torch.Tensor,
    width_noise: float,
    xml_path: str,
    particle: str,
    log_transform: bool = False,
    voxel_energy_cutoff: float | None = None,
) -> np.ndarray:
    """Convert CINN internal representation to 772D classifier input.

    Replicates the full ``_preprocess_batch`` pipeline from the
    classifier's ``LargeHDF5MLPDataModule``, including optional
    ``log1p`` transform and voxel energy cutoff.

    Parameters
    ----------
    x_internal : np.ndarray or torch.Tensor  shape (N, 730)
        CINN internal representation (720 shower cells + 10 extra dims).
        This is the raw output of ``model.sample()``.
    c : np.ndarray or torch.Tensor  shape (N, 1)
        Incident energies in GeV.
    layer_boundaries : list[int]
        Cumulative cell indices defining layer boundaries.
    q : torch.Tensor
        Quantile thresholds for postprocessing.
    width_noise : float
        Uniform noise width subtracted before postprocessing.
    xml_path : str
        Path to the XML binning file.
    particle : str
        Particle name matching the XML (e.g. "pion").
    log_transform : bool
        If True, apply ``np.log1p`` to energy-normalised cells.
        Must match the classifier's training config.
    voxel_energy_cutoff : float, optional
        Zero out cells below this MeV threshold before HLF computation.
        Must match the classifier's training config.

    Returns
    -------
    np.ndarray  shape (N, 772)
        Classifier input: [X_proc(720), cond_proc(1), X_hlf(51)].
        Ready to be fed to ``ClassifierWrapper.forward()``.
    """
    # Convert torch → numpy if needed
    if isinstance(x_internal, torch.Tensor):
        x_np = x_internal.cpu().numpy().copy()
    else:
        x_np = np.array(x_internal, copy=True)
    if isinstance(c, torch.Tensor):
        c_np = c.cpu().numpy().copy()
    else:
        c_np = np.array(c, copy=True)

    # Ensure 2D
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    if c_np.ndim == 1:
        c_np = c_np.reshape(-1, 1)

    # Step 1: Subtract width noise (matches predict_step / generate_single_energy)
    x_np = x_np - width_noise

    # Step 2: Postprocess CINN internal → physical layers in GeV
    data = data_util.postprocess(
        x_np,
        c_np,
        layer_boundaries=layer_boundaries,
        quantiles=q.detach().cpu().numpy(),
        threshold=width_noise,
    )
    # data = {"energy": (N,1) in GeV, "layer_0": (N, cells_0), ..., "layer_9": (N, cells_9)}

    # Step 3: Reconstruct flat shower array in MeV
    n_layers = len(layer_boundaries) - 1
    layers_gev = [data[f"layer_{i}"] for i in range(n_layers)]
    X_gev = np.concatenate(layers_gev, axis=1)  # (N, 720) in GeV
    X_mev = X_gev * 1e3                          # (N, 720) in MeV
    Einc_gev = data["energy"]                     # (N, 1) in GeV
    Einc_mev = Einc_gev * 1e3                     # (N, 1) in MeV

    # Step 4: Apply voxel energy cutoff (before HLF, matches _preprocess_batch)
    X_mev = _apply_voxel_cutoff(X_mev, voxel_energy_cutoff)

    # Step 5: Compute high-level features (requires MeV, after cutoff)
    X_hlf = _compute_hlf(X_mev, Einc_mev, xml_path, particle)  # (N, 51)

    # Step 6: Scale shower (energy-normalise, MeV→GeV for condition)
    X_proc, cond_proc = _scale_shower(X_mev, Einc_mev)          # (N, 720), (N, 1)

    # Step 7: Optional log transform (matches log_transform config)
    if log_transform:
        X_proc = np.log1p(X_proc)

    # Step 8: Stack into classifier input
    classifier_input = np.concatenate(
        [X_proc, cond_proc, X_hlf],
        axis=1,
    ).astype(np.float32)

    return classifier_input  # (N, 772)
