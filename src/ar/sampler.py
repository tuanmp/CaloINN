"""Rejection-Acceptance sampler with classifier-based density ratio.

Uses the CINN as a proposal distribution ``q(x|c)`` and a trained
classifier to estimate the density ratio ``r(x,c) = p(x|c)/q(x|c)``.
The acceptance probability simplifies to:

    p = min(1, r(x,c)/M)

which requires only the classifier, not the true density ``p``. M is a 
normalization constant
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch

from mcmc.calibration import BaseCalibrator

class ARSampler:
    """Rejection-Acceptance sampler with classifier-based density ratio.

    Uses the CINN as a proposal distribution ``q(x|c)`` and a trained
    classifier to estimate the density ratio ``r(x,c) = p(x|c)/q(x|c)``.
    The acceptance probability simplifies to:

        p = min(1, r(x,c)/M)

    which requires only the classifier, not the true density ``p``. M is a 
    normalization constant.
    """

    def __init__(
        self,
        model,
        classifier,
        calibrator: BaseCalibrator,
        Z: float = 0.95,
        conversion_fn: Callable | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.classifier = classifier
        self.calibrator = calibrator
        self._device = device
        self.Z = Z
        self.conversion_fn = conversion_fn

        self.model.to(self._device)

        if not calibrator.is_fitted:
            raise ValueError("Calibrator must be fitted before use.")
        
    @torch.inference_mode()
    def _sample(
        self, 
        conditions: torch.Tensor,
        max_steps: int = 1000,
    ):
        """Run AR chains for multiple conditions (energies).

        Parameters
        ----------
        conditions : torch.Tensor  shape (N, 1)
            Incident energies in GeV.
        max_steps : int
            Maximum number of steps to run.

        Returns
        -------
        dict
            - ``"samples"`` : np.ndarray shape (n_chains, 730)
            - ``"acceptance_rate"`` : float
            - ``"density_ratios"`` : np.ndarray shape (n_kept, n_chains)
            - ``"profile"`` : dict (only if profile=True)
        """
        self.model.eval()
        device = self._device

        # Broadcast energy
        c = conditions.reshape(-1, 1).to(device)  # (N, 1)

        # manual seed torch
        torch.manual_seed(int(time.time() * 1000) % (2**32 - 1))

        # sampling loop
        step = 0
        n_sample = 0
        n_requested = c.shape[0]
        out_samples, out_conditions, out_ratios = [], [], []
        while step < max_steps and n_sample < n_requested:
            # -- Sample from CINN ------------------------------------------------
            x = self.model.sample(1, c).squeeze(1)  # (N, D)
            x = x.to(device)

            # -- Convert to classifier input -------------------------------------
            if self.conversion_fn is not None:
                x_clf = self.conversion_fn(x, c)
            else:
                x_clf = x

            # -- Accept/reject ---------------------------------------------------
            r = self._density_ratio(x_clf)
            u = torch.rand_like(r)
            accept_mask = u < (r / self.Z)
            out_samples.append(x[accept_mask])
            out_conditions.append(c[accept_mask])
            out_ratios.append(r[accept_mask])

            n_sample += accept_mask.sum().item()
            step += 1
            
            c = c[~accept_mask]  # keep only rejected conditions for next step
            x = x[~accept_mask]  # keep only rejected samples for next step
            r = r[~accept_mask]  # keep only rejected ratios for next step

        if n_sample < n_requested:
            out_samples.append(x)
            out_conditions.append(c)
            out_ratios.append(r)
        
        out_samples = torch.cat(out_samples, dim=0)
        out_conditions = torch.cat(out_conditions, dim=0)
        out_ratios = torch.cat(out_ratios, dim=0)

        assert out_samples.shape[0] == out_conditions.shape[0] == out_ratios.shape[0] == n_requested, \
            "Inconsistent shapes in sampled outputs"
        return out_samples, out_conditions, out_ratios

    def _density_ratio(self, x_clf: torch.Tensor) -> torch.Tensor:
        """Compute density ratio r(x,c) = p(x|c)/q(x|c) using the classifier.

        Parameters
        ----------
        x_clf : torch.Tensor shape (N, D)
            Samples from the proposal distribution q(x|c).
            Should be x concatenated with conditions c, as expected by the classifier.
        """
        # -- Compute density ratios ------------------------------------------
        eps = 1e-10  # small constant to avoid division by zero

        logits = self.classifier.forward(x_clf).squeeze(-1)  # shape (N,)
        calibrated_logits = self.calibrator.transform_logits(logits)
        D_cal = torch.sigmoid(calibrated_logits)
        r = D_cal / (1 - D_cal + eps)  # density ratio r(x,c) = p(x|c)/q(x|c)
        return r

    def sample_single_energy(
        self,
        energy_gev: float | torch.Tensor,
        n_samples: int,
        max_steps: int = 1000,
    ) :
        """Run IMH chains at a single incident energy.

        Parameters
        ----------
        energy_gev : float or torch.Tensor
            Incident energy in GeV.
        n_samples : int
            Number of samples to generate.
        max_steps : int
            Maximum number of resampling steps.

        """

        c = torch.full((n_samples, 1), float(energy_gev))
        return self._sample(c, max_steps=max_steps)

    def sample_multiple_energies(
        self,
        energies_gev: torch.Tensor,
        max_steps: int = 1000,
    ):
        """Run IMH chains at multiple incident energies.

        Parameters
        ----------
        energies_gev : torch.Tensor shape (N,)
            Incident energies in GeV.
        max_steps : int
            Maximum number of resampling steps.

        """
        return self._sample(energies_gev.to(torch.float32), max_steps=max_steps)