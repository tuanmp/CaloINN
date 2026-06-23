"""Independent Metropolis-Hastings sampler with classifier-based density ratio.

Uses the CINN as a proposal distribution ``q(x|c)`` and a trained
classifier to estimate the density ratio ``r(x,c) = p(x|c)/q(x|c)``.
The acceptance ratio simplifies to:

    α = min(1, r(x',c) / r(x,c))

which requires only the classifier, not the true density ``p``.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from src.mcmc.calibration import TemperatureCalibrator


class IMHSampler:
    """Independent Metropolis-Hastings with density ratio reweighting.

    Parameters
    ----------
    model : CINN
        Trained CINN model.  Must provide ``sample(num_pts, condition)``.
    classifier : ClassifierWrapper
        Trained classifier that outputs raw logits.
    calibrator : TemperatureCalibrator
        Fitted temperature calibrator for density ratio computation.
    conversion_fn : callable
        Function ``f(x_internal, c, ...) -> classifier_input`` that
        converts CINN internal representation to the 772D format
        expected by the classifier.
    device : str or torch.device
        Device for computation.
    r_clip : float, optional
        Clip density ratios to [1/r_clip, r_clip] as a safety measure
        against extreme values from miscalibrated classifiers.
        Default 100.0 (i.e., r ∈ [0.01, 100]).
    """

    def __init__(
        self,
        model,
        classifier,
        calibrator: TemperatureCalibrator,
        conversion_fn: Callable,
        device: str | torch.device = "cpu",
        r_clip: float = 100.0,
    ):
        self.model = model
        self.clf = classifier
        self.calibrator = calibrator
        self.convert = conversion_fn
        self._device = torch.device(device)
        self._r_clip = r_clip

        if calibrator.T is None:
            raise ValueError("Calibrator must be fitted before use.")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample(
        self,
        energy_gev: float,
        n_chains: int = 100,
        n_steps: int = 500,
        burn_in: int = 100,
        thin: int = 5,
        seed: int | None = None,
    ) -> dict:
        """Run IMH chains at a single incident energy.

        Parameters
        ----------
        energy_gev : float
            Incident energy in GeV.
        n_chains : int
            Number of independent Markov chains.
        n_steps : int
            Total MH steps per chain (including burn-in).
        burn_in : int
            Number of initial steps to discard.
        thin : int
            Keep every ``thin``-th post-burn-in sample.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            - ``"samples"`` : np.ndarray shape (n_kept, 730)
              CINN-internal samples.  Postprocess with ``data_util.postprocess``.
            - ``"acceptance_rate"`` : float
              Overall acceptance rate (fraction of proposals accepted).
            - ``"density_ratios"`` : np.ndarray shape (n_kept,)
              Calibrated density ratio for each kept sample.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model.eval()
        device = self._device

        # Broadcast single energy to all chains
        c = torch.full((n_chains, 1), energy_gev, device=device)

        # --- Initialise chains from CINN samples -----------------------
        x_current = self._propose(c)                     # (n_chains, 730)
        r_current = self._density_ratio(x_current, c)    # (n_chains,)

        # --- Storage ---------------------------------------------------
        n_kept = max(1, (n_steps - burn_in) // thin)
        samples = torch.zeros((n_kept, n_chains, x_current.shape[1]), device=device)
        ratios = torch.zeros((n_kept, n_chains), device=device)
        total_accepted = 0
        total_proposed = 0
        storage_idx = 0

        # --- MH loop ---------------------------------------------------
        for step in range(n_steps):
            x_next, r_next, accepted = self._step(x_current, c, r_current)

            total_accepted += int(accepted.sum().item())
            total_proposed += n_chains

            x_current = x_next
            r_current = r_next

            # Store post-burn-in, thinned
            post_burn = step >= burn_in
            at_thin_interval = (step - burn_in) % thin == 0
            if post_burn and at_thin_interval:
                samples[storage_idx] = x_current
                ratios[storage_idx] = r_current
                storage_idx += 1

        # --- Aggregate -------------------------------------------------
        acceptance_rate = total_accepted / max(total_proposed, 1)
        flat_samples = samples.cpu().numpy().reshape(-1, samples.shape[-1])  # (n_kept * n_chains, 730)
        flat_ratios = ratios.cpu().numpy().reshape(-1)                        # (n_kept * n_chains,)

        return {
            "samples": flat_samples,
            "acceptance_rate": acceptance_rate,
            "density_ratios": flat_ratios,
        }

    # ------------------------------------------------------------------
    #  Single MH step
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _step(
        self,
        x_current: torch.Tensor,
        c: torch.Tensor,
        r_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one MH step for all chains.

        Parameters
        ----------
        x_current : torch.Tensor  shape (N, 730)
            Current chain states (CINN internal).
        c : torch.Tensor  shape (N, 1)
            Incident energies in GeV.
        r_current : torch.Tensor  shape (N,)
            Density ratios for current states.

        Returns
        -------
        x_next : torch.Tensor  shape (N, 730)
        r_next : torch.Tensor  shape (N,)
        accepted : torch.Tensor  shape (N,)  bool
        """
        # 1. Propose
        x_proposed = self._propose(c)                     # (N, 730)

        # 2. Evaluate density ratios
        r_proposed = self._density_ratio(x_proposed, c)   # (N,)

        # 3. Acceptance ratio  α = min(1, r' / r)
        alpha = torch.clamp(
            r_proposed / (r_current + 1e-10),
            max=1.0,
        )

        # 4. Accept / reject
        u = torch.rand_like(alpha)
        accept = u < alpha                               # (N,) bool

        x_next = torch.where(
            accept[:, None], x_proposed, x_current
        )
        r_next = torch.where(accept, r_proposed, r_current)

        return x_next, r_next, accept

    # ------------------------------------------------------------------
    #  Proposal
    # ------------------------------------------------------------------

    def _propose(self, c: torch.Tensor) -> torch.Tensor:
        """Draw proposals from the CINN.

        Parameters
        ----------
        c : torch.Tensor  shape (N, 1)

        Returns
        -------
        torch.Tensor  shape (N, 730)
        """
        # model.sample(num_pts, condition) → (len(condition), num_pts, dims)
        # For N chains at the same energy: broadcast to 1 condition, N pts
        samples = self.model.sample(c.shape[0], c[:1])   # (1, N, 730)
        return samples.squeeze(0)                         # (N, 730)

    # ------------------------------------------------------------------
    #  Density ratio via classifier
    # ------------------------------------------------------------------

    def _density_ratio(
        self,
        x_internal: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibrated density ratio r(x,c) = D_cal / (1 - D_cal).

        Parameters
        ----------
        x_internal : torch.Tensor  shape (N, 730)
            CINN internal samples.
        c : torch.Tensor  shape (N, 1)
            Incident energies in GeV.

        Returns
        -------
        torch.Tensor  shape (N,)
            Density ratios (clipped).
        """
        # Convert CINN internal → classifier input (numpy path)
        temp = self.calibrator.T
        eps = 1e-10

        x_np = x_internal.cpu().numpy()
        c_np = c.cpu().numpy()

        # The conversion_fn signature includes all the extra args
        # stored in the closure.  Simplest: pass them explicitly.
        z = self.convert(x_np, c_np)                      # (N, 772) np.ndarray

        # Classifier forward → raw logits
        z_t = torch.from_numpy(z).to(self._device)
        logits = self.clf.forward(z_t).squeeze(-1)        # (N,)

        # Temperature calibrate: D_cal = σ(logit / T)
        D_cal = torch.sigmoid(logits / temp)               # (N,)

        # Density ratio  r = D / (1 - D)
        r = D_cal / (1.0 - D_cal + eps)                   # (N,)

        # Clip for safety
        r = torch.clamp(r, 1.0 / self._r_clip, self._r_clip)

        return r
