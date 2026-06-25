"""Independent Metropolis-Hastings sampler with classifier-based density ratio.

Uses the CINN as a proposal distribution ``q(x|c)`` and a trained
classifier to estimate the density ratio ``r(x,c) = p(x|c)/q(x|c)``.
The acceptance ratio simplifies to:

    α = min(1, r(x',c) / r(x,c))

which requires only the classifier, not the true density ``p``.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch

from .calibration import TemperatureCalibrator


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
    
    @torch.inference_mode()
    def _sample(
        self,
        conditions: torch.Tensor,
        n_steps: int = 500,
        burn_in: int = 10,
        thin: int = 1,
        profile: bool = False,
    ):  
        """Run IMH chains for multiple conditions (energies).

        Parameters
        ----------
        conditions : torch.Tensor  shape (N, 1)
            Incident energies in GeV.
        n_steps : int
            Total MH steps per chain (including burn-in).
        burn_in : int
            Number of initial steps to discard.
        thin : int
            Keep every ``thin``-th post-burn-in sample.
        profile : bool
            If True, collect per-component wall-clock timings.

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
        n_chains = c.shape[0]

        # --- Profiling accumulators ------------------------------------
        if profile:
            t_propose = 0.0
            t_density = 0.0
            t_gpu_to_cpu = 0.0
            t_convert = 0.0
            t_cpu_to_gpu = 0.0
            t_clf_fwd = 0.0
            t_ratio = 0.0
            t_ar = 0.0

        # --- Initialise chains from CINN samples -----------------------
        x_current = self._propose(c)                     # (n_chains, 730)
        r_current = self._density_ratio(
            x_current, c,
            _profile_return=(profile, None) if not profile else None,
        )[0] if profile else self._density_ratio(x_current, c)

        # --- Storage ---------------------------------------------------
        n_kept = max(1, (n_steps - burn_in) // thin)
        ratios = torch.zeros((n_kept, n_chains), device=device)
        total_accepted = 0
        total_proposed = 0
        storage_idx = 0

        # --- MH loop ---------------------------------------------------
        for step in range(n_steps):
            if profile:
                t0 = time.perf_counter()
                x_proposed = self._propose(c)
                t_p = time.perf_counter()

                _, d_times = self._density_ratio(x_proposed, c, _profile_return=True)
                r_proposed = d_times["result"]
                t_d = time.perf_counter()

                # Acceptance ratio
                alpha = torch.clamp(
                    r_proposed / (r_current + 1e-10), max=1.0
                )
                u = torch.rand_like(alpha)
                accept = u < alpha
                t_ar_step = time.perf_counter()

                x_current = torch.where(accept[:, None], x_proposed, x_current)
                r_current = torch.where(accept, r_proposed, r_current)

                # Accumulate
                t_propose += t_p - t0
                t_gpu_to_cpu += d_times["gpu_to_cpu"]
                t_convert += d_times["convert"]
                t_cpu_to_gpu += d_times["cpu_to_gpu"]
                t_clf_fwd += d_times["clf_forward"]
                t_ratio += d_times["ratio"]
                t_density += t_d - t_p
                t_ar += time.perf_counter() - t_ar_step

                total_accepted += int(accept.sum().item())
                total_proposed += n_chains
            else:
                x_next, r_next, accepted = self._step(x_current, c, r_current)
                total_accepted += int(accepted.sum().item())
                total_proposed += n_chains
                x_current = x_next
                r_current = r_next

            # Store post-burn-in, thinned
            post_burn = step >= burn_in
            at_thin_interval = (step - burn_in) % thin == 0
            if post_burn and at_thin_interval:
                ratios[storage_idx] = r_current
                storage_idx += 1

        # --- Aggregate -------------------------------------------------
        acceptance_rate = total_accepted / max(total_proposed, 1)
        flat_samples = x_current.cpu().numpy()    # (n_chains, 730)
        flat_ratios = ratios.cpu().numpy()        # (n_kept, n_chains)

        result = {
            "samples": flat_samples,
            "acceptance_rate": acceptance_rate,
            "density_ratios": flat_ratios,
        }

        if profile:
            result["profile"] = {
                "n_chains": n_chains,
                "n_steps": n_steps,
                "t_propose": t_propose,
                "t_density_ratio": t_density,
                "  gpu_to_cpu": t_gpu_to_cpu,
                "  convert_np": t_convert,
                "  cpu_to_gpu": t_cpu_to_gpu,
                "  clf_forward": t_clf_fwd,
                "  compute_ratio": t_ratio,
                "t_accept_reject": t_ar,
                "t_total": t_propose + t_density + t_ar,
                "ms_per_step": 1000 * (t_propose + t_density + t_ar) / n_steps,
            }

        return result
        

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_single_energy(
        self,
        energy_gev: float | torch.Tensor,
        n_chains: int = 100,
        n_steps: int = 500,
        burn_in: int = 10,
        thin: int = 1,
        seed: int | None = None,
    ) -> dict:
        """Run IMH chains at a single incident energy.

        Parameters
        ----------
        energy_gev : float or torch.Tensor
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
        else:
            from datetime import datetime
            now = datetime.now().timestamp()
            torch.manual_seed(int(now) % (2**32 - 1))
            np.random.seed(int(now) % (2**32 - 1))

        c = torch.full((n_chains, 1), float(energy_gev), device=self._device)
        return self._sample(c, n_steps, burn_in, thin)

    @torch.inference_mode()
    def sample_multiple_energies(
        self,
        energy_gev: torch.Tensor,
        n_steps: int,
        burn_in: int,
        thin: int,
        seed: int | None = None,
        profile: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run MCMC for multiple energies. Each energy value is used to run a 
        separate MCMC chain

        Parameters
        ----------
        energy_gev : list of float or torch.Tensor
            Incident energies in GeV.
        n_chains : int
            Number of chains to run.
        n_steps : int
            Number of steps to run each chain.
        burn_in : int
            Number of burn-in steps to discard.
        thin : int
            Thinning factor for the samples.
        seed : int | None, optional
            Random seed for reproducibility.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the samples, acceptance rate, and density ratios.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            from datetime import datetime
            now = datetime.now().timestamp()
            torch.manual_seed(int(now) % (2**32 - 1))
            np.random.seed(int(now) % (2**32 - 1))
        
        c = energy_gev.reshape(-1, 1).to(self._device)
        return self._sample(c, n_steps, burn_in, thin, profile=profile)

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
        """Execute one MH step for all chains simultaneously.

        All N chains are processed in a single batched operation:
        one ``model.sample()`` call, one classifier forward pass,
        and one vectorised accept/reject — no Python loop over chains.

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
        _profile_return: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute calibrated density ratio r(x,c) = D_cal / (1 - D_cal).

        Parameters
        ----------
        x_internal : torch.Tensor  shape (N, 730)
            CINN internal samples.
        c : torch.Tensor  shape (N, 1)
            Incident energies in GeV.
        _profile_return : bool
            If True, return (result, timing_dict) instead of just result.

        Returns
        -------
        torch.Tensor  shape (N,)
            Density ratios (clipped).
        or tuple (torch.Tensor, dict) when _profile_return=True.

        Notes
        -----
        The conversion path (postprocess + HLF) runs on CPU via numpy,
        causing a GPU→CPU→GPU round-trip per MH step.
        """
        temp = self.calibrator.T
        eps = 1e-10
        times = {} if _profile_return else None

        # GPU → CPU
        t0 = time.perf_counter() if _profile_return else 0
        x_np = x_internal.cpu().numpy()
        c_np = c.cpu().numpy()
        if _profile_return:
            torch.cuda.synchronize()
            times["gpu_to_cpu"] = time.perf_counter() - t0

        # Conversion (numpy postprocess + HLF)
        t1 = time.perf_counter() if _profile_return else 0
        z = self.convert(x_np, c_np)                      # (N, 772)
        if _profile_return:
            times["convert"] = time.perf_counter() - t1

        # CPU → GPU + classifier forward
        t2 = time.perf_counter() if _profile_return else 0
        z_t = torch.from_numpy(z).to(self._device)
        if _profile_return:
            torch.cuda.synchronize()
            times["cpu_to_gpu"] = time.perf_counter() - t2

        t3 = time.perf_counter() if _profile_return else 0
        logits = self.clf.forward(z_t).squeeze(-1)        # (N,)
        if _profile_return:
            torch.cuda.synchronize()
            times["clf_forward"] = time.perf_counter() - t3

        # Density ratio
        t4 = time.perf_counter() if _profile_return else 0
        D_cal = torch.sigmoid(logits / temp)               # (N,)
        r = D_cal / (1.0 - D_cal + eps)                   # (N,)
        r = torch.clamp(r, 1.0 / self._r_clip, self._r_clip)
        if _profile_return:
            torch.cuda.synchronize()
            times["ratio"] = time.perf_counter() - t4
            times["result"] = r
            return r, times

        return r
