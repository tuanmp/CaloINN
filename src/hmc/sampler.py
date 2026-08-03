"""The Hamiltonian Monte Carlo sampling loop.

For a target with log-density ``log_prob(q)`` we treat the potential as
``U(q) = -log_prob(q)`` and draw independent unit-normal momenta ``p``. A
leapfrog trajectory proposes a move ``(q, p)``, accepted with the standard
Metropolis probability ``min(1, exp(H(q, p) - H(q', p')))``.
"""

import numpy as np

from hmc.integrator import leapfrog_trajectory


class _TargetAdapter:
    """Bridge the `targets` interface to potential/gradient callables."""

    def __init__(self, target):
        self.target = target

    def potential(self, q):
        return -self.target.log_prob(q)

    def grad(self, q):
        return -self.target.grad_log_prob(q)


def _adapt_step_size(step_size, acceptance, target=0.8):
    """Simple multiplicative step-size adaptation toward a target acceptance."""
    return step_size * np.exp(1.0 / 2 * (acceptance - target))


def sample_hmc(
    target,
    n_samples=10_000,
    n_warmup=1_000,
    step_size=0.25,
    n_leapfrog=10,
    seed=None,
    adapt_step_size=True,
    target_acceptance=0.8,
):
    """Draw samples from ``target`` using Hamiltonian Monte Carlo.

    Parameters
    ----------
    target : object
        Any object exposing ``log_prob(q)`` and ``grad_log_prob(q)``
        (e.g. an instance from ``hmc.targets``).
    n_samples : int
        Number of post-warmup samples to return.
    n_warmup : int
        Number of burn-in iterations, discarded (also used for adaptation).
    step_size : float
        Leapfrog step size. Must be positive.
    n_leapfrog : int
        Number of leapfrog steps per trajectory. Must be >= 1.
    seed : int | numpy.random.Generator, optional
        Random seed / generator for reproducibility.
    adapt_step_size : bool
        Whether to tune ``step_size`` during warmup toward
        ``target_acceptance``.
    target_acceptance : float
        Target Metropolis acceptance rate used during adaptation.

    Returns
    -------
    samples : np.ndarray of shape (n_samples, dim)
    info : dict
        Diagnostics: ``acceptance_rate`` (over all iterations),
        ``step_size`` (final, after adaptation).
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if n_leapfrog < 1:
        raise ValueError(f"n_leapfrog must be >= 1, got {n_leapfrog}")

    rng = np.random.default_rng(seed)
    adapter = _TargetAdapter(target)
    q0 = rng.normal(size=target.dim)
    dim = target.dim

    total = n_warmup + n_samples
    samples = np.empty((n_samples, dim))
    eps = step_size
    accepted = 0

    q = q0
    for i in range(total):
        p = rng.normal(size=dim)
        qs, ps, energies = leapfrog_trajectory(
            adapter.grad, adapter.potential, q, p, eps, n_leapfrog
        )
        q_prop, p_prop = qs[-1], ps[-1]
        log_accept = -np.inf
        if np.isfinite(energies[0]) and np.isfinite(energies[-1]):
            log_accept = energies[0] - energies[-1]
            if log_accept > 0 or np.log(rng.uniform()) < log_accept:
                q = q_prop
                accepted += 1

        if i >= n_warmup:
            samples[i - n_warmup] = q
        elif adapt_step_size:
            eps = _adapt_step_size(eps, min(1.0, np.exp(log_accept)), target_acceptance)

    info = {
        "acceptance_rate": accepted / total,
        "step_size": eps,
    }
    return samples, info
