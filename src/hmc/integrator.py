"""Hamiltonian Monte Carlo: leapfrog integration of Hamilton's equations."""

import numpy as np
import torch


def leapfrog_step(f_grad: torch.Tensor, q: torch.Tensor, p: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Take one leapfrog step for a Hamiltonian with unit mass.

    Parameters
    ----------
    f_grad : callable
        Gradient of the potential energy U(q) = -log_prob(q), i.e. dU/dq.
    q, p : torch.Tensor
        Current position and momentum.
    eps : float
        Step size.

    Returns
    -------
    (q_new, p_new) : tuple[torch.Tensor, torch.Tensor]
    """
    p_half = p - 0.5 * eps * f_grad(q)
    q_new = q + eps * p_half
    p_new = p_half - 0.5 * eps * f_grad(q_new)
    return q_new, p_new


def leapfrog_trajectory(f_grad: torch.Tensor, potential: torch.Tensor, q: torch.Tensor, p: torch.Tensor, eps: float, n_steps: int, 
    return_energy=False, return_trajectory=False):
    """Integrate a leapfrog trajectory of ``n_steps`` steps.

    Parameters
    ----------
    f_grad : callable
        Gradient of the potential energy, dU/dq.
    potential : callable
        Potential energy U(q) = -log_prob(q).
    q, p : torch.Tensor
        Starting position and momentum.
    eps : float
        Step size.
    n_steps : int
        Number of leapfrog steps.

    Returns
    -------
    qs, ps : torch.Tensor of shape (n_steps + 1, dim)
        Position and momentum history along the trajectory.
    energies : torch.Tensor of shape (n_steps + 1,)
        Hamiltonian H = U(q) + 0.5 p @ p at each point.
    """
    dim = q.shape
    qs = torch.empty(dim)
    ps = torch.empty(dim)
    if return_trajectory:
        qs = torch.empty((n_steps + 1, dim))
        ps = torch.empty((n_steps + 1, dim))
    for i in range(1, n_steps + 1):
        q, p = leapfrog_step(f_grad, q, p, eps)
        if return_trajectory:
            qs[i] = q
            ps[i] = p
        
    energies = torch.tensor(
        [potential(qi) + 0.5 * torch.dot(pi, pi) for qi, pi in zip(qs, ps)]
    )
    return qs, ps, energies
