"""Diagnostics for MCMC chains: acceptance rate, autocorrelation, ESS."""

import numpy as np


def acceptance_rate(acceptances):
    """Fraction of accepted proposals in a sequence of 0/1 flags."""
    acceptances = np.asarray(acceptances)
    return float(np.mean(acceptances))


def lagged_autocorrelation(x, lag):
    """Pearson autocorrelation of a 1-D chain at a given lag."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[: x.size - lag] * x[lag:]) / denom)


def effective_sample_size(x):
    """Effective sample size per dimension using the Geyer initial-positive
    estimator (monotone sequence of autocorrelation sums).

    Parameters
    ----------
    x : np.ndarray of shape (n,) or (n, dim)

    Returns
    -------
    float or np.ndarray of shape (dim,)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _ess_1d(x)
    return np.array([_ess_1d(col) for col in x.T])


def _ess_1d(x):
    n = x.size
    x = x - np.mean(x)
    if np.all(x == 0):
        return 0.0

    var = np.mean(x * x)
    if var == 0:
        return 0.0

    rho = []
    for lag in range(1, n):
        rho.append(np.sum(x[: n - lag] * x[lag:]) / (n - lag) / var)
        if rho[-1] <= 0:
            break
    rho = np.asarray(rho)

    rho_pos = np.maximum(rho, 0)
    rho_mono = np.minimum.accumulate(rho_pos)
    tau = 1 + 2 * np.sum(rho_mono)
    if tau <= 0:
        return float(n)
    return float(n / tau)
