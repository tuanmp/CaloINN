"""Calibration utilities for binary classifier probability estimates.

Ported from caloxtreme_clf/calibration/__init__.py.  Provides:

- ``expected_calibration_error`` — ECE metric
- ``TemperatureCalibrator`` — fit-once, apply-to-many temperature scaling
- ``calibrate_platt`` — Platt scaling (logistic regression on logits)
- ``compare_calibration_methods`` — driver that evaluates all methods

For MCMC we primarily use temperature scaling because it preserves the
rank ordering of density ratios and requires storing only a single float.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════
# 1.  Expected Calibration Error (ECE)
# ═══════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 100,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into *n_bins* equal-width bins across [0, 1],
    then computes the weighted average of |accuracy - confidence| per bin.

    Parameters
    ----------
    y_true : np.ndarray  shape (N,)
        Binary ground-truth labels.
    y_pred : np.ndarray  shape (N,)
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of equal-width bins (default 100 for fine-grained ECE).

    Returns
    -------
    float
        ECE value in [0, 1].  Lower is better calibrated.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        else:
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (n_bin / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


# ═══════════════════════════════════════════════════════════════════════
# 0.  Abstract base calibrator
# ═══════════════════════════════════════════════════════════════════════

class BaseCalibrator(ABC):
    """Abstract base for classifier probability calibrators.

    Subclasses must implement ``transform_logits``, ``save``, and a
    ``load`` classmethod.  The unified contract is that
    ``transform_logits`` accepts raw classifier logits and returns
    calibrated logits such that ``sigmoid(calibrated_logits)`` gives
    well-calibrated probabilities.
    """

    @abstractmethod
    def transform_logits(self, logits):
        """Convert raw logits to calibrated logits.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor
            Raw logits from the classifier.

        Returns
        -------
        np.ndarray or torch.Tensor (same type as input)
            Calibrated logits.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """Return True if the calibrator has been fitted."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save calibrator to file."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> "BaseCalibrator":
        """Factory: load a calibrator from file, auto-detecting method.

        Reads the file, inspects the ``"method"`` key (or infers
        ``"temperature"`` for legacy files), and returns the appropriate
        subclass instance.
        """
        path = Path(path)

        if path.suffix == ".npz":
            from sklearn.isotonic import IsotonicRegression

            data = np.load(path)
            # Forward reference — defined later in this module
            from mcmc.calibration import IsotonicCalibrator  # type: ignore[import-not-found]

            iso = IsotonicCalibrator()
            iso._iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso._iso_reg.X_thresholds_ = data["X_thresholds"]
            iso._iso_reg.y_thresholds_ = data["y_thresholds"]
            iso._X_thresholds = torch.from_numpy(data["X_thresholds"])
            iso._y_thresholds = torch.from_numpy(data["y_thresholds"])
            iso._fitted = True
            return iso

        with open(path) as f:
            data = json.load(f)

        method = data.get("method", "temperature")  # legacy files have no "method" key
        if method == "temperature":
            return TemperatureCalibrator(T=float(data["T"]))
        elif method == "platt":
            # Forward reference — defined later in this module
            from mcmc.calibration import PlattCalibrator  # type: ignore[import-not-found]

            return PlattCalibrator(a=float(data["a"]), b=float(data["b"]))
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


# ═══════════════════════════════════════════════════════════════════════
# 2.  Temperature scaling calibrator
# ═══════════════════════════════════════════════════════════════════════

class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibration (Guo et al. 2017).

    Learns a single temperature parameter *T* on validation data by
    minimising negative log-likelihood, then applies it to new scores:

        D_cal(x) = σ( logit(x) / T )

    *T* > 1 softens overconfident predictions; *T* < 1 sharpens
    underconfident ones.  *T* = 1 leaves the model unchanged.

    Parameters
    ----------
    T : float, optional
        Pre-fitted temperature.  If *None*, call ``fit()`` first.
    """

    def __init__(self, T: float | None = None):
        self.T: float | None = T

    @property
    def is_fitted(self) -> bool:
        return self.T is not None

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        yhat_val: np.ndarray,
        y_val: np.ndarray,
        T_bounds: tuple[float, float] = (0.1, 10.0),
    ) -> "TemperatureCalibrator":
        """Fit *T* on a held-out validation set.

        Parameters
        ----------
        yhat_val : np.ndarray  shape (N_val,)
            Raw predicted probabilities (before calibration) on validation set.
        y_val : np.ndarray  shape (N_val,)
            True binary labels for validation set.
        T_bounds : tuple[float, float]
            (lower, upper) bounds for the scalar optimisation.

        Returns
        -------
        self
        """
        logit_val = _prob_to_logit(yhat_val)
        eps = 1e-8

        def nll(T: float) -> float:
            t = max(T, 1e-6)
            calibrated = 1.0 / (1.0 + np.exp(-logit_val / t))
            calibrated = np.clip(calibrated, eps, 1.0 - eps)
            return -float(np.mean(
                y_val * np.log(calibrated)
                + (1.0 - y_val) * np.log(1.0 - calibrated)
            ))

        result = minimize_scalar(nll, bounds=T_bounds, method="bounded")
        self.T = float(result.x)
        return self

    # ------------------------------------------------------------------
    #  Apply
    # ------------------------------------------------------------------

    def transform(self, yhat: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw probabilities.

        Parameters
        ----------
        yhat : np.ndarray  shape (N,)
            Raw predicted probabilities.

        Returns
        -------
        np.ndarray  shape (N,)
            Calibrated probabilities.
        """
        if self.T is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        logits = _prob_to_logit(yhat)
        calibrated = 1.0 / (1.0 + np.exp(-logits / self.T))
        return np.clip(calibrated, 1e-8, 1.0 - 1e-8).astype(np.float64)

    def transform_logits(self, logits: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply temperature scaling directly to logits.

        This is the preferred method during MCMC since the classifier
        outputs raw logits and we want to avoid the double-sigmoid.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor  shape (N, 1) or (N,)
            Raw logits from the classifier.

        Returns
        -------
        Same type as input.
            Calibrated logits (divided by T).
        """
        if self.T is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        if isinstance(logits, np.ndarray):
            return logits / self.T
        else:
            # torch.Tensor
            return logits / self.T

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save T to a JSON file."""
        if self.T is None:
            raise RuntimeError("Calibrator not fitted. Nothing to save.")
        with open(path, "w") as f:
            json.dump({"method": "temperature", "T": self.T}, f)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Platt scaling (alternative calibration)
# ═══════════════════════════════════════════════════════════════════════

def calibrate_platt(
    yhat_val: np.ndarray,
    y_val: np.ndarray,
    yhat_test: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Platt scaling calibration (Platt 1999).

    Fits a logistic regression model on the validation logits,
    then applies it to test scores.

    Parameters
    ----------
    yhat_val : np.ndarray  shape (N_val,)
        Raw predicted probabilities on validation set.
    y_val : np.ndarray  shape (N_val,)
        True labels on validation set.
    yhat_test : np.ndarray  shape (N_test,)
        Raw predicted probabilities to calibrate.

    Returns
    -------
    calibrated : np.ndarray  shape (N_test,)
    metadata : dict  with keys "method", "coef", "intercept"
    """
    logit_val = _prob_to_logit(yhat_val).reshape(-1, 1)
    logit_test = _prob_to_logit(yhat_test).reshape(-1, 1)

    lr = LogisticRegression(C=np.inf, solver="lbfgs")
    lr.fit(logit_val, y_val)

    calibrated = lr.predict_proba(logit_test)[:, 1]
    metadata = {
        "method": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }
    return calibrated, metadata


# ═══════════════════════════════════════════════════════════════════════
# 4.  Method comparison driver
# ═══════════════════════════════════════════════════════════════════════

def compare_calibration_methods(
    yhat_val: np.ndarray,
    y_val: np.ndarray,
    yhat_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compare raw, isotonic, temperature, and Platt scaling on test set.

    Parameters
    ----------
    yhat_val, yhat_test : np.ndarray
        Raw predictions on validation / test sets.
    y_val, y_test : np.ndarray
        True labels.
    n_bins : int
        Number of bins for ECE.

    Returns
    -------
    dict
        Method name → {"ece": float, "brier": float, "scores": np.ndarray, ...}
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    results: dict = {}

    # Raw
    results["raw"] = {
        "ece": expected_calibration_error(y_test, yhat_test, n_bins),
        "brier": float(brier_score_loss(y_test, yhat_test)),
        "scores": yhat_test,
    }

    # Isotonic
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(yhat_val, y_val)
    cal_iso = ir.transform(yhat_test)
    results["isotonic"] = {
        "ece": expected_calibration_error(y_test, cal_iso, n_bins),
        "brier": float(brier_score_loss(y_test, cal_iso)),
        "scores": cal_iso,
    }

    # Temperature
    calib = TemperatureCalibrator()
    calib.fit(yhat_val, y_val)
    cal_temp = calib.transform(yhat_test)
    results["temperature"] = {
        "ece": expected_calibration_error(y_test, cal_temp, n_bins),
        "brier": float(brier_score_loss(y_test, cal_temp)),
        "scores": cal_temp,
        "T": calib.T,
    }

    # Platt
    cal_platt, platt_meta = calibrate_platt(yhat_val, y_val, yhat_test)
    results["platt"] = {
        "ece": expected_calibration_error(y_test, cal_platt, n_bins),
        "brier": float(brier_score_loss(y_test, cal_platt)),
        "scores": cal_platt,
        "coef": platt_meta["coef"],
        "intercept": platt_meta["intercept"],
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5.  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _prob_to_logit(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert probability scores to logits (inverse sigmoid)."""
    prob = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(prob / (1.0 - prob))


# Import needed for transform_logits type hint
import torch  # noqa: E402
