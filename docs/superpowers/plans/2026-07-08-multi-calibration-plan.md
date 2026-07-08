# Multi-Method Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Platt scaling and isotonic regression as calibration methods for MCMC/AR density ratio estimation, alongside the existing temperature scaling.

**Architecture:** Introduce a `BaseCalibrator` ABC with `transform_logits(logits) → calibrated_logits` as the unifying interface. Refactor the existing `TemperatureCalibrator` to inherit from it. Add `PlattCalibrator` (affine logit transform via sklearn logistic regression) and `IsotonicCalibrator` (piecewise interpolation with dual torch-native/sklearn backends). Samplers call `calibrator.transform_logits(logits)` instead of hardcoding `logits / T`.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn (IsotonicRegression, LogisticRegression), scipy (minimize_scalar), numpy

---

### Task 1: Define BaseCalibrator ABC and refactor TemperatureCalibrator

**Files:**
- Modify: `src/mcmc/calibration.py`

- [ ] **Step 1: Add the ABC import and BaseCalibrator class before TemperatureCalibrator**

Insert after the existing imports (line 21) and before the ECE section:

```python
from abc import ABC, abstractmethod
```

Insert the BaseCalibrator class just before the `TemperatureCalibrator` class definition (before line 83):

```python
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
            return PlattCalibrator(a=float(data["a"]), b=float(data["b"]))
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"
```

- [ ] **Step 2: Refactor TemperatureCalibrator to inherit from BaseCalibrator**

Change the class declaration (line 83) from:
```python
class TemperatureCalibrator:
```
to:
```python
class TemperatureCalibrator(BaseCalibrator):
```

Add `is_fitted` property to `TemperatureCalibrator` (after `__init__`, before `fit`):

```python
    @property
    def is_fitted(self) -> bool:
        return self.T is not None
```

Update `save` method (line 196) to include the `"method"` key. Change:
```python
            json.dump({"T": self.T}, f)
```
to:
```python
            json.dump({"method": "temperature", "T": self.T}, f)
```

Remove the existing `load` classmethod (lines 203-208) — it's now handled by `BaseCalibrator.load()`.

Remove the existing `__repr__` (lines 210-212) — the base class handles it.

- [ ] **Step 3: Verify the file is syntactically valid**

```bash
uv run python -c "from mcmc.calibration import BaseCalibrator, TemperatureCalibrator; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/mcmc/calibration.py
git commit -m "refactor: add BaseCalibrator ABC, make TemperatureCalibrator a subclass"
```

---

### Task 2: Add PlattCalibrator class

**Files:**
- Modify: `src/mcmc/calibration.py`

- [ ] **Step 1: Add PlattCalibrator class**

Insert after `TemperatureCalibrator` (after its `__repr__`), before the existing `calibrate_platt` function:

```python
# ═══════════════════════════════════════════════════════════════════════
# 3.  Platt scaling calibrator
# ═══════════════════════════════════════════════════════════════════════

class PlattCalibrator(BaseCalibrator):
    """Platt scaling calibration (Platt 1999).

    Fits a logistic regression on validation logits::

        D_cal(x) = σ(a · logit(x) + b)

    At inference, ``transform_logits`` returns ``a * logits + b``, which
    is directly usable as calibrated logits.

    Parameters
    ----------
    a : float, optional
        Logistic regression coefficient (slope).
    b : float, optional
        Logistic regression intercept.
    """

    def __init__(self, a: float | None = None, b: float | None = None):
        self._a = a
        self._b = b

    @property
    def is_fitted(self) -> bool:
        return self._a is not None and self._b is not None

    @property
    def a(self) -> float:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted.")
        return self._a

    @property
    def b(self) -> float:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted.")
        return self._b

    def fit(
        self,
        yhat_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "PlattCalibrator":
        """Fit Platt scaling on validation data.

        Parameters
        ----------
        yhat_val : np.ndarray  shape (N_val,)
            Raw predicted probabilities on validation set.
        y_val : np.ndarray  shape (N_val,)
            True binary labels.

        Returns
        -------
        self
        """
        logit_val = _prob_to_logit(yhat_val).reshape(-1, 1)

        lr = LogisticRegression(C=np.inf, solver="lbfgs")
        lr.fit(logit_val, y_val)

        self._a = float(lr.coef_[0, 0])
        self._b = float(lr.intercept_[0])
        return self

    def transform_logits(self, logits):
        """Apply Platt scaling to logits: ``a * logits + b``.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor
            Raw classifier logits.

        Returns
        -------
        Same type as input.
            Calibrated logits.
        """
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted. Call fit() first.")

        if isinstance(logits, np.ndarray):
            return self._a * logits + self._b
        else:
            return self._a * logits + self._b

    def save(self, path: str | Path) -> None:
        """Save to JSON."""
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted.")
        with open(path, "w") as f:
            json.dump({"method": "platt", "a": self._a, "b": self._b}, f)
```

- [ ] **Step 2: Verify the module loads**

```bash
uv run python -c "from mcmc.calibration import PlattCalibrator; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mcmc/calibration.py
git commit -m "feat: add PlattCalibrator class"
```

---

### Task 3: Add IsotonicCalibrator class

**Files:**
- Modify: `src/mcmc/calibration.py`

- [ ] **Step 1: Add IsotonicCalibrator class**

Insert after `PlattCalibrator`, before the existing `calibrate_platt` function:

```python
# ═══════════════════════════════════════════════════════════════════════
# 4.  Isotonic regression calibrator
# ═══════════════════════════════════════════════════════════════════════

class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration.

    Fits a non-parametric monotonic function on validation probabilities.
    Supports two backends:

    - **Torch-native** (default): pure-PyTorch piecewise linear
      interpolation using ``X_thresholds_`` / ``y_thresholds_``.  Stays
      on GPU, zero CPU transfers.
    - **sklearn fallback**: uses the fitted ``IsotonicRegression``
      object for correctness reference.

    At inference, ``transform_logits`` does::

        probs = σ(logits)
        cal_probs = isotonic_fn(probs)
        return logit(cal_probs)

    Parameters
    ----------
    use_torch : bool
        If True (default), use the torch-native interpolation path.
        If False, use sklearn ``.transform()``.
    """

    def __init__(self, use_torch: bool = True):
        self._iso_reg = None
        self._X_thresholds = None
        self._y_thresholds = None
        self._use_torch = use_torch
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        yhat_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression on validation probabilities.

        Parameters
        ----------
        yhat_val : np.ndarray  shape (N_val,)
            Raw predicted probabilities.
        y_val : np.ndarray  shape (N_val,)
            True binary labels.

        Returns
        -------
        self
        """
        self._iso_reg = IsotonicRegression(out_of_bounds="clip")
        self._iso_reg.fit(yhat_val.ravel(), y_val.ravel())

        # Extract thresholds for torch-native path
        self._X_thresholds = torch.from_numpy(
            self._iso_reg.X_thresholds_.copy()
        ).float()
        self._y_thresholds = torch.from_numpy(
            self._iso_reg.y_thresholds_.copy()
        ).float()
        self._fitted = True

        # Validate torch-native against sklearn on a small probe set
        if self._use_torch and len(yhat_val) > 0:
            probe = torch.linspace(0.0, 1.0, 101)
            torch_out = self._isotonic_transform_torch(probe).numpy()
            sklearn_out = self._iso_reg.transform(probe.numpy().reshape(-1, 1)).ravel()
            max_err = np.abs(torch_out - sklearn_out).max()
            if max_err > 1e-5:
                import warnings
                warnings.warn(
                    f"IsotonicCalibrator: torch vs sklearn max error = {max_err:.2e}. "
                    f"Consider use_torch=False for this calibrator."
                )
        return self

    def transform_logits(self, logits):
        """Apply isotonic calibration to logits.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor
            Raw classifier logits.

        Returns
        -------
        Same type as input.
            Calibrated logits: ``logit( f(σ(logits)) )``.
        """
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator not fitted. Call fit() first.")

        # Work in torch space for consistency; convert numpy inputs
        is_numpy = isinstance(logits, np.ndarray)
        if is_numpy:
            logits = torch.from_numpy(np.asarray(logits, dtype=np.float64))

        probs = torch.sigmoid(logits)

        if self._use_torch:
            cal_probs = self._isotonic_transform_torch(probs)
        else:
            cal_probs = self._isotonic_transform_sklearn(probs)

        # Convert calibrated probabilities back to logits
        eps = 1e-8
        cal_probs = torch.clamp(cal_probs, eps, 1.0 - eps)
        calibrated_logits = torch.log(cal_probs / (1.0 - cal_probs))

        if is_numpy:
            return calibrated_logits.numpy()
        return calibrated_logits

    def _isotonic_transform_torch(self, probs: "torch.Tensor") -> "torch.Tensor":
        """Torch-native piecewise linear interpolation.

        Mirrors sklearn's IsotonicRegression.transform() using
        searchsorted + linear interpolation within each bin.
        """
        X = self._X_thresholds.to(probs.device)
        Y = self._y_thresholds.to(probs.device)
        n_thresh = len(X)

        if n_thresh == 0:
            return probs

        # Find the right bin for each probability
        idx = torch.searchsorted(X, probs)
        idx = idx.clamp(1, n_thresh - 1)

        x_lo = X[idx - 1]
        x_hi = X[idx]
        y_lo = Y[idx - 1]
        y_hi = Y[idx]

        # Linear interpolation: y = y_lo + (y_hi - y_lo) * (x - x_lo) / (x_hi - x_lo)
        t = (probs - x_lo) / (x_hi - x_lo + 1e-10)
        t = t.clamp(0.0, 1.0)
        return y_lo + t * (y_hi - y_lo)

    def _isotonic_transform_sklearn(self, probs: "torch.Tensor") -> "torch.Tensor":
        """sklearn fallback: CPU roundtrip through IsotonicRegression.transform."""
        probs_np = probs.cpu().numpy().reshape(-1, 1)
        cal_np = self._iso_reg.transform(probs_np).ravel()
        return torch.from_numpy(cal_np).to(probs.device)

    def save(self, path: str | Path) -> None:
        """Save thresholds to .npz."""
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator not fitted.")
        np.savez(
            path,
            X_thresholds=self._iso_reg.X_thresholds_,
            y_thresholds=self._iso_reg.y_thresholds_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        """Load thresholds from .npz."""
        # Delegated to BaseCalibrator.load() via factory
        return BaseCalibrator.load(path)
```

- [ ] **Step 2: Add sklearn import at top of file**

If not already present, ensure `from sklearn.isotonic import IsotonicRegression` is at the top. The existing `compare_calibration_methods` already does a local import. Move it to a module-level import:

At line 21 (after `from sklearn.linear_model import LogisticRegression`), add:
```python
from sklearn.isotonic import IsotonicRegression
```

Then remove the local import inside `compare_calibration_methods` (line 285).

- [ ] **Step 3: Verify module loads and basic functionality**

```bash
uv run python -c "
import numpy as np
from mcmc.calibration import IsotonicCalibrator
# Quick smoke test
np.random.seed(42)
probs = np.random.rand(100).astype(np.float32)
labels = (probs > 0.5).astype(np.float32)
cal = IsotonicCalibrator(use_torch=True)
cal.fit(probs, labels)
out = cal.transform_logits(np.array([0.3, 0.5, 0.7]))
print('Isotonic output shape:', out.shape)
print('OK')
"
```

Expected: `Isotonic output shape: (3,)` then `OK`

- [ ] **Step 4: Commit**

```bash
git add src/mcmc/calibration.py
git commit -m "feat: add IsotonicCalibrator with torch-native + sklearn backends"
```

---

### Task 4: Update IMHSampler to use BaseCalibrator

**Files:**
- Modify: `src/mcmc/sampler.py`

- [ ] **Step 1: Change import and type annotation**

Line 21: Change import from:
```python
from .calibration import TemperatureCalibrator
```
to:
```python
from .calibration import BaseCalibrator
```

Line 55: Change constructor parameter type from:
```python
calibrator: TemperatureCalibrator,
```
to:
```python
calibrator: BaseCalibrator,
```

- [ ] **Step 2: Change constructor validation**

Lines 69-70: Change from:
```python
if calibrator.T is None:
    raise ValueError("Calibrator must be fitted before use.")
```
to:
```python
if not calibrator.is_fitted:
    raise ValueError("Calibrator must be fitted before use.")
```

- [ ] **Step 3: Update _density_ratio_numpy (lines 464-468)**

Change:
```python
        # Density ratio
        t4 = time.perf_counter() if _profile_return else 0
        D_cal = torch.sigmoid(logits / temp)               # (N,)
        r = D_cal / (1.0 - D_cal + eps)                   # (N,)
        r = torch.clamp(r, 1.0 / self._r_clip, self._r_clip)
```
to:
```python
        # Density ratio
        t4 = time.perf_counter() if _profile_return else 0
        calibrated_logits = self.calibrator.transform_logits(logits)
        D_cal = torch.sigmoid(calibrated_logits)           # (N,)
        r = D_cal / (1.0 - D_cal + eps)                   # (N,)
        r = torch.clamp(r, 1.0 / self._r_clip, self._r_clip)
```

Also remove the `temp = self.calibrator.T` line (line 484 in the current file, or wherever it appears before the density ratio block). In `_density_ratio_numpy`, this variable is accessed but may be defined near the top of the method or inside. Search and remove any `temp = self.calibrator.T` lines.

- [ ] **Step 4: Update _density_ratio_torch (lines 503-506)**

Change:
```python
        # Density ratio
        t2 = time.perf_counter() if _profile_return else 0
        D_cal = torch.sigmoid(logits / temp)               # (N,)
        r = D_cal / (1.0 - D_cal + eps)                   # (N,)
        r = torch.clamp(r, 1.0 / self._r_clip, self._r_clip)
```
to:
```python
        # Density ratio
        t2 = time.perf_counter() if _profile_return else 0
        calibrated_logits = self.calibrator.transform_logits(logits)
        D_cal = torch.sigmoid(calibrated_logits)           # (N,)
        r = D_cal / (1.0 - D_cal + eps)                   # (N,)
        r = torch.clamp(r, 1.0 / self._r_clip, self._r_clip)
```

Also remove the `temp = self.calibrator.T` line at the top of `_density_ratio_torch` (line 484).

- [ ] **Step 5: Verify syntax**

```bash
uv run python -c "from mcmc.sampler import IMHSampler; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/mcmc/sampler.py
git commit -m "refactor: IMHSampler uses BaseCalibrator.transform_logits instead of hardcoded T"
```

---

### Task 5: Update ARSampler to use BaseCalibrator

**Files:**
- Modify: `src/ar/sampler.py`

- [ ] **Step 1: Change import**

Line 22: Change from:
```python
from mcmc.calibration import TemperatureCalibrator
```
to:
```python
from mcmc.calibration import BaseCalibrator
```

- [ ] **Step 2: Change type annotation and validation**

Line 41: Change from:
```python
calibrator: TemperatureCalibrator,
```
to:
```python
calibrator: BaseCalibrator,
```

Lines 55-56: Change from:
```python
if calibrator.T is None:
    raise ValueError("Calibrator must be fitted before use.")
```
to:
```python
if not calibrator.is_fitted:
    raise ValueError("Calibrator must be fitted before use.")
```

- [ ] **Step 3: Update _density_ratio method (lines 144-149)**

Change:
```python
        # -- Compute density ratios ------------------------------------------
        T = self.calibrator.T
        eps = 1e-10  # small constant to avoid division by zero

        logits = self.classifier.forward(x_clf).squeeze(-1)  # shape (N,)
        D_cal = torch.sigmoid(logits / T)
        r = D_cal / (1 - D_cal + eps)  # density ratio r(x,c) = p(x|c)/q(x|c)
        return r
```
to:
```python
        # -- Compute density ratios ------------------------------------------
        eps = 1e-10  # small constant to avoid division by zero

        logits = self.classifier.forward(x_clf).squeeze(-1)  # shape (N,)
        calibrated_logits = self.calibrator.transform_logits(logits)
        D_cal = torch.sigmoid(calibrated_logits)
        r = D_cal / (1 - D_cal + eps)  # density ratio r(x,c) = p(x|c)/q(x|c)
        return r
```

- [ ] **Step 4: Verify syntax**

```bash
uv run python -c "from ar.sampler import ARSampler; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/ar/sampler.py
git commit -m "refactor: ARSampler uses BaseCalibrator.transform_logits instead of hardcoded T"
```

---

### Task 6: Write unit tests for calibrators

**Files:**
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write the test file**

```python
"""Unit tests for calibration classes."""
import numpy as np
import torch
import pytest
import json
import tempfile
from pathlib import Path

from mcmc.calibration import (
    BaseCalibrator,
    TemperatureCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    expected_calibration_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibration_data():
    """Generate synthetic calibration data.

    Returns raw probabilities and labels where the classifier is
    overconfident (raw probs are too extreme compared to true labels).
    """
    rng = np.random.RandomState(42)
    n = 1000

    # Simulate: classifier outputs overly confident probabilities
    raw_probs = rng.beta(2, 2, size=n)  # centered around 0.5, but some extremes

    # True labels: regress toward 0.5 (well-calibrated would match raw_probs exactly)
    # Shift toward 0.5 to create miscalibration
    true_labels = (rng.rand(n) < (raw_probs * 0.6 + 0.2)).astype(np.float64)

    return raw_probs, true_labels


# ---------------------------------------------------------------------------
# TemperatureCalibrator
# ---------------------------------------------------------------------------

class TestTemperatureCalibrator:
    def test_fit_reduces_nll(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)
        assert cal.is_fitted
        assert cal.T > 0

    def test_transform_logits_numpy(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)

        logits = np.log(raw_probs / (1 - raw_probs + 1e-8))
        out = cal.transform_logits(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == logits.shape
        # Temperature scaling on logits should be logits / T
        np.testing.assert_allclose(out, logits / cal.T, rtol=1e-6)

    def test_transform_logits_torch(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)

        logits = torch.from_numpy(np.log(raw_probs / (1 - raw_probs + 1e-8)))
        out = cal.transform_logits(logits)
        assert isinstance(out, torch.Tensor)
        assert out.shape == logits.shape
        torch.testing.assert_close(out, logits / cal.T)

    def test_save_load_roundtrip(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, TemperatureCalibrator)
            assert loaded.T == cal.T
        finally:
            Path(path).unlink()

    def test_legacy_json_load(self):
        """Load a JSON without 'method' key (legacy format)."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"T": 2.5}, f)
            path = f.name
        try:
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, TemperatureCalibrator)
            assert loaded.T == 2.5
        finally:
            Path(path).unlink()

    def test_unfitted_raises(self):
        cal = TemperatureCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform_logits(np.array([0.0]))

    def test_save_unfitted_raises(self):
        cal = TemperatureCalibrator()
        with pytest.raises(RuntimeError):
            cal.save("/tmp/unused.json")


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------

class TestPlattCalibrator:
    def test_fit(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)
        assert cal.is_fitted
        assert isinstance(cal.a, float)
        assert isinstance(cal.b, float)

    def test_transform_logits_numpy(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)

        logits = np.log(raw_probs / (1 - raw_probs + 1e-8)).astype(np.float64)
        out = cal.transform_logits(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == logits.shape
        np.testing.assert_allclose(out, cal.a * logits + cal.b, rtol=1e-6)

    def test_transform_logits_torch(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)

        logits = torch.from_numpy(np.log(raw_probs / (1 - raw_probs + 1e-8)))
        out = cal.transform_logits(logits)
        assert isinstance(out, torch.Tensor)
        assert out.shape == logits.shape

    def test_save_load_roundtrip(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, PlattCalibrator)
            assert loaded._a == cal._a
            assert loaded._b == cal._b
        finally:
            Path(path).unlink()

    def test_unfitted_raises(self):
        cal = PlattCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform_logits(np.array([0.0]))


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_fit(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)
        assert cal.is_fitted
        assert cal._iso_reg is not None
        assert cal._X_thresholds is not None
        assert cal._y_thresholds is not None

    def test_torch_matches_sklearn_on_simple_case(self):
        """Torch-native interpolation should match sklearn exactly."""
        # Create a simple monotonic calibration scenario
        rng = np.random.RandomState(42)
        raw_probs = np.sort(rng.rand(500).astype(np.float64))
        labels = (rng.rand(500) < raw_probs).astype(np.float64)

        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        # Test on a dense grid of probabilities
        test_probs = np.linspace(0.0, 1.0, 101)
        logits = np.log(test_probs / (1 - test_probs + 1e-8))

        # Torch path
        torch_out = cal._isotonic_transform_torch(torch.from_numpy(test_probs))

        # Sklearn path
        sklearn_out = cal._iso_reg.transform(test_probs.reshape(-1, 1)).ravel()

        np.testing.assert_allclose(torch_out.numpy(), sklearn_out, rtol=1e-5, atol=1e-7)

    def test_torch_matches_sklearn_on_calibration_data(self, calibration_data):
        """Torch vs sklearn on realistic calibration data."""
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        test_probs = np.linspace(0.01, 0.99, 200)
        torch_out = cal._isotonic_transform_torch(torch.from_numpy(test_probs))
        sklearn_out = cal._iso_reg.transform(test_probs.reshape(-1, 1)).ravel()

        max_err = np.abs(torch_out.numpy() - sklearn_out).max()
        assert max_err < 1e-5, f"Max error {max_err:.2e} exceeds tolerance"

    def test_transform_logits_torch_path(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        logits = torch.tensor([-2.0, 0.0, 2.0])  # sigmoid: ~0.12, 0.5, ~0.88
        out = cal.transform_logits(logits)
        assert isinstance(out, torch.Tensor)
        assert out.shape == logits.shape
        assert torch.isfinite(out).all()

    def test_transform_logits_sklearn_path(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=False)
        cal.fit(raw_probs, labels)

        logits = torch.tensor([-2.0, 0.0, 2.0])
        out = cal.transform_logits(logits)
        assert out.shape == logits.shape
        assert torch.isfinite(out).all()

    def test_torch_and_sklearn_paths_agree(self, calibration_data):
        """transform_logits should produce identical output regardless of backend."""
        raw_probs, labels = calibration_data
        cal_torch = IsotonicCalibrator(use_torch=True)
        cal_sklearn = IsotonicCalibrator(use_torch=False)
        cal_torch.fit(raw_probs, labels)
        cal_sklearn.fit(raw_probs, labels)

        logits = torch.linspace(-3.0, 3.0, 50)
        out_torch = cal_torch.transform_logits(logits)
        out_sklearn = cal_sklearn.transform_logits(logits)

        torch.testing.assert_close(out_torch, out_sklearn, rtol=1e-5, atol=1e-6)

    def test_numpy_input(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        logits = np.array([-2.0, 0.0, 2.0])
        out = cal.transform_logits(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_save_load_roundtrip(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, IsotonicCalibrator)
            assert loaded.is_fitted

            # Verify transform gives same output
            test_logits = torch.linspace(-3.0, 3.0, 20)
            original_out = cal.transform_logits(test_logits)
            loaded_out = loaded.transform_logits(test_logits)
            torch.testing.assert_close(original_out, loaded_out)
        finally:
            Path(path).unlink()

    def test_unfitted_raises(self):
        cal = IsotonicCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform_logits(np.array([0.0]))


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration(self):
        """Perfectly calibrated: prob = label."""
        n = 1000
        rng = np.random.RandomState(0)
        probs = rng.rand(n).astype(np.float64)
        labels = (rng.rand(n) < probs).astype(np.float64)
        ece = expected_calibration_error(labels, probs, n_bins=10)
        assert ece < 0.1  # should be very low for well-calibrated

    def test_miscalibrated(self):
        """Overconfident predictions should have higher ECE."""
        n = 1000
        y_true = np.ones(n, dtype=np.float64)
        y_pred = np.full(n, 0.6, dtype=np.float64)  # always 0.6, but labels are all 1
        ece = expected_calibration_error(y_true, y_pred, n_bins=10)
        assert ece > 0.3  # heavily miscalibrated → high ECE

    def test_empty_input(self):
        ece = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0
```

- [ ] **Step 2: Run the tests**

```bash
HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/test_calibration.py -v
```

Expected: All tests pass (20-22 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_calibration.py
git commit -m "test: add unit tests for calibrator classes"
```

---

### Task 7: Update fit_calibration.py with --method flag

**Files:**
- Modify: `scripts/fit_calibration.py`

- [ ] **Step 1: Add --method argument to parse_args()**

After line 79 (`--max-val-samples` block), add:

```python
    parser.add_argument(
        "--method",
        choices=["temperature", "platt", "isotonic", "all"],
        default="temperature",
        help="Calibration method to save (default: temperature). "
             "Use 'all' to save all methods to separate files.",
    )
```

- [ ] **Step 2: Replace the save logic at the end of main()**

Replace lines 168-189 (the section starting with `# -- Save best temperature calibrator ---`) with:

```python
    # -- Save calibrators based on --method ----------------------------------
    output_base = Path(args.output)
    saved = []

    if args.method in ("temperature", "all"):
        calib = TemperatureCalibrator()
        calib.fit(yhat_fit, y_fit)
        path = output_base if args.method == "temperature" else output_base.with_stem(f"{output_base.stem}_temperature")
        calib.save(str(path))
        saved.append(("temperature", str(path), calib.T))
        print(f"💾 Temperature calibrator (T={calib.T:.4f}) → {path}")

    if args.method in ("platt", "all"):
        path = output_base if args.method == "platt" else output_base.with_stem(f"{output_base.stem}_platt")
        platt = PlattCalibrator()
        platt.fit(yhat_fit, y_fit)
        platt.save(str(path))
        saved.append(("platt", str(path), platt))
        print(f"💾 Platt calibrator (a={platt.a:.3f}, b={platt.b:.3f}) → {path}")

    if args.method in ("isotonic", "all"):
        if args.method == "all":
            path = output_base.with_stem(f"{output_base.stem}_isotonic").with_suffix(".npz")
        else:
            path = output_base.with_suffix(".npz")
        iso = IsotonicCalibrator(use_torch=True)
        iso.fit(yhat_fit, y_fit)
        iso.save(str(path))
        saved.append(("isotonic", str(path), iso))
        print(f"💾 Isotonic calibrator ({len(iso._X_thresholds)} thresholds) → {path}")
```

Also update the import at line 44-47 to include the new classes:
```python
from mcmc.calibration import (
    TemperatureCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    compare_calibration_methods,
)
```

- [ ] **Step 3: Verify syntax**

```bash
uv run python -c "
import ast
with open('scripts/fit_calibration.py') as f:
    ast.parse(f.read())
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Quick smoke test with synthetic data (no dataset needed)**

```bash
uv run python -c "
import numpy as np
import tempfile, json
from pathlib import Path
from mcmc.calibration import TemperatureCalibrator, PlattCalibrator, IsotonicCalibrator, BaseCalibrator

rng = np.random.RandomState(42)
probs = rng.rand(200).astype(np.float64)
labels = (rng.rand(200) < (probs * 0.7 + 0.15)).astype(np.float64)  # miscalibrated

# Test temperature roundtrip
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    t_path = f.name
cal = TemperatureCalibrator(); cal.fit(probs, labels); cal.save(t_path)
loaded = BaseCalibrator.load(t_path)
assert isinstance(loaded, TemperatureCalibrator) and loaded.T == cal.T
Path(t_path).unlink()

# Test platt roundtrip
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    p_path = f.name
cal = PlattCalibrator(); cal.fit(probs, labels); cal.save(p_path)
loaded = BaseCalibrator.load(p_path)
assert isinstance(loaded, PlattCalibrator) and loaded._a == cal._a
Path(p_path).unlink()

# Test isotonic roundtrip
with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
    i_path = f.name
cal = IsotonicCalibrator(); cal.fit(probs, labels); cal.save(i_path)
loaded = BaseCalibrator.load(i_path)
assert isinstance(loaded, IsotonicCalibrator)
Path(i_path).unlink()

print('All roundtrip tests passed')
"
```

Expected: `All roundtrip tests passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/fit_calibration.py
git commit -m "feat: add --method flag to fit_calibration.py for platt/isotonic/temperature/all"
```

---

### Task 8: Run existing parity tests to verify no regressions

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: All existing tests pass with no regressions. Calibration tests from Task 6 also pass.

- [ ] **Step 2: (Optional) Run specific calibration tests**

```bash
HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/test_calibration.py -v
```

Expected: All 20-22 calibration tests pass.

- [ ] **Step 3: No commit needed (verification only)**

---
