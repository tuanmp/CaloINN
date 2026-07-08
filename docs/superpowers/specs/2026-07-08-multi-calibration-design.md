# Multi-Method Calibration for MCMC/AR Density Ratio Estimation

**Date**: 2026-07-08
**Status**: Design approved
**Scope**: `src/mcmc/calibration.py`, `src/mcmc/sampler.py`, `src/ar/sampler.py`, `scripts/fit_calibration.py`

## Problem

The MCMC and AR samplers use a classifier as a density ratio estimator:
`r(x,c) = D(x,c) / (1 - D(x,c))`. Currently, only temperature scaling is
available to calibrate the classifier probabilities `D(x,c)`. The calibration
module (`src/mcmc/calibration.py`) already evaluates Platt scaling and isotonic
regression in the offline `compare_calibration_methods()` driver, but these
methods cannot be plugged into the sampling pipeline — the samplers hardcode
`TemperatureCalibrator` and access `self.calibrator.T` directly.

## Goals

1. Make all three calibration methods (temperature, Platt, isotonic) usable at
   sampling time, not just in offline evaluation.
2. Minimize changes to the sampler code — the calibrator should be a drop-in
   black box from the sampler's perspective.
3. Preserve GPU performance for the hot path (density ratio computation per
   MCMC step).

## Design

### 1. Abstract Base Class

A new ABC `BaseCalibrator` provides the unified interface. Every calibrator
must implement:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `transform_logits` | `(logits: Tensor) → Tensor` | Core interface — produces calibrated logits from raw classifier logits |
| `save` | `(path: str) → None` | Serialize to file |
| `load` | `(path: str) → BaseCalibrator` (classmethod) | Factory — detects method from file content and instantiates the correct subclass |
| `is_fitted` | `→ bool` (property) | Samplers check this instead of `calibrator.T is None` |

`transform_logits` returns calibrated logits. The sampler then applies sigmoid
and computes the density ratio uniformly, regardless of method:

```
calibrated_logits = calibrator.transform_logits(logits)
D_cal = sigmoid(calibrated_logits)
r = D_cal / (1 - D_cal + eps)
```

### 2. Calibrator Implementations

All three live in `src/mcmc/calibration.py`.

#### TemperatureCalibrator (existing, refactored)

Inherits from `BaseCalibrator`. Internal change: `transform_logits` becomes the
canonical method (returns `logits / T`). The existing `transform()` and
`transform_logits()` methods are consolidated. Serialization unchanged (JSON
with `"T"` key), with `"method": "temperature"` added.

#### PlattCalibrator (new)

Wraps the existing `calibrate_platt()` logic into a class.

- **Fit**: Logistic regression on validation logits → learns coefficients `a` (coef) and `b` (intercept).
- **transform_logits**: `a * logits + b` — a pure affine transform on logits.
- **Serialization**: `{"method": "platt", "a": 0.812, "b": -0.34}`

This is the simplest addition — affine transforms on logits correspond to
logistic regression on the probability scale, and the torch implementation is a
single `logits * a + b`.

#### IsotonicCalibrator (new)

Wraps `sklearn.isotonic.IsotonicRegression` with a dual-backend design.

- **Fit**: Uses sklearn `IsotonicRegression(out_of_bounds="clip")` fitted on
  raw probabilities (sigmoid of logits) vs true labels. Extracts
  `X_thresholds_` and `y_thresholds_` arrays.
- **Serialization**: Saves thresholds to `.npz` (binary, preserves float64
  precision for the piecewise interpolation).
- **transform_logits**: Two paths, selected at construction time:

  1. **Torch-native** (default for GPU): `sigmoid(logits)` → `searchsorted` +
     piecewise linear interpolation using the threshold arrays → `logit` to
     convert back to logit space → return.
  2. **sklearn fallback** (CPU/reference): `sigmoid(logits)` → move to numpy →
     `iso_reg.transform()` → `logit` → move back to torch.

  At fit time, the calibrator validates the torch-native path against the
  sklearn reference on a few random samples and warns (or falls back) on
  discrepancies.

### 3. Sampler Changes

Both `IMHSampler` and `ARSampler` change in exactly one place — the density
ratio computation:

**Before** (3 identical blocks in `_density_ratio_numpy`, `_density_ratio_torch`, `_density_ratio`):
```python
T = self.calibrator.T
D_cal = torch.sigmoid(logits / T)
r = D_cal / (1.0 - D_cal + eps)
```

**After**:
```python
calibrated_logits = self.calibrator.transform_logits(logits)
D_cal = torch.sigmoid(calibrated_logits)
r = D_cal / (1.0 - D_cal + eps)
```

Constructor changes:
- Type annotation: `TemperatureCalibrator` → `BaseCalibrator`
- Validation: `calibrator.T is None` → `not calibrator.is_fitted`

No other sampler code changes. Proposal, MCMC acceptance/rejection, burn-in,
thinning, profiling, and the AR resampling loop are untouched.

### 4. Fit Script Changes (`scripts/fit_calibration.py`)

New `--method` flag:

```
--method temperature   Save only temperature calibrator (default, backward-compatible)
--method platt         Save only Platt calibrator
--method isotonic      Save only isotonic calibrator
--method all           Save all four (raw excluded) to separate files
```

Files produced by `--method all`:
```
{output}_temperature.json
{output}_platt.json
{output}_isotonic.npz
{output}.comparison.json       (existing sidecar, unchanged)
```

The existing `compare_calibration_methods()` call is always run (its ECE/Brier
table is useful regardless), and the chosen method's calibrator is fitted and
saved afterward.

### 5. File-Level Plan

| File | Change | Approx. lines |
|------|--------|---------------|
| `src/mcmc/calibration.py` | Add `BaseCalibrator` ABC; refactor `TemperatureCalibrator` to inherit; add `PlattCalibrator`, `IsotonicCalibrator` classes | +160 |
| `src/mcmc/sampler.py` | Type annotation change; 3 blocks of `T / logits` → `transform_logits` | ~6 changed |
| `src/ar/sampler.py` | Same pattern as sampler.py | ~4 changed |
| `scripts/fit_calibration.py` | Add `--method` arg; save logic per method | +30 |

**No new files.** All classes added to the existing `calibration.py`.

### 6. Backward Compatibility

- `TemperatureCalibrator` retains its existing public API (`fit`, `transform`,
  `transform_logits`, `save`, `load`) — only internal implementation moves to
  inheritance.
- Existing temperature calibration JSON files (`{"T": 1.234}`) remain loadable
  by `BaseCalibrator.load()` (missing `"method"` key implies temperature).
- Samplers that pass a `TemperatureCalibrator` instance continue to work —
  they're now passing a `BaseCalibrator` subclass.
- `compare_calibration_methods()` is unchanged.

### 7. Testing Strategy

No dedicated test file — calibration correctness is validated through the
sampler parity tests (`tests/test_lightning_parity.py`):

- Run MCMC with each calibrator method (temperature/platt/isotonic) at a single
  energy and verify output shapes, ratio ranges, and acceptance rates are
  within expected bounds.
- For isotonic, add a unit-level assertion that the torch-native path matches
  sklearn output to within `1e-6` relative tolerance on a fixed set of logits.
- The isotonic torch implementation gets a standalone `pytest` parametrized
  test with known-answer threshold arrays.

## Decisions Recorded

| Question | Decision |
|----------|----------|
| Calibrator selection workflow | Separate files per method; user passes the file path |
| Isotonic implementation | Torch-native with sklearn fallback |
| Interface style | Abstract `BaseCalibrator` class |
