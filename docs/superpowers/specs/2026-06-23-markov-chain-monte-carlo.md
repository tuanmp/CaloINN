# MCMC Density Ratio Reweighting — Implementation Plan

**Date:** 2026-06-23  
**Status:** Design complete, awaiting classifier checkpoint  
**Branch:** `feature/mcmc-density-ratio`

---

## 1. Overview

### 1.1 Goal

Correct CINN-generated showers toward the true Geant4 distribution using Independent Metropolis-Hastings (IMH) MCMC with a classifier-based density ratio estimator.

### 1.2 Key Insight

The CINN provides both a proposal distribution `q(x|c)` (via `model.sample()`) and exact log-density `log q(x|c)` (via `model.log_prob()`). A classifier `D(x,c)` trained to distinguish real from generated showers gives the density ratio:

$$r(x,c) = \frac{p(x|c)}{q(x|c)} = \frac{D(x,c)}{1 - D(x,c)}$$

IMH then uses this ratio for acceptance without needing the true density `p(x|c)` or a bounding constant on `r`:

$$\alpha = \min\!\left(1, \frac{r(x',c)}{r(x,c)}\right)$$

### 1.3 Why MCMC Over Importance Sampling

| Approach | Output | Problem |
|----------|--------|---------|
| Importance sampling | Weighted samples | A few samples with huge weights dominate → low effective sample size |
| **IMH** | **Unweighted samples** | Rejects low-ratio proposals, keeps high-ratio ones → naturally upweights high-p regions |

---

## 2. Architecture

### 2.1 Data Flow (End-to-End)

```
CINN sample (730D internal)
    │  model.sample(1, c)
    ▼
Postprocess (data_util.postprocess)
    │  → dict: {"energy": E_GeV, "layer_0": ..., "layer_9": ...}
    ▼
Convert to classifier input
    │  1. Concatenate layers → X_MeV (720 cells, in MeV)
    │  2. Einc_MeV = energy * 1e3
    │  3. X_proc = X_MeV / Einc_MeV                    [scale_shower]
    │  4. cond_proc = Einc_MeV / 1000.0                  [MeV→GeV]
    │  5. X_hlf = get_high_level_features(X_MeV, Einc_MeV)  [51 HLF dims]
    │  6. concat([X_proc, cond_proc, X_hlf]) → 772D
    ▼
Classifier forward → raw logits
    ▼
Temperature calibrate: logits / T → sigmoid → D_cal(x,c)
    ▼
r(x,c) = D_cal / (1 - D_cal + 1e-8)
    ▼
MH step: α = min(1, r(x',c) / r(x,c)) → accept/reject
```

### 2.2 File Layout

```
caloinn_lightning/                          (this repo)
├── src/
│   ├── mcmc/
│   │   ├── __init__.py                     # Public interface
│   │   ├── sampler.py                      # IMHSampler class
│   │   ├── classifier.py                   # MLP + MLPClassifier (ported)
│   │   ├── calibration.py                  # Temperature scaling + ECE (ported)
│   │   └── convert.py                      # CINN internal → classifier input
│   └── lightning_module.py                 # MODIFY: add mcmc_sample()
├── scripts/
│   └── fit_calibration.py                  # Fit T on validation set
├── tests/
│   └── test_mcmc_sampler.py                # Unit + integration tests
└── docs/superpowers/specs/
    └── 2026-06-23-markov-chain-monte-carlo.md  # This document

caloxtreme_clf/                             (source of ported code)
├── module/classifier.py                    # → src/mcmc/classifier.py
├── module/lightning.py                     # → MLPClassifier class extracted
├── calibration/__init__.py                 # → src/mcmc/calibration.py
└── data/HighLevelFeatures.py               # → copy to src/mcmc/hlf.py
```

### 2.3 Key Classes

#### `IMHSampler` (`src/mcmc/sampler.py`)

```python
class IMHSampler:
    """Independent Metropolis-Hastings sampler with density ratio correction."""

    def __init__(self, model, classifier, calibrator, conversion_fn, device):
        self.model = model            # CINN
        self.clf = classifier         # MLPClassifier (ported)
        self.calibrator = calibrator  # Temperature scaling params
        self.convert = conversion_fn  # CINN output → classifier input
        self.device = device

    @torch.inference_mode()
    def sample(self, energies, n_chains=100, n_steps=500,
               burn_in=100, thin=5):
        """Run IMH and return post-burn-in, thinned samples.

        Args:
            energies: (N, 1) incident energies in GeV
            n_chains: Number of independent chains
            n_steps: MH steps per chain (including burn-in)
            burn_in: Number of initial steps to discard
            thin: Keep every thin-th sample

        Returns:
            samples: (n_kept, N, 730) CINN-internal samples
            acceptance_rate: float
            density_ratios: (n_kept,) calibrated density ratios
        """
        ...

    def _step(self, x_current, c):
        """Single MH step: propose → evaluate → accept/reject.

        Returns:
            x_next: accepted proposal or current state
            accepted: bool tensor
            r_value: density ratio of accepted state
        """
        ...
```

#### `DensityRatioConverter` (`src/mcmc/convert.py`)

Pure function that replicates the classifier's preprocessing pipeline:

```python
def cinn_sample_to_classifier_input(
    x_internal, c, layer_boundaries, q, width_noise,
    xml_path, particle
) -> np.ndarray:
    """Convert CINN internal representation to classifier input.

    x_internal: (N, 730) — CINN internal (720 cells + 10 extra dims)
    c: (N, 1) — incident energy in GeV

    Returns: (N, 772) — [X_proc(720), cond_proc(1), X_hlf(51)]
    """
    ...
```

---

## 3. Calibration

### 3.1 Why Temperature Scaling

Without calibration, the raw classifier outputs D(x) tend toward 0 or 1 for most inputs, producing extreme density ratios r(x) ∈ {0.001, 999}. This breaks MCMC — the chain either accepts everything or rejects everything, never mixing properly.

Temperature scaling (Guo et al. 2017) applies a single parameter `T`:

$$D_{\text{cal}}(x) = \sigma\!\left(\frac{\text{logit}(x)}{T}\right)$$

- `T > 1` softens overconfident predictions
- `T` fit once on a held-out validation set, then frozen
- Preserves rank ordering — the MH acceptance ordering doesn't change, only the sharpness

### 3.2 Fitting Procedure

```
fit_calibration.py:
  1. Load trained CINN → generate validation samples
  2. Load real validation data (from truth HDF5, not seen by classifier)
  3. Run classifier on both → collect raw predictions
  4. Fit T by minimizing NLL on validation set
  5. Save T (single float) alongside classifier checkpoint
```

The calibration set must contain both real and CINN-generated samples with their true labels. This matches the distribution the classifier was trained to distinguish.

### 3.3 When to Re-fit

| Event | Re-fit needed? |
|-------|---------------|
| Same CINN + same classifier | No — use stored T |
| CINN retrained | Yes — generated distribution changed |
| New classifier checkpoint | Yes — uncalibrated scores changed |

---

## 4. IMH Algorithm (Detailed)

### 4.1 Per-Step Pseudocode

```python
def _step(x_current, c):
    # 1. Propose from CINN (independent of current state)
    x_proposed = model.sample(1, c).squeeze(1)  # (N, 730)

    # 2. Convert both to classifier input
    z_current  = convert(x_current,  c)  # (N, 772)
    z_proposed = convert(x_proposed, c)

    # 3. Evaluate classifier + calibrate + compute ratio
    logit_current  = clf(z_current)       # raw logits
    logit_proposed = clf(z_proposed)

    D_current  = sigmoid(logit_current / T)
    D_proposed = sigmoid(logit_proposed / T)

    r_current  = D_current  / (1 - D_current  + 1e-8)
    r_proposed = D_proposed / (1 - D_proposed + 1e-8)

    # 4. Acceptance ratio (IMH simplification — q cancels)
    alpha = torch.clamp(r_proposed / (r_current + 1e-8), max=1.0)

    # 5. Accept/reject
    u = torch.rand_like(alpha)
    accept = u < alpha
    x_next = torch.where(accept[:, None], x_proposed, x_current)
    r_next = torch.where(accept, r_proposed, r_current)

    return x_next, accept, r_next
```

### 4.2 Initialization

Chains are initialized from CINN samples: `x_0 ~ q(·|c)`. This guarantees the initial state is in the support of q (required for IMH).

### 4.3 Burn-in and Thinning

- **Burn-in**: First `burn_in` steps are discarded. Chains need time to forget their initialization and converge to the stationary distribution.
- **Thinning**: Keep every `thin`-th sample to reduce autocorrelation. IMH with a good proposal may not need aggressive thinning.

### 4.4 Acceptance Rate Monitoring

Low acceptance rate (< 10%) indicates the CINN is far from the true distribution in regions the classifier can detect. Options:
- Use temperature annealing: start with higher T (even more softened), then reduce
- Accept that the CINN needs retraining
- Switch to RWMH with local proposals

High acceptance rate (> 90%) indicates either the CINN is already near-perfect or the classifier is poorly trained (AUC ≈ 0.5).

---

## 5. Implementation Phases

### Phase 0: Port Infrastructure (no CINN needed yet)

**Files:** `src/mcmc/classifier.py`, `src/mcmc/calibration.py`, `src/mcmc/hlf.py`

1. Copy `MLP` + `MLPClassifier` from `caloxtreme_clf/module/` (stripping Lightning dependency)
2. Copy temperature scaling + ECE from `caloxtreme_clf/calibration/`
3. Copy `HighLevelFeatures` + `XMLHandler` from `caloxtreme_clf/data/`
4. Write unit test: can load a checkpoint and run inference

**Validation:** `uv run python -c "from src.mcmc.classifier import MLPClassifier; ..."` succeeds.

### Phase 1: Conversion Function

**File:** `src/mcmc/convert.py`

1. Implement `cinn_sample_to_classifier_input()` — replicates `scale_shower` + `get_high_level_features`
2. Write unit test: given a known CINN sample + postprocessed output, verify conversion produces the expected 772D tensor
3. Test that converted input produces the same classifier output as going through `LargeHDF5MLPDataModule`

**Validation:** Round-trip test: CINN sample → convert → classifier → matches direct path.

### Phase 2: IMH Sampler

**File:** `src/mcmc/sampler.py`

1. Implement `IMHSampler` class with `sample()` and `_step()`
2. Write unit tests:
   - Sanity: if classifier always outputs D=0.5, acceptance rate = 100%
   - Sanity: if classifier is perfect (D=1 for real, D=0 for fake), the chain should reject fake-looking proposals
   - Small integration: 10 chains × 50 steps on a toy energy

**Validation:** Acceptance rate in reasonable range (10-90%) on real data.

### Phase 3: Lightning Integration

**File:** `src/lightning_module.py` (modify)

1. Add `mcmc_sample(energies, n_chains, n_steps, ...)` method
2. Loads classifier checkpoint + calibration from disk
3. Creates `IMHSampler`, runs chains, postprocesses results
4. Returns dict of physical showers (matching `generate_single_energy` output format)

**Validation:** `uv run python -c "from src.lightning_module import ..."` — MCMC method callable.

### Phase 4: Calibration Script

**File:** `scripts/fit_calibration.py`

1. Standalone script that loads CINN + classifier, generates validation samples, fits T, saves to disk
2. CLI: `uv run python scripts/fit_calibration.py --ckpt <classifier.ckpt> --cinn-ckpt <cinn.ckpt> --truth-data <truth.hdf5> --output <T.json>`

### Phase 5: Integration Tests

**File:** `tests/test_mcmc_sampler.py`

1. Test that IMH output distribution is closer to truth than raw CINN samples (KS test or Wasserstein on HLF)
2. Test that acceptance rate is stable across energies
3. Test that calibrator is correctly applied (ECE of calibrated scores < ECE of raw scores)

---

## 6. Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| CINN mode collapse | **Critical** | MCMC cannot recover missing modes — proposals never reach them. CINN must cover p's support. |
| Classifier quality | **High** | If AUC ≈ 0.5, IMH provides no correction. Validate AUC before MCMC. |
| 772D input space | **Medium** | Curse of dimensionality for classifier. OK if data lies on lower-dimensional manifold (it does — CINN is a flow). |
| Postprocessing overhead per step | **Low** | `data_util.postprocess` + HLF computation per proposed sample. ~1ms per sample — acceptable for 500-step chains. |
| Calibration drift at extreme energies | **Low** | Single T covers full energy range since energy is an input feature. |

---

## 7. Dependencies from External Repo

| Source (`caloxtreme_clf/`) | Destination | Purpose |
|---------------------------|-------------|---------|
| `module/classifier.py` → `MLP` class | `src/mcmc/classifier.py` | Classifier architecture |
| `module/lightning.py` → `MLPClassifier` class | `src/mcmc/classifier.py` | Classifier with `get_input_from_batch` |
| `calibration/__init__.py` → temp scaling + ECE | `src/mcmc/calibration.py` | Calibration |
| `data/utils.py` → `get_high_level_features`, `scale_shower` | `src/mcmc/convert.py` | Preprocessing for classifier input |
| `data/HighLevelFeatures.py` | `src/mcmc/hlf.py` | HLF computation |
| `data/XMLHandler.py` | `src/mcmc/hlf.py` | XML binning reader |
| Classifier checkpoint (`.ckpt`) | `checkpoints/` | Trained weights |
| Temperature `T` (float) | `checkpoints/` | Calibration parameter |

---

## 8. Key Design Decisions

1. **Data-space IMH, not latent-space.** The CINN's `model.sample()` gives data-space proposals directly. Latent-space MCMC would require computing the reverse Jacobian, adding complexity with no benefit.

2. **Classifier operates on postprocessed data, not CINN internal.** The CINN internal representation (730D) includes extra dimensions that are reconstruction metadata, not physical features. Postprocessing removes these and produces physical showers that match what the classifier was trained on.

3. **Temperature scaling, not isotonic regression.** Single-parameter calibration is simpler to store, apply, and reason about. Isotonic regression would require storing the full fitted regressor object.

4. **Independent chains per energy, not a single chain.** Each energy gets its own set of chains. This naturally parallelizes and avoids the need for energy-jumping proposals.

5. **Frozen calibration at inference.** T is fit once on validation data and never updated during MCMC. This follows standard practice and avoids introducing bias from the MCMC samples themselves.
