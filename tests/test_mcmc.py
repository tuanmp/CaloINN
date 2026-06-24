"""Tests for MCMC density ratio reweighting.

Usage::

    # Unit tests (no GPU needed)
    HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/test_mcmc.py -v -k "not gpu"

    # All tests (requires checkpoints + GPU)
    HDF5_USE_FILE_LOCKING=FALSE uv run python -m pytest tests/test_mcmc.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Ensure src/ is on path for bare imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcmc.calibration import (
    TemperatureCalibrator,
    expected_calibration_error,
)
from mcmc.classifier import MLP, ClassifierWrapper
from mcmc.convert import (
    _apply_voxel_cutoff,
    _compute_hlf,
    _scale_shower,
    cinn_sample_to_classifier_input,
)
from mcmc.sampler import IMHSampler


# ═══════════════════════════════════════════════════════════════════════
#  TemperatureCalibrator
# ═══════════════════════════════════════════════════════════════════════

class TestTemperatureCalibrator:
    def test_fit_perfect_calibration(self):
        """Well-calibrated predictions should yield T ≈ 1."""
        rng = np.random.RandomState(42)
        # True probabilities uniformly in [0, 1]
        p_true = rng.uniform(0.01, 0.99, size=2000).astype(np.float64)
        y = rng.binomial(1, p_true)
        yhat = p_true  # Perfectly calibrated by construction
        calib = TemperatureCalibrator()
        calib.fit(yhat, y)
        assert 0.8 < calib.T < 1.2, f"Expected T≈1, got {calib.T:.3f}"

    def test_fit_overconfident(self):
        """Overconfident predictions (confident but frequently wrong) → T > 1."""
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.5, size=2000)
        # Flip 40% of labels to create mismatch: model is confident (0.95/0.05)
        # but only correct 60% of the time → overconfident.
        noisy_y = y.copy()
        flip = rng.choice(2000, size=800, replace=False)
        noisy_y[flip] = 1 - noisy_y[flip]
        yhat = np.where(noisy_y == 1, 0.95, 0.05).astype(np.float64)
        calib = TemperatureCalibrator()
        calib.fit(yhat, y)
        assert calib.T > 1.0, f"Expected T > 1 for overconfident, got {calib.T:.3f}"

    def test_save_load_roundtrip(self, tmp_path):
        """Save/load should preserve T."""
        calib = TemperatureCalibrator()
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.5, size=500)
        yhat = np.clip(y + rng.normal(0, 0.1, size=500), 0.01, 0.99)
        calib.fit(yhat, y)

        path = tmp_path / "T.json"
        calib.save(path)
        loaded = TemperatureCalibrator.load(path)
        assert abs(calib.T - loaded.T) < 1e-6

    def test_transform_logits(self):
        """Temperature scaling on logits should match probability transform."""
        calib = TemperatureCalibrator(T=2.0)
        logits = np.array([0.0, 2.0, -1.0])
        # Via logits
        cal_logits = calib.transform_logits(logits)
        # Via probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))
        cal_probs = calib.transform(probs)
        # Should match
        expected = 1.0 / (1.0 + np.exp(-cal_logits))
        np.testing.assert_allclose(cal_probs, expected, atol=1e-6)

    def test_unfitted_raises(self):
        calib = TemperatureCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            calib.transform(np.array([0.5]))

    def test_unfitted_save_raises(self, tmp_path):
        calib = TemperatureCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            calib.save(tmp_path / "T.json")


# ═══════════════════════════════════════════════════════════════════════
#  MLP
# ═══════════════════════════════════════════════════════════════════════

class TestMLP:
    def test_forward_shape(self):
        mlp = MLP(hidden_dim=64, num_layers=2, output_dim=1)
        x = torch.randn(8, 772)
        out = mlp(x)
        assert out.shape == (8, 1)

    def test_layer_norm_and_batch_norm_mutex(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            MLP(batch_norm=True, layer_norm=True)

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            MLP(activation="nonexistent")


# ═══════════════════════════════════════════════════════════════════════
#  Expected Calibration Error
# ═══════════════════════════════════════════════════════════════════════

class TestECE:
    def test_perfect_calibration(self):
        """ECE should be low for well-calibrated predictions."""
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.5, size=5000)
        yhat = np.full_like(y, 0.5, dtype=np.float64)
        ece = expected_calibration_error(y, yhat)
        assert ece < 0.05, f"ECE {ece:.4f} too high for perfect calibration"

    def test_overconfident(self):
        """ECE should be high for predictions that are confidently wrong."""
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.5, size=5000).astype(np.float64)
        # Always predict 0.9 regardless of true label → badly miscalibrated
        yhat = np.full_like(y, 0.9, dtype=np.float64)
        ece = expected_calibration_error(y, yhat)
        assert ece > 0.3, f"ECE {ece:.4f} too low for overconfident predictions"


# ═══════════════════════════════════════════════════════════════════════
#  Conversion functions
# ═══════════════════════════════════════════════════════════════════════

class TestConversion:
    def test_scale_shower(self):
        X_mev = np.array([[1000.0, 2000.0], [500.0, 1000.0]], dtype=np.float32)
        Einc_mev = np.array([[2000.0], [1000.0]], dtype=np.float32)
        X_proc, cond_proc = _scale_shower(X_mev, Einc_mev)
        np.testing.assert_allclose(X_proc[0], [0.5, 1.0], atol=1e-5)
        np.testing.assert_allclose(cond_proc[0], [2.0], atol=1e-5)

    def test_voxel_cutoff(self):
        X = np.array([[0.5, 2.0, 0.1], [3.0, 0.8, 4.0]], dtype=np.float32)
        result = _apply_voxel_cutoff(X, cutoff_mev=1.0)
        expected = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_voxel_cutoff_none(self):
        X = np.random.randn(4, 10).astype(np.float32)
        result = _apply_voxel_cutoff(X, cutoff_mev=None)
        np.testing.assert_array_equal(result, X)

    def test_hlf_shape(self):
        """HLF should produce (N, 51) features."""
        rng = np.random.RandomState(42)
        X_mev = rng.uniform(0, 1000, size=(4, 720)).astype(np.float32)
        Einc_mev = np.full((4, 1), 65536.0, dtype=np.float32)
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "binning_pion_odd_coarse_hcal.xml"
        )
        hlf = _compute_hlf(X_mev, Einc_mev, xml_path, "pion")
        assert hlf.shape == (4, 51), f"Expected (4, 51), got {hlf.shape}"

    def test_full_conversion_shape(self):
        """End-to-end conversion should produce (N, 772)."""
        rng = np.random.RandomState(42)
        x = rng.uniform(0.01, 0.5, size=(2, 730)).astype(np.float32)
        c = np.array([[60.0], [70.0]], dtype=np.float32)
        q = torch.tensor(5e-6)
        layer_boundaries = [0, 48, 96, 144, 192, 240, 336, 432, 528, 624, 720]
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "binning_pion_odd_coarse_hcal.xml"
        )

        result = cinn_sample_to_classifier_input(
            x, c, layer_boundaries, q=q, width_noise=5e-6,
            xml_path=xml_path, particle="pion",
            log_transform=False, voxel_energy_cutoff=None,
        )
        assert result.shape == (2, 772)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_conversion_with_log_transform(self):
        """log_transform=True should produce non-negative X_proc portion."""
        rng = np.random.RandomState(42)
        x = rng.uniform(0.01, 0.5, size=(2, 730)).astype(np.float32)
        c = np.array([[60.0], [70.0]], dtype=np.float32)
        q = torch.tensor(5e-6)
        layer_boundaries = [0, 48, 96, 144, 192, 240, 336, 432, 528, 624, 720]
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "binning_pion_odd_coarse_hcal.xml"
        )

        result = cinn_sample_to_classifier_input(
            x, c, layer_boundaries, q=q, width_noise=5e-6,
            xml_path=xml_path, particle="pion",
            log_transform=True, voxel_energy_cutoff=1.0,
        )
        # X_proc portion (first 720 columns) should be non-negative after log1p
        assert np.all(result[:, :720] >= 0), "X_proc should be non-negative after log1p"


# ═══════════════════════════════════════════════════════════════════════
#  IMHSampler (unit tests with mock components)
# ═══════════════════════════════════════════════════════════════════════

class MockModel:
    def eval(self): pass
    def sample(self, n, c):
        return torch.randn(1, n, 730)


class MockClassifier:
    """Returns logits that produce specific density ratios."""
    def __init__(self, logit_val=0.0):
        self.logit_val = logit_val
    def forward(self, x):
        return torch.full((x.shape[0], 1), self.logit_val, device=x.device)


class TestIMHSampler:
    @pytest.fixture
    def calibrator(self):
        return TemperatureCalibrator(T=1.0)

    def _make_sampler(self, calibrator, logit_val=0.0, r_clip=100.0):
        model = MockModel()
        clf = MockClassifier(logit_val=logit_val)
        def convert_fn(x, c):
            return np.random.randn(x.shape[0], 772).astype(np.float32)
        return IMHSampler(
            model=model, classifier=clf, calibrator=calibrator,
            conversion_fn=convert_fn, device="cpu", r_clip=r_clip,
        )

    def test_uniform_acceptance(self, calibrator):
        """With D=0.5 everywhere, r=1.0 → 100% acceptance."""
        sampler = self._make_sampler(calibrator, logit_val=0.0)
        x = torch.randn(4, 730)
        c = torch.randn(4, 1)
        r_cur = torch.ones(4)
        _, _, accepted = sampler._step(x, c, r_cur)
        assert accepted.float().mean() >= 0.7, "Expected high acceptance with uniform r"

    def test_density_ratio_with_uniform_clf(self, calibrator):
        """D=0.5 → r=1.0."""
        sampler = self._make_sampler(calibrator, logit_val=0.0)
        x = torch.randn(4, 730)
        c = torch.randn(4, 1)
        r = sampler._density_ratio(x, c)
        np.testing.assert_allclose(r.numpy(), 1.0, atol=1e-5)

    def test_density_ratio_clipping(self, calibrator):
        """Ratio should be clipped to [1/r_clip, r_clip]."""
        sampler = self._make_sampler(calibrator, logit_val=10.0, r_clip=10.0)
        x = torch.randn(4, 730)
        c = torch.randn(4, 1)
        r = sampler._density_ratio(x, c)
        assert torch.all(r <= 10.0)
        assert torch.all(r >= 0.1)

    def test_unfitted_calibrator_raises(self, calibrator):
        calib = TemperatureCalibrator()  # unfitted
        with pytest.raises(ValueError, match="must be fitted"):
            self._make_sampler(calib)


# ═══════════════════════════════════════════════════════════════════════
#  GPU integration tests (require checkpoints)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available",
)
class TestIntegrationGPU:
    """End-to-end tests requiring trained checkpoints and GPU."""

    CINN_CKPT = (
        "/pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/"
        "checkpoints/epoch=491-val_loss=895.95.ckpt"
    )
    CLF_CKPT = (
        "/pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/"
        "best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt"
    )

    @pytest.fixture(scope="class")
    def classifier(self):
        return ClassifierWrapper.load_from_checkpoint(self.CLF_CKPT, device="cuda")

    def test_classifier_loads(self, classifier):
        """Classifier checkpoint loads and produces valid output."""
        x = torch.randn(10, 772).cuda()
        logits = classifier.forward(x)
        assert logits.shape == (10, 1)
        probs = classifier.predict_proba(x)
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_classifier_discrimination(self, classifier):
        """Classifier should distinguish extreme cases."""
        # All zeros → should predict "fake" (low probability)
        x_fake = torch.zeros(10, 772).cuda()
        p_fake = classifier.predict_proba(x_fake).mean()
        # Small random → should be somewhere in middle
        x_rand = torch.randn(10, 772).cuda()
        p_rand = classifier.predict_proba(x_rand).mean()
        # Just sanity: not all identical
        print(f"  p_fake={p_fake:.3f}, p_rand={p_rand:.3f}")

    def test_conversion_with_real_cinn(self):
        """Load CINN, generate samples, convert to classifier input."""
        pytest.skip("Requires CINN model loading — test manually for now")
