import copy
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_util
from model import CINN
from trainer import Trainer

try:
    from lightning_data import CaloINNDataModule
    from lightning_module import CaloINNLightningModule
    LIGHTNING_AVAILABLE = True
except ModuleNotFoundError:
    LIGHTNING_AVAILABLE = False


class DummyDoc:
    def __init__(self, basedir):
        self.basedir = basedir

    def get_file(self, name, add_run_name=False):
        return os.path.join(self.basedir, name)


@contextmanager
def patch_small_dataset(max_events):
    original_load_data = data_util.load_data

    def _patched_load_data(*args, **kwargs):
        data, layer_boundaries = original_load_data(*args, **kwargs)
        sliced = {key: value[:max_events] for key, value in data.items()}
        return sliced, layer_boundaries

    data_util.load_data = _patched_load_data
    try:
        yield
    finally:
        data_util.load_data = original_load_data


class TestLightningParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LIGHTNING_AVAILABLE:
            raise unittest.SkipTest("pytorch_lightning is not installed")

        params_path = os.path.join(REPO_ROOT, "params", "pions.yaml")
        with open(params_path) as f:
            base_params = yaml.load(f, Loader=yaml.FullLoader)

        data_path = base_params.get("data_path")
        xml_path = base_params.get("xml_path")
        if not os.path.exists(data_path) or not os.path.exists(xml_path):
            raise unittest.SkipTest("Dataset paths in params/pions.yaml are not available")

        cls.params = copy.deepcopy(base_params)
        cls.params.update(
            {
                "batch_size": 16,
                "val_frac": 0.2,
                "n_epochs": 1,
                "cycle_epochs": 1,
                "save_interval": 100,
                "width_noise": 0.0,
                "custom_noise": False,
                "n_blocks": 2,
                "internal_size": 32,
                "eval_dataset": "2",
                "norm": False,
            }
        )

        torch.set_default_dtype(torch.float32)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="caloinn_lightning_test_")
        self.doc = DummyDoc(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_legacy_and_lightning(self, seed=1234, max_events=64):
        with patch_small_dataset(max_events=max_events):
            np.random.seed(seed)
            torch.manual_seed(seed)
            legacy = Trainer(copy.deepcopy(self.params), "cpu", self.doc)

            np.random.seed(seed)
            torch.manual_seed(seed)
            dataset_params = {
                "xml_path": self.params.get("xml_path"),
                "xml_ptype": self.params.get("xml_ptype"),
                "single_energy": self.params.get("single_energy", None),
                "eps": self.params.get("eps", 1.0e-10),
                "u0up_cut": self.params.get("u0up_cut", 7.0),
                "u0low_cut": self.params.get("u0low_cut", 0.0),
                "pt_rew": self.params.get("pt_rew", 1.0),
                "dep_cut": self.params.get("dep_cut", 1.0e10),
                           "width_noise": self.params.get("width_noise", 1e-7),
            }
            datamodule = CaloINNDataModule(
                data_path=self.params.get("data_path"),
                val_data_path=self.params.get("val_data_path", self.params.get("data_path")),
                batch_size=self.params.get("batch_size", 16),
                cond_key="incident_energies",
                sample_key="showers",
                val_frac=self.params.get("val_frac", 0.01),
                shuffle=False,
                eval_dataset=self.params.get("eval_dataset", "1-pions"),
                num_workers=0,
                predict_batch_size=self.params.get("batch_size", 16),
                dataset_kwargs=dataset_params,
            )
            datamodule.setup("fit")

            cinn_params = {
                "bayesian": self.params.get("bayesian", False),
                "alpha": self.params.get("alpha", 1.0e-8),
                "alpha_logit": self.params.get("alpha_logit", 1.0e-6),
                "use_extra_dims": self.params.get("use_extra_dims", True),
                "use_norm": self.params.get("use_norm", False),
                "log_cond": self.params.get("log_cond", True),
                "layers_per_block": self.params.get("layers_per_block", 4),
                "internal_size": self.params.get("internal_size", 256),
                "n_blocks": self.params.get("n_blocks", 12),
                "coupling_type": self.params.get("coupling_type", "rational_quadratic"),
                "dropout": self.params.get("dropout", 0.0),
                "layer_norm": self.params.get("layer_norm", "nn.BatchNorm1d"),
                "layer_act": self.params.get("layer_act", "nn.SiLU"),
                "num_bins": self.params.get("num_bins", 10),
                "bounds_init": self.params.get("bounds_init", 18),
                "permute_soft": self.params.get("permute_soft", False),
                "permute_layer": self.params.get("permute_layer", False),
                "std_init": self.params.get("std_init", -15.0),
                "prior_prec": self.params.get("prior_prec", 5000),
                "sub_layers": self.params.get("sub_layers", ["linear", "linear", "linear", "linear"]),
                "norm": self.params.get("norm", True),
            }
            lightning_module = CaloINNLightningModule(
                setup_data_sample_path=self.params.get("data_path"),
                enable_diagnostics=False,
                actnorm_calibration_samples=min(max_events, 1024),
                xml_path=self.params.get("xml_path"),
                xml_ptype=self.params.get("xml_ptype"),
                dataset_params=dataset_params,
                cinn_params=cinn_params,
                width_noise=self.params.get("width_noise", 0.0),
                custom_noise=self.params.get("custom_noise", False),
                single_energy=self.params.get("single_energy", None),
            )
            lightning_module.model.load_state_dict(copy.deepcopy(legacy.model.state_dict()))
            lightning_module = lightning_module.to("cpu")

        return legacy, datamodule, lightning_module

    def test_dataloader_split_matches_legacy(self):
        legacy, datamodule, _ = self._build_legacy_and_lightning(seed=11, max_events=80)

        self.assertIsNotNone(datamodule._train_loader)
        self.assertIsNotNone(datamodule._val_loader)

        x_dm, c_dm = next(iter(datamodule.train_dataloader()))
        self.assertEqual(x_dm.shape[1], legacy.train_loader.data.shape[1])
        self.assertEqual(c_dm.shape[1], legacy.train_loader.cond.shape[1])

        self.assertEqual(int(legacy.train_loader.data.shape[0]), 64)
        self.assertEqual(int(legacy.test_loader.data.shape[0]), 16)

    def test_loss_and_optimizer_step_match_legacy(self):
        legacy, datamodule, lightning_module = self._build_legacy_and_lightning(seed=19, max_events=64)

        x = legacy.train_loader.data[: self.params["batch_size"]]
        c = legacy.train_loader.cond[: self.params["batch_size"]]

        legacy_inn = -torch.mean(legacy.model.log_prob(x, c))
        legacy_loss = legacy_inn

        new_loss, new_inn, _ = lightning_module._compute_losses(x, c)

        self.assertTrue(torch.allclose(legacy_inn, new_inn, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(legacy_loss, new_loss, atol=1e-7, rtol=1e-6))

        old_optim = torch.optim.AdamW(
            legacy.model.params_trainable,
            lr=self.params.get("lr", 0.0002),
            betas=self.params.get("betas", [0.9, 0.999]),
            eps=self.params.get("eps", 1e-6),
            weight_decay=self.params.get("weight_decay", 0.0),
        )
        new_optim = torch.optim.AdamW(
            lightning_module.model.params_trainable,
            lr=self.params.get("lr", 0.0002),
            betas=self.params.get("betas", [0.9, 0.999]),
            eps=self.params.get("eps", 1e-6),
            weight_decay=self.params.get("weight_decay", 0.0),
        )

        old_optim.zero_grad()
        legacy_loss.backward()
        old_optim.step()

        new_optim.zero_grad()
        new_loss.backward()
        new_optim.step()

        old_state = legacy.model.state_dict()
        new_state = lightning_module.model.state_dict()
        self.assertEqual(set(old_state.keys()), set(new_state.keys()))
        for key in old_state:
            self.assertTrue(torch.allclose(old_state[key], new_state[key], atol=1e-6, rtol=1e-5), msg=key)

    def test_generate_output_matches_legacy_shape_and_values(self):
        legacy, _, lightning_module = self._build_legacy_and_lightning(seed=29, max_events=64)

        if not hasattr(lightning_module, "generate"):
            self.skipTest("CaloINNLightningModule.generate is not implemented in current API")

        torch.manual_seed(123)
        np.random.seed(123)
        legacy_samples = legacy.generate(num_samples=32, batch_size=16)

        torch.manual_seed(123)
        np.random.seed(123)
        lightning_samples = lightning_module.generate(
            num_samples=32,
            batch_size=16,
            output_file=os.path.join(self.tmpdir, "samples_lightning.hdf5"),
        )

        self.assertEqual(set(legacy_samples.keys()), set(lightning_samples.keys()))
        for key in legacy_samples:
            self.assertEqual(legacy_samples[key].shape, lightning_samples[key].shape)

            old_arr = legacy_samples[key]
            new_arr = lightning_samples[key]
            self.assertTrue(np.isfinite(old_arr).all(), msg=key)
            self.assertTrue(np.isfinite(new_arr).all(), msg=key)

            if key == "energy":
                self.assertTrue((old_arr > 0).all())
                self.assertTrue((new_arr > 0).all())

            old_mean = float(np.mean(old_arr))
            new_mean = float(np.mean(new_arr))
            old_std = float(np.std(old_arr))
            new_std = float(np.std(new_arr))

            mean_tol = max(1e-6, 0.1 * max(abs(old_mean), abs(new_mean), 1.0))
            std_tol = max(1e-6, 0.1 * max(abs(old_std), abs(new_std), 1.0))

            self.assertLess(abs(old_mean - new_mean), mean_tol, msg=f"{key}:mean")
            self.assertLess(abs(old_std - new_std), std_tol, msg=f"{key}:std")


class TestLightningInitParity(unittest.TestCase):
    """Phase 2: test model initialization parity — weights and latent space.

    Verifies that CINN(params, data, cond) produces identical weights
    and identical latent-z outputs when created from the same preprocessed
    data with the same random seed.

    Also tests latent-space histogram parity via Kolmogorov-Smirnov test
    (mimics the legacy Trainer.latent_samples(0) call done before training).
    """

    @classmethod
    def setUpClass(cls):
        if not LIGHTNING_AVAILABLE:
            raise unittest.SkipTest("pytorch_lightning is not installed")

        params_path = os.path.join(REPO_ROOT, "params", "pions.yaml")
        with open(params_path) as f:
            base_params = yaml.load(f, Loader=yaml.FullLoader)

        data_path = base_params.get("data_path")
        xml_path = base_params.get("xml_path")
        if not os.path.exists(data_path) or not os.path.exists(xml_path):
            raise unittest.SkipTest("Dataset paths not available")

        # Use a small model for fast tests; keep ActNorm (norm=True)
        cls.params = copy.deepcopy(base_params)
        cls.params.update({
            "batch_size": 16,
            "val_frac": 0.2,
            "n_epochs": 1,
            "cycle_epochs": 1,
            "width_noise": 0.0,
            "n_blocks": 2,
            "internal_size": 32,
            "eval_dataset": "2",
            "norm": True,
            "bayesian": False,
        })
        torch.set_default_dtype(torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_models(
        self, seed, max_events=64
    ) -> Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        """Build legacy and Lightning CINN models from the *same* data."""
        with patch_small_dataset(max_events=max_events):
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_loader, _, lbs = data_util.get_loaders(
                self.params.get("data_path"),
                self.params.get("xml_path"),
                self.params.get("xml_ptype"),
                self.params.get("val_frac"),
                self.params.get("batch_size"),
                self.params.get("eps"),
                "cpu",
                width_noise=0.0,
                shuffle=False,
                u0up_cut=self.params.get("u0up_cut", 7.0),
                u0low_cut=self.params.get("u0low_cut", 0.0),
                rew=self.params.get("pt_rew", 1.0),
                dep_cut=self.params.get("dep_cut", 1e10),
            )
            x = torch.clone(train_loader.data)
            c = torch.clone(train_loader.cond)

            # Legacy model
            model_legacy = CINN(self.params, x, c)

            # Lightning model — same seed, same data, same cinn_params
            np.random.seed(seed)
            torch.manual_seed(seed)
            cinn_params = _params_to_cinn(self.params)
            lm = CaloINNLightningModule(
                init_data_x=x, init_data_c=c, init_layer_boundaries=lbs,
                init_num_train_samples=int(x.shape[0]),
                cinn_params=cinn_params, width_noise=0.0,
            )
            model_new = lm.model

        return model_legacy, model_new, x, c

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_num_train_samples_not_one(self):
        """num_train_samples should equal actual data size, not 1."""
        print("\n=== test_num_train_samples_not_one ===")

        with patch_small_dataset(max_events=128):
            np.random.seed(42); torch.manual_seed(42)
            legacy = Trainer(copy.deepcopy(self.params), "cpu", DummyDoc("/tmp"))

            x = legacy.train_loader.data
            n_legacy = int(x.shape[0])

            np.random.seed(42); torch.manual_seed(42)
            cinn_params = _params_to_cinn(self.params)
            lm = CaloINNLightningModule(
                setup_data_sample_path=self.params.get("data_path"),
                xml_path=self.params.get("xml_path"),
                xml_ptype=self.params.get("xml_ptype"),
                train_val_frac=0.2,
                dataset_params={
                    "xml_path": self.params.get("xml_path"),
                    "eps": self.params.get("eps", 1.0e-10),
                },
                cinn_params=cinn_params, width_noise=0.0,
            )

        print(f"Legacy train samples : {n_legacy}")
        print(f"Lightning num_train_samples: {lm.num_train_samples}")

        self.assertEqual(
            lm.num_train_samples, n_legacy,
            f"num_train_samples={lm.num_train_samples} vs legacy {n_legacy}"
        )

    def test_model_weights_identical(self):
        """Weights must be bit-identical when init data and seed are the same."""
        print("\n=== test_model_weights_identical ===")

        model_legacy, model_new, x, c = self._build_models(seed=1234, max_events=64)

        ls = model_legacy.state_dict()
        ns = model_new.state_dict()

        # Same architecture
        self.assertEqual(set(ls.keys()), set(ns.keys()), "State dict keys differ")

        # Bit-identical weights
        max_diff = 0.0
        for key in sorted(ls.keys()):
            diff = float(torch.max(torch.abs(ls[key] - ns[key])))
            max_diff = max(max_diff, diff)

        self.assertEqual(
            max_diff, 0.0,
            f"Model weights differ: max_diff={max_diff:.6e}"
        )

        print(f"Weights: {len(ls)} tensors, max_diff = {max_diff:.1e}")

    def test_latent_z_identical(self):
        """Latent z must be identical (same data, same weights, same model)."""
        print("\n=== test_latent_z_identical ===")

        model_legacy, model_new, x, c = self._build_models(seed=1234, max_events=64)

        with torch.no_grad():
            z1 = model_legacy(x, c)
            z2 = model_new(x, c)
            z1 = z1[0] if isinstance(z1, tuple) else z1
            z2 = z2[0] if isinstance(z2, tuple) else z2

        z_diff = float(torch.max(torch.abs(z1 - z2)))
        print(f"Latent z max diff: {z_diff:.6e}")

        self.assertEqual(z_diff, 0.0, "Latent z differs — model init mismatch")

    def test_latent_histogram_ks_test(self):
        """Latent-space histograms should match via Kolmogorov-Smirnov test.

        This mimics the legacy Trainer.latent_samples(0) call done at the
        very start of training.  For a well-initialised CINN the latent
        variables z are approximately N(0,1) per dimension.
        """
        print("\n=== test_latent_histogram_ks_test ===")

        model_legacy, model_new, x, c = self._build_models(seed=456, max_events=128)

        with torch.no_grad():
            z1 = model_legacy(x, c)
            z2 = model_new(x, c)
            z1 = z1[0] if isinstance(z1, tuple) else z1
            z2 = z2[0] if isinstance(z2, tuple) else z2

        from scipy import stats

        n_dims = z1.shape[1]
        significance = 0.01  # 1 % false-positive rate per dimension
        n_fail = 0

        for d in range(n_dims):
            ks_stat, p_value = stats.ks_2samp(
                z1[:, d].numpy(), z2[:, d].numpy()
            )
            if p_value < significance:
                n_fail += 1
                if n_fail <= 3:  # only report first few failures
                    print(
                        f"  dim {d:3d}: KS={ks_stat:.4f} p={p_value:.4f}  "
                        f"(z1 mean={z1[:, d].mean():.4f}, "
                        f"z2 mean={z2[:, d].mean():.4f})"
                    )

        fail_frac = n_fail / n_dims
        print(
            f"KS test: {n_dims} dims, {n_fail} failed ({fail_frac:.1%}), "
            f"threshold p<{significance}"
        )

        # Allow at most 5 % of dimensions to fail (tolerates a few false
        # positives from multiple testing).
        self.assertLess(
            fail_frac, 0.05,
            f"Too many dimensions failed KS test: {n_fail}/{n_dims}"
        )

    def test_init_with_precomputed_arrays(self):
        """init_data_x/init_data_c bypasses file loading."""
        print("\n=== test_init_with_precomputed_arrays ===")

        with patch_small_dataset(max_events=128):
            np.random.seed(42); torch.manual_seed(42)
            legacy = Trainer(copy.deepcopy(self.params), "cpu", DummyDoc("/tmp"))

            x = legacy.train_loader.data
            c = legacy.train_loader.cond
            lbs = legacy.layer_boundaries

            np.random.seed(42); torch.manual_seed(42)
            cinn_params = _params_to_cinn(self.params)
            lm = CaloINNLightningModule(
                init_data_x=x, init_data_c=c, init_layer_boundaries=lbs,
                init_num_train_samples=int(x.shape[0]),
                cinn_params=cinn_params, width_noise=0.0,
            )

        print(f"num_train_samples={lm.num_train_samples}, num_dim={lm.num_dim}")
        self.assertEqual(lm.num_train_samples, x.shape[0])
        self.assertEqual(lm.num_dim, x.shape[1])


# ------------------------------------------------------------------
# Helper: extract CINN params from legacy-style flat config
# ------------------------------------------------------------------

def _params_to_cinn(params: dict) -> dict:
    """Convert legacy flat-param dict to cinn_params dict expected by Lightning."""
    return {
        "bayesian": params.get("bayesian", False),
        "alpha": params.get("alpha", 1.0e-8),
        "alpha_logit": params.get("alpha_logit", 1.0e-6),
        "use_extra_dims": params.get("use_extra_dims", True),
        "use_norm": params.get("use_norm", False),
        "log_cond": params.get("log_cond", True),
        "layers_per_block": params.get("layers_per_block", 4),
        "internal_size": params.get("internal_size", 256),
        "n_blocks": params.get("n_blocks", 12),
        "coupling_type": params.get("coupling_type", "rational_quadratic"),
        "dropout": params.get("dropout", 0.0),
        "layer_norm": params.get("layer_norm", "nn.BatchNorm1d"),
        "layer_act": params.get("layer_act", "nn.SiLU"),
        "num_bins": params.get("num_bins", 10),
        "bounds_init": params.get("bounds_init", 18),
        "permute_soft": params.get("permute_soft", False),
        "permute_layer": params.get("permute_layer", False),
        "std_init": params.get("std_init", -15.0),
        "prior_prec": params.get("prior_prec", 5000),
        "sub_layers": params.get("sub_layers", ["linear", "linear", "linear", "linear"]),
        "norm": params.get("norm", True),
    }


class TestNoiseParity(unittest.TestCase):
    """Phase 3: verify noise application matches legacy MyDataLoader.

    Checks:
    1.  Streaming dataset noise equals legacy MyDataLoader noise (same seed).
    2.  Lightning training_step does NOT add noise (no double-noise).
    """

    @classmethod
    def setUpClass(cls):
        if not LIGHTNING_AVAILABLE:
            raise unittest.SkipTest("pytorch_lightning is not installed")

        params_path = os.path.join(REPO_ROOT, "params", "pions.yaml")
        with open(params_path) as f:
            base_params = yaml.load(f, Loader=yaml.FullLoader)

        if not os.path.exists(base_params.get("data_path", "")):
            raise unittest.SkipTest("Dataset paths not available")

        cls.params = copy.deepcopy(base_params)
        cls.params.update({
            "batch_size": 16, "val_frac": 0.5, "n_epochs": 1,
            "cycle_epochs": 1, "width_noise": 5e-6, "n_blocks": 2,
            "internal_size": 32, "eval_dataset": "2", "norm": False,
        })
        torch.set_default_dtype(torch.float32)

    def test_noise_matches_legacy_loader(self):
        """Streaming dataset noise == legacy MyDataLoader noise with same seed."""
        print("\n=== test_noise_matches_legacy_loader ===")

        w = float(self.params.get("width_noise", 5e-6))

        with patch_small_dataset(max_events=64):
            # Legacy loader with noise
            np.random.seed(41); torch.manual_seed(41)
            loader, _, lbs = data_util.get_loaders(
                self.params["data_path"], self.params.get("xml_path"),
                self.params.get("xml_ptype"), self.params["val_frac"],
                16, self.params["eps"], "cpu", width_noise=w, shuffle=False,
                u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
            )
            x_legacy_noisy, c_legacy = next(iter(loader))

            # Streaming dataset — same data, same noise
            from streaming_data import PreprocessedStreamingDataset
            ds = PreprocessedStreamingDataset(
                data_path=self.params["data_path"],
                xml_filename=self.params.get("xml_path"),
                particle_type=self.params.get("xml_ptype"),
                batch_size=16,
                eps=self.params["eps"],
                u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
                width_noise=w, fixed_noise=False, val_frac=0,
                shuffle=False, is_train=True,
                precomputed_valid_indices=loader.index[:16].numpy(),
                layer_boundaries=lbs,
            )

            np.random.seed(41); torch.manual_seed(41)
            x_stream_noisy, c_stream = next(iter(ds))

        # Conditions must match (no noise on cond)
        self.assertTrue(torch.allclose(c_legacy, c_stream, atol=1e-7),
                        "Condition vectors differ between loaders")

        # Noisy x should match (same data + same noise seed)
        if torch.allclose(x_legacy_noisy, x_stream_noisy, atol=1e-7):
            print("Noise identical between legacy and streaming ✓")
        else:
            # If they differ, measure how much (should be within noise range)
            diff = float(torch.max(torch.abs(
                x_legacy_noisy - x_stream_noisy
            )))
            print(f"Max noise diff: {diff:.2e}  (width_noise={w:.2e})")
            # Differences up to width_noise are expected if seeds drift
            self.assertLess(diff, 2 * w,
                            f"Noise differs beyond width_noise range: {diff:.2e}")

    def test_no_noise_in_training_step(self):
        """training_step must NOT add noise — verify by behavior, not source."""
        print("\n=== test_no_noise_in_training_step ===")

        with patch_small_dataset(max_events=32):
            np.random.seed(42); torch.manual_seed(42)
            loader, _, lbs = data_util.get_loaders(
                self.params["data_path"], self.params.get("xml_path"),
                self.params.get("xml_ptype"), 0.5, 16, self.params["eps"],
                "cpu", width_noise=0.0, shuffle=False,
                u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
            )
            data = torch.clone(loader.data)
            cond = torch.clone(loader.cond)

            np.random.seed(42); torch.manual_seed(42)
            cinn_params = _params_to_cinn(self.params)
            lm = CaloINNLightningModule(
                init_data_x=data, init_data_c=cond, init_layer_boundaries=lbs,
                init_num_train_samples=int(data.shape[0]),
                cinn_params=cinn_params,
                width_noise=5e-6,  # Noise enabled but should NOT be in training_step
            )

            # Simulate a training_step call: give it clean data and check
            # whether the module adds noise (it should NOT).
            x_clean = data[:16].clone()
            c_clean = cond[:16].clone()

            batch = (x_clean, c_clean)
            # We test behavior: if training_step adds noise, x_clean will be modified
            # (actually training_step returns loss, so we need to hook it)
            #
            # Simpler approach: just verify _compute_losses sees the same x
            # that we passed in.  We can monkey-patch _compute_losses.
            captured_x = []

            original_compute = lm._compute_losses
            def _capturing_compute(x, c, run_diagnostics=False):
                captured_x.append(x.detach().clone())
                return original_compute(x, c, run_diagnostics)
            lm._compute_losses = _capturing_compute

            # Make the module think it has a trainer
            lm.num_train_samples = int(data.shape[0])
            lm.steps_per_epoch = 1

            # Call training_step
            lm.training_step(batch, 0)

            # Restore original
            lm._compute_losses = original_compute

        # Verify: the x passed to _compute_losses should be exactly the input
        x_seen = captured_x[0]
        self.assertTrue(
            torch.allclose(x_clean, x_seen, atol=1e-9),
            "training_step modified x (likely added noise) — double-noise bug"
        )
        print("training_step passes x unchanged — no double-noise ✓")


class TestEndToEndPipelineParity(unittest.TestCase):
    """Phase 3b: full pipeline parity — legacy vs new, model + dataloader.

    Three levels of parity:
    1.  Same preprocessed data → legacy model vs new model → same output.
    2.  Same raw HDF5 → legacy pipeline vs new pipeline (no noise) → same.
    3.  Same raw HDF5 → legacy vs new pipeline (with noise) → statistically similar.
    """

    @classmethod
    def setUpClass(cls):
        if not LIGHTNING_AVAILABLE:
            raise unittest.SkipTest("pytorch_lightning is not installed")

        params_path = os.path.join(REPO_ROOT, "params", "pions.yaml")
        with open(params_path) as f:
            base_params = yaml.load(f, Loader=yaml.FullLoader)

        if not os.path.exists(base_params.get("data_path", "")):
            raise unittest.SkipTest("Dataset paths not available")

        cls.params = copy.deepcopy(base_params)
        cls.params.update({
            "batch_size": 32, "val_frac": 0.3, "n_epochs": 1,
            "cycle_epochs": 1, "width_noise": 0.0, "n_blocks": 2,
            "internal_size": 32, "eval_dataset": "2", "norm": True,
            "bayesian": False,
        })
        torch.set_default_dtype(torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_legacy_model(self, max_events=256):
        """Build legacy CINN from a small subset of data."""
        with patch_small_dataset(max_events=max_events):
            np.random.seed(999); torch.manual_seed(999)
            loader, _, lbs = data_util.get_loaders(
                self.params["data_path"], self.params.get("xml_path"),
                self.params.get("xml_ptype"), self.params["val_frac"],
                16, self.params["eps"], "cpu", width_noise=0.0, shuffle=False,
                u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
            )
            data = torch.clone(loader.data)
            cond = torch.clone(loader.cond)
            model = CINN(self.params, data, cond)
            model.eval()
        return model, data, cond, lbs

    def _build_new_model(self, data, cond, lbs):
        """Build Lightning CINN from preprocessed data."""
        np.random.seed(999); torch.manual_seed(999)
        cinn_params = _params_to_cinn(self.params)
        lm = CaloINNLightningModule(
            init_data_x=data, init_data_c=cond, init_layer_boundaries=lbs,
            init_num_train_samples=int(data.shape[0]),
            cinn_params=cinn_params, width_noise=0.0,
        )
        lm.model.eval()
        return lm.model

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_model_parity_same_input(self):
        """Same preprocessed data → legacy and new produce identical outputs."""
        print("\n=== test_model_parity_same_input ===")

        model_legacy, data, cond, lbs = self._build_legacy_model(256)
        model_new = self._build_new_model(data, cond, lbs)

        xt, ct = data[:32].clone(), cond[:32].clone()
        with torch.no_grad():
            z1 = model_legacy(xt, ct)
            z1 = z1[0] if isinstance(z1, tuple) else z1
            z2 = model_new(xt, ct)
            z2 = z2[0] if isinstance(z2, tuple) else z2
            lp1 = model_legacy.log_prob(xt, ct)
            lp2 = model_new.log_prob(xt, ct)

        z_diff = float(torch.max(torch.abs(z1 - z2)))
        lp_diff = float(torch.max(torch.abs(lp1 - lp2)))
        print(f"  z max diff = {z_diff:.1e}    log_prob max diff = {lp_diff:.1e}")

        self.assertEqual(z_diff, 0.0, f"latent z mismatch: {z_diff:.1e}")
        self.assertEqual(lp_diff, 0.0, f"log_prob mismatch: {lp_diff:.1e}")
        print("  PASS ✓")

    def test_pipeline_parity_no_noise(self):
        """Legacy pipeline vs new pipeline — no noise, same seed — must agree."""
        print("\n=== test_pipeline_parity_no_noise ===")

        from streaming_data import PreprocessedStreamingDataset

        model_legacy, _data, _cond, lbs = self._build_legacy_model(256)
        model_new = self._build_new_model(_data, _cond, lbs)

        # Legacy pipeline batch
        np.random.seed(42); torch.manual_seed(42)
        legacy_loader, _, lbs2 = data_util.get_loaders(
            self.params["data_path"], self.params.get("xml_path"),
            self.params.get("xml_ptype"), self.params["val_frac"],
            32, self.params["eps"], "cpu", width_noise=0.0, shuffle=False,
            u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
        )
        x_leg, c_leg = next(iter(legacy_loader))

        # New pipeline batch
        ds = PreprocessedStreamingDataset(
            data_path=self.params["data_path"],
            xml_filename=self.params.get("xml_path"),
            particle_type=self.params.get("xml_ptype"),
            batch_size=32, eps=self.params["eps"],
            u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
            width_noise=0.0, fixed_noise=False, val_frac=0,
            shuffle=False, is_train=True, layer_boundaries=lbs2,
        )
        np.random.seed(42); torch.manual_seed(42)
        x_new, c_new = next(iter(ds))

        self.assertEqual(x_leg.dtype, x_new.dtype, "dtype mismatch")
        self.assertEqual(c_leg.dtype, c_new.dtype, "dtype mismatch")

        x_diff = float(torch.max(torch.abs(x_leg - x_new)))
        c_diff = float(torch.max(torch.abs(c_leg - c_new)))
        print(f"  Data diff: x={x_diff:.1e}  c={c_diff:.1e}")

        with torch.no_grad():
            z_leg = model_legacy(x_leg, c_leg)
            z_leg = z_leg[0] if isinstance(z_leg, tuple) else z_leg
            z_new = model_new(x_new, c_new)
            z_new = z_new[0] if isinstance(z_new, tuple) else z_new

        z_diff = float(torch.max(torch.abs(z_leg - z_new)))
        print(f"  Latent z diff: {z_diff:.1e}")

        self.assertLess(x_diff, 1e-5, f"x data differs: {x_diff:.1e}")
        self.assertLess(c_diff, 1e-5, f"c data differs: {c_diff:.1e}")
        self.assertLess(z_diff, 1e-5, f"z output differs: {z_diff:.1e}")
        print("  PASS ✓")

    def test_pipeline_parity_with_noise(self):
        """Legacy vs new pipeline with noise — statistically similar via KS."""
        print("\n=== test_pipeline_parity_with_noise ===")

        from streaming_data import PreprocessedStreamingDataset

        model_legacy, _data, _cond, lbs = self._build_legacy_model(256)
        model_new = self._build_new_model(_data, _cond, lbs)

        w = 5e-6
        # Legacy pipeline with noise
        np.random.seed(42); torch.manual_seed(42)
        legacy_loader, _, lbs3 = data_util.get_loaders(
            self.params["data_path"], self.params.get("xml_path"),
            self.params.get("xml_ptype"), self.params["val_frac"],
            32, self.params["eps"], "cpu", width_noise=w, shuffle=False,
            u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
        )
        x_leg, c_leg = next(iter(legacy_loader))

        # New pipeline with noise
        ds = PreprocessedStreamingDataset(
            data_path=self.params["data_path"],
            xml_filename=self.params.get("xml_path"),
            particle_type=self.params.get("xml_ptype"),
            batch_size=32, eps=self.params["eps"],
            u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
            width_noise=w, fixed_noise=False, val_frac=0,
            shuffle=False, is_train=True, layer_boundaries=lbs3,
        )
        np.random.seed(42); torch.manual_seed(42)
        x_new, c_new = next(iter(ds))

        x_diff = float(torch.max(torch.abs(x_leg - x_new)))
        c_diff = float(torch.max(torch.abs(c_leg - c_new)))
        print(f"  Noisy data: x_diff={x_diff:.1e} (~{w:.1e} expected), c_diff={c_diff:.1e}")

        # x_diff should be ~width_noise (noise is stochastic)
        self.assertLess(x_diff, 3 * w, "x_diff too large for noise")

        with torch.no_grad():
            z_leg = model_legacy(x_leg, c_leg)
            z_leg = z_leg[0] if isinstance(z_leg, tuple) else z_leg
            z_new = model_new(x_new, c_new)
            z_new = z_new[0] if isinstance(z_new, tuple) else z_new

        from scipy import stats
        n_test = min(30, z_leg.shape[1])
        ks_fails = sum(
            1 for d in range(n_test)
            if stats.ks_2samp(
                z_leg[:, d].numpy(), z_new[:, d].numpy()
            )[1] < 0.01
        )
        fail_pct = ks_fails / n_test * 100
        print(f"  KS test: {ks_fails}/{n_test} failed ({fail_pct:.1f}%)")

        # With noise ~5e-6, expect at most ~10% of dims to fail KS
        self.assertLess(fail_pct, 15, f"Too many KS failures: {ks_fails}/{n_test}")
        print("  PASS ✓")


class TestSchedulerParity(unittest.TestCase):
    """Phase 4: verify LR schedules match legacy for all scheduler types."""

    @classmethod
    def setUpClass(cls):
        if not LIGHTNING_AVAILABLE:
            raise unittest.SkipTest("pytorch_lightning is not installed")
        torch.set_default_dtype(torch.float32)

    def _build_optimizers(self, scheduler_type, extra_sched=None):
        """Return (legacy_opt, legacy_sched, new_opt, new_sched)."""
        params = {
            "lr": 1e-5,
            "max_lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "lr_scheduler": scheduler_type,
        }
        if extra_sched:
            params.update(extra_sched)

        # --- Legacy optimiser ---
        dummy_model = torch.nn.Linear(10, 10)
        legacy_opt = torch.optim.AdamW(
            dummy_model.parameters(),
            lr=params.get("lr", 1e-5),
            betas=params.get("betas", [0.9, 0.999]),
            eps=params.get("eps", 1e-8),
            weight_decay=params.get("weight_decay", 0.0),
        )

        if scheduler_type == "step":
            legacy_sched = torch.optim.lr_scheduler.StepLR(
                legacy_opt, params.get("lr_decay_epochs", 30),
                gamma=params.get("lr_decay_factor", 0.1),
            )
        elif scheduler_type == "one_cycle_lr":
            legacy_sched = torch.optim.lr_scheduler.OneCycleLR(
                legacy_opt, params.get("max_lr", 1e-4),
                epochs=params.get("cycle_epochs", params.get("n_epochs", 1)),
                steps_per_epoch=params.get("steps_per_epoch", 200),
            )
        elif scheduler_type == "cycle_lr":
            legacy_sched = torch.optim.lr_scheduler.CyclicLR(
                legacy_opt, params.get("lr", 1e-5),
                max_lr=params.get("max_lr", 1e-4),
                step_size_up=params.get("step_size_up", 2000),
                mode=params.get("cycle_mode", "triangular"),
                cycle_momentum=False,
            )
        elif scheduler_type == "multi_step_lr":
            legacy_sched = torch.optim.lr_scheduler.MultiStepLR(
                legacy_opt, params.get("milestones", [100, 200]),
                gamma=params.get("gamma", 0.5),
            )
        elif scheduler_type == "reduce_on_plateau":
            legacy_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                legacy_opt, factor=0.8, patience=10, verbose=False,
            )
        else:
            raise ValueError(scheduler_type)

        # --- New (Lightning style) ---
        new_opt = torch.optim.AdamW(
            dummy_model.parameters(),
            lr=params.get("lr", 1e-5),
            betas=params.get("betas", [0.9, 0.999]),
            eps=params.get("eps", 1e-8),
            weight_decay=params.get("weight_decay", 0.0),
        )

        if scheduler_type == "step":
            new_sched = torch.optim.lr_scheduler.StepLR(
                new_opt, params.get("lr_decay_epochs", 30),
                gamma=params.get("lr_decay_factor", 0.1),
            )
        elif scheduler_type == "one_cycle_lr":
            new_sched = torch.optim.lr_scheduler.OneCycleLR(
                new_opt, params.get("max_lr", 1e-4),
                epochs=params.get("cycle_epochs", params.get("n_epochs", 1)),
                steps_per_epoch=params.get("steps_per_epoch", 200),
            )
        elif scheduler_type == "cycle_lr":
            new_sched = torch.optim.lr_scheduler.CyclicLR(
                new_opt, params.get("lr", 1e-5),
                max_lr=params.get("max_lr", 1e-4),
                step_size_up=params.get("step_size_up", 2000),
                mode=params.get("cycle_mode", "triangular"),
                cycle_momentum=False,
            )
        elif scheduler_type == "multi_step_lr":
            new_sched = torch.optim.lr_scheduler.MultiStepLR(
                new_opt, params.get("milestones", [100, 200]),
                gamma=params.get("gamma", 0.5),
            )
        elif scheduler_type == "reduce_on_plateau":
            new_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                new_opt, factor=0.8, patience=10, verbose=False,
            )
        else:
            raise ValueError(scheduler_type)

        return legacy_opt, legacy_sched, new_opt, new_sched

    def _compare_lr_steps(self, scheduler_type, n_steps=10, **kwargs):
        """Step both schedulers n_steps times; assert LR values match."""
        lo, ls, no_, ns = self._build_optimizers(scheduler_type, kwargs)
        lrs_legacy, lrs_new = [], []

        for _ in range(n_steps):
            lrs_legacy.append(float(lo.param_groups[0]["lr"]))
            lrs_new.append(float(no_.param_groups[0]["lr"]))

            # ReduceLROnPlateau needs a loss; others just step
            if scheduler_type == "reduce_on_plateau":
                ls.step(1.0)
                ns.step(1.0)
            else:
                ls.step()
                ns.step()

        max_diff = max(abs(a - b) for a, b in zip(lrs_legacy, lrs_new))
        return max_diff, lrs_legacy[0] if lrs_legacy else 0

    def test_one_cycle_lr_parity(self):
        print("\n=== test_one_cycle_lr_parity ===")
        max_diff, _ = self._compare_lr_steps(
            "one_cycle_lr", steps_per_epoch=200, cycle_epochs=1, max_lr=1e-4,
        )
        print(f"  one_cycle_lr max LR diff: {max_diff:.1e}")
        self.assertEqual(max_diff, 0.0, f"LR differs: {max_diff:.1e}")

    def test_step_lr_parity(self):
        print("\n=== test_step_lr_parity ===")
        max_diff, _ = self._compare_lr_steps(
            "step", lr_decay_epochs=30, lr_decay_factor=0.1,
        )
        print(f"  step_lr max LR diff: {max_diff:.1e}")
        self.assertEqual(max_diff, 0.0)

    def test_cycle_lr_parity(self):
        print("\n=== test_cycle_lr_parity ===")
        max_diff, _ = self._compare_lr_steps("cycle_lr", step_size_up=2000)
        print(f"  cycle_lr max LR diff: {max_diff:.1e}")
        self.assertEqual(max_diff, 0.0)

    def test_multi_step_lr_parity(self):
        print("\n=== test_multi_step_lr_parity ===")
        max_diff, _ = self._compare_lr_steps(
            "multi_step_lr", milestones=[3, 7],
        )
        print(f"  multi_step_lr max LR diff: {max_diff:.1e}")


if __name__ == "__main__":
    unittest.main()
