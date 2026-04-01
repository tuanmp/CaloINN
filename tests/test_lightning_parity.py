import copy
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_util
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


if __name__ == "__main__":
    unittest.main()
