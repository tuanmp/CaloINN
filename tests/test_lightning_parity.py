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
            datamodule = CaloINNDataModule(copy.deepcopy(self.params))
            datamodule.setup("fit")

            lightning_module = CaloINNLightningModule(
                copy.deepcopy(self.params),
                train_data=datamodule.train_data,
                train_cond=datamodule.train_cond,
                layer_boundaries=datamodule.layer_boundaries,
            )
            lightning_module.model.load_state_dict(copy.deepcopy(legacy.model.state_dict()))
            lightning_module = lightning_module.to("cpu")

        return legacy, datamodule, lightning_module

    def test_dataloader_split_matches_legacy(self):
        legacy, datamodule, _ = self._build_legacy_and_lightning(seed=11, max_events=80)

        self.assertTrue(torch.allclose(datamodule.train_data, legacy.train_loader.data))
        self.assertTrue(torch.allclose(datamodule.train_cond, legacy.train_loader.cond))
        self.assertTrue(torch.allclose(datamodule.val_data, legacy.test_loader.data))
        self.assertTrue(torch.allclose(datamodule.val_cond, legacy.test_loader.cond))
        self.assertEqual(list(datamodule.layer_boundaries), list(legacy.layer_boundaries))

    def test_loss_and_optimizer_step_match_legacy(self):
        legacy, datamodule, lightning_module = self._build_legacy_and_lightning(seed=19, max_events=64)

        x = datamodule.train_data[: self.params["batch_size"]]
        c = datamodule.train_cond[: self.params["batch_size"]]

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
