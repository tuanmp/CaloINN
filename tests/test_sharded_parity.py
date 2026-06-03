"""Parity tests for sharded_data.py — map-style Dataset vs legacy pipeline."""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "caloch_eval"))

import numpy as np
import torch
import yaml

import data_util
from sharded_data import (
    ShardedCaloINNDataModule,
    _CaloINNDataset,
    _RawHDF5Source,
    _build_data_dict,
)


class TestShardedDataParity(unittest.TestCase):

    CONFIG_PATH = os.path.join(REPO_ROOT, "params", "pions.yaml")

    @classmethod
    def setUpClass(cls):
        with open(cls.CONFIG_PATH) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if not os.path.exists(cfg.get("data_path", "")):
            raise unittest.SkipTest("pions dataset not available")
        cls.cfg = cfg

    # ------------------------------------------------------------------
    #  1.  Same data from legacy get_loaders and sharded dataset
    # ------------------------------------------------------------------

    def test_data_parity_same_indices(self):
        """Same HDF5 indices → sharded produces same x, c as legacy."""
        print("\n=== test_data_parity_same_indices ===")

        seed = 42
        xml_path = self.cfg.get("xml_path")

        # Legacy loader
        np.random.seed(seed); torch.manual_seed(seed)
        loader, _, lbs = data_util.get_loaders(
            self.cfg["data_path"], xml_path, self.cfg.get("xml_ptype"),
            0.1, 32, 1e-10, "cpu", width_noise=0.0, shuffle=False,
            u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
        )
        xl, cl = next(iter(loader))

        # Sharded — same indices (first 32 of train set, no shuffle)
        source = _RawHDF5Source(self.cfg["data_path"])
        ds = _CaloINNDataset(
            source=source, indices=np.arange(32),
            layer_boundaries=lbs, xml_filename=xml_path,
            particle_type=self.cfg.get("xml_ptype"),
            eps=1e-10, u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1e10,
            width_noise=0.0, fixed_noise=False,
        )
        xs, cs = ds._get_numpy_batch(np.arange(32))
        xs = torch.from_numpy(xs)
        cs = torch.from_numpy(cs)

        xd = float(torch.max(torch.abs(xl - xs)))
        cd = float(torch.max(torch.abs(cl - cs)))
        print(f"  x max diff: {xd:.2e}, c max diff: {cd:.2e}")

        self.assertLess(xd, 1e-5, f"x mismatch: {xd:.1e}")
        self.assertLess(cd, 1e-5, f"c mismatch: {cd:.1e}")
        print("  PASS ✓")

    # ------------------------------------------------------------------
    #  2.  Same noise with fixed_noise=True
    # ------------------------------------------------------------------

    def test_fixed_noise_parity(self):
        """fixed_noise=True → same noise for same sample across calls."""
        print("\n=== test_fixed_noise_parity ===")

        xml_path = self.cfg.get("xml_path")
        source = _RawHDF5Source(self.cfg["data_path"])

        # Two datasets with same indices, fixed_noise
        ds1 = _CaloINNDataset(
            source=source, indices=np.arange(64),
            layer_boundaries=np.array([0, 240, 540]),  # rough
            xml_filename=xml_path,
            particle_type=self.cfg.get("xml_ptype"),
            width_noise=5e-6, fixed_noise=True,
        )
        ds2 = _CaloINNDataset(
            source=source, indices=np.arange(64),
            layer_boundaries=np.array([0, 240, 540]),
            xml_filename=xml_path,
            particle_type=self.cfg.get("xml_ptype"),
            width_noise=5e-6, fixed_noise=True,
        )

        x1, c1 = ds1._get_numpy_batch(np.arange(32))
        x2, c2 = ds2._get_numpy_batch(np.arange(32))
        xd = float(np.max(np.abs(x1 - x2)))
        print(f"  x max diff (fixed_noise): {xd:.2e}")
        self.assertEqual(xd, 0.0, "fixed_noise should produce identical noise")
        print("  PASS ✓")

    # ------------------------------------------------------------------
    #  3.  Fresh noise per call with fixed_noise=False
    # ------------------------------------------------------------------

    def test_fresh_noise_per_call(self):
        """fixed_noise=False → different noise each get_numpy_batch call."""
        print("\n=== test_fresh_noise_per_call ===")

        xml_path = self.cfg.get("xml_path")
        source = _RawHDF5Source(self.cfg["data_path"])

        ds = _CaloINNDataset(
            source=source, indices=np.arange(64),
            layer_boundaries=np.array([0, 240, 540]),
            xml_filename=xml_path,
            particle_type=self.cfg.get("xml_ptype"),
            width_noise=5e-6, fixed_noise=False,
        )

        x1, _ = ds._get_numpy_batch(np.arange(32))
        x2, _ = ds._get_numpy_batch(np.arange(32))
        same = np.array_equal(x1, x2)
        print(f"  Same x? {same}")
        self.assertFalse(same, "fresh noise should differ per call")
        print("  PASS ✓")

    # ------------------------------------------------------------------
    #  4.  DataModule setup produces train/val splits
    # ------------------------------------------------------------------

    def test_datamodule_splits(self):
        """ShardedCaloINNDataModule.setup creates train + val datasets."""
        print("\n=== test_datamodule_splits ===")

        dm = ShardedCaloINNDataModule(
            data_path=self.cfg["data_path"],
            val_data_path=self.cfg["data_path"],
            batch_size=64, val_frac=0.05,
            xml_path=self.cfg.get("xml_path"),
            xml_ptype=self.cfg.get("xml_ptype"),
            eps=1e-10, u0up_cut=7.0, u0low_cut=0.0,
            rew=1.0, dep_cut=1e10, width_noise=0.0, shuffle=True,
        )
        dm.setup("fit")

        self.assertIsNotNone(dm._train_dataset)
        self.assertIsNotNone(dm._val_dataset)
        self.assertGreater(len(dm._train_dataset), 0)
        self.assertGreater(dm.num_train_samples, 0)

        train_batch = next(iter(dm.train_dataloader()))
        self.assertEqual(train_batch[0].shape[0], 64)

        print(f"  Train: {dm.num_train_samples}, "
              f"batch: {train_batch[0].shape}")
        print("  PASS ✓")


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    unittest.main(verbosity=2)