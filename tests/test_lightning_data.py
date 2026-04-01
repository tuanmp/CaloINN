import os
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_util
from lightning_data import CaloINNDataModule, HDF5IterableDataset


class TestHDF5IterableDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.tmpdir.name, "toy.h5")

        # 10 samples, 6 features
        energies = np.arange(10, dtype=np.float32).reshape(-1, 1)
        showers = np.arange(60, dtype=np.float32).reshape(10, 6)
        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset("incident_energies", data=energies)
            f.create_dataset("showers", data=showers)

        self.params = {
            "xml_ptype": "pion",
            "xml_path": "dummy.xml",
            "single_energy": None,
            "eps": 1.0e-10,
            "u0up_cut": 7.0,
            "u0low_cut": 0.0,
            "pt_rew": 1.0,
            "dep_cut": 1.0e10,
        }

        self.seen_indices = []
        self._orig_load_data = data_util.load_data
        self._orig_preprocess = data_util.preprocess

        def _fake_load_data(data_file, particle_type, xml_filename, threshold=1e-5, energy=None, indices=None):
            self.assertIsInstance(data_file, h5py.File)
            self.assertEqual(particle_type, self.params["xml_ptype"])
            self.assertEqual(xml_filename, self.params["xml_path"])
            self.assertIsNotNone(indices)
            self.seen_indices.extend(indices.tolist())

            x = data_file["showers"][indices]
            c = data_file["incident_energies"][indices]
            data = {
                "energy": c,
                "layer_0": x,
            }
            layer_boundaries = np.array([0, x.shape[1]])
            return data, layer_boundaries

        def _fake_preprocess(data, layer_boundaries, eps=1.0e-10, u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1.0e10):
            return data["layer_0"], data["energy"]

        data_util.load_data = _fake_load_data
        data_util.preprocess = _fake_preprocess

    def tearDown(self):
        data_util.load_data = self._orig_load_data
        data_util.preprocess = self._orig_preprocess
        self.tmpdir.cleanup()

    def test_num_samples_with_index_range(self):
        dataset = HDF5IterableDataset(
            file_path=self.h5_path,
            batch_size=4,
            params=self.params,
            index_start=2,
            index_stop=9,
            shuffle=False,
        )
        self.assertEqual(dataset.num_samples, 7)

    def test_iter_yields_tensor_batches_and_drops_incomplete_chunk(self):
        dataset = HDF5IterableDataset(
            file_path=self.h5_path,
            batch_size=4,
            params=self.params,
            index_start=0,
            index_stop=10,
            shuffle=False,
        )

        batches = list(iter(dataset))
        self.assertEqual(len(batches), 2)

        for x, c in batches:
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(c, torch.Tensor)
            self.assertEqual(x.dtype, torch.get_default_dtype())
            self.assertEqual(c.dtype, torch.get_default_dtype())
            self.assertEqual(tuple(x.shape), (4, 6))
            self.assertEqual(tuple(c.shape), (4, 1))

        self.assertEqual(self.seen_indices, list(range(8)))


class TestCaloINNDataModule(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.train_h5 = os.path.join(self.tmpdir.name, "train.h5")
        self.val_h5 = os.path.join(self.tmpdir.name, "val.h5")

        with h5py.File(self.train_h5, "w") as f:
            f.create_dataset("incident_energies", data=np.arange(20, dtype=np.float32).reshape(-1, 1))
            f.create_dataset("showers", data=np.random.randn(20, 6).astype(np.float32))

        with h5py.File(self.val_h5, "w") as f:
            f.create_dataset("incident_energies", data=np.arange(12, dtype=np.float32).reshape(-1, 1))
            f.create_dataset("showers", data=np.random.randn(12, 6).astype(np.float32))

        self.params = {
            "data_path": self.train_h5,
            "val_data_path": self.val_h5,
            "batch_size": 5,
            "val_frac": 0.25,
            "num_workers": 0,
            "cond_key": "incident_energies",
            "sample_key": "showers",
            "xml_ptype": "pion",
            "xml_path": "dummy.xml",
        }

        # Setup mocks for data loading
        self._orig_load_data = data_util.load_data
        self._orig_preprocess = data_util.preprocess

        def _fake_load_data(data_file, particle_type, xml_filename, threshold=1e-5, energy=None, indices=None):
            # Sort indices for h5py (which requires increasing order)
            sorted_indices = np.sort(indices)
            x = data_file["showers"][sorted_indices]
            c = data_file["incident_energies"][sorted_indices]
            data = {
                "energy": c,
                "layer_0": x,
            }
            layer_boundaries = np.array([0, x.shape[1]])
            return data, layer_boundaries

        def _fake_preprocess(data, layer_boundaries, eps=1.0e-10, u0up_cut=7.0, u0low_cut=0.0, rew=1.0, dep_cut=1.0e10):
            return data["layer_0"], data["energy"]

        data_util.load_data = _fake_load_data
        data_util.preprocess = _fake_preprocess

    def tearDown(self):
        data_util.load_data = self._orig_load_data
        data_util.preprocess = self._orig_preprocess
        self.tmpdir.cleanup()

    def test_setup_creates_train_and_val_datasets(self):
        """Test that setup() correctly computes splits and creates datasets."""
        dm = CaloINNDataModule(self.params)
        dm.setup("fit")

        # Verify datasets are created
        self.assertIsNotNone(dm._train_dataset)
        self.assertIsNotNone(dm._val_dataset)
        self.assertIsInstance(dm._train_dataset, HDF5IterableDataset)
        self.assertIsInstance(dm._val_dataset, HDF5IterableDataset)

        # Verify split computation: 20 samples, val_frac=0.25 -> train=15, val=5
        self.assertEqual(dm._train_dataset.num_samples, 15)
        self.assertEqual(dm._val_dataset.num_samples, 5)

        # Verify index ranges
        self.assertEqual(dm._train_dataset.index_start, 0)
        self.assertEqual(dm._train_dataset.index_stop, 15)
        self.assertEqual(dm._val_dataset.index_start, 15)
        self.assertEqual(dm._val_dataset.index_stop, 20)

    def test_get_dataset_with_explicit_split(self):
        """Test that get_dataset() accepts split tuples and creates correct datasets."""
        dm = CaloINNDataModule(self.params)

        # Explicitly call get_dataset with split tuples (as called by setup())
        train_ds = dm.get_dataset("train", (0, 15))
        val_ds = dm.get_dataset("val", (15, 20))
        test_ds = dm.get_dataset("test")

        self.assertIsInstance(train_ds, HDF5IterableDataset)
        self.assertIsInstance(val_ds, HDF5IterableDataset)
        self.assertIsInstance(test_ds, HDF5IterableDataset)

        # Verify file paths
        self.assertEqual(train_ds.file_path, self.train_h5)
        self.assertEqual(val_ds.file_path, self.train_h5)
        self.assertEqual(test_ds.file_path, self.val_h5)

        # Verify shuffle settings
        self.assertTrue(train_ds.shuffle)
        self.assertTrue(val_ds.shuffle)
        self.assertFalse(test_ds.shuffle)

        # Verify index ranges
        self.assertEqual(train_ds.index_start, 0)
        self.assertEqual(train_ds.index_stop, 15)
        self.assertEqual(val_ds.index_start, 15)
        self.assertEqual(val_ds.index_stop, 20)

    def test_train_and_val_dataloaders(self):
        """Test that dataloaders are created from stored datasets."""
        dm = CaloINNDataModule(self.params)
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        self.assertIsInstance(train_loader.dataset, HDF5IterableDataset)
        self.assertIsInstance(val_loader.dataset, HDF5IterableDataset)
        self.assertEqual(train_loader.batch_size, None)
        self.assertEqual(val_loader.batch_size, None)
        self.assertEqual(train_loader.num_workers, self.params["num_workers"])

    def test_train_dataloader_batch_sizes(self):
        """Test that train dataloader returns batches of correct size (5)."""
        dm = CaloINNDataModule(self.params)
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        batches = list(train_loader)

        # 15 train samples with batch_size=5 -> 3 complete batches
        self.assertEqual(len(batches), 3)

        for x_batch, c_batch in batches:
            self.assertEqual(x_batch.shape[0], 5, f"Expected batch size 5, got {x_batch.shape[0]}")
            self.assertEqual(c_batch.shape[0], 5, f"Expected batch size 5, got {c_batch.shape[0]}")
            self.assertEqual(x_batch.shape[1], 6, "Expected feature dimension 6 for showers")
            self.assertEqual(c_batch.shape[1], 1, "Expected feature dimension 1 for energies")

    def test_val_dataloader_batch_sizes(self):
        """Test that val dataloader returns batches of correct size (5)."""
        dm = CaloINNDataModule(self.params)
        dm.setup("fit")

        val_loader = dm.val_dataloader()
        batches = list(val_loader)

        # 5 val samples with batch_size=5 -> 1 complete batch
        self.assertEqual(len(batches), 1)

        x_batch, c_batch = batches[0]
        self.assertEqual(x_batch.shape[0], 5, f"Expected batch size 5, got {x_batch.shape[0]}")
        self.assertEqual(c_batch.shape[0], 5, f"Expected batch size 5, got {c_batch.shape[0]}")
        self.assertEqual(x_batch.shape[1], 6, "Expected feature dimension 6 for showers")
        self.assertEqual(c_batch.shape[1], 1, "Expected feature dimension 1 for energies")

    def test_total_samples_from_dataloaders(self):
        """Test that total samples from dataloaders match expected counts."""
        dm = CaloINNDataModule(self.params)
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Count total samples from train dataloader
        train_samples = sum(x.shape[0] for x, _ in train_loader)
        self.assertEqual(train_samples, 15, "Train dataloader should return 15 total samples")

        # Count total samples from val dataloader
        val_samples = sum(x.shape[0] for x, _ in val_loader)
        self.assertEqual(val_samples, 5, "Val dataloader should return 5 total samples")


if __name__ == "__main__":
    unittest.main()
