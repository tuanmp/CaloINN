"""Sharded HDF5 datamodule for CaloINN — map-style Dataset with multi-worker support.

Inspired by caloxtreme_clf's LargeHDF5MLPDataModule architecture.
Key improvements over the IterableDataset in streaming_data.py:
- Map-style Dataset → standard DataLoader with num_workers > 0
- Each worker opens its own HDF5 handle (pickle-safe via __getstate__)
- Vectorised __getitems__ for efficient batch preprocessing
- Same preprocessing + noise as legacy MyDataLoader

Data flow:
  Raw HDF5 → split indices → worker opens HDF5 → __getitems__
    → _read_rows (sort/unsort) → data_util.preprocess → add noise → return tensors
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import tqdm
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import data_util

# HDF5 read-cache settings — small enough for per-worker handles
_H5_RDCC = {"rdcc_nbytes": 64 * 1024 * 1024, "rdcc_nslots": 4093}


# ═══════════════════════════════════════════════════════════════════════
# 1.  Efficient sorted HDF5 row reader
# ═══════════════════════════════════════════════════════════════════════

def _read_rows(dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows with internal sort/unsort for I/O speed."""
    if len(indices) == 0:
        return dataset[:0]

    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.size == 1:
        return np.asarray(dataset[ordered])

    # Fast path: already monotonic (common with DataLoader sequential access)
    if np.all(ordered[1:] >= ordered[:-1]):
        return np.asarray(dataset[ordered])

    # Sort → read → unsort to restore caller's order
    sort_order = np.argsort(ordered)
    sorted_indices = ordered[sort_order]
    data = np.asarray(dataset[sorted_indices])
    inverse = np.argsort(sort_order)
    return data[inverse]


# ═══════════════════════════════════════════════════════════════════════
# 2.  Lazy per-worker HDF5 source
# ═══════════════════════════════════════════════════════════════════════

class _RawHDF5Source:
    """Lazy HDF5 reader — one per worker, pickle-safe.

    Opens the file on first access in each worker.  __getstate__ drops
    the file handle so DataLoader multiprocessing works.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        with h5py.File(self.file_path, "r") as handle:
            self._length = int(handle["showers"].shape[0])

    def __len__(self) -> int:
        return self._length

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r", **_H5_RDCC)
        return self._file

    def read_rows(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        handle = self._get_file()
        showers = _read_rows(handle["showers"], indices)
        energies = _read_rows(handle["incident_energies"], indices)
        return showers, energies

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None  # don't pickle the file handle
        return state


# ═══════════════════════════════════════════════════════════════════════
# 3.  Map-style Dataset — preprocessing on access
# ═══════════════════════════════════════════════════════════════════════

class _CaloINNDataset(Dataset):
    """Processes raw HDF5 data on-the-fly per batch.

    Stores ONLY split indices (no data).  Preprocessing happens in
    __getitems__, which is called once per batch by the DataLoader.

    Preprocessing pipeline (matches legacy data_util.preprocess):
      raw showers → _build_data_dict → preprocess → add noise → tensor
    """

    def __init__(
        self,
        source: _RawHDF5Source,
        indices: np.ndarray,
        layer_boundaries: np.ndarray,
        xml_filename: str,
        particle_type: str,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        rew: float = 1.0,
        dep_cut: float = 1e10,
        width_noise: float = 0.0,
        fixed_noise: bool = False,
    ):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.layer_boundaries = layer_boundaries
        self.xml_filename = xml_filename
        self.particle_type = particle_type
        self.eps = eps
        self.u0up_cut = u0up_cut
        self.u0low_cut = u0low_cut
        self.rew = rew
        self.dep_cut = dep_cut
        self.width_noise = width_noise
        self.fixed_noise = fixed_noise

        # Fast noise distribution (same as legacy MyDataLoader)
        self._noise_dist = torch.distributions.Uniform(
            torch.tensor(0.0), torch.tensor(1.0)
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    # ------------------------------------------------------------------
    #  Per-element (DataLoader fallback — not the fast path)
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        x, c = self._get_numpy_batch(np.array([index]))
        return (
            torch.from_numpy(x[0]),
            torch.from_numpy(c[0]),
        )

    # ------------------------------------------------------------------
    #  Per-batch vectorised access (fast path via DataLoader)
    # ------------------------------------------------------------------

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        x, c = self._get_numpy_batch(idx)
        return [
            (torch.from_numpy(x[i]), torch.from_numpy(c[i]))
            for i in range(len(idx))
        ]

    # ------------------------------------------------------------------
    #  Core preprocessing — the same as legacy data_util pipeline
    # ------------------------------------------------------------------

    def _get_numpy_batch(
        self, local_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw data, preprocess, add noise — all vectorised."""
        source_indices = self.indices[local_indices]
        showers, energies = self.source.read_rows(source_indices)

        # Build data dict (matches legacy load_data pattern)
        data = _build_data_dict(showers, energies, self.layer_boundaries)

        # Legacy preprocessing
        x, c = data_util.preprocess(
            data,
            self.layer_boundaries,
            self.eps,
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut,
            rew=self.rew,
            dep_cut=self.dep_cut,
        )

        # Convert to float32 tensors for noise + output
        x = x.astype(np.float32, copy=False)
        c = c.astype(np.float32, copy=False)

        # Add noise (matches legacy MyDataLoader)
        if self.width_noise > 0:
            if self.fixed_noise:
                # Deterministic noise per HDF5 index
                rng = np.random.RandomState(source_indices.astype(np.int64))
                noise = rng.uniform(0, 1, x.shape).astype(np.float32) * self.width_noise
            else:
                noise = np.random.uniform(0, 1, x.shape).astype(np.float32) * self.width_noise
            x = x + noise

        return x, c


# ═══════════════════════════════════════════════════════════════════════
# 4.  Lightning DataModule
# ═══════════════════════════════════════════════════════════════════════

class ShardedCaloINNDataModule(LightningDataModule):
    """Lightning DataModule backed by map-style HDF5 datasets.

    Replaces IterableDataset-based LegacyStreamingDataModule with
    proper multi-worker DataLoader support.  Preprocessing and noise
    happen on-the-fly in workers.

    Can be used directly with LightningCLI or via CaloINNDataModule
    with use_sharded=True.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        batch_size: int,
        xml_path: str,
        xml_ptype: str = "pion",
        val_frac: float = 0.01,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        rew: float = 1.0,
        dep_cut: float = 1e10,
        width_noise: float = 0.0,
        fixed_noise: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        predict_batch_size: int = 1000,
        eval_dataset: str = "1-pions",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.xml_path = xml_path
        self.xml_ptype = xml_ptype
        self.val_frac = val_frac
        self.eps = eps
        self.u0up_cut = u0up_cut
        self.u0low_cut = u0low_cut
        self.rew = rew
        self.dep_cut = dep_cut
        self.width_noise = width_noise
        self.fixed_noise = fixed_noise
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.predict_batch_size = predict_batch_size
        self.eval_dataset = eval_dataset

        # Load layer boundaries once (shared across splits)
        from caloch_eval.XMLHandler import XMLHandler
        xml_handler = XMLHandler(particle_name=xml_ptype, filename=xml_path)
        self.layer_boundaries = np.unique(xml_handler.GetBinEdges())

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.num_train_samples = 0

    # ------------------------------------------------------------------
    #  Split index computation (with caching)
    # ------------------------------------------------------------------

    def _compute_filter_indices(self, data_path: str) -> np.ndarray:
        """Compute valid indices with caching."""
        import hashlib

        param_str = (
            f"{self.u0up_cut}_{self.u0low_cut}_{self.dep_cut}_{self.eps}_"
            f"{self.xml_ptype}_{self.xml_path}"
        )
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        cache_path = f"{data_path}.filter_{param_hash}.npy"

        if os.path.exists(cache_path):
            return np.load(cache_path)

        source = _RawHDF5Source(data_path)
        n_samples = len(source)
        chunk_size = 100000
        all_valid = []

        pbar = tqdm.tqdm(
            total=n_samples, unit="samples",
            desc="Computing filter indices (will cache)",
        )

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            idx = np.arange(start, end)
            showers, energies = source.read_rows(idx)

            data = _build_data_dict(showers, energies, self.layer_boundaries)

            energy, layers = data_util.get_energy_and_sorted_layers(data)
            x_raw = np.concatenate(layers, axis=1)
            c_raw = energy
            c_raw, extra_dims = data_util.get_energy_dims(
                x_raw, c_raw, self.layer_boundaries, self.eps
            )

            mask = np.sum(x_raw, axis=1) >= 0
            mask &= extra_dims[:, 0] < self.u0up_cut
            mask &= extra_dims[:, 0] >= self.u0low_cut
            mask &= ((x_raw < self.dep_cut).prod(-1) != 0)

            all_valid.append(idx[mask])
            pbar.update(len(idx))

        pbar.close()
        valid = np.concatenate(all_valid)
        np.save(cache_path, valid)
        return valid

    # ------------------------------------------------------------------
    #  Lightning DataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        train_valid = self._compute_filter_indices(self.data_path)

        # One-time shuffle before split (matching legacy)
        if self.shuffle:
            rng = np.random.RandomState(42)
            train_valid = train_valid[rng.permutation(len(train_valid))]

        n_total = len(train_valid)
        n_val = int(n_total * self.val_frac)
        n_train = n_total - n_val

        self.num_train_samples = n_train

        if stage in (None, "fit"):
            source = _RawHDF5Source(self.data_path)
            ds_kwargs = dict(
                source=source,
                layer_boundaries=self.layer_boundaries,
                xml_filename=self.xml_path,
                particle_type=self.xml_ptype,
                eps=self.eps,
                u0up_cut=self.u0up_cut,
                u0low_cut=self.u0low_cut,
                rew=self.rew,
                dep_cut=self.dep_cut,
                width_noise=self.width_noise,
                fixed_noise=self.fixed_noise,
            )
            self._train_dataset = _CaloINNDataset(
                indices=train_valid[:n_train], **ds_kwargs
            )
            self._val_dataset = _CaloINNDataset(
                indices=train_valid[n_train:],
                width_noise=self.width_noise,  # legacy val also gets noise
                fixed_noise=self.fixed_noise,
                **{k: v for k, v in ds_kwargs.items()
                   if k not in ("width_noise", "fixed_noise")},
            )

        if stage in (None, "test", "predict"):
            val_valid = self._compute_filter_indices(self.val_data_path)
            val_source = _RawHDF5Source(self.val_data_path)
            self._test_dataset = _CaloINNDataset(
                source=val_source,
                indices=val_valid,
                layer_boundaries=self.layer_boundaries,
                xml_filename=self.xml_path,
                particle_type=self.xml_ptype,
                eps=self.eps,
                u0up_cut=self.u0up_cut,
                u0low_cut=self.u0low_cut,
                rew=self.rew,
                dep_cut=self.dep_cut,
                width_noise=0.0,
                fixed_noise=False,
            )

    def _dl_kwargs(self, shuffle: bool = False) -> dict:
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_dataloader(self):
        return DataLoader(self._train_dataset, **self._dl_kwargs(shuffle=self.shuffle))

    def val_dataloader(self):
        return DataLoader(self._val_dataset, **self._dl_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_data_dict(
    showers: np.ndarray,
    energies: np.ndarray,
    layer_boundaries: np.ndarray,
) -> dict:
    """Build data dict from raw arrays, matching legacy load_data."""
    E_SCALE = 1.e3  # Python float → float64, matches legacy
    data = {}
    data["energy"] = energies.reshape(-1, 1) / E_SCALE
    for li, (ls, le) in enumerate(
        zip(layer_boundaries[:-1], layer_boundaries[1:])
    ):
        data[f"layer_{li}"] = showers[..., ls:le] / E_SCALE
    return data