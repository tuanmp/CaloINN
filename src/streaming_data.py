"""
Memory-efficient streaming datasets for CaloINN training.

This module provides HDF5-backed iterable datasets that avoid loading
the full preprocessed dataset into RAM, solving the memory bottleneck
of the legacy TensorDataset approach.

Key classes:
- HDF5CaloDataset: Base streaming dataset for HDF5 calorimeter data
- PreprocessedStreamingDataset: Applies legacy preprocessing on-the-fly
- LegacyStreamingDataModule: DataModule-compatible class for Lightning integration

Design notes:
- Noise (width_noise) is applied in the dataset's __iter__ method, matching
  the legacy MyDataLoader behavior. The Lightning module's training_step
  should NOT add noise again — see Phase 3 of REFACTOR_PLAN.md.
"""

import math
import os
from typing import Iterator, Tuple, Optional

import h5py
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, IterableDataset


class HDF5CaloDataset(IterableDataset):
    """
    Base memory-efficient HDF5 dataset that streams data without loading
    the full array into RAM.

    IMPORTANT: This dataset applies NO preprocessing. It just reads raw HDF5
    and yields (showers, energies) tuples. Preprocessing is handled by
    PreprocessedStreamingDataset.

    Usage:
        dataset = HDF5CaloDataset(
            file_path="/path/to/data.hdf5",
            indices=valid_indices,  # which samples to use (after filtering)
            batch_size=512,
        )
        for x, c in dataset:
            print(x.shape)  # (batch_size, n_features)
    """

    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        include_geom_features: bool = False,
    ):
        """
        Args:
            file_path: Path to HDF5 file
            indices: Array of valid sample indices (after preprocessing filter)
            batch_size: Batch size for iteration
            shuffle: Whether to shuffle indices each epoch
            drop_last: If True, drop last batch if smaller than batch_size
            include_geom_features: If True, also load phi/theta as conditions
        """
        super().__init__()
        self.file_path = file_path
        self.indices = indices.astype(np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.include_geom_features = include_geom_features

        # Precompute max batch count
        if self.drop_last:
            self.max_batch = len(self.indices) // batch_size
        else:
            self.max_batch = math.ceil(len(self.indices) / batch_size)

    def __len__(self) -> int:
        return self.max_batch

    def _get_shuffled_indices(self, epoch_seed: Optional[int] = None) -> np.ndarray:
        """Get shuffled index array, using seeded RNG for reproducibility."""
        if epoch_seed is not None:
            # Seed-based shuffling for reproducibility across runs
            rng = np.random.RandomState(epoch_seed)
            return rng.permutation(self.indices)
        elif self.shuffle:
            return np.random.permutation(self.indices)
        else:
            return self.indices.copy()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Use a generator to avoid storing large arrays
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError(
                "HDF5CaloDataset does not support multi-worker DataLoader. "
                "Use num_workers=0 for streaming datasets."
            )

        # Open HDF5 file in this worker (lazily)
        with h5py.File(self.file_path, 'r') as f:
            showers = f['showers']
            energies = f['incident_energies']

            # Shuffle indices for this epoch
            epoch_seed = torch.randint(0, 2**31, (1,)).item() if self.shuffle else None
            shuffled_idx = self._get_shuffled_indices(epoch_seed)

            batch_idx = 0
            while batch_idx < self.max_batch:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(shuffled_idx))
                batch_indices = shuffled_idx[start:end]

                # Read batch from HDF5 (this is the streaming part - only load batch, not full dataset)
                batch_showers = showers[batch_indices]  # shape: (batch, n_features)
                batch_energies = energies[batch_indices]  # shape: (batch, 1)

                # Convert to tensors (copy to ensure contiguous)
                x = torch.from_numpy(batch_showers.astype(np.float32)).contiguous()
                c = torch.from_numpy(batch_energies.astype(np.float32)).contiguous()

                yield x, c
                batch_idx += 1


class PreprocessedStreamingDataset(IterableDataset):
    """
    Streaming dataset that applies CaloINN preprocessing on-the-fly.

    This applies the same preprocessing as legacy data_util.preprocess():
    - Layer normalization
    - Extra dimensions computation
    - Filtering (u0up_cut, u0low_cut, dep_cut)

    But does it per-batch rather than loading the full preprocessed array.

    The preprocessing filter is computed on the first pass, then stored
    as an index array. Only valid indices are streamed.

    Usage:
        dataset = PreprocessedStreamingDataset(
            file_path="/path/to/data.hdf5",
            xml_path="./binning.xml",
            particle_type="pion",
            batch_size=512,
            preprocess_kwargs={...},
        )
        for x, c in dataset:
            print(x.shape)  # (batch_size, num_preprocessed_features)
    """

    def __init__(
        self,
        data_path: str,
        xml_filename: str,
        particle_type: str,
        batch_size: int,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        rew: float = 1.0,
        dep_cut: float = 1e10,
        width_noise: float = 0.0,
        fixed_noise: bool = False,
        val_frac: float = 0.01,
        shuffle: bool = True,
        is_train: bool = True,
        drop_last: bool = False,
        precomputed_valid_indices: Optional[np.ndarray] = None,
        layer_boundaries: Optional[np.ndarray] = None,
    ):
        """
        Args:
            data_path: Path to HDF5 data file
            xml_filename: Path to XML binning file
            particle_type: Particle type string (e.g., "pion")
            batch_size: Batch size for iteration
            eps: Epsilon for numerical stability in normalization
            u0up_cut: Upper cut on first extra dimension
            u0low_cut: Lower cut on first extra dimension
            rew: Reweighting power (1.0 = no reweighting)
            dep_cut: Deposit cut threshold
            width_noise: Width of uniform noise to add per sample
            fixed_noise: If True, generate noise once at init and reuse
            val_frac: Fraction of data to use for validation
            shuffle: Whether to shuffle training data each epoch
            is_train: If True, use training split; else use validation split
            drop_last: If True, drop last batch if smaller than batch_size
            precomputed_valid_indices: If provided, skip filter computation and use these indices
            layer_boundaries: If provided, use these layer boundaries; else compute from XML
        """
        super().__init__()
        self.data_path = data_path
        self.xml_filename = xml_filename
        self.particle_type = particle_type
        self.batch_size = batch_size
        self.eps = eps
        self.u0up_cut = u0up_cut
        self.u0low_cut = u0low_cut
        self.rew = rew
        self.dep_cut = dep_cut
        self.width_noise = width_noise
        self.fixed_noise = fixed_noise
        self.val_frac = val_frac
        self.shuffle = shuffle
        self.is_train = is_train
        self.drop_last = drop_last

        # Import here to avoid circular imports
        import data_util

        # Load layer boundaries
        if layer_boundaries is None:
            from caloch_eval.XMLHandler import XMLHandler
            xml_handler = XMLHandler(particle_name=particle_type, filename=xml_filename)
            self.layer_boundaries = np.unique(xml_handler.GetBinEdges())
        else:
            self.layer_boundaries = layer_boundaries

        # Compute valid indices if not provided
        if precomputed_valid_indices is None:
            self.valid_indices, self.num_original = self._compute_filter_indices(data_path)
            # One-time shuffle before split (matching legacy get_loaders:444-454)
            if self.shuffle:
                rng = np.random.RandomState(42)
                self.valid_indices = self.valid_indices[rng.permutation(len(self.valid_indices))]
        else:
            self.valid_indices = precomputed_valid_indices
            self.num_original = len(precomputed_valid_indices)  # Approximation

        # Compute train/val split
        n_total = len(self.valid_indices)
        n_val = int(n_total * val_frac)
        n_train = n_total - n_val

        if is_train:
            self.indices = self.valid_indices[:n_train]
        else:
            self.indices = self.valid_indices[n_train:]

        # Precompute max batch
        if self.drop_last:
            self.max_batch = len(self.indices) // batch_size
        else:
            self.max_batch = math.ceil(len(self.indices) / batch_size)

        # Noise generator setup (matches old MyDataLoader)
        self.noise_distribution = torch.distributions.Uniform(
            torch.tensor(0.0),
            torch.tensor(1.0)
        )
        self.num_preprocessed_features = None  # Set after first batch

    def _compute_filter_indices(self, data_path: str) -> Tuple[np.ndarray, int]:
        """First pass: compute which samples pass the filter.

        This reads the full HDF5 but only keeps indices, not data.
        Returns (valid_indices, num_original).
        """
        import data_util

        with h5py.File(data_path, 'r') as f:
            n_samples = f['incident_energies'].shape[0]

        chunk_size = 100000
        all_valid_indices = []
        num_original = 0

        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        pbar = tqdm.tqdm(
            total=n_samples, unit="samples",
            desc="Computing filter indices", leave=False
        )

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            indices_chunk = np.arange(start, end)

            with h5py.File(self.data_path, 'r') as f:
                chunk_showers = f['showers'][indices_chunk]
                chunk_energies = f['incident_energies'][indices_chunk]

            data = self._build_data_dict(chunk_showers, chunk_energies)

            x_raw = np.concatenate(
                [data[f'layer_{i}'] for i in range(len(data)-1)], axis=1
            )
            energy_raw, _ = data_util.get_energy_and_sorted_layers(data)
            x_raw_energy = np.concatenate(
                [data[f'layer_{i}'] for i in range(len(data)-1)], axis=1
            )
            c_raw = energy_raw
            c_raw, extra_dims = data_util.get_energy_dims(
                x_raw_energy, c_raw, self.layer_boundaries, self.eps
            )

            binary_mask = np.sum(x_raw, axis=1) >= 0
            binary_mask &= extra_dims[:, 0] < self.u0up_cut
            binary_mask &= extra_dims[:, 0] >= self.u0low_cut
            binary_mask &= ((x_raw < self.dep_cut).prod(-1) != 0)

            valid_in_chunk = indices_chunk[binary_mask]
            all_valid_indices.append(valid_in_chunk)
            num_original += len(indices_chunk)

            pbar.update(len(indices_chunk))

        pbar.close()
        valid_indices = np.concatenate(all_valid_indices) if all_valid_indices else np.array([], dtype=np.int64)
        return valid_indices, num_original

    def _build_data_dict(self, showers: np.ndarray, energies: np.ndarray) -> dict:
        """Build data dict from raw arrays (matching HDF5 structure).

        Creates layer_N keys like load_data does, for preprocessing compatibility.

        IMPORTANT: uses Python float 1.e3 (float64) to match legacy dtype
        promotion.  The legacy load_data divides by 1.e3 which promotes
        HDF5 float32 → numpy float64 through the entire preprocessing
        pipeline.  We intentionally replicate this for bit-reproducibility.
        Final casting to float32 happens in the tensor conversion step,
        exactly matching `torch.tensor(x, dtype=torch.get_default_dtype())`.
        """
        E_SCALE = 1.e3  # Python float → float64, matches legacy behavior
        data = {}
        data["energy"] = energies.reshape(-1, 1) / E_SCALE

        # Split showers into layers using layer_boundaries
        for layer_index, (layer_start, layer_end) in enumerate(
            zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])
        ):
            data[f"layer_{layer_index}"] = (
                showers[..., layer_start:layer_end] / E_SCALE
            )

        return data

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterator that yields (x, c) tuples with preprocessing applied.

        IMPORTANT: Matches old MyDataLoader iteration logic exactly:
        - Uses torch.randperm for shuffling (same as old line 66)
        - Adds noise via torch.rand_like * width_noise (same as old line 81)
        - Returns clones to prevent in-place modifications
        """
        import data_util

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError(
                "PreprocessedStreamingDataset does not support multi-worker DataLoader. "
                "Use num_workers=0 for streaming datasets."
            )

        with h5py.File(self.data_path, 'r') as f:
            showers_dataset = f['showers']
            energies_dataset = f['incident_energies']

            # Shuffle indices (same logic as old MyDataLoader line 66)
            if self.shuffle:
                index = torch.randperm(len(self.indices), device='cpu')
            else:
                index = torch.arange(len(self.indices), device='cpu')

            batch_num = 0

            while batch_num < self.max_batch:
                first = batch_num * self.batch_size
                last = min(first + self.batch_size, len(self.indices))
                idx = index[first:last].numpy()

                # Get HDF5 indices in shuffled order (matching legacy MyDataLoader)
                hdf5_unsorted = self.indices[idx]            # shuffled HDF5 rows
                sort_order = np.argsort(hdf5_unsorted)       # permutation that sorts
                hdf5_sorted = hdf5_unsorted[sort_order]      # sorted for efficient HDF5 read

                # Read raw data from HDF5 (sequential read for HDF5 compliance)
                raw_x = showers_dataset[hdf5_sorted]         # (batch, n_features)
                raw_c = energies_dataset[hdf5_sorted]        # (batch, 1)

                # Build data dict for preprocessing
                data = self._build_data_dict(raw_x, raw_c)

                # Apply preprocessing
                x, c = data_util.preprocess(
                    data,
                    self.layer_boundaries,
                    self.eps,
                    u0up_cut=self.u0up_cut,
                    u0low_cut=self.u0low_cut,
                    rew=self.rew,
                    dep_cut=self.dep_cut,
                    verbose=False
                )

                # Restore shuffled batch order (undo the HDF5 sort)
                # Legacy MyDataLoader preserves randperm order within batches;
                # np.argsort(sort_order) inverts the sort to recover it.
                unsort_order = np.argsort(sort_order)
                x = x[unsort_order]
                c = c[unsort_order]

                # Convert to tensors
                x = torch.from_numpy(x.astype(np.float32))
                c = torch.from_numpy(c.astype(np.float32))

                # Apply noise (matches old MyDataLoader lines 80-84)
                # legacy: if not fixed_noise → add fresh noise; else → skip
                if not self.fixed_noise and self.width_noise > 0:
                    noise = self.noise_distribution.sample(x.shape) * self.width_noise
                    x = x + noise

                # Return clones (same as old line 81: torch.clone(self.add_noise(self.data[idx])))
                yield x.clone(), c.clone()

                batch_num += 1


class LegacyStreamingDataModule:
    """
    DataModule-compatible class that provides streaming dataloaders.

    This replaces CaloINNDataModule for memory-efficient training on large datasets.
    It provides the same interface (train_dataloader, val_dataloader) but uses
    streaming datasets that never load the full preprocessed array into RAM.

    Usage:
        dm = LegacyStreamingDataModule(
            data_path="/path/to/train.hdf5",
            val_data_path="/path/to/val.hdf5",
            xml_path="./binning.xml",
            xml_ptype="pion",
            batch_size=2048,
            ...
        )
        dm.setup("fit")
        for x, c in dm.train_dataloader():
            print(x.shape)  # (2048, 540)
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        batch_size: int,
        xml_path: str,
        xml_ptype: str,
        val_frac: float = 0.01,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        rew: float = 1.0,
        dep_cut: float = 1e10,
        width_noise: float = 0.0,
        fixed_noise: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,  # Must be 0 for streaming
        predict_batch_size: int = 1000,
        eval_dataset: str = "1-pions",
        **kwargs,  # Ignore extra kwargs for compatibility
    ):
        """
        Args:
            data_path: Path to training HDF5 file
            val_data_path: Path to validation HDF5 file
            batch_size: Training batch size
            xml_path: Path to XML binning file
            xml_ptype: Particle type string
            val_frac: Validation fraction
            eps: Epsilon for normalization
            u0up_cut: Upper cut on first extra dimension
            u0low_cut: Lower cut on first extra dimension
            rew: Reweighting power
            dep_cut: Deposit cut threshold
            width_noise: Width of noise to add per sample
            fixed_noise: If True, generate noise once at init
            shuffle: Whether to shuffle training data
            num_workers: Must be 0 for streaming datasets
            predict_batch_size: Batch size for prediction
            eval_dataset: Dataset identifier for evaluation
        """
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

        # These will be set during setup()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.num_train_samples = 0

        # Precompute layer boundaries (same as legacy)
        from caloch_eval.XMLHandler import XMLHandler
        xml_handler = XMLHandler(particle_name=xml_ptype, filename=xml_path)
        self.layer_boundaries = np.unique(xml_handler.GetBinEdges())

    def _build_data_dict(self, showers: np.ndarray, energies: np.ndarray) -> dict:
        """Build data dict from raw arrays (matching HDF5 structure).

        Creates layer_N keys like load_data does, for preprocessing compatibility.

        IMPORTANT: uses Python float 1.e3 (float64) to match legacy dtype
        promotion.  The legacy load_data divides by 1.e3 which promotes
        HDF5 float32 → numpy float64 through the entire preprocessing
        pipeline.  We intentionally replicate this for bit-reproducibility.
        Final casting to float32 happens in the tensor conversion step,
        exactly matching `torch.tensor(x, dtype=torch.get_default_dtype())`.
        """
        E_SCALE = 1.e3  # Python float → float64, matches legacy behavior
        data = {}
        data["energy"] = energies.reshape(-1, 1) / E_SCALE

        # Split showers into layers using layer_boundaries
        for layer_index, (layer_start, layer_end) in enumerate(
            zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])
        ):
            data[f"layer_{layer_index}"] = (
                showers[..., layer_start:layer_end] / E_SCALE
            )

        return data

    def _compute_filter_indices_with_cache(self, data_path: str) -> np.ndarray:
        """Compute valid indices with caching to avoid recomputation."""
        cache_path = data_path + ".valid_indices_cache.npy"

        if os.path.exists(cache_path):
            return np.load(cache_path)

        import data_util

        with h5py.File(data_path, 'r') as f:
            n_samples = f['incident_energies'].shape[0]

        chunk_size = 100000
        all_valid_indices = []

        pbar = tqdm.tqdm(
            total=n_samples, unit="samples",
            desc="Computing filter indices (will cache)",
        )

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            indices_chunk = np.arange(start, end)

            with h5py.File(data_path, 'r') as f:
                chunk_showers = f['showers'][indices_chunk]
                chunk_energies = f['incident_energies'][indices_chunk]

            data = self._build_data_dict(chunk_showers, chunk_energies)

            energy, layers = data_util.get_energy_and_sorted_layers(data)
            x_raw = np.concatenate(layers, axis=1)
            c_raw = energy
            c_raw, extra_dims = data_util.get_energy_dims(
                x_raw, c_raw, self.layer_boundaries, self.eps
            )

            binary_mask = np.sum(x_raw, axis=1) >= 0
            binary_mask &= extra_dims[:, 0] < self.u0up_cut
            binary_mask &= extra_dims[:, 0] >= self.u0low_cut
            binary_mask &= ((x_raw < self.dep_cut).prod(-1) != 0)

            all_valid_indices.append(indices_chunk[binary_mask])

            pbar.update(len(indices_chunk))

        pbar.close()
        valid_indices = np.concatenate(all_valid_indices)
        np.save(cache_path, valid_indices)
        return valid_indices

    def setup(self, stage=None):
        """Set up the datasets (same pattern as Lightning DataModule).

        On first call, this may take ~30s to compute valid indices.
        Subsequent calls use cached indices for near-instant setup.
        """
        # Compute valid indices (with caching for performance)
        train_valid = self._compute_filter_indices_with_cache(self.data_path)

        # One-time shuffle before train/val split (matching legacy get_loaders
        # lines 444-454).  Without this, train/val ordering is HDF5-native
        # which may be biased if the file has any structure.
        n_total = len(train_valid)
        if self.shuffle:
            rng = np.random.RandomState(42)
            train_valid = train_valid[rng.permutation(n_total)]

        # Train/val split
        n_val = int(n_total * self.val_frac)
        n_train = n_total - n_val

        train_indices = train_valid[:n_train]
        val_indices = train_valid[n_train:]

        self.num_train_samples = n_train

        # Create streaming datasets
        self._train_dataset = PreprocessedStreamingDataset(
            data_path=self.data_path,
            xml_filename=self.xml_path,
            particle_type=self.xml_ptype,
            batch_size=self.batch_size,
            eps=self.eps,
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut,
            rew=self.rew,
            dep_cut=self.dep_cut,
            width_noise=self.width_noise,
            fixed_noise=self.fixed_noise,
            val_frac=0,  # Already split
            shuffle=self.shuffle,
            is_train=True,
            drop_last=False,
            precomputed_valid_indices=train_indices,
            layer_boundaries=self.layer_boundaries,
        )

        self._val_dataset = PreprocessedStreamingDataset(
            data_path=self.data_path,
            xml_filename=self.xml_path,
            particle_type=self.xml_ptype,
            batch_size=self.batch_size,
            eps=self.eps,
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut,
            rew=self.rew,
            dep_cut=self.dep_cut,
            width_noise=self.width_noise,  # matches legacy: val loader also gets noise
            fixed_noise=False,
            val_frac=0,
            shuffle=False,
            is_train=False,
            drop_last=False,
            precomputed_valid_indices=val_indices,
            layer_boundaries=self.layer_boundaries,
        )

        # Test dataset uses validation file
        test_valid = self._compute_filter_indices_with_cache(self.val_data_path)
        self._test_dataset = PreprocessedStreamingDataset(
            data_path=self.val_data_path,
            xml_filename=self.xml_path,
            particle_type=self.xml_ptype,
            batch_size=self.predict_batch_size,
            eps=self.eps,
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut,
            rew=self.rew,
            dep_cut=self.dep_cut,
            width_noise=0,
            fixed_noise=False,
            val_frac=0,
            shuffle=False,
            is_train=False,  # Use all for test
            drop_last=False,
            precomputed_valid_indices=test_valid,
            layer_boundaries=self.layer_boundaries,
        )

    def train_dataloader(self):
        """Return training dataloader (num_workers must be 0 for streaming)."""
        return DataLoader(
            self._train_dataset,
            batch_size=None,  # Handled by dataset
            shuffle=False,  # Shuffling handled by dataset
            num_workers=0,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self._test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def predict_dataloader(self):
        """Return prediction dataloader (same as test)."""
        return self.test_dataloader()