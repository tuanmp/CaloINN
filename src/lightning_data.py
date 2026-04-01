import h5py
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import data_util


class HDF5IterableDataset(IterableDataset):
    """Iterable dataset that reads from an HDF5 file on the fly."""

    def __init__(self, file_path, batch_size, params, cond_key: str="incident_energies", sample_key: str="showers", index_start: int=None, index_stop:int=None, shuffle: bool=True):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.chunk_size = batch_size
        self.cond_key = cond_key
        self.sample_key = sample_key
        self.index_start = index_start
        self.index_stop = index_stop
        self.shuffle = shuffle
        self.params = params
        # Debug instrumentation for notebook diagnostics.
        self.last_rng_seed = None
        self.last_indices_head = None

        
        # Extract width_noise from params (legacy approach: add noise during data loading)
        self.width_noise = params.get("width_noise", 1e-7)
        if self.index_start is not None and self.index_stop is not None:
            self.num_samples = self.index_stop - self.index_start
        else:
            with data_util.h5py.File(self.file_path, "r") as f:
                self.num_samples = f[self.cond_key].shape[0]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        range_start = int(self.index_start) if self.index_start is not None else 0
        range_stop = int(self.index_stop) if self.index_stop is not None else range_start + int(self.num_samples)
        total_samples = max(0, range_stop - range_start)

        # Use a local RNG seeded from torch's worker seed to make shuffle reproducible.
        if worker_info is None:
            rng_seed = int(torch.initial_seed())
        else:
            rng_seed = int(worker_info.seed)
        self.last_rng_seed = rng_seed
        rng = np.random.default_rng(rng_seed)

        with h5py.File(self.file_path, "r") as f:
            if worker_info is None:
                start_idx, end_idx = range_start, range_stop
            else:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                samples_per_worker = (total_samples + num_workers - 1) // num_workers
                start_idx = range_start + worker_id * samples_per_worker
                end_idx = min(start_idx + samples_per_worker, range_stop)

            indices = np.arange(start_idx, end_idx)
            if self.shuffle:
                indices = rng.permutation(indices)
            self.last_indices_head = indices[: min(16, len(indices))].copy()

            for chunk_start in range(0, len(indices), self.chunk_size):
                idx = indices[chunk_start: chunk_start + self.chunk_size]
                if len(idx) == 0:
                    continue
                # chunk_cond = cond_data[chunk_start:chunk_end]    # read once from disk
                # chunk_samples = sample_data[chunk_start:chunk_end]

                data, layer_boundaries = data_util.load_data(
                    f,
                    self.params.get("xml_ptype"),
                    xml_filename=self.params.get("xml_path"),
                    energy=self.params.get("single_energy", None),
                    indices=idx,
                )

                x, c = data_util.preprocess(
                    data,
                    layer_boundaries,
                    self.params.get("eps", 1.0e-10),
                    u0up_cut=self.params.get("u0up_cut", 7.0),
                    u0low_cut=self.params.get("u0low_cut", 0.0),
                    rew=self.params.get("pt_rew", 1.0),
                    dep_cut=self.params.get("dep_cut", 1.0e10),
                )

                dtype = torch.get_default_dtype()
                
                # Apply noise to x (following legacy MyDataLoader logic)
                x_tensor = torch.tensor(x, dtype=dtype)
                if self.width_noise > 0:
                    noise = torch.rand_like(x_tensor) * self.width_noise
                    x_tensor = x_tensor + noise
                c_tensor = torch.tensor(c, dtype=dtype)

                yield x_tensor, c_tensor



class CaloINNDataModule(pl.LightningDataModule):
    """Lightning data module that mirrors the legacy data pipeline."""

    def __init__(self, data_path: str, val_data_path: str, 
                 batch_size, cond_key: str="incident_energies", 
                 sample_key: str="showers", val_frac=0.01, 
                 shuffle=True, eval_dataset: str="1-pions", 
                 num_workers=8,
                 predict_batch_size=1000, dataset_kwargs={}):
        super().__init__()
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.shuffle = shuffle
        self.eval_dataset = eval_dataset
        self.predict_batch_size = predict_batch_size
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.cond_key = cond_key
        self.sample_key = sample_key
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            with data_util.h5py.File(self.data_path, "r") as f:
                num_samples = f[self.cond_key].shape[0]

                train_samples = int(num_samples * (1 - self.val_frac))

                self._train_dataset = self.get_dataset('train', (0, train_samples))
                self._val_dataset = self.get_dataset('val', (train_samples, num_samples))
                self.num_train_samples = train_samples
        if stage in (None, "test", "predict"):
            self._test_dataset = self.get_dataset('test')

    def get_dataset(self, dataset="train", split: tuple=None):
        if dataset in ["train", "val"]:
            return HDF5IterableDataset(
                file_path=self.data_path,
                batch_size=self.batch_size,
                params=self.dataset_kwargs,
                cond_key=self.cond_key,
                sample_key=self.sample_key,
                index_start=split[0],
                index_stop=split[1],
                shuffle=self.shuffle,
            )
        elif dataset == "test":
            return HDF5IterableDataset(
                file_path=self.val_data_path,
                batch_size=self.predict_batch_size,
                params=self.dataset_kwargs,
                cond_key=self.cond_key,
                sample_key=self.sample_key,
                shuffle=False,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )
