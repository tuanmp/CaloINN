import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

import data_util


class CaloINNDataModule(pl.LightningDataModule):
    """Lightning data module that mirrors the legacy data pipeline."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.batch_size = params.get("batch_size", 512)
        self.val_frac = params.get("val_frac", 0.01)
        self.shuffle = params.get("shuffle", True)
        self.drop_last = params.get("drop_last", False)

        self.layer_boundaries = None
        self.train_data = None
        self.train_cond = None
        self.val_data = None
        self.val_cond = None

        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage=None):
        if self._train_dataset is not None and self._val_dataset is not None:
            return

        data, layer_boundaries = data_util.load_data(
            self.params.get("data_path"),
            self.params.get("xml_ptype"),
            xml_filename=self.params.get("xml_path"),
            energy=self.params.get("single_energy", None),
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

        number_of_samples = len(x)
        if self.shuffle:
            full_index = np.random.choice(number_of_samples, number_of_samples, replace=False)
        else:
            full_index = np.arange(number_of_samples)

        number_of_val_samples = int(number_of_samples * self.val_frac)
        number_of_trn_samples = number_of_samples - number_of_val_samples

        trn_index = full_index[:number_of_trn_samples]
        val_index = full_index[number_of_trn_samples:]

        x_trn = x[trn_index]
        c_trn = c[trn_index]
        x_val = x[val_index]
        c_val = c[val_index]

        dtype = torch.get_default_dtype()
        self.train_data = torch.tensor(x_trn, dtype=dtype)
        self.train_cond = torch.tensor(c_trn, dtype=dtype)
        self.val_data = torch.tensor(x_val, dtype=dtype)
        self.val_cond = torch.tensor(c_val, dtype=dtype)

        self.layer_boundaries = layer_boundaries
        self._train_dataset = TensorDataset(self.train_data, self.train_cond)
        self._val_dataset = TensorDataset(self.val_data, self.val_cond)

    @property
    def num_dim(self):
        return int(self.train_data.shape[1])

    @property
    def num_train_samples(self):
        return int(self.train_data.shape[0])

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=0,
        )
