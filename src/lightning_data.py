import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import data_util


class CaloINNDataModule(pl.LightningDataModule):
    """Lightning data module mirroring the legacy preprocessing/split logic."""

    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        batch_size,
        cond_key: str = "incident_energies",
        sample_key: str = "showers",
        val_frac: float = 0.01,
        shuffle: bool = True,
        eval_dataset: str = "1-pions",
        num_workers=8,
        predict_batch_size=1000,
        dataset_kwargs={},
    ):
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

    def _legacy_kwargs(self):
        return {
            "xml_filename": self.dataset_kwargs.get("xml_path"),
            "particle_type": self.dataset_kwargs.get("xml_ptype"),
            "eps": self.dataset_kwargs.get("eps", 1.0e-10),
            "energy": self.dataset_kwargs.get("single_energy", None),
            "u0up_cut": self.dataset_kwargs.get("u0up_cut", 7.0),
            "u0low_cut": self.dataset_kwargs.get("u0low_cut", 0.0),
            "rew": self.dataset_kwargs.get("pt_rew", 1.0),
            "dep_cut": self.dataset_kwargs.get("dep_cut", 1e10),
        }

    def _load_preprocessed_arrays(self, file_path: str):
        kwargs = self._legacy_kwargs()
        data, layer_boundaries = data_util.load_data(
            file_path,
            kwargs["particle_type"],
            xml_filename=kwargs["xml_filename"],
            energy=kwargs["energy"],
        )
        x, c = data_util.preprocess(
            data,
            layer_boundaries,
            kwargs["eps"],
            u0up_cut=kwargs["u0up_cut"],
            u0low_cut=kwargs["u0low_cut"],
            rew=kwargs["rew"],
            dep_cut=kwargs["dep_cut"],
        )
        return x, c, layer_boundaries

    def setup(self, stage=None):
        dtype = torch.get_default_dtype()

        if stage in (None, "fit"):
            x, c, layer_boundaries = self._load_preprocessed_arrays(
                self.data_path
            )
            self.layer_boundaries = layer_boundaries

            number_of_samples = len(x)
            if self.shuffle:
                full_index = np.random.choice(
                    number_of_samples,
                    number_of_samples,
                    replace=False,
                )
            else:
                full_index = np.arange(number_of_samples)

            number_of_val_samples = int(number_of_samples * self.val_frac)
            number_of_trn_samples = number_of_samples - number_of_val_samples

            trn_index = full_index[:number_of_trn_samples]
            val_index = full_index[number_of_trn_samples:]

            x_trn = torch.tensor(x[trn_index], dtype=dtype)
            c_trn = torch.tensor(c[trn_index], dtype=dtype)
            x_val = torch.tensor(x[val_index], dtype=dtype)
            c_val = torch.tensor(c[val_index], dtype=dtype)

            self._train_dataset = TensorDataset(x_trn, c_trn)
            self._val_dataset = TensorDataset(x_val, c_val)
            self.num_train_samples = int(number_of_trn_samples)

        if stage in (None, "test", "predict"):
            x, c, _ = self._load_preprocessed_arrays(self.val_data_path)
            x_tensor = torch.tensor(x, dtype=dtype)
            c_tensor = torch.tensor(c, dtype=dtype)
            self._test_dataset = TensorDataset(x_tensor, c_tensor)

    def train_dataloader(self):
        assert self._train_dataset is not None, (
            "Call setup('fit') before train_dataloader"
        )
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        assert self._val_dataset is not None, (
            "Call setup('fit') before val_dataloader"
        )
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        assert self._test_dataset is not None, (
            "Call setup('test'|'predict') before requesting test_dataloader"
        )
        return DataLoader(
            self._test_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def predict_dataloader(self):
        assert self._test_dataset is not None, (
            "Call setup('predict') before requesting predict_dataloader"
        )
        return DataLoader(
            self._test_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=0,
        )
