import os
import sys
from datetime import datetime

import lightning as pl
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from lightning_data import CaloINNDataModule
from lightning_module import CaloINNLightningModule

try:
    from lightning.pytorch.cli import LightningCLI
except ImportError:
    from pytorch_lightning.cli import LightningCLI

from training_utils.trainer import Trainer


class ParamFileDataModule(CaloINNDataModule):
    def __init__(self, param_file: str):
        with open(param_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        super().__init__(params)


class ParamFileLightningModule(CaloINNLightningModule):
    def __init__(
        self,
        param_file: str,
        predict_num_samples: int = 100000,
        predict_batch_size: int = 1000,
        predict_output_file: str = "samples.hdf5",
    ):
        with open(param_file, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        dtype = params.get("dtype", "")
        if dtype == "float64":
            torch.set_default_dtype(torch.float64)
        elif dtype == "float16":
            torch.set_default_dtype(torch.float16)
        elif dtype == "float32":
            torch.set_default_dtype(torch.float32)

        pl.seed_everything(params.get("seed", 0), workers=True)

        super().__init__(
            params=params,
            predict_num_samples=predict_num_samples,
            predict_batch_size=predict_batch_size,
            predict_output_file=predict_output_file,
        )


class CaloINNLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.val_frac", "model.train_val_frac")
        parser.link_arguments("data.batch_size", "model.train_batch_size")
        parser.link_arguments("data.shuffle", "model.train_shuffle")



def main():
    CaloINNLightningCLI(
        model_class=CaloINNLightningModule,
        datamodule_class=CaloINNDataModule,
        trainer_class=Trainer,
        auto_configure_optimizers=False,
        trainer_defaults={"num_sanity_val_steps": 0},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
