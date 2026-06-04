import os
import sys

import lightning as pl
import torch

# Force float32 — legacy preprocess promotes float32→float64 via 1.e3,
# and CINN model must receive float32 inputs.  Must be set before CLI
# instantiates anything.
torch.set_default_dtype(torch.float32)

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


class CaloINNLightningCLI(LightningCLI):
    """LightningCLI with legacy argument linking for train/val split parity."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.val_frac", "model.train_val_frac")
        parser.link_arguments("data.batch_size", "model.train_batch_size")
        parser.link_arguments("data.shuffle", "model.train_shuffle")


def main():
    CaloINNLightningCLI(
        model_class=CaloINNLightningModule,
        # datamodule_class=CaloINNDataModule,
        trainer_class=Trainer,
        auto_configure_optimizers=False,
        trainer_defaults={"num_sanity_val_steps": 0},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
