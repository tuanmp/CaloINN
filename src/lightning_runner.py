import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from lightning_data import CaloINNDataModule
from lightning_module import CaloINNLightningModule


class LegacyStateDictCheckpoint(Callback):
    """Save model state dicts using legacy file names."""

    def __init__(self, doc, save_interval, n_epochs):
        super().__init__()
        self.doc = doc
        self.save_interval = int(save_interval)
        self.n_epochs = int(n_epochs)

    def _save(self, pl_module, suffix=""):
        torch.save(
            {"net": pl_module.model.state_dict()},
            self.doc.get_file(f"model{suffix}.pt"),
        )

    def on_train_start(self, trainer, pl_module):
        self._save(pl_module, "")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.save_interval == 0 or epoch == self.n_epochs:
            self._save(pl_module, str(epoch))

    def on_train_end(self, trainer, pl_module):
        self._save(pl_module, "_last")


class LightningRunner:
    """Compatibility layer around Lightning for existing CLI flow."""

    def __init__(self, params, device, doc):
        self.params = params
        self.device = device
        self.doc = doc

        self.datamodule = CaloINNDataModule(params)
        self.datamodule.setup("fit")

        self.module = CaloINNLightningModule(
            params,
            train_data=self.datamodule.train_data,
            train_cond=self.datamodule.train_cond,
            layer_boundaries=self.datamodule.layer_boundaries,
        )
        self.module = self.module.to(device)

        accelerator = "gpu" if str(device).startswith("cuda") else "cpu"
        grad_clip = params.get("grad_clip", None)

        callbacks = [
            LegacyStateDictCheckpoint(
                doc=doc,
                save_interval=params.get("save_interval", 20),
                n_epochs=params.get("n_epochs", 1),
            )
        ]

        trainer_kwargs = {
            "max_epochs": params.get("n_epochs", 1),
            "accelerator": accelerator,
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "callbacks": callbacks,
            "log_every_n_steps": 1,
        }
        if grad_clip is not None:
            trainer_kwargs["gradient_clip_val"] = grad_clip

        self.trainer = pl.Trainer(**trainer_kwargs)

    def train(self):
        self.trainer.fit(self.module, datamodule=self.datamodule)

    def save(self, epoch=""):
        torch.save({"net": self.module.model.state_dict()}, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        state_dicts = torch.load(self.doc.get_file(f"model{epoch}.pt"), map_location=self.device)
        self.module.model.load_state_dict(state_dicts["net"])
        self.module = self.module.to(self.device)

    def generate(self, num_samples):
        return self.module.generate(
            num_samples=num_samples,
            output_file=self.doc.get_file("samples.hdf5"),
        )

    def plot_default_from_caloch(self, sample_name="samples.hdf5", eval_name="final", cut=1.515e-3):
        self.module.plot_default_from_caloch(
            base_dir=self.doc.basedir,
            sample_name=sample_name,
            eval_name=eval_name,
            cut=cut,
        )
