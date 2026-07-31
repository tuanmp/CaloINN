from collections.abc import Mapping
from functools import partial
from typing import Any, Optional, Union

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from sklearn.metrics import brier_score_loss
from torchmetrics import AUROC, MetricCollection

from lightning_module import CaloINNLightningModule
from mcmc.calibration import expected_calibration_error
from mcmc.convert import cinn_sample_to_classifier_input_torch


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int=256,
        num_layers: int=4,
        batch_norm: bool=False,
        layer_norm: bool=True,
        output_dim: int=1,
        dropout: float=0.,
        activation: str="relu",
    ):
        super().__init__()

        assert not (batch_norm and layer_norm), "Batch normalization and layer normalization cannot be used together."

        activation_layers = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in activation_layers:
            raise ValueError(f"Unsupported activation '{activation}'. Available: {sorted(activation_layers.keys())}")

        modules = []

        input_projector = nn.LazyLinear(hidden_dim)
        modules.append(input_projector)

        for _ in range(num_layers):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_dim))
            modules.append(activation_layers[activation]())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*modules)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.output_layer(x)
        return x


class CaloINNCLF(pl.LightningModule):
    """CaloINN with classifier density ratio correction.

    This class extends CaloINNLightningModule to include a classifier
    for density ratio estimation, enabling Acceptance-Rejection sampling.
    The class loads a pre-trained CaloINN model and adds a Multi-Layer Perceptron (MLP)
    """

    def __init__(self, 
        hidden_dim: int=256,
        num_layers: int=4,
        batch_norm: bool=False,
        layer_norm: bool=True,
        output_dim: int=1,
        dropout: float=0.,
        activation: str="relu",
        generator_ckpt_path: str=None,
        voxel_energy_cutoff: float=None,
        log_transform: bool=False,
        lr: float=1e-3,
        step_size: int=10,
        gamma: float=0.95,
        amsgrad: bool=True,
        kw_overrides: Optional[dict]={},
        **kwargs
    ):
        super().__init__(**kwargs)
        if generator_ckpt_path is None:
            raise ValueError("generator_ckpt_path must be provided for CaloINNCLF.")
        rank_zero_info(f"🔄 Loading generator from {generator_ckpt_path}")
        # ckpt = torch.load(generator_ckpt_path, map_location="cpu")
        # self.generator = CaloINNLightningModule(**ckpt["hyper_parameters"])
        # self.generator.load_state_dict(ckpt["state_dict"])
        self.generator = CaloINNLightningModule.load_from_checkpoint(generator_ckpt_path, **kw_overrides)
        rank_zero_info(f"   ✅ Generator loaded. Model has {sum(p.numel() for p in self.generator.parameters())} parameters.")
        self.net = MLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation
        )

        self.convert_func = partial(
            cinn_sample_to_classifier_input_torch,
            layer_boundaries=self.generator.layer_boundaries,
            q=self.generator.q,
            width_noise=self.generator.width_noise,
            xml_path=self.generator.xml_path,
            particle=self.generator.xml_ptype,
            log_transform=log_transform,
            voxel_energy_cutoff=voxel_energy_cutoff
        )

        self.metrics = MetricCollection({
            "auroc": AUROC(task="binary")
        })

        # Accumulators for epoch-level Brier / ECE (not batch-level)
        self._val_y_hat: list = []
        self._val_y: list = []
        self._test_y_hat: list = []
        self._test_y: list = []

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.amsgrad = amsgrad

        self.save_hyperparameters(ignore=["generator", "net", "convert_func", "metrics", "_val_y_hat", "_val_y", "_test_y_hat", "_test_y"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def get_input_from_batch(self, batch):

        x_truth, c = batch
        x_gen = self.generator.conditional_generate(c, measure_gen_time=False).squeeze(1)

        y_truth = torch.ones(x_truth.shape[0], 1, device=x_truth.device)
        y_gen = torch.zeros(x_gen.shape[0], 1, device=x_gen.device)

        X = torch.cat([x_truth, x_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(2, 1)

        X_input = self.convert_func(X, c)

        return X_input, y
    
    def criterion(self, logits, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    
    def training_step(self, batch, batch_idx):
        X_input, y = self.get_input_from_batch(batch)
        logits = self(X_input)
        loss = self.criterion(logits, y)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.log('train_loss', outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    
    def validation_step(self, batch, batch_idx):
        X_input, y = self.get_input_from_batch(batch)
        logits = self(X_input)
        loss = self.criterion(logits, y)
        return {
            "loss": loss,
            "y_hat": logits,
            "y": y
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('val_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics['auroc'].update(y_hat, y)
        # Accumulate for epoch-level Brier/ECE (not computed per batch)
        self._val_y_hat.append(torch.sigmoid(y_hat).detach().cpu().numpy().flatten())
        self._val_y.append(y.detach().cpu().numpy().flatten())

    def on_validation_epoch_end(self) -> None:
        # AUROC
        self.log("val_aucroc", self.metrics["auroc"].compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        # Brier / ECE — computed once over all batches
        y_hat_all = np.concatenate(self._val_y_hat)
        y_all = np.concatenate(self._val_y)
        self.log('val_brier', brier_score_loss(y_all, y_hat_all), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ece', expected_calibration_error(y_all, y_hat_all), on_epoch=True, prog_bar=True, logger=True)
        self._val_y_hat.clear()
        self._val_y.clear()
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('test_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics['auroc'].update(y_hat, y)
        # Accumulate for epoch-level Brier/ECE (not computed per batch)
        self._test_y_hat.append(torch.sigmoid(y_hat).detach().cpu().numpy().flatten())
        self._test_y.append(y.detach().cpu().numpy().flatten())

    def on_test_epoch_end(self) -> None:
        self.log("test_aucroc", self.metrics["auroc"].compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        y_hat_all = np.concatenate(self._test_y_hat)
        y_all = np.concatenate(self._test_y)
        self.log('test_brier', brier_score_loss(y_all, y_hat_all), on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ece', expected_calibration_error(y_all, y_hat_all), on_epoch=True, prog_bar=True, logger=True)
        self._test_y_hat.clear()
        self._test_y.clear()

class CaloINNBaseCLF(CaloINNCLF):
    """CaloINN with classifier density ratio correction.

    This class extends CaloINNLightningModule to include a classifier
    for density ratio estimation, enabling Acceptance-Rejection sampling.
    The class loads a pre-trained CaloINN model and adds a Multi-Layer Perceptron (MLP)
    """

    @property
    def q0(self):
        return torch.distributions.Normal(0, 1)

    @torch.inference_mode()
    def _get_latent(self, x, c):
        z = self.generator.model(x, c, rev=False)[0]
        return z

    def get_input_from_batch(self, batch):

        x_truth, c = batch

        z_truth = self._get_latent(x_truth, c)
        z_gen = self.q0.sample(z_truth.shape).to(z_truth.device)

        y_truth = torch.ones(z_truth.shape[0], 1, device=z_truth.device)
        y_gen = torch.zeros(z_gen.shape[0], 1, device=z_gen.device)

        X = torch.cat([z_truth, z_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(2, 1)

        X_input = torch.cat([X, c], dim=1)

        return X_input, y