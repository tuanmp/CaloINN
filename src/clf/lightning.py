from typing import Any, Mapping

from lightning import LightningModule
import torch.nn.functional as F
from clf.classifier import MLP
import torch
import numpy as np
from sklearn.metrics import brier_score_loss
from torchmetrics import AUROC, MetricCollection
from mcmc.calibration import expected_calibration_error


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.metrics = MetricCollection({
            "auroc": AUROC(task="binary")
        })

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y, reduction='mean')

    def get_input_from_batch(self, batch):
        x = torch.concat(batch[:-1], dim=-1)
        y = batch[-1]
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss}
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.log('train_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {
            "loss": loss,
            "y_hat": y_hat.detach(),
            "y": y.detach(),
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # update metrics
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('val_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics['auroc'].update(y_hat, y)
        y_hat_np = torch.sigmoid(y_hat).cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        brier = brier_score_loss(y_np, y_hat_np)
        self.log('val_brier', brier, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        ece = expected_calibration_error(y_np, y_hat_np)
        self.log('val_ece', ece, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_aucroc", self.metrics["auroc"].compute(), on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('test_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        y_hat_np = torch.sigmoid(y_hat).cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        brier = brier_score_loss(y_np, y_hat_np)
        self.log('test_brier', brier, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        ece = expected_calibration_error(y_np, y_hat_np)
        self.log('test_ece', ece, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def predict_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        y_hat = self(x)
        y_hat = torch.sigmoid(y_hat)
        return {"y_hat": y_hat, "y": y}

class MLPClassifier(BaseModel):
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

        self.net = MLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
        )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.net(x)

class MLPLowLevelClassifier(MLPClassifier):

    def get_input_from_batch(self, batch):
        X_proc, cond_proc, X_hlf, y = batch
        X = torch.concat([X_proc, cond_proc], dim=-1)
        y = y.unsqueeze(1)
        return X, y
    
class MLPHighLevelClassifier(MLPClassifier):

    def get_input_from_batch(self, batch):
        X_proc, cond_proc, X_hlf, y = batch
        X = torch.concat([X_hlf, cond_proc], dim=-1)
        y = y.unsqueeze(1)
        return X, y

class MLPLatentClassifier(MLPClassifier):

    def get_input_from_batch(self, batch):
        X_proc, cond_proc, y = batch
        X = torch.concat([X_proc, cond_proc], dim=-1)
        y = y.reshape(-1, 1)
        return X, y


class MultiMLPClassifier(BaseModel):
    def __init__(
        self,
        hidden_dim: int=256,
        num_layers: int=4,
        num_heads: int=1,
        batch_norm: bool=False,
        layer_norm: bool=True,
        output_dim: int=1,
        dropout: float=0.,
        activation: str="relu",
    ):
        super().__init__()

        def make_net():
            return MLP(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                output_dim=output_dim,
                dropout=dropout,
                activation=activation,
            )

        self.nets = torch.nn.ModuleList([make_net() for _ in range(num_heads)])

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        yhats = [net(x) for net in self.nets]
        yhats = torch.concat(yhats, dim=-1)
        yhats = torch.mean(yhats, dim=-1, keepdim=True)
        return yhats

    def training_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        losses = [self.criterion(net(x), y) for net in self.nets]
        loss = sum(losses) / len(losses)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input_from_batch(batch)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        yhats = [net(x) for net in self.nets]
        losses = [self.criterion(yhat, y) for yhat in yhats]
        loss = sum(losses) / len(losses)
        yhats = torch.concat(yhats, dim=-1)
        yhats = torch.sigmoid(yhats)
        yhats_mean = torch.mean(yhats, dim=-1, keepdim=True)
        yhat_std = torch.std(yhats, dim=-1, keepdim=True)
        
        return {
            "loss": loss,
            "y_hat": yhats_mean.detach(),
            "y": y.detach(),
            "y_hat_std": yhat_std.detach(),
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
        self.log('yhat_std', torch.mean(outputs["y_hat_std"]), on_epoch=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log("test_loss", outputs["loss"], on_epoch=True, prog_bar=True, logger=True)
        self.log("yhat_std", torch.mean(outputs["y_hat_std"]), on_epoch=True, logger=True)


    