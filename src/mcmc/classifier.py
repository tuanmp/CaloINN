"""Classifier for density ratio estimation — ported from caloxtreme_clf.

Provides the MLP backbone and a Lightning-free wrapper that loads a
trained MLPClassifier checkpoint and exposes ``forward(x) → logits``
and ``predict_proba(x) → probabilities``.

Usage:
    from src.mcmc.classifier import ClassifierWrapper

    clf = ClassifierWrapper.load_from_checkpoint(
        "checkpoints/mlp-epoch=25-val_loss=0.38819.ckpt",
        device="cuda",
    )
    logits = clf(x)                    # raw logits
    probs = clf.predict_proba(x)       # sigmoid(logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
# 1.  MLP backbone (pure nn.Module, no Lightning dependency)
# ═══════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Feed-forward MLP with optional normalisation and configurable activation.

    Architecture:
        LazyLinear(input_dim → hidden_dim)  →  N × [Linear, Norm, Act, Dropout]  →  Linear(hidden_dim → output_dim)

    Parameters
    ----------
    hidden_dim : int
        Width of hidden layers.
    num_layers : int
        Number of hidden layers.
    batch_norm : bool
        Apply BatchNorm1d after each hidden Linear.
    layer_norm : bool
        Apply LayerNorm after each hidden Linear (mutually exclusive with batch_norm).
    output_dim : int
        Output dimensionality (1 for binary classification).
    dropout : float
        Dropout probability (0 = disabled).
    activation : str
        One of "relu", "gelu", "silu", "elu", "leaky_relu", "tanh", "sigmoid".
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        if batch_norm and layer_norm:
            raise ValueError(
                "BatchNorm and LayerNorm cannot be used together."
            )

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Available: {sorted(self._ACTIVATIONS.keys())}"
            )

        act_cls = self._ACTIVATIONS[activation]

        modules: list[nn.Module] = [nn.LazyLinear(hidden_dim)]

        for _ in range(num_layers):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_dim))
            modules.append(act_cls())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*modules)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → raw logits."""
        x = self.net(x)
        return self.output_layer(x)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Lightning-free classifier wrapper
# ═══════════════════════════════════════════════════════════════════════

class ClassifierWrapper:
    """Loads a Lightning ``MLPClassifier`` checkpoint and exposes a clean API.

    The original Lightning module wraps an ``MLP`` inside ``self.net``.
    This wrapper extracts the MLP weights, discards the Lightning
    boilerplate, and provides pure ``forward()`` and ``predict_proba()``.

    Parameters
    ----------
    mlp : MLP
        The underlying MLP backbone.
    device : str or torch.device
        Device for inference.
    """

    def __init__(self, mlp: MLP, device: str | torch.device = "cpu"):
        self.mlp = mlp
        self._device = torch.device(device)
        self.mlp.to(self._device)
        self.mlp.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        **mlp_kwargs,
    ) -> "ClassifierWrapper":
        """Load a Lightning ``MLPClassifier`` checkpoint.

        The checkpoint state_dict uses ``net.*`` prefix (Lightning
        module attribute).  This method strips the prefix and loads
        into a plain ``MLP``.

        Parameters
        ----------
        checkpoint_path : str
            Path to a ``.ckpt`` file saved by ``MLPClassifier``.
        device : str or torch.device
            Device to place the model on.
        **mlp_kwargs
            Passed to ``MLP.__init__`` (if the checkpoint does not
            contain ``hyper_parameters``, these are required).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Try to infer MLP config from Lightning hyperparameters
        hp = ckpt.get("hyper_parameters", {})
        if hp:
            mlp_kwargs = {
                k: hp[k]
                for k in (
                    "hidden_dim",
                    "num_layers",
                    "batch_norm",
                    "layer_norm",
                    "output_dim",
                    "dropout",
                    "activation",
                )
                if k in hp
            }

        mlp = MLP(**mlp_kwargs)

        # Lightning state_dict keys are prefixed with "net."
        state_dict = ckpt.get("state_dict", ckpt)
        stripped = {
            k.removeprefix("net."): v
            for k, v in state_dict.items()
            if k.startswith("net.")
        }
        if not stripped:
            # Fallback: try raw state_dict (older checkpoints)
            stripped = {
                k: v for k, v in state_dict.items() if "loss" not in k.lower()
            }

        mlp.load_state_dict(stripped, strict=False)
        return cls(mlp, device=device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → raw logits.

        Parameters
        ----------
        x : torch.Tensor  shape (N, input_dim)
            Preprocessed classifier input (concatenated X_proc + cond_proc + X_hlf).

        Returns
        -------
        torch.Tensor  shape (N, 1)
            Raw logits (before sigmoid).
        """
        x = x.to(self._device)
        return self.mlp(x)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → calibrated probabilities.

        Does NOT apply temperature scaling — that's handled separately
        by ``TemperatureCalibrator``.  This returns raw sigmoid(logits).

        Parameters
        ----------
        x : torch.Tensor  shape (N, input_dim)

        Returns
        -------
        torch.Tensor  shape (N, 1)
            Probability that each sample is real (in [0, 1]).
        """
        return torch.sigmoid(self.forward(x))
