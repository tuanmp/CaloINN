import torch.nn as nn


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
