import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BBoxHead(nn.Module):
    """Predict normalized [cx, cy, w, h] bbox."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B, S, E, d] -> boxes: [B, S, E, 4] in [0,1]"""
        return torch.sigmoid(self.mlp(q))
