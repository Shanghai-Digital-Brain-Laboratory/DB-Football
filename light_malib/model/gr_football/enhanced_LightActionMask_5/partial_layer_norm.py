import torch
import torch.nn as nn


class PartialLayernorm(nn.Module):
    def __init__(self, in_dim, layer):
        super().__init__()
        self.layer = layer
        self.dim = self.layer.normalized_shape[0]
        self.layer2 = nn.LayerNorm(in_dim - self.dim)

    def forward(self, x):
        x1 = x[..., :self.dim]
        y1 = self.layer(x1)
        x2 = x[..., self.dim:]
        y2 = self.layer2(x2)
        y = torch.concat([y1, y2], dim=-1)
        return y
