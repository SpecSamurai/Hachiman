from typing import override
import torch

class LayerNorm(torch.nn.Module):
    eps: float
    scale: torch.nn.Parameter
    shift: torch.nn.Parameter

    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    @override
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps) # eps prevents division by zero

        return self.scale * norm_x + self.shift
