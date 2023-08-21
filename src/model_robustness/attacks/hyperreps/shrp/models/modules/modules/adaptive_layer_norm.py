import torch
from torch import nn


class ContextAdaNorm(nn.Module):
    def __init__(self, normalized_shape: int, ln_eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Adaptive layer normalization parameters
        self.layer_norm = nn.LayerNorm(
            normalized_shape, elementwise_affine=False, eps=ln_eps
        )
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                normalized_shape, 3 * normalized_shape
            ),  # 3 due to shift, scale, gate outputs
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        shift, scale, gate = self.adaptive_modulation(context).chunk(3, dim=1)
        return self.layer_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
