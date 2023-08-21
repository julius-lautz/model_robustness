from typing import Optional

# PyTorch
import torch
from torch import nn


# Scaling hyper representations
from shrp.models.modules.adaptive_layer_norm import ContextAdaNorm


class DiffusionTransformerOutputLayer(nn.Module):
    def __init__(self, embedded_token_length: int, out_features: int, iddpm_output:bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iddpm_output = iddpm_output
        original_embedded_token_length = embedded_token_length

        if self.iddpm_output:
            out_features *= 2
            

        self.linear = nn.Linear(
            in_features=embedded_token_length, out_features=out_features
        )

        # Adaptive Layer Norm which handles both context and diffusion timestep embedding
        self.ada_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=original_embedded_token_length,
                out_features=embedded_token_length*2,
                bias=True,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # Final ada norm
        shift, scale = self.ada_norm(context).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = self.linear(x)
        if self.iddpm_output:
            # Split tensor in two halves and stack to form channels
            x = torch.stack(x.chunk(2, axis=2), axis=1)
        return x
