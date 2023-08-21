from typing import Optional

# PyTorch
import torch
from torch import nn

# ViT imports
from timm.models.vision_transformer import Attention, Mlp

# Scaling hyper representations
from shrp.models.modules.adaptive_layer_norm import ContextAdaNorm


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        embedded_token_length: int,
        num_attention_heads,
        attention_dropout: float = 0.0,
        ffnn_dropout: float = 0.0,
        ffnn_ratio: int = 4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Transformer Modules
        self.mh_attention = Attention(
            dim=embedded_token_length,
            num_heads=num_attention_heads,
            qkv_bias=True,
            attn_drop=attention_dropout
        )
        ffnn_out_features = embedded_token_length * ffnn_ratio
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffnn = Mlp(
            in_features=embedded_token_length,
            hidden_features=ffnn_out_features,
            act_layer=approx_gelu,
            drop=ffnn_dropout
        )

        # Adaptive Layer Norm which handles both context and diffusion timestep embedding
        self.first_context_ada_norm = ContextAdaNorm(
            normalized_shape=embedded_token_length,
        )
        self.second_context_ada_norm = ContextAdaNorm(
            normalized_shape=embedded_token_length,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # First half with multi-head attention
        x_intermediary, gate = self.first_context_ada_norm(
            x=x,
            context=context,
        )
        # Skip connection
        x = x + self.mh_attention(x_intermediary) * gate
        # Second half feed-forward neural network
        x_intermediary, gate = self.second_context_ada_norm(
            x=x,
            context=context,
        )
        # Skip connection
        x = x + self.ffnn(x_intermediary) * gate
        return x
