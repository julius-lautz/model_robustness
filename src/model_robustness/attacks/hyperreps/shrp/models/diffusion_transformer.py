from typing import List, Optional

# PyTorch
import torch
from torch import nn

# Scaling hyper representations
from shrp.models.modules.input_embedding import InputEmbedding
from shrp.models.modules.diffusion_timestep_embedding import DiffusionTimestepEmbedding
from shrp.models.modules.diffusion_transformer_block import DiffusionTransformerBlock
from shrp.models.modules.diffusion_transformer_output_layer import (
    DiffusionTransformerOutputLayer,
)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_token_length: int,
        embedded_token_length: int,
        num_attention_heads: int,
        depth: int,
        attention_dropout: float = 0.0,
        ffnn_dropout: float = 0.0,
        context_embedding: Optional[nn.Module] = None,
        max_positions: Optional[List[int]] = None,
        iddpm_output: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Embedders for input, diffusion timestep, and context
        self.input_embedding = InputEmbedding(
            input_token_length=input_token_length,
            emdedded_token_length=embedded_token_length,
            max_positions=max_positions,
        )
        self.timestep_embedding = DiffusionTimestepEmbedding(
            out_features=embedded_token_length,
        )
        self.context_embedding = context_embedding

        # Diffusion transformer blocks
        self.diffusion_transformer_blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    embedded_token_length=embedded_token_length,
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    ffnn_dropout=ffnn_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Create output layer
        self.output_layer = DiffusionTransformerOutputLayer(
            embedded_token_length=embedded_token_length,
            out_features=input_token_length,
            iddpm_output=iddpm_output,
        )

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        # Embed input into tokens
        x = self.input_embedding(x, p)

        # Embed diffusion timestep
        t = self.timestep_embedding(t)

        # Embed context if it was given
        if context is not None:
            context = t + self.context_embedding(context)
        else:
            context = t

        # Iterate through all transformer blocks
        for block in self.diffusion_transformer_blocks:
            x = block(x, context)

        x = self.output_layer(x, context)
        return x
