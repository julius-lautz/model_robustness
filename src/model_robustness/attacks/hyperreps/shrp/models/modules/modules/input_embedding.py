from typing import List, Optional

# PyTorch
import torch
from torch import nn

# Scaling hyper representations
from shrp.models.modules.positional_embedding import PositionalEmbedding
from shrp.datasets.augmentations import WindowCutter


class InputEmbedding(nn.Module):
    def __init__(
        self,
        input_token_length: int,
        emdedded_token_length: int, 
        max_positions: Optional[List[int]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not max_positions:
            max_positions = [48, 256]
        self.linear_transform = nn.Linear(
            in_features=input_token_length,
            out_features=emdedded_token_length,
        )
        self.positional_embedding = PositionalEmbedding(
            embedding_dim=emdedded_token_length, max_positions=max_positions
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        x = self.linear_transform(x)
        x = self.positional_embedding(inputs=x, pos=p)
        return x
