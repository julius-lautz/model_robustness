import math

# PyTorch
import torch
from torch import nn


class DiffusionTimestepEmbedding(nn.Module):
    def __init__(
        self, out_features: int, timestep_embedding_size: int = 256, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ffnn = nn.Sequential(
            nn.Linear(in_features=timestep_embedding_size, out_features=out_features),
            nn.SiLU(),
            nn.Linear(in_features=out_features, out_features=out_features),
        )
        self.timestep_embedding_size = timestep_embedding_size

    def forward(self, t: torch.Tensor):
        t = sinusoidal_timestep_embedding(
            timestep=t, embedding_size=self.timestep_embedding_size
        )
        t = self.ffnn(t)
        return t


def sinusoidal_timestep_embedding(
    timestep: torch.Tensor, embedding_size, max_period=10000
):
    """
    Create sinusoidal timestep embeddings.
    https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

    :param timestep: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_size: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = embedding_size // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timestep.device)
    args = timestep[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_size % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
