import torch.nn as nn
import torch
from einops import repeat


class AE(nn.Module):
    def __init__(self, config):
        # TODO
        super(AE, self).__init__()
        # instanciate components
        i_dim = config.get("ae:i_dim", 201)
        d_model = config.get("ae:d_model", 512)
        nhead = config.get("ae:nhead", 8)
        num_layers = config.get("ae:num_layers", 6)
        lat_dim = config.get("ae:lat_dim", 16)

        assert (
            d_model % nhead == 0
        ), f"invalid transformer config with d_model {d_model} and n_heads {nhead}"

        # mapping to token_dim
        self.tokenizer = nn.Linear(i_dim, d_model)
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.encoder_comp = nn.Linear(d_model, lat_dim)

        # decoder
        # mapping from token_dim to original dim
        self.detokenizer = nn.Linear(d_model, i_dim)
        # decoder is built of __ENcoder__ layers
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )
        self.decoder_comp = nn.Linear(lat_dim, d_model)

        # position encoder
        max_positions = config.get("ae:max_positions", [48, d_model])
        self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)

        # projection head?
        # self.projection_head = ProjectionHead(
        #     d_model=lat_dim, nhead=4, num_layers=2, odim=30
        # )
        self.projection_head = SimpleProjectionHead(
            # d_model=lat_dim, n_tokens=windowsize, odim=30
            d_model=lat_dim,
            n_tokens=max_positions[1],  # expect 1 token per layer
            odim=30,
        )
        # initialize compression tokens
        n_layers = max_positions[1]
        self.comp_token = nn.Parameter(torch.randn(1, n_layers, d_model))

    #
    def forward(self, x: torch.tensor, p: torch.tensor):
        """
        passes sequence of embeddings through encoder / decoder transformer
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
        Returns:
            z: torch.tensor sequence of latent representations
            y: torch.tensor sequence of reconstructions
        """
        # create masks for compression tokens (1 if p[1]==idx else 0)
        mask = get_comp_token_mask(p, max_pos=self.comp_token.shape[1])
        # get compression tokens
        copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=x.shape[0])
        # Multiply mask with compression tokens (only those with positions present in sequence will be kept)
        copm_tokens = copm_tokens * mask
        # append masked compression tokens to input
        x = torch.cat((x, copm_tokens), dim=1)
        # pass through encoder
        z = self.forward_encoder(x, p)
        # extract compression tokens
        zp = z[:, -self.comp_token.shape[1] :, :]
        # pass compression tokens to projection head
        zp = self.projection_head(zp)
        # decode to weights
        y = self.forward_decoder(z, p)
        return z, y, zp

    def forward_encoder(self, x: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
        Returns:
            z: torch.tensor sequence of latent representations
        """
        # apply position encodings
        x = self.tokenizer(x)
        x = self.pe(x, p)
        x = self.transformer_encoder(x)
        x = self.encoder_comp(x)
        return x

    def forward_decoder(self, z: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Args:
            z: torch.tensor sequence of latent representations
            p: torch.tensor sequence of positions
        Returns:
            y: torch.tensor sequence of reconstructions
        """
        # apply position encodings
        z = self.decoder_comp(z)
        z = self.pe(z, p)
        z = self.transformer_encoder(z)
        # remove compression tokens, so that sequence is same length as input
        z = z[:, : -self.comp_token.shape[1], :]
        z = self.detokenizer(z)
        return z

    def forward_embeddings(self, x: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
        Returns:
            z: torch.tensor sequence of latent representations
        """
        mask = get_comp_token_mask(p)
        # get compression tokens
        copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=x.shape[0])
        # Multiply mask with compression tokens (only those with positions present in sequence will be kept)
        copm_tokens = copm_tokens * mask
        # append masked compression tokens to input
        x = torch.cat((x, copm_tokens), dim=1)
        # pass through encoder
        z = self.forward_encoder(x, p)
        # extract compression tokens
        z = z[:, -self.comp_token.shape[1] :, :]
        return z


class PositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.
    Attributes:
        posemb_init: positional embedding initializer.
        max_positions: maximum number of positions to embed.
        embedding_dim: dimension of the input embeddings.
    """

    def __init__(self, max_positions=[48, 256], embedding_dim=128):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        if len(max_positions) == 2:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)  # add 1 + 2
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)  # add 1 + 2
            self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)  # cat 1+2 & 3

    def forward(self, inputs, pos):
        """Applies the AddPositionEmbs module.
        Args:
            inputs: Inputs to the layer, shape `(batch_size, seq_len, emb_dim)`.
            pos: Position of the first token in each sequence, shape `(batch_size,seq_len,2)`.
        Returns:
            Output tensor with shape `(batch_size, seq_len, emb_dim + 2)`.
        """
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but it is {inputs.ndim}"
        assert pos.shape[2] == len(
            self.max_positions
        ), "Position tensors should have as many demsions as max_positions"
        assert (
            pos.shape[0] == inputs.shape[0]
        ), "Position tensors should have the same batch size as inputs"
        assert (
            pos.shape[1] == inputs.shape[1]
        ), "Position tensors should have the same seq length as inputs"

        pos_emb1 = self.pe1(pos[:, :, 0])
        pos_emb2 = self.pe2(pos[:, :, 1])
        if self.pe3 is not None:
            pos_emb3 = self.pe3(pos[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)

        out = inputs + pos_emb
        return out


class ProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(
        self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, odim: int = 50
    ):
        super(ProjectionHead, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, odim, bias=False)
        self.comp_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # init compression token
        b, n, _ = z.shape
        copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
        z = torch.cat((copm_tokens, z), dim=1)
        # pass through
        z = self.encoder(z)
        # take only comp_token
        z = z[:, 0, :].squeeze()
        # pass through head
        z = self.head(z)
        # return
        return z


class SimpleProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(self, d_model: int = 512, n_tokens: int = 12, odim: int = 50):
        super(SimpleProjectionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model * n_tokens, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
            nn.Linear(odim, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
        )

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # avereage tokens
        # z = z.mean(dim=1)
        z = z.view(z.shape[0], -1)
        # pass through head
        z = self.head(z)
        # return
        return z


def get_comp_token_mask(p: torch.Tensor, max_pos: int) -> torch.tensor:
    """
    maps a batch of positions to a mask of shape [n_samples, max_pos]
    If a position is present in p, the corresponding index in the mask is set to 1
    Args:
        p: tensor of shape [n_samples, n_positions]
        max_pos: maximum number of positions
    Returns:
        mask: tensor of shape [n_samples, max_pos]
    """
    # initialize mask with zeros
    mask = torch.zeros(p.size(0), max_pos)
    # add 1 at the index of value p along dim=1 (per sample)
    mask = mask.scatter_(dim=1, index=p, value=1.0)

    return mask
