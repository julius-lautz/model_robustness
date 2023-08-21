import torch
import torch.nn as nn
import numpy as np


### just an identity model, maps input on itself
class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x, p):
        """
        just flatten the weight tokens
        this ignores the fact that there is a lot of padding involved here
        """
        return x.flatten(start_dim=1)

    def forward_embeddings(self, x, p):
        return x.flatten(start_dim=1)


class LayerQuintiles(nn.Module):
    def __init__(
        self,
    ):
        super(LayerQuintiles, self).__init__()

    def forward(self, x, p):
        z = self.compute_layer_quintiles(x, p)
        return z

    def forward_embeddings(self, x, p):
        z = self.compute_layer_quintiles(x, p)
        return z

    def compute_layer_quintiles(self, weights, pos):
        # weights need to be on the cpu for numpy
        weights = weights.to(torch.device("cpu"))
        quantiles = [1, 25, 50, 75, 99]
        features = []
        # get unique layers, assume pos.shape = [n_samples, n_tokens, 3]
        layers = torch.unique(pos[0, :, 1])
        # iterate over layers
        for idx, layer in enumerate(layers):
            # get slices
            index_kernel = torch.where(pos[0, :, 1] == layer)[0]
            # compute kernel stat values
            wtmp = weights[:, index_kernel].detach().flatten(start_dim=1).numpy()
            # print(wtmp.shape)
            # # compute kernel stat values
            features_ldx_weights = np.percentile(
                a=wtmp,
                q=quantiles,
                axis=1,
            )
            features_ldx_weights = torch.tensor(features_ldx_weights)
            # # transpose to [n_samples, n_features]
            features_ldx_weights = torch.einsum("ij->ji", features_ldx_weights)
            # print(features_ldx_weights.shape)
            mean_ldx_weights = torch.tensor(np.mean(a=wtmp, axis=1)).unsqueeze(dim=1)
            var_ldx_weights = torch.tensor(np.var(a=wtmp, axis=1)).unsqueeze(dim=1)
            # print(mean_ldx_weights.shape)
            # print(var_ldx_weights.shape)
            features.extend([mean_ldx_weights, var_ldx_weights, features_ldx_weights])
        # put together
        z = torch.cat(features, dim=1)
        return z
