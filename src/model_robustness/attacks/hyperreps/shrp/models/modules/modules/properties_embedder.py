from torch import nn


class PropertiesEmbedder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, dropout_p: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(in_features=input_size, out_features=embedding_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, batch):
        return self.dropout(self.linear(batch))
