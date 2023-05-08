import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1, 64)
        self.fc_2 = nn.Linear(64, 128)
        self.fc_3 = nn.Linear(128, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)

        return out


class ConvNetSmall(nn.Module):
    def __init__(
            self,
            channels_in,
            nlin="leakyrelu",
            dropout=0.2,
            init_type="uniform"
    ):
        super().__init__()

        self.module_list = nn.ModuleList()

        # Conv layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Conv layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Conv layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))

        # Flatten output
        self.module_list.append(nn.Flatten())

        # Linear layer 1
        self.module_list.append(nn.Linear(3*3*4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Linear layer 2
        self.module_list.append(nn.Linear(20, 10))

        # Initialize weights
        self.initialize_weights(init_type)

    def get_nonlin(self, nlin):
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def initialize_weights(self, init_type):
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

    # TODO: Check why this is needed
    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class ConvNetLarge(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        self.module_list = nn.ModuleList()

        # Conv layer 1
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Conv layer 2
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Conv layer 3
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Flatten output
        self.module_list.append(nn.Flatten())

        # Linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))

        # Linear layer 2
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations
