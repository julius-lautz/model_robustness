import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock


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


class ResNet18(ResNet):
    def __init__(
        self, 
        channels_in=3,
        out_dim=10,
        nlin="relu",
        dropout=0.2,
        init_type="kaiming_uniform",
    ):
        super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=out_dim)
        # adapt first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            bias=False
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)
        
    def initialize_weights(self, init_type):
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
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
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m


class CNN_ARD(nn.Module):

    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        self.module_list = nn.ModuleList()

        ## compose layer 1
        self.module_list.append(Conv2dARD(in_channels=channels_in,out_channels=8, kernel_size=5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(Conv2dARD(in_channels=8, out_channels=6, kernel_size=5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(Conv2dARD(in_channels=6, out_channels=4, kernel_size=2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(LinearARD(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(LinearARD(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == LinearARD or type(m) == Conv2dARD:
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
                try:
                    # set bias to some small non-zero value
                    m.bias.data.fill_(0.01)
                except Exception as e:
                    print(e)

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


def get_ard_reg(module):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearARD) or isinstance(module, Conv2dARD):
        return module.get_reg()
    elif hasattr(module, 'children'):
        return sum([get_ard_reg(submodule) for submodule in module.children()])
    return 0


def _get_dropped_params_cnt(module):
    if hasattr(module, 'get_dropped_params_cnt'):
        return module.get_dropped_params_cnt()
    elif hasattr(module, 'children'):
        return sum([_get_dropped_params_cnt(submodule) for submodule in module.children()])
    return 0


def _get_params_cnt(module):
    if any([isinstance(module, l) for l in [LinearARD, Conv2dARD]]):
        return reduce(operator.mul, module.weight.shape, 1)
    elif hasattr(module, 'children'):
        return sum(
            [_get_params_cnt(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio(model):
    return _get_dropped_params_cnt(model) * 1.0 / _get_params_cnt(model)

""" LOSS FUNCTION"""

class ELBOLoss(nn.Module):
    def __init__(self, net, loss_fn):
        super(ELBOLoss, self).__init__()
        self.loss_fn = loss_fn
        self.net = net

    def forward(self, input, target, loss_weight=1., kl_weight=1.):
        assert not target.requires_grad
        # Estimate ELBO
        return loss_weight * self.loss_fn(input, target)  \
            + kl_weight * get_ard_reg(self.net)


""" FULLY CONNECTED LAYER"""
class LinearARD(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True, thresh=3, ard_init=-10):
        super(LinearARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.ard_init = ard_init
        self.log_sigma2 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma2_bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            W_mu = F.linear(input, self.weight)
            std_w = torch.exp(self.log_alpha).permute(1,0)
            W_std = torch.sqrt((input.pow(2)).matmul(std_w*(self.weight.permute(1,0)**2)) + 1e-15)

            epsilon = W_std.new(W_std.shape).normal_()
            output = W_mu + W_std * epsilon
            if self.bias is not None: 
              output += self.bias
        else:
            W = self.weights_clipped
            output = F.linear(input, W) + self.bias
        return output

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        if self.bias is not None:
            self.bias.data.zero_()
        self.log_sigma2.data.fill_(self.ard_init)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - \
            0.5 * torch.log1p(torch.exp(-self.log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)


""" CONVOLUTIONAL LAYER """

class Conv2dARD(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-10, thresh=3,bias=True):
        # bias = False  # Goes to nan if bias = True
        super(Conv2dARD, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        # self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ard_init = ard_init
        self.log_sigma2 = nn.Parameter(ard_init * torch.ones_like(self.weight))
        self.log_sigma2_bias = nn.Parameter(ard_init * torch.ones_like(self.bias))
        # self.log_sigma2 = Parameter(2 * torch.log(torch.abs(self.weight) + eps).clone().detach()+ard_init*torch.ones_like(self.weight))
        
    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        if self.training == False:
            weights_clipped = self.weights_clipped
            # bias_clipped = self.bias_clipped()
            bias_clipped = self.bias_clipped()
            return F.conv2d(input, weights_clipped,
                            bias_clipped, self.stride,
                            self.padding, self.dilation, self.groups)
            # return F.conv2d(input, self.weights_clipped,
            #                 self.bias, self.stride,
            #                 self.padding, self.dilation, self.groups)
        W = self.weight
        b = self.bias
        conved_mu = F.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        log_alpha = self.log_alpha
        log_alpha_bias = self.log_alpha_bias()
        conved_si = torch.sqrt(1e-15 + F.conv2d(input * input,
                                                torch.exp(log_alpha) * W *
                                                W, torch.exp(log_alpha_bias)*b*b, self.stride,
                                                self.padding, self.dilation, self.groups))
        
        # conved_si = torch.sqrt(1e-15 + F.conv2d(input * input,
        #                                         torch.exp(log_alpha) * W *
        #                                         W, self.bias, self.stride,
        #                                         self.padding, self.dilation, self.groups))
        conved = conved_mu + \
            conved_si * \
            torch.normal(torch.zeros_like(conved_mu),
                         torch.ones_like(conved_mu))
        return conved

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)
    
    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)
    
    # @property
    def bias_clipped(self):
        clip_mask = self.get_clip_mask_bias()
        return torch.where(clip_mask, torch.zeros_like(self.bias), self.bias)

    def get_clip_mask_bias(self):
        log_alpha_bias = self.log_alpha_bias()
        return torch.ge(log_alpha_bias, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        # add bias
        log_alpha_bias = self.log_alpha_bias()
        mdkl_b = k1 * torch.sigmoid(k2 + k3 * log_alpha_bias) - \
            0.5 * torch.log1p(torch.exp(-log_alpha_bias)) + C
        return -torch.sum(mdkl) - torch.sum(mdkl_b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -8, 8)
    
    # @property
    def log_alpha_bias(self):
        log_alpha_bias = self.log_sigma2_bias - 2 * \
            torch.log(torch.abs(self.bias) + 1e-15)
        return torch.clamp(log_alpha_bias, -8, 8)


###############################################################################
# define net
# ##############################################################################
def compute_outdim(i_dim, stride, kernel, padding, dilation):
    o_dim = (i_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return o_dim
