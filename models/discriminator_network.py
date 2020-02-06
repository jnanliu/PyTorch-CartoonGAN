# -*- coding : utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

def initial(layers) :
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u

## ------------------------------- Conv2d with Spectral Normalization ------------------------------ ##
class SNConv2d(conv._ConvNd) :

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True) :
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

## --------------------------------- discriminator network ----------------------------------------- ##
class DiscriminatorNetwork(nn.Module) :

    def __init__(self) :
        super(DiscriminatorNetwork, self).__init__()
        self.functions_1 = nn.Sequential(
            SNConv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        initial(self.functions_1)

        self.functions_2 = nn.Sequential(
            SNConv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SNConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        initial(self.functions_2)

        self.functions_3 = nn.Sequential(
            SNConv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SNConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        initial(self.functions_3)

        self.functions_4 = nn.Sequential(
            SNConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        initial(self.functions_4)

        self.functions_5 = nn.Sequential(
            SNConv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        initial(self.functions_5)

    def forward(self, inputs) :
        x = self.functions_1(inputs)
        x = self.functions_2(x)
        x = self.functions_3(x)
        x = self.functions_4(x)
        x = self.functions_5(x)
        return x