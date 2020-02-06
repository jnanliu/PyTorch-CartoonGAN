# -*- coding : utf-8 -*-
import math
import torch.nn as nn

def initial(layers) :
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

## ------------------- flat convolution ------------------ ##
class FlatConvolution(nn.Module) :

    def __init__(self) :
        super(FlatConvolution, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        initial(self.layers)

    def forward(self, inputs) :
        x = self.layers(inputs)
        return x

## ------------------- down convolution ------------------ ##
class DownConvlution(nn.Module) :

    def __init__(self) :
        super(DownConvlution, self).__init__()
        self.layers_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        initial(self.layers_1)
        self.layers_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        initial(self.layers_2)

    def forward(self, inputs) :
        x = self.layers_1(inputs)
        x = self.layers_2(x)
        return x

## ------------------------------- residual block ------------------------------ ##
class ResidualBlock(nn.Module) :

    def __init__(self) :
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256)
        )
        initial(self.layers)

        self.shortcut = nn.Sequential()

    def forward(self, inputs) :
        x = nn.ReLU(True)(self.layers(inputs) + self.shortcut(inputs))
        return x

## ------------------------- up convolution ------------------------ ##
class UpConvolution(nn.Module) :

    def __init__(self) :
        super(UpConvolution, self).__init__()
        self.layers_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        initial(self.layers_1)

        self.layers_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        initial(self.layers_2)

    def forward(self, inputs) :
        x = self.layers_1(inputs)
        x = self.layers_2(x)
        return x

## ------------------------- final convolution ------------------------ ##
class FinalConvolution(nn.Module) :

    def __init__(self) :
        super(FinalConvolution, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
        )
        initial(self.layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x

## ------------------------- generator network ------------------------ ##
class GeneratorNetwork(nn.Module) :

    def __init__(self, residual_block_num) :
        super(GeneratorNetwork, self).__init__()
        self.functions = nn.Sequential()

        self.functions.add_module("flat_convolution", FlatConvolution())
        self.functions.add_module("down_convolution", DownConvlution())
        for i in range(residual_block_num) :
            self.functions.add_module("residual_block_{}".format(i), ResidualBlock())
        self.functions.add_module("up_convolution", UpConvolution())
        self.functions.add_module("final_convolution", FinalConvolution())

    def forward(self, inputs):
        x = self.functions(inputs)
        return x