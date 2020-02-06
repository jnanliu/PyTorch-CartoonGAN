# -*- coding : utf-8 -*-
import torch.nn as nn
import torchvision.models as models

vgg19 = models.vgg19(pretrained=True).features
ef_layers = [4, 4]

## ----------------------------------- extract feature network --------------------------------- ##
class ExtractFeaturesNetwork(nn.Module) :

    def __init__(self) :
        super(ExtractFeaturesNetwork, self).__init__()
        conv2d_block_index, conv2d_in_block_index, maxpool2d_index, relu_index = 1, 0, 0, 0
        self.functions = nn.Sequential()

        for layer in vgg19 :
            if isinstance(layer, nn.Conv2d) :
                conv2d_in_block_index += 1
                self.functions.add_module("conv{}_{}".format(
                    conv2d_block_index, conv2d_in_block_index
                ), layer)
                # conv4_4
                if conv2d_block_index == ef_layers[0] and conv2d_in_block_index == ef_layers[1] :
                    break

            if isinstance(layer, nn.MaxPool2d) :
                maxpool2d_index += 1
                conv2d_block_index += 1
                conv2d_in_block_index = 0
                self.functions.add_module("maxpool_{}".format(maxpool2d_index), layer)

            if isinstance(layer, nn.ReLU) :
                relu_index += 1
                self.functions.add_module("relu_{}".format(relu_index), layer)

    def forward(self, inputs) :
        x = self.functions(inputs)
        return x