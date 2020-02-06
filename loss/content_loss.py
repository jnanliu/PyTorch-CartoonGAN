# -*- coding : utf-8 -*-
import torch
import torch.nn as nn

class ContentLoss(nn.Module) :

    def __init__(self, weight) :
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss()

    def forward(self, fake_feature_map, real_feature_map) :
        self.loss = self.weight * self.criterion(fake_feature_map, real_feature_map)
        return self.loss