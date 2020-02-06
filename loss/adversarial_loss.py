# -*- coding : utf-8 -*-
import torch.nn as nn

class AdversarialLoss(nn.Module) :

    def __init__(self, weight) :
        super(AdversarialLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.BCELoss()

    def forward(self, inputs, labels) :
        self.loss = self.weight * self.criterion(inputs, labels)
        return self.loss