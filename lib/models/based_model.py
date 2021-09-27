from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

class Base_Model(nn.Module):

    def __init__(self, model, criterion):
        super(Base_Model, self).__init__()
        self.model = model
        self.loss_list = ['cls_loss']
        self.criterion = criterion

    def forward(self, x, *nargs):
        x = self.model(x)
        return x


    def loss(self, output, target, *nargs):
        loss_strategy = {}
        loss = self.criterion(output, target)
        loss_strategy['cls_loss'] = loss
        return loss_strategy
