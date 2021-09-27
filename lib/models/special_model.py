from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class Special_Model(nn.Module):

    def __init__(self, model, train_strategy, mid_ratio=None):
        super(Special_Model, self).__init__()
        self.model = model
        self.model.init_loss(train_strategy)
        self.loss_list = self.model.loss_list

    def forward(self, input, target):
        x = self.model(input, target)
        return x

    def loss(self, output, target, back_tag):
        self.model.optimize_parameters(back_tag)
        loss_strategy = {}
        if back_tag:
            for _, loss_name in enumerate(self.loss_list):
                loss_strategy[loss_name] = getattr(self.model, loss_name)
        else:
            loss_strategy['cls_loss'] = getattr(self.model, 'cls_loss')
        return loss_strategy

    def train_optimizer(self, train_strategy):
        self.model.init_optimizer(train_strategy)
