from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class Advanced_Model(nn.Module):

    def __init__(self, model, mid_ratio=None):
        super(Advanced_Model, self).__init__()
        self.model = model
        self.loss_list = ['cls_loss']
        if mid_ratio is not None:
            assert mid_ratio < 1.
            assert mid_ratio > 0.
            self.mid_ratio = [1 - mid_ratio, mid_ratio]
        else:
            self.mid_ratio = [1., 1.]

    def forward(self, x, *nargs):
        x = self.model(x)
        return x

    def loss(self, outputs, target, *nargs):
        loss = 0
        loss_strategy = {}
        for i, output in enumerate(outputs):
            loss += self.mid_ratio[i] * F.cross_entropy(output, target)
        loss_strategy['cls_loss'] = loss
        return loss_strategy
