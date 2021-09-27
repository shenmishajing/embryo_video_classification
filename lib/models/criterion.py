import torch
from torch.nn import CrossEntropyLoss
from lib.utils.focal_loss_softmax import FocalLoss

__sets = {}

__sets['cross_entropy'] = CrossEntropyLoss
__sets['focal_loss'] = FocalLoss

class Criterion(object):
    def __init__(self, loss_name, num_classes, weight=None):
        self.loss_name = loss_name.lower()
        if self.loss_name == 'cross_entropy':
            self.criterion = CrossEntropyLoss(weight)
        elif self.loss_name == 'focal_loss':
            self.criterion = FocalLoss(num_classes)
        else:
            print('Not exists such loss caled: {}'.format(self.loss_name))
            raise ValueError

    def __call__(self, input, target):
        '''

        :param input: [N, C]
        :param target: [N]
        :return:
        '''
        return self.criterion(input, target)