from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nets.resnet import resnet50, resnet101
from .nets.vgg import vgg16_bn
from .nets.densenet import densenet121, densenet169
from .nets.p3d_model import P3D199
from .nets.i3dpt import i3d

__sets = {}

__sets['res50'] = resnet50
__sets['res101'] = resnet101
__sets['vgg16bn'] = vgg16_bn
__sets['dense121'] = densenet121
__sets['dense169'] = densenet169
__sets['p3d199'] = P3D199
__sets['i3d'] = i3d

def get_model(name):
    '''
    :param name:
    :return:
    '''
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    else:
        return __sets[name]
