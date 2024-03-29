from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# Number of classes
__C.NUM_CLASSES = 3
__C.GPU = True
__C.CLASSES = ('__background__', 'l', 'h')

# Config Path set
__C.ModelSave_Path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'save_models'))
__C.TensorboardSave_Path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tensorboardx'))

# Solver
__C.SOLVER = edict()
__C.SOLVER.WEIGHT_DECAY = 0.0001
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.BASE_LR = 0.01
__C.SOLVER.LR_DECAY = 0.1
__C.SOLVER.MAX_EPOCH = 120
__C.SOLVER.STEPS = [0, 120000, 160000]
__C.SOLVER.MAX_DECAY = 3
__C.SOLVER.SAVE_INTERVAL = 10

# Train
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.SCALE = 224
__C.TRAIN.PRE_TRAIN = True
__C.TRAIN.FIX_PARAMETER = True

# TEST
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 16

# Data
__C.DATA = edict()
# __C.DATA.DATA_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cache'))
# print('CACHE: {}'.format(__C.DATA.DATA_CACHE))




def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v