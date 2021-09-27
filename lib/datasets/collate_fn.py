from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

def classify_collate(batch):
  '''
  :param batch: [imgs, boxes, labels] dtype = np.ndarray
  imgs:
    shape = (C H W)
  :return:
  '''

  imgs = [x[0] for x in batch]
  labels = [x[1] for x in batch]
  # imgs_path = [x[2] for x in batch]

  # return torch.stack(imgs, 0), np.asarray(labels).astype(np.int64), imgs_path
  return torch.stack(imgs, 0), np.asarray(labels).astype(np.int64)


def test():
  # MsgOfImg = namedtuple(
  #   'MsgOfImg', [
  #     'height', 'width', 'scale', 'height_b_scale', 'width_b_scale'
  #   ]
  # )
  # w = MsgOfImg(
  #   height=10,
  #   width=4,
  #   scale=None,
  #   height_b_scale=None,
  #   width_b_scale=None
  # )
  # print(w)
  # w = w._replace(height_b_scale=100)
  # print(w)
  pass

if __name__ == '__main__':
  test()

