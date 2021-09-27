from lib.datasets.dataset.video import TSNDataSet

__sets = {}

__sets['HUST_video'] = TSNDataSet

def get_imdb(data_category):
  '''
  :param name:
  :return:
  '''
  data_key_name = data_category
  if data_category not in __sets:
    raise KeyError('Unknown dataset: {}'.format(data_key_name))
  else:
    return __sets[data_key_name]





