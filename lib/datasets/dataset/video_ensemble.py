import torch.utils.data as data

from PIL import Image
import os
import os.path
import json
import numpy as np
import torch
import pdb
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNEnsemble(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split', list_file[0]),
                              os.path.join(self.root_path, 'data_split', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split', list_file)
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl[0].format(idx))).convert('RGB')]

    def _load_flow(self, directory, idx):
        idx = idx + 1
        x_img = Image.open(os.path.join(directory, self.image_tmpl[1].format('x', idx))).convert('L')
        y_img = Image.open(os.path.join(directory, self.image_tmpl[1].format('y', idx))).convert('L')

        return [x_img, y_img]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record['num_frames'] - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(0, average_duration, size=self.num_segments)
        elif record['num_frames'] > self.num_segments:
            offsets = np.sort(randint(record['num_frames'] - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if record['num_frames'] > self.num_segments + self.new_length[0] - 1:
            tick_img = (record['num_frames'] - self.new_length[0] + 1) / float(self.num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets_img = np.array([int(tick_img / 2.0 + tick_img * x) for x in range(self.num_segments)])
        else:
            offsets_img = np.zeros((self.num_segments,))

        if record['num_frames'] > self.num_segments + self.new_length[1] - 1:
            tick_flow = (record['num_frames'] - self.new_length[1] + 1) / float(self.num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets_flow = np.array([int(tick_flow / 2.0 + tick_flow * x) for x in range(self.num_segments)])
        else:
            offsets_flow = np.zeros((self.num_segments,))
        return offsets_img + 1, offsets_flow + 1

    def _get_test_indices(self, record):

        tick_img = (record['num_frames'] - self.new_length[0] + 1) / float(self.num_segments)
        offsets_img = np.array([int(tick_img / 2.0 + tick_img * x) for x in range(self.num_segments)])

        tick_flow = (record['num_frames'] - self.new_length[1] + 1) / float(self.num_segments)
        offsets_flow = np.array([int(tick_flow / 2.0 + tick_flow * x) for x in range(self.num_segments)])

        return offsets_img + 1, offsets_flow + 1

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        flows = list()
        for seg_ind in indices[0]:
            p = int(seg_ind)
            for i in range(self.new_length[0]):
                try:
                    seg_imgs = self._load_image(os.path.join(self.root_path, 'Images', video_name), record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    exit()
                images.extend(seg_imgs)
                if p < record['num_frames']:
                    p += 1

        for seg_ind in indices[1]:
            p = int(seg_ind)
            for i in range(self.new_length[1]):
                try:
                    seg_flows = self._load_flow(os.path.join(self.root_path, 'OpticalFlow', video_name), record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    exit()
                flows.extend(seg_flows)
                if p < record['num_frames']:
                    p += 1

        process_images = self.transform[0](images)
        process_flows = self.transform[1](flows)
        return process_images, process_flows, record['label']

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''
    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        imgs = [x[0] for x in batch]
        flows = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        return torch.stack(imgs, 0), torch.stack(flows, 0), np.asarray(labels).astype(np.int64)
