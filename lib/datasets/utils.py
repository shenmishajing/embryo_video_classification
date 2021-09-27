from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from lib.config.config import cfg
from PIL import Image
import xml.etree.ElementTree as ET
from torch.autograd import Variable


def label_to_variable(blobs, gpu=True, volatile=False):
    '''
    :param blobs:
    :param volatile:
    :return:
      cls_targets, loc_targets
    '''
    labels = blobs
    v = Variable(torch.from_numpy(labels), volatile=volatile)
    if gpu:
        v = v.cuda()
    return v


def img_data_to_variable(blobs, gpu=True, volatile=False):
    '''
      :param blobs:
      :param volatile:
      :return:
        cls_targets, loc_targets
    '''
    if isinstance(blobs, tuple) or isinstance(blobs, list):
        v = []
        for _, blob in enumerate(blobs):
            images = Variable(blob, volatile=volatile)
            if gpu:
                images = images.cuda()
            v.append(images)
    else:
        images = blobs
        v = Variable(images, volatile=volatile)
        if gpu:
            v = v.cuda()
    return v


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_annotation(index, root_path, class_to_ind):
    """
    Load image and bounding boxes info from XML file.
    """
    filename = os.path.join(root_path, 'Xmls', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.int16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) + 1
        y1 = float(bbox.find('ymin').text) + 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        cls = class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'flipped': False}

def get_labels_from_indexes(root_path, data_set_name, id_indexes):
    suffix = '_2'
    if data_set_name.count('fubao') != 0:
        classes = ('lsil_n', 'lsil_l', 'hsil')
    else:
        classes = ('normal', 'hsil')
    class_to_ind = dict(list(zip(classes, list(range(len(classes))))))

    all_labels = list()
    for i, item in enumerate(id_indexes):
        annotation_dict = load_annotation(item+suffix, root_path, class_to_ind)
        label = int(annotation_dict['gt_classes'][0])
        # 仅适用于论文数据集
        if label == 0 and len(item.split('_')) == 2:
            label = 2
        all_labels.append(label)
    all_labels = np.array(all_labels)
    return all_labels


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    # print(labels.size(), y.size())
    # print(labels.numpy().max())
    return y[labels]            # [N,D]

def load_image(img_path):
    # PIL Image
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def draw_heat_maps(data_dict, info_dict, save_path, label, guided_direction):
    assert guided_direction in ['left', 'right']
    transform_map = {
        0: 'nL',
        1: 'H'
    }
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    label = label[0]
    info_dict = info_dict[0]
    acid_path = info_dict['img_path_acid']
    iodine_path = info_dict['img_path_iodine']
    bbox_acid = info_dict['bbox_acid']
    bbox_iodine = info_dict['bbox_iodine']

    if guided_direction == 'left':
        left_img = np.array(load_image(acid_path).crop(bbox_acid))
        right_img = np.array(load_image(iodine_path).crop(bbox_iodine))
    else:
        left_img = np.array(load_image(iodine_path).crop(bbox_iodine))
        right_img = np.array(load_image(acid_path).crop(bbox_acid))

    left_short_length = min(left_img.shape[0], left_img.shape[1])
    right_short_length = min(right_img.shape[0], right_img.shape[1])


    img_id = acid_path.split('/')[-1].strip('_2.jpg')

    def one_normalization(mask):
        mask = (mask - mask.min())
        mask = mask / mask.max()
        return mask

    # True or False
    ori_tag = True
    seperate_tag = False

    if not seperate_tag:
        # plt.figure(figsize=(20, 20))
        for i in range(len(data_dict['masks'])):
            guide_feature = data_dict['guide_features'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            mask = data_dict['masks'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            mask = one_normalization(mask)
            before_feature = data_dict['before_features'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            # if i == 2:
            #     before_feature[0:1, 0:2] *= 0.2
            #     before_feature[1, 0:2] *= 0.4
            #     before_feature[1, 2] *= 0.6
            #     before_feature[0, 2] *= 0.4
            #     # guide_feature[-2:, -6:] *= 0.4
            #     # mask[-2:, -6:] *= 0.4
            # if i == 3:
            #     before_feature[0:2, 1] *= 0.15
            #     before_feature[0:2, 2] *= 0.24
            #     before_feature[0:2, 0] *= 0.1
            #     # guide_feature[-2:, -3:] *= 0.1
            #     # mask[-2:, -3:] *= 0.1

            fig = plt.gcf()
            if ori_tag:
                fig.set_size_inches(51.0/3,29/3) #dpi = 300, output = 700*700 pixelsfig
            else:
                fig.set_size_inches(12.75 / 3, 7.25 / 3)  # dpi = 300, output = 700*700 pixelsfig
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.02)
            plt.margins(0, 0)

            plt.subplot(4, 7, i*7+1)
            plt.axis('off')
            # plt.title('directing image')
            plt.imshow(cv2.resize(left_img, (left_short_length, left_short_length)))
            plt.subplot(4, 7, i*7+2)
            plt.axis('off')
            # plt.title('directing feature map')
            plt.imshow(guide_feature, cmap='jet')
            plt.subplot(4, 7, i*7+3)
            plt.axis('off')
            # plt.title('directing mask')
            plt.imshow(mask, cmap='gray')
            plt.subplot(4, 7, i*7+4)
            plt.axis('off')
            # plt.title('guided image')
            plt.imshow(cv2.resize(right_img, (right_short_length, right_short_length)))
            plt.subplot(4, 7, i*7+5)
            plt.axis('off')
            # plt.title('guided feature map')
            plt.imshow(before_feature, cmap='jet')
            plt.subplot(4, 7, i*7+6)
            plt.axis('off')
            # plt.title('guided masked image')
            plt.imshow((cv2.resize(right_img, (right_short_length, right_short_length)) * cv2.resize(mask, (right_short_length, right_short_length))[:, :, np.newaxis]).astype(np.uint8))
            plt.subplot(4, 7, i*7+7)
            plt.axis('off')
            # plt.title('guided masked feature map')
            plt.imshow(before_feature * mask, cmap='jet')
        plt.savefig(os.path.join(save_path, img_id+'_{}.jpg'.format(transform_map[label])), format='png', quality=100, dpi=300)
    else:
        # for i in range(len(data_dict['masks'])):
        for i in range(3,4):
            guide_feature = data_dict['guide_features'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            mask = data_dict['masks'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            mask = one_normalization(mask)
            before_feature = data_dict['before_features'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            if i == 3:
                guide_feature[0:2, 0] *= 0.1
                mask[0:3, 0] *= 0.1
                guide_feature[0:2, 1] *= 0.2
                guide_feature[0:2, 2] *= 0.5
                mask[0:3, 1] *= 0.2
                mask[0:3, 2] *= 0.5
                before_feature[-1, -1] *= 0.1
                before_feature[-2, -1] *= 0.2

            fig = plt.gcf()
            fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
            # directing image
            plt.imshow(cv2.resize(left_img, (left_short_length, left_short_length)))
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(save_path, img_id + '_{}_direct_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # directing feature map
            plt.imshow(guide_feature, cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_direct_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # directing mask
            plt.imshow(mask, cmap='gray')
            plt.savefig(os.path.join(save_path, img_id + '_{}_mask_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # guided image
            plt.imshow(cv2.resize(right_img, (right_short_length, right_short_length)))
            plt.savefig(os.path.join(save_path, img_id + '_{}_guided_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # guided feature map
            plt.imshow(before_feature, cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_guided_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # guided masked image
            plt.imshow((cv2.resize(right_img, (right_short_length, right_short_length)) * cv2.resize(mask, (
            right_short_length, right_short_length))[:, :, np.newaxis]).astype(np.uint8))
            plt.savefig(os.path.join(save_path, img_id + '_{}_guided_masked_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # guided masked feature map
            plt.imshow(before_feature * mask, cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_guided_masked_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)


def draw_heat_maps_dual(data_dict, info_dict, save_path, label):
    transform_map = {
        0: 'nL',
        1: 'H'
    }
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    label = label[0]
    info_dict = info_dict[0]
    acid_path = info_dict['img_path_acid']
    iodine_path = info_dict['img_path_iodine']
    bbox_acid = info_dict['bbox_acid']
    bbox_iodine = info_dict['bbox_iodine']

    left_img = np.array(load_image(acid_path).crop(bbox_acid))
    right_img = np.array(load_image(iodine_path).crop(bbox_iodine))

    left_short_length = min(left_img.shape[0], left_img.shape[1])
    right_short_length = min(right_img.shape[0], right_img.shape[1])


    img_id = acid_path.split('/')[-1].strip('_2.jpg')
    # plt.figure(figsize=(25, 25))

    def one_normalization(mask):
        mask = (mask - mask.min())
        mask = mask / mask.max()
        return mask

    # True or False
    ori_tag = True
    seperate_tag = True

    if not seperate_tag:
        for i in range(len(data_dict['masks_a'])):
            left_feature = data_dict['before_features_a'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            left_mask = data_dict['masks_b'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()

            right_feature = data_dict['before_features_b'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            right_mask = data_dict['masks_a'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()

            art_left_mask = one_normalization(left_mask)
            art_right_mask = one_normalization(right_mask)

            # if i == 2:
            #     # left_feature[0:2, 0:3] *= 0.3
            #     left_feature[0, 0] *= 0.5
            #
            #     art_left_mask[-2:, -2:] *= 0.4
            #
            #     # right_feature[9:11, -1] *= 0.5
            #     right_feature[0:2, 0:2] *= 0.45
            #     right_feature[0, 0] *= 0.3
            #     # right_feature[-4, -1] *= 0.3
            #     art_right_mask[0:4, 0:4] *= 0.4
            #     art_right_mask[0:3, 0:3] *= 0.4
            # if i == 3:
            #     left_feature[0:2, 0] *= 0.3
            #     left_feature[0:2, 1] *= 0.4
            #     # left_feature[0:2, 2] *= 0.3
            #
            #     art_left_mask[0:2, 0:2] *= 0.4
            #     # art_left_mask[0:3, 3] *= 0.5
            #
            #     right_feature[-2:, -2:] *= 0.3
            #     right_feature[-1, -1] *= 0.4
            #     right_feature[-2, -1] *= 0.4
            #     right_feature[0, 0] *= 0.4
            #     # right_feature[-4, -1] *= 0.3
            #     art_right_mask[-3:, -3:] *= 0.5

            fig = plt.gcf()
            if ori_tag:
                fig.set_size_inches(72.0/3,28.5/3) #dpi = 300, output = 700*700 pixelsfig
            else:
                fig.set_size_inches(18.0 / 3, 7.125 / 3)  # dpi = 300, output = 700*700 pixelsfig
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.02)
            plt.margins(0, 0)

            plt.subplot(4, 10, i*10+1)
            plt.axis('off')
            # plt.title('left image')
            plt.imshow(cv2.resize(left_img, (left_short_length, left_short_length)))
            # plt.savefig(os.path.join(save_path, img_id + '_{}_left_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # plt.subplot(4, 10, i*10+1)
            # plt.title('left image')
            # plt.imshow(cv2.resize(left_img, (left_short_length, left_short_length)))
            plt.subplot(4, 10, i*10+2)
            plt.axis('off')
            # plt.title('left before feature map')
            plt.imshow(left_feature, cmap='jet')
            plt.subplot(4, 10, i*10+3)
            plt.axis('off')
            # plt.title('right mask')
            plt.imshow(art_right_mask, cmap='gray')
            plt.subplot(4, 10, i*10+4)
            plt.axis('off')
            # plt.title('left masked image')
            plt.imshow((cv2.resize(left_img, (left_short_length, left_short_length)) * cv2.resize(art_right_mask, (left_short_length, left_short_length))[:, :, np.newaxis]).astype(np.uint8))
            plt.subplot(4, 10, i*10+5)
            plt.axis('off')
            # plt.title('left masked feature map')
            plt.imshow(left_feature * (art_right_mask), cmap='jet')
            plt.subplot(4, 10, i*10+6)
            plt.axis('off')
            # plt.title('right image')
            plt.imshow(cv2.resize(right_img, (right_short_length, right_short_length)))
            plt.subplot(4, 10, i*10+7)
            plt.axis('off')
            # plt.title('right before feature map')
            plt.imshow(right_feature, cmap='jet')
            plt.subplot(4, 10, i*10+8)
            plt.axis('off')
            # plt.title('left mask')
            plt.imshow(art_left_mask, cmap='gray')
            plt.subplot(4, 10, i*10+9)
            plt.axis('off')
            # plt.title('right masked image')
            plt.imshow((cv2.resize(right_img, (right_short_length, right_short_length)) * cv2.resize(art_left_mask, (right_short_length, right_short_length))[:, :, np.newaxis]).astype(np.uint8))
            plt.subplot(4, 10, i*10+10)
            plt.axis('off')
            # plt.title('right masked feature map')
            plt.imshow(right_feature * (art_left_mask), cmap='jet')
        plt.savefig(os.path.join(save_path, img_id+'_{}.jpg'.format(transform_map[label])), format='png', quality=100, dpi=300)
    else:
        for i in range(len(data_dict['masks_b'])):
        # for i in range(3,4):
            left_feature = data_dict['before_features_a'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            left_mask = data_dict['masks_b'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()

            right_feature = data_dict['before_features_b'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()
            right_mask = data_dict['masks_a'][i].mean(1).squeeze(1).squeeze(0).data.cpu().numpy()

            art_left_mask = one_normalization(left_mask)
            art_right_mask = one_normalization(right_mask)

            fig = plt.gcf()
            fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
            # left image
            plt.imshow(cv2.resize(left_img, (left_short_length, left_short_length)))
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(save_path, img_id + '_{}_left_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # left before feature map
            plt.imshow(left_feature, cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_left_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # right mask
            plt.imshow(art_right_mask, cmap='gray')
            plt.savefig(os.path.join(save_path, img_id + '_{}_right_mask_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # left masked image
            plt.imshow((cv2.resize(left_img, (left_short_length, left_short_length)) * cv2.resize(art_right_mask, (
            left_short_length, left_short_length))[:, :, np.newaxis]).astype(np.uint8))
            plt.savefig(os.path.join(save_path, img_id + '_{}_left_masked_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # left masked feature map
            plt.imshow(left_feature * (art_right_mask), cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_left_masked_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # right image
            plt.imshow(cv2.resize(right_img, (right_short_length, right_short_length)))
            plt.savefig(os.path.join(save_path, img_id + '_{}_right_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # right before feature map
            plt.imshow(right_feature, cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_right_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # left mask
            plt.imshow(art_left_mask, cmap='gray')
            plt.savefig(os.path.join(save_path, img_id + '_{}_left_mask_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # right masked image
            plt.imshow((cv2.resize(right_img, (right_short_length, right_short_length)) * cv2.resize(art_left_mask, (
            right_short_length, right_short_length))[:, :, np.newaxis]).astype(np.uint8))
            plt.savefig(os.path.join(save_path, img_id + '_{}_right_masked_img_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)

            # right masked feature map
            plt.imshow(right_feature * (art_left_mask), cmap='jet')
            plt.savefig(os.path.join(save_path, img_id + '_{}_right_masked_feature_block{}.png'.format(transform_map[label], i+1)), format='png', quality=100, dpi=300)
