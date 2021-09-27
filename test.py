import argparse
import os

from torch.utils.data import DataLoader
from lib.config.config import cfg_from_file, cfg
from lib.datasets.factory import get_imdb
from lib.models.factory import get_model
from lib.models.based_model import Base_Model
from lib.utils.transform import *
from lib.datasets.dataset.video import TSNDataSet
from lib.datasets.dataset.video_ensemble import TSNEnsemble
from lib.utils.init_function import weights_init_xavier, weights_init_he, weights_init_normal

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a network')
    # model architecture
    parser.add_argument('arc', choices=['single'],
                        help='the architecture of model')
    parser.add_argument('--net', dest='net',
                        choices=['vgg16bn', 'res50', 'res101', 'dense121', 'dense169', 'p3d199', 'i3d'],
                        default='res50', type=str)
    parser.add_argument('--init_func', dest='init_func', default='xavier',
                        choices=['xavier', 'he', 'normal'], type=str,
                        help='initialization function of model')
    # data parameters
    parser.add_argument('--num_segments', dest='num_segments',
                        help='tag of the model',
                        default=16, type=int)
    parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
    parser.add_argument('--flow_prefix', type=str, help="prefix of Flow frames", default='flow_')
    parser.add_argument('--modality', type=str, help='input modality of networks', default='RGB', choices=['RGB', 'Flow'
                        , 'RGBDiff'])
    # testing strategy
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pre-trained model weights',
                        default=None,
                        type=str)
    parser.add_argument('-tf', '--tf_learning', dest='tf_learning', action='store_true',
                        help='whether to use transfer learning')
    parser.add_argument('--resume', dest='model_check_point',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--experiment_name', dest='experiment_name',
                        help='the name of the experimetn',
                        default=None, type=str)

    parser.add_argument('--kfold', dest='kfold', default=None, type=int,
                        help='whether to use kfold testing')
    parser.add_argument('--test_set', dest='test_set', default='test', type=str,
                        help='the data set which used to test')
    parser.add_argument('--ensemble', dest='ensemble', default=False, action='store_true',
                        help='whether to use ensemble while testing')
    parser.add_argument('-m', '--merge_valid', dest='merge_valid', action='store_true',
                        help='whether to merge valid and test set during evluation')
    args = parser.parse_args()
    return args

def choose_model(args, init_function, criterion=None):
    if args.net == 'i3d':
        net = Base_Model(get_model(args.net)(cfg.NUM_CLASSES, init_function, args.modality.lower(), args.new_length), criterion)
    elif args.net == 'p3d199':
        net = Base_Model(get_model(args.net)(cfg.NUM_CLASSES, init_function, args.modality), criterion)
    net_arc = args.net
    return net, net_arc

if __name__ == '__main__':
    args = parse_args()

    args.data_category = 'HUST_video'
    args.data_set_name = 'first_travel_VideoImage'

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        # data_length = 5
        data_length = 1

    if not args.ensemble:
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]

        if args.modality == 'RGB':
            pass
        elif args.modality == 'Flow':
            # in old, mean and std is the same as 'RGB'
            input_mean = [0.5]
            input_std = [np.mean(input_std)]
        elif args.modality == 'RGBDiff':
            input_mean = [0.485, 0.456, 0.406] + [0] * 3 * args.new_length
            input_std = input_std + [np.mean(input_std) * 2] * 3 * args.new_length
        else:
            raise ValueError

        if args.modality != 'RGBDiff':
            normalize = GroupNormalize(input_mean, input_std)
        else:
            normalize = IdentityTransform()
        args.new_length = data_length
    else:
        input_mean_rgb = [0.485, 0.456, 0.406]
        input_std_rgb = [0.229, 0.224, 0.225]
        input_mean_flow = [0.5]
        input_std_flow = [np.mean(input_std_rgb)]
        normalize = [GroupNormalize(input_mean_rgb, input_std_rgb), GroupNormalize(input_mean_flow, input_std_flow)]
        args.new_length = [1, 1]

    scale_size = cfg.TRAIN.SCALE * 256 // 224

    args.cfg_file = './experiments/{}_{}_{}.yml'.format(args.arc, args.data_category, args.data_set_name)

    # indicate the experiment name by yourself
    if args.ensemble:
        # args.experiment_name = ['single_i3d_224_tf_i3d_test4', 'single_i3d_224_tf_i3d_flow_1e-2']

        # valid acc 72%
        # args.experiment_name = ['single_p3d199_224_tf_Kinetics600_refine_run1', 'single_p3d199_Flow_224_Segments_16_Clip_1_tf_1e-3_run1']

        # args.experiment_name = ['single_p3d199_RGB_224_Segments_16_Clip_1_tf_fixed_1e-3_run1', 'single_p3d199_Flow_224_Segments_16_Clip_1_tf_1e-3_run3']

        args.experiment_name = ['single_p3d199_RGB_224_Segments_16_Clip_1_tf_new_Agu_fixed_1e-3_run1', 'single_p3d199_Flow_224_Segments_16_Clip_1_tf_1e-3_run3']
    else:
        if args.experiment_name is None:
            # Flow
            # args.experiment_name = 'single_i3d_224_tf_i3d_flow_1e-2'
            # args.experiment_name = 'single_i3d_224_Segments_16_Clip_5_tf_i3d_flow_5e-2'
            # args.experiment_name = 'single_p3d199_Flow_224_Segments_16_Clip_1_tf_1e-3_run1'
            # args.experiment_name = 'single_p3d199_Flow_224_Segments_16_Clip_1_tf_1e-3_run3'

            # RGB
            # args.experiment_name = 'single_i3d_224_tf_i3d_test4'
            # args.experiment_name = 'single_i3d_224_Segments_16_Clip_1_tf_i3d_rgb_5e-3'
            # args.experiment_name = 'single_p3d199_224_tf_Kinetics600_refine_run1'
            # args.experiment_name = 'single_p3d199_RGB_224_Segments_16_Clip_1_tf_1e-3_run1'
            # args.experiment_name = 'single_p3d199_RGB_224_Segments_16_Clip_1_tf_fixed_1e-3_run1'
            args.experiment_name = 'single_p3d199_RGB_224_Segments_16_Clip_1_tf_p3d_rgb_v1.0'

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    from lib.train.eval import test

    DataSetClass = get_imdb(args.data_category)

    # init function
    if args.init_func == 'xavier':
        init_function = weights_init_xavier
    elif args.init_func == 'he':
        init_function = weights_init_he
    elif args.init_func == 'normal':
        init_function = weights_init_normal
    else:
        raise ValueError

    print('Called with args:')
    print(args)

    # net choose
    if args.ensemble:
        # netA, _ = choose_model(args, 'rgb')
        args.modality = 'RGB'
        args.net = 'p3d199'
        netA, _ = choose_model(args, init_function)
        args.modality = 'Flow'
        args.net = 'p3d199'
        netB, _ = choose_model(args, init_function)
        net = [netA, netB]
    else:
        net, net_arc = choose_model(args, init_function)

    if args.ensemble:
        test_dataset = TSNEnsemble("/home/mxj/data/IVF_HUST/first_travel_VideoImage", ['valid.json', 'test.json'] if args.merge_valid else '{}.json'.format(args.test_set),
                                  num_segments=args.num_segments,
                                  new_length=args.new_length,
                                  modality=args.modality,
                                  image_tmpl=["{:d}.jpg", args.flow_prefix + "{}_{:05d}.jpg"],
                                  random_shift=False,
                                  transform=[
                                      torchvision.transforms.Compose([
                                      GroupScale(int(scale_size)),
                                      GroupCenterCrop(cfg.TRAIN.SCALE),
                                      Stack(roll=False),
                                      ToTorchFormatTensor(args.num_segments, cfg.TRAIN.SCALE, div=True, modality='RGB', new_length=args.new_length[0]),
                                      normalize[0],
                                  ]),
                                      torchvision.transforms.Compose([
                                          GroupScale(int(scale_size)),
                                          GroupCenterCrop(cfg.TRAIN.SCALE),
                                          Stack(roll=False),
                                          ToTorchFormatTensor(args.num_segments, cfg.TRAIN.SCALE, div=True, modality='Flow', new_length=args.new_length[1]),
                                          normalize[1],
                                      ])],
                                   )
    else:
        test_dataset = TSNDataSet("/home/mxj/data/IVF_HUST/first_travel_VideoImage", ['valid.json', 'test.json'] if args.merge_valid else '{}.json'.format(args.test_set), num_segments=args.num_segments,
                       new_length=args.new_length,
                       modality=args.modality,
                       image_tmpl="{:d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                       random_shift=False,
                       transform=torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(cfg.TRAIN.SCALE),
                           Stack(roll=False),
                           ToTorchFormatTensor(args.num_segments, cfg.TRAIN.SCALE, div=True, modality=args.modality, new_length=args.new_length),
                           normalize,
                       ]))

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                  num_workers=2, collate_fn=test_dataset.classify_collate, pin_memory=True)

    data_tag = os.path.join(args.data_category, args.data_set_name)

    strategy_dict = {
        'experiment_name': args.experiment_name,
        'resume': args.model_check_point,
        'data_tag': data_tag,
        'test_set': args.test_set,
        'ensemble': args.ensemble
    }

    test(net, test_dataloader, strategy_dict)
