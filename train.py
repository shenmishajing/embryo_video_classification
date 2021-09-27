import argparse
import os

from lib.utils.transform import *
from torch.utils.data import DataLoader
from lib.config.config import cfg_from_file, cfg
from lib.datasets.factory import get_imdb
from lib.models.factory import get_model
from lib.models.based_model import Base_Model
from lib.datasets.dataset.video import TSNDataSet
from lib.utils.init_function import weights_init_xavier, weights_init_he, weights_init_normal
from lib.models.criterion import Criterion

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a network')
    # model architecture('attention' -> 'cross_attention')
    parser.add_argument('arc', choices=['single'],
                        help='the architecture of model')
    parser.add_argument('--net', dest='net',
                        choices=['vgg16bn', 'res50', 'res101', 'dense121', 'dense169', 'p3d199', 'i3d'],
                        default='res50', type=str)
    parser.add_argument('--init_func', dest='init_func', default='xavier',
                        choices=['xavier', 'he', 'normal'], type=str,
                        help='initialization function of model')
    # data parameters
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--num_segments', dest='num_segments',
                        help='tag of the model',
                        default=16, type=int)
    parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
    parser.add_argument('--flow_prefix', type=str, help="prefix of Flow frames", default='flow_')
    parser.add_argument('--modality', type=str, help='input modality of networks', default='RGB', choices=['RGB', 'Flow'
                        , 'RGBDiff'])
    parser.add_argument('-rd', '--random_sample', dest='random_sample', action='store_true',
                        help='whether to use random sampling strategy during each segment')
    # training strategy
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pre-trained model weights',
                        default=None,
                        type=str)
    parser.add_argument('-tf', '--tf_learning', dest='tf_learning', action='store_true',
                        help='whether to use transfer learning')
    parser.add_argument('--resume', dest='resume',
                        help='resume checkpoint',
                        default=None, type=int)
    parser.add_argument('--auto', dest='auto', action='store_true',
                        help='whether to use auto learning decay')
    parser.add_argument('--auto_metric', dest='auto_metric',
                        help='when use "auto", select a metric as every epoch s judgement',
                        default='none', choices=['none', 'loss', 'acc', 'weighted_f1_score'])
    parser.add_argument('-rs', '--regularly_save', dest='regularly_save', action='store_true',
                        help='whether to save checkpoint timely')
    parser.add_argument('--kfold', dest='kfold', default=None, type=int,
                        help='Use k-fold cross-validation')
    parser.add_argument('--lr', dest='lr', default=None, type=float, help='learning rate of training')
    parser.add_argument('--loss', dest='loss', default='Cross_Entropy', type=str, choices=['Cross_Entropy', 'Focal_Loss'],
                        help='loss type used in training stage')
    parser.add_argument('--class_weights', dest='class_weights', type=int, nargs='+', default=None,
                        help='Loss weights which are used to optimize the less class')
    parser.add_argument('-m', '--merge_valid', dest='merge_valid', action='store_true',
                        help='whether to merge valid and test set during evluation')
    parser.add_argument('-dl', '--dynamic_lr', dest='dynamic_lr', action='store_true',
                        help='whether to use dynamic lr during training(p3d)')
    # gpu setting
    parser.add_argument('--gpus', dest='gpus', default=None, nargs='+', type=int,
                        help='When use dataparallel, it is needed to set gpu Ids')

    # if len(sys.argv) == 1:
    #   parser.print_help()
    #   sys.exit(1)

    args = parser.parse_args()
    return args

def choose_model(args, init_function, criterion):
    if args.net == 'i3d':
        net = Base_Model(get_model(args.net)(cfg.NUM_CLASSES, init_function, args.modality.lower(), args.new_length), criterion)
    elif args.net == 'p3d199':
        net = Base_Model(get_model(args.net)(cfg.NUM_CLASSES, init_function, args.modality), criterion)
    net_arc = args.net
    return net, net_arc

def get_augmentation(modality, input_size):
    if modality == 'RGB':
        # return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
        #                                        GroupRandomHorizontalFlip(is_flow=False)])

        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False),
                                               GroupRandomVerticalFlip(),
                                               GroupRandomBrightness(),
                                               GroupShiftScaleRotate()])
    elif modality == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=True)])
    elif modality == 'RGBDiff':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=False)])

if __name__ == '__main__':
    args = parse_args()

    args.data_category = 'HUST_video'
    args.data_set_name = 'first_travel_VideoImage'

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        # data_length = 5
        data_length = 1
    args.new_length = data_length

    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    if args.modality == 'RGB':
        pass
    elif args.modality == 'Flow':
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

    scale_size = cfg.TRAIN.SCALE * 256 // 224

    args.cfg_file = './experiments/{}_{}_{}.yml'.format(args.arc, args.data_category, args.data_set_name)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    from lib.train.train import train

    data_set_types = ['train', 'valid', 'test']

    DataSetClass = get_imdb(args.data_category)

    train_augmentation = get_augmentation(args.modality, cfg.TRAIN.SCALE)

    # init function
    if args.init_func == 'xavier':
        init_function = weights_init_xavier
    elif args.init_func == 'he':
        init_function = weights_init_he
    elif args.init_func == 'normal':
        init_function = weights_init_normal
    else:
        raise ValueError

    # pre-trained model path set
    if args.net == 'res50':
        args.weight = '/home/mxj/.torch/models/resnet50-19c8e357.pth'
    elif args.net == 'res101':
        args.weight = '/home/mxj/.torch/models/resnet101-5d3b4d8f.pth'
    elif args.net == 'vgg16bn':
        args.weight = '/home/mxj/.torch/models/vgg16-397923af.pth'
    elif args.net == 'dense121':
        args.weight = '/home/mxj/.torch/models/densenet121-241335ed.pth'
    elif args.net == 'dense169':
        args.weight = '/home/mxj/.torch/models/densenet169-6f0f7f60.pth'
    elif args.net == 'res50_ame':
        args.weight = '/home/mxj/.torch/models/resnet50-19c8e357.pth'
    elif args.net == 'p3d199':
        if args.modality == 'RGB':
            # Kinetics600
            # args.weight = '/home/mxj/data/CommonWeights/p3d_rgb_199.checkpoint.pth.tar'
            args.weight = '/data/mxj/CommonWeights/P3D199_rgb_299x299_model_best.pth.tar'
        else:
            args.weight = '/data/mxj/CommonWeights/p3d_flow_199.checkpoint.pth.tar'
    elif args.net == 'i3d':
        if args.modality == 'RGB':
            args.weight = '/data/grq/project/I3D/model/model_rgb.pth'
        else:
            args.weight = '/data/grq/project/I3D/model/model_flow.pth'
    else:
        print('unexpected BaseModel!')
        raise KeyError

    # net choose
    criterion = Criterion(args.loss, cfg.NUM_CLASSES, args.class_weights)
    net, net_arc = choose_model(args, init_function, criterion)

    # tags factory
    auto_tag = (lambda: '' if not args.auto else '_'.join(['', 'auto', args.auto_metric]))()
    tf_tag = (lambda: '_'.join(['', 'ori']) if not args.tf_learning else '_'.join(['', 'tf']))()
    exp_data_tag = '_'.join(['', str(cfg.TRAIN.SCALE), 'Segments', str(args.num_segments), 'Clip', str(data_length)])

    # example: {single}_{res50}_{crop}{_exp_data_tag}{_text}{_tf}{_auto_loss}_{1}
    experiment_name = '{}_{}_{}{}{}{}_{}'.format(args.arc, net_arc, args.modality, exp_data_tag, tf_tag, auto_tag, args.tag)
    data_tag = os.path.join(args.data_category, args.data_set_name)

    print('Called with args:')
    print(args)

    train_dataset = TSNDataSet("/data/mxj/IVF_HUST/first_travel_VideoImage", 'train.json', num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="{:d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   # change sampling setting
                   random_shift=args.random_sample,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=False),
                       ToTorchFormatTensor(args.num_segments, cfg.TRAIN.SCALE, div=True, modality=args.modality, new_length=args.new_length),
                       normalize,
                   ]))
    train_dataloader = DataLoader(train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=train_dataset.classify_collate, pin_memory=True)

    valid_dataset = TSNDataSet("/data/mxj/IVF_HUST/first_travel_VideoImage", ['valid.json', 'test.json'] if args.merge_valid else 'valid.json', num_segments=args.num_segments,
                   new_length=data_length,
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
    valid_dataloader = DataLoader(valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=valid_dataset.classify_collate, pin_memory=True)

    strategy_dict = {
        'weight': args.weight,
        'resume': args.resume,
        'experiment_name': experiment_name,
        'data_tag': data_tag,
        'batch_size': cfg.TRAIN.BATCH_SIZE,
        'tf_learning': args.tf_learning,
        'auto': args.auto,
        'auto_metric': args.auto_metric,
        'regularly_save': args.regularly_save,
        'gpu_ids': args.gpus,
        'net':args.net,
        'new_length': args.new_length,
        'modality': args.modality,
        'dynamic_lr': args.dynamic_lr
    }

    train(net, [train_dataloader, valid_dataloader], strategy_dict)