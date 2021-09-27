from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import os
import time
import sys
import numpy as np
import pdb

from lib.models.nets.p3d_model import get_optim_policies

from datetime import datetime
from sklearn import metrics
from lib.datasets.utils import AverageMeter
from lib.datasets.utils import label_to_variable, img_data_to_variable
from lib.utils.summary import LogSummary
from lib.utils.comparator import Comparator
from lib.config.config import cfg

auto_op_mapping = {'loss': 'less_than', 'acc': 'larger_than', 'weighted_f1_score': 'larger_than', 'none': 'equal'}

def load_model(net, path_name, model_name):
    root_path = cfg.ModelSave_Path
    load_path = os.path.join(root_path, path_name, model_name)
    assert os.path.exists(load_path), 'Not exists requested checkpoint!'
    pre_model = torch.load(load_path)
    pre_dict = pre_model['state_dict']
    pre_dict = {k: v for k, v in pre_dict.items() if ('num_batches_tracked' not in k)}
    # state_dict = net.state_dict()
    # assert pre_dict.keys() == state_dict.keys()
    # state_dict.update(pre_dict)
    net.load_state_dict(pre_dict)
    print('Load Model From :{}'.format(load_path))
    print('Iters :{}'.format(pre_model['iters']))
    return pre_model['iters'], pre_model['lr']


def save_model(net, path_name, model_name, iters, lr, parallel=False):
    root_path = cfg.ModelSave_Path
    load_path = os.path.join(root_path, path_name)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    load_path = os.path.join(load_path, model_name)
    if parallel:
        save_dict = {
            'iters': iters,
            'state_dict': net.module.state_dict(),
            'lr': lr
        }
    else:
        save_dict = {
            'iters': iters,
            'state_dict': net.state_dict(),
            'lr': lr
        }
    torch.save(save_dict, load_path)


def init_optimizer(net, lr, strategy):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    parameters = []
    for para in net.parameters():
        if para.requires_grad == True:
            parameters.append(para)
    if strategy['net'] == 'p3d199' and strategy['dynamic_lr']:
        optimizer = torch.optim.SGD(get_optim_policies(net.model, strategy['modality']), momentum=cfg.SOLVER.MOMENTUM, lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(parameters, momentum=cfg.SOLVER.MOMENTUM, lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(net, data_input, strategy_dict):
    parallel_tag = False
    start_step = 0
    end_step = cfg.SOLVER.MAX_EPOCH * len(data_input[0])
    learning_rate = cfg.SOLVER.BASE_LR
    resume_path = os.path.join(cfg.ModelSave_Path, strategy_dict['data_tag'], strategy_dict['experiment_name'])
    tb_dir = os.path.join(cfg.TensorboardSave_Path, strategy_dict['data_tag'], datetime.now().strftime('%b%d_%H-%M-%S_')
                          + strategy_dict['experiment_name'])

    # TensorboardX
    tensor_writer = LogSummary(tb_dir)

    # resume
    if strategy_dict['resume'] is not None:
        model_name = 'last_checkpoint.pth'
        start_step = strategy_dict['resume'] + 1
        _, old_lr = load_model(net, resume_path, model_name)
        learning_rate = old_lr
    else:
        if strategy_dict['weight'] is not None:
            if strategy_dict['tf_learning']:
                print('Use transfer learning strategy!')
                net.model.load_model(strategy_dict['weight'])
        else:
            pass

    # auto learning indicator
    # warning: can't resume last parameters except old learning_rate
    lr_adaptiver = Comparator(learning_rate, strategy_dict['auto'], auto_op_mapping[strategy_dict['auto_metric']],
                              (np.array(cfg.SOLVER.STEPS) * len(data_input[0])).tolist(), decay_ratio=cfg.SOLVER.LR_DECAY, max_decay=cfg.SOLVER.MAX_DECAY)
    lr_indication = None

    # print net's parameters' num
    num_params = 0
    for param in net.model.parameters():
        num_params += param.numel()
    print('#'*10)
    print('Total number of parameters: %.3f M' % (num_params / 1e6))

    if cfg.GPU:
        net = net.cuda()
        if strategy_dict['gpu_ids'] is not None:
            assert isinstance(strategy_dict['gpu_ids'], list) and len(strategy_dict['gpu_ids']) > 1
            net = torch.nn.DataParallel(net, device_ids=strategy_dict['gpu_ids'])
            parallel_tag = True

    # froze and fix
    if cfg.TRAIN.FIX_PARAMETER:
        if not parallel_tag:
            # net.model.fix_model()
            net.model.froze_bn()
            print('='*10)
            print('Now, using frozen BN')
            print('='*10)
        else:
            # net.module.model.fix_model()
            net.module.model.froze_bn()
            print('='*10)
            print('Now, using frozen BN')
            print('='*10)

    # optimizer
    optimizer = init_optimizer(net, learning_rate, strategy_dict)

    # dataloader
    train_dataloader = data_input[0]
    train_iter = iter(train_dataloader)
    valid_dataloader = data_input[1]
    valid_iter = iter(valid_dataloader)

    valid_interval = len(valid_dataloader)
    train_interval = 50
    log_interval = len(train_dataloader)
    eval_interval = len(train_dataloader)

    if not parallel_tag:
        loss_list = net.loss_list
    else:
        loss_list = net.module.loss_list

    train_loss_counters = []
    # The first is always 'cls_loss'
    for _, loss_name in enumerate(loss_list):
        train_loss_counters.append(AverageMeter())

    train_predict_list = []
    train_target_list = []
    valid_best_weighted_f1_score = 0.
    valid_best_acc = 0.

    epoch_batch_count = 0

    for step in range(start_step, end_step + 1):
        net.train()
        # update lr if no auto decay
        if not strategy_dict['auto']:
            lr_indication = lr_adaptiver(step)
            if lr_indication == 'decay':
                learning_rate = lr_adaptiver.lr
                adjust_learning_rate(optimizer, learning_rate)
            elif lr_indication == 'invariable':
                pass
            else:
                raise ValueError

        # Data
        blobs, targets = next(train_iter)
        epoch_batch_count += 1
        image_data = img_data_to_variable(blobs, gpu=cfg.GPU)
        cls_targets = label_to_variable(targets, gpu=cfg.GPU, volatile=False)

        # reset iter
        if epoch_batch_count >= len(train_dataloader):
            train_iter = iter(train_dataloader)
            epoch_batch_count = 0

        # forward
        t0 = time.time()
        # if strategy_dict['net'] == 'i3d':
        #     out, cls_preds = net(image_data, cls_targets)
        # else:
        cls_preds = net(image_data, cls_targets)
        if not parallel_tag:
            train_loss_strategy = net.loss(cls_preds, cls_targets, True)
        else:
            train_loss_strategy = net.module.loss(cls_preds, cls_targets, True)

        if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
            pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
        else:
            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

        train_predict_list += list(pred.cpu().numpy())
        train_target_list += list(targets)

        for i, loss_name in enumerate(train_loss_strategy):
            train_loss_counters[i].update(train_loss_strategy[loss_name].data, cfg.TRAIN.BATCH_SIZE)

        # backward and update
        optimizer.zero_grad()
        train_loss = train_loss_strategy['cls_loss']
        train_loss.backward()
        optimizer.step()
        t1 = time.time()

        if step % train_interval == 0:
            pstring = 'timer: %.4f sec.' % (t1 - t0)
            print(pstring)
            if 'total_loss' not in train_loss_strategy.keys():
                show_loss = train_loss_strategy['cls_loss'].data[0]
            else:
                show_loss = train_loss_strategy['total_loss'].data[0]
            pstring = 'iter ' + repr(step) + ' || Loss: %.4f || lr: %.5f || acc: %.4f' % (show_loss, learning_rate,
                metrics.accuracy_score(
                    train_target_list,
                    train_predict_list))
            print(pstring)

        if step % log_interval == 0 and step != start_step:
            metric_names = []
            scalars = []
            metric_names += ['recall-{}'.format(cls) for cls in range(cfg.NUM_CLASSES)]
            scalars += [metrics.recall_score(train_target_list, train_predict_list, average='macro', labels=[cls])
                        for cls in range(cfg.NUM_CLASSES)]
            metric_names += ['acc']
            scalars += [metrics.accuracy_score(train_target_list, train_predict_list)]
            metric_names += ['weighted_f1_score']
            scalars += [metrics.f1_score(train_target_list, train_predict_list, average='weighted')]
            for i, loss_name in enumerate(train_loss_strategy):
                metric_names += [loss_name]
                scalars += [train_loss_counters[i].avg]
            metric_names += ['learning_rate']
            scalars += [learning_rate]
            tensor_writer.write_scalars(scalars, metric_names, step, tag='train')

            # reset the loss counter
            for _, loss_name in enumerate(loss_list):
                train_loss_counters.append(AverageMeter())
            train_predict_list, train_target_list = [], []

        if step % eval_interval == 0 and step != start_step:
            net.eval()
            test_loss_counters = []
            # The first is always 'cls_loss'
            for _, loss_name in enumerate(loss_list):
                test_loss_counters.append(AverageMeter())
            test_target_list, test_predict_list = [], []
            for test_step in range(valid_interval):
                # Data
                # blobs, targets, imgs_paths = next(valid_iter)
                blobs, targets = next(valid_iter)
                # valid_epoch_batch_count += 1
                image_data = img_data_to_variable(blobs, gpu=cfg.GPU, volatile=True)
                cls_targets = label_to_variable(targets, gpu=cfg.GPU, volatile=True)

                with torch.no_grad():
                    # forward
                    # if strategy_dict['net'] == 'i3d':
                    #     out, cls_preds = net(image_data, cls_targets)
                    # else:
                    cls_preds = net(image_data, cls_targets)
                    if not parallel_tag:
                        test_loss_strategy = net.loss(cls_preds, cls_targets, False)
                    else:
                        test_loss_strategy = net.module.loss(cls_preds, cls_targets, False)

                if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                    pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

                test_predict_list += list(pred.cpu().numpy())
                test_target_list += list(targets)

                for i, loss_name in enumerate(test_loss_strategy):
                    test_loss_counters[i].update(test_loss_strategy[loss_name].data, cfg.TEST.BATCH_SIZE)
            valid_iter = iter(valid_dataloader)
                # del cls_preds
                # del image_data

            metric_names = []
            scalars = []
            metric_names += ['recall-{}'.format(cls) for cls in range(cfg.NUM_CLASSES)]
            scalars += [metrics.recall_score(test_target_list, test_predict_list, average='macro', labels=[cls])
                        for cls in range(cfg.NUM_CLASSES)]
            metric_names += ['acc']
            scalars += [metrics.accuracy_score(test_target_list, test_predict_list)]
            metric_names += ['weighted_f1_score']
            scalars += [metrics.f1_score(test_target_list, test_predict_list, average='weighted')]
            for i, loss_name in enumerate(test_loss_strategy):
                metric_names += [loss_name]
                scalars += [test_loss_counters[i].avg]

            tensor_writer.write_scalars(scalars, metric_names, step, tag='val')

            # save best checkpoint, the metric is weighted f1 score
            # consider later checkpoint having stronger robustness, choose later
            if scalars[metric_names.index('weighted_f1_score')] > valid_best_weighted_f1_score:
                valid_best_weighted_f1_score = scalars[metric_names.index('weighted_f1_score')]
                print('Saving best valid f1 state, iter:', step)
                save_model(net, resume_path, 'best_valid_f1_checkpoint.pth', step, learning_rate, parallel_tag)

            if scalars[metric_names.index('acc')] > valid_best_acc:
                valid_best_acc = scalars[metric_names.index('acc')]
                print('Saving best valid acc state, iter:', step)
                save_model(net, resume_path, 'best_valid_acc_checkpoint.pth', step, learning_rate, parallel_tag)

            if strategy_dict['auto']:
                lr_indication = lr_adaptiver(scalars[metric_names.index(strategy_dict['auto_metric'])])
                if lr_indication == 'decay':
                    learning_rate = lr_adaptiver.lr
                    adjust_learning_rate(optimizer, learning_rate)
                elif lr_indication == 'early_stop':
                    print('Early stop!')
                    sys.exit()
                elif lr_indication == 'invariable':
                    pass
                else:
                    raise ValueError

        # save_model
        if step % (1 * len(train_dataloader)) == 0 and step != start_step:
            print('Saving last state, iter:', step)
            save_model(net, resume_path, 'last_checkpoint.pth', step, learning_rate, parallel_tag)

        # save_model regularly
        if strategy_dict['regularly_save']:
            if step % (cfg.SOLVER.SAVE_INTERVAL * len(train_dataloader)) == 0 and step != start_step:
                print('Saving state regularly, iter:', step)
                save_model(net, resume_path, '{}_checkpoint.pth'.format(step), step, learning_rate, parallel_tag)
    return valid_best_acc