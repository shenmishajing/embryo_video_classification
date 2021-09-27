import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import metrics

from lib.datasets.utils import img_data_to_variable, label_to_variable
from lib.train.train import load_model
from lib.config.config import cfg

def if_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.train()

def test(net, test_dataloader, strategy_dict):
    if strategy_dict['ensemble']:
        # model_name = ['best_valid_checkpoint.pth', 'best_valid_checkpoint.pth']
        model_name = ['best_valid_acc_checkpoint.pth', 'best_valid_checkpoint.pth']
        alpha = 0.7

        ensemble_ratio = [1-alpha, alpha]

        resume_path_A = os.path.join(cfg.ModelSave_Path, strategy_dict['data_tag'], strategy_dict['experiment_name'][0])
        resume_path_B = os.path.join(cfg.ModelSave_Path, strategy_dict['data_tag'], strategy_dict['experiment_name'][1])
        _, _ = load_model(net[0], resume_path_A, model_name[0])
        _, _ = load_model(net[1], resume_path_B, model_name[1])

        if cfg.GPU:
            net[0] = net[0].cuda()
            net[1] = net[1].cuda()

        net[0].eval()
        net[1].eval()
    else:
        # load model
        if strategy_dict['resume'] is not None:
            model_name = '{}_checkpoint.pth'.format(strategy_dict['resume'])
        else:
            # model_name = 'best_valid_checkpoint.pth'
            model_name = 'best_valid_acc_checkpoint.pth'

        resume_path = os.path.join(cfg.ModelSave_Path, strategy_dict['data_tag'], strategy_dict['experiment_name'])
        _, old_lr = load_model(net, resume_path, model_name)

        if cfg.GPU:
            net = net.cuda()

        net.eval()

    test_iter = iter(test_dataloader)
    test_interval = len(test_dataloader)
    test_target_list, test_predict_list = [], []

    tb = time.time()
    for test_step in range(test_interval):
        # Data
        t0 = time.time()
        if strategy_dict['ensemble']:
            imgs, flows, targets = next(test_iter)
            image_data = img_data_to_variable(imgs, gpu=cfg.GPU, volatile=True)
            flow_data = img_data_to_variable(flows, gpu=cfg.GPU, volatile=True)
            cls_targets = label_to_variable(targets, gpu=cfg.GPU, volatile=True)

            # forward
            with torch.no_grad():
                cls_preds_img = net[0](image_data, cls_targets)
                cls_preds_flow = net[1](flow_data, cls_targets)
            # cls_preds = 0.4 * F.softmax(cls_preds_img, dim=1) + 0.6 * F.softmax(cls_preds_flow, dim=1)
            cls_preds = ensemble_ratio[0] * F.softmax(cls_preds_img, dim=1) + ensemble_ratio[1] * F.softmax(cls_preds_flow, dim=1)
        else:
            blobs, targets = next(test_iter)

            image_data = img_data_to_variable(blobs, gpu=cfg.GPU, volatile=True)
            cls_targets = label_to_variable(targets, gpu=cfg.GPU, volatile=True)

            # forward
            with torch.no_grad():
                cls_preds = net(image_data, cls_targets)

        # print(cls_preds, F.softmax(cls_preds_img, dim=1))
        if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
            pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
        else:
            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

        test_predict_list += list(pred.cpu().numpy())
        test_target_list += list(targets)

        t1 = time.time()
        print('Test: {:d}/{:d} {:.3f}s {:.3f}s'.format(test_step + 1, test_interval, t1 - t0, t1 - tb))

    confusion_matrix = metrics.confusion_matrix(test_target_list, test_predict_list,
                                                labels=[cls for cls in range(cfg.NUM_CLASSES)])

    print('===' * 10)
    print('Evaluation')
    print('===' * 10)

    print('Confusion Matrix:')
    for line in confusion_matrix:
        line_s = '  '.join([str(i) for i in line])
        print(line_s)


    metricsTest = {'recall-{}'.format(cfg.CLASSES[cls]): metrics.recall_score(test_target_list, test_predict_list,
                                                                              average='macro', labels=[cls])
                   for cls in range(cfg.NUM_CLASSES)}
    metricsTest['acc'] = metrics.accuracy_score(test_target_list, test_predict_list)

    print('Accuracy And Recall:')
    for key, value in metricsTest.items():
        print('{}: {:.5f}'.format(key, value))
