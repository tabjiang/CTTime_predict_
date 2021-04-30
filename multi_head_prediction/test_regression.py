import os
import re
import argparse
import time
import datetime
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataset import *
import logging
from sklearn.metrics import *
import model_zoo

logging.basicConfig(level=logging.INFO)

def main(FLAGS, ckpt, i_fold):
    #SEEDS
    torch.backends.cudnn.enabled = False
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    if not os.path.isdir(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    testLoader, df = load_HEdataset(FLAGS.path, FLAGS.guide_csv, False, FLAGS.batch_size, FLAGS.num_workers,
                                    FLAGS.img_mode, FLAGS.img_size, FLAGS.img_depth)

    if FLAGS.model_name == 'densenet':
        model = model_zoo.generate_model(model_depth=121, n_input_channels=1, num_classes=1,
                                         attention_type=FLAGS.attention_type)
    else:
        raise Exception('Undefined model!')

    state_dict = {k.replace('module.', ''):v for k, v in ckpt.items()}
    model.load_state_dict(state_dict, True)
    model = model.cuda()
    if(FLAGS.multi_gpu):
        model = nn.DataParallel(model)
    
    model.eval()

    lossFunction = nn.L1Loss()
    runningLoss = 0
    running_pred = []
    running_label = []

    with torch.no_grad():
        for i, (img, label, lenth) in enumerate(testLoader):
            img, label = img.cuda(), label.cuda()
            b, t, c, h, w = img.shape
            img = torch.transpose(img, 1, 2)

            label = label.view(b, 1)
            pred = model(img)

            loss = lossFunction(pred, label)

            runningLoss += loss.item()

            pred_cpu = np.concatenate(pred.clone().detach().cpu().numpy(), axis=0)
            label_cpu = np.concatenate(label.clone().detach().cpu().numpy(), axis=0)
            running_pred.append(pred_cpu)
            running_label.append(label_cpu)

            logging.info('Instance:[{} / {}], loss:{:.2f}'.format(i, len(testLoader), loss.item()))

    running_pred = np.concatenate(running_pred, axis=0)
    running_label = np.concatenate(running_label, axis=0)

    runningLoss /= len(testLoader)
    runningMAE, runningRMSE, runningR2, runningP = eval_metrics_regression(running_label, running_pred)

    testDictSlice = dict(loss=[runningLoss], MAE=[runningMAE], RMSE=[runningRMSE], R2=[runningR2], P=[runningP])

    pd.DataFrame(data=testDictSlice).to_csv(
        os.path.join(FLAGS.save_path, 'Test_stage_{}.csv'.format(i_fold)))

    logging.info('Test complete, loss:{:.2f}, MAE:{:.2f}, RMSE:{:.2f}, R2:{:.2f}, P:{:.2f}' \
                 .format(runningLoss, runningMAE, runningRMSE, runningR2, runningP))

    logging.info('Test complete, result save at {}.'.format(FLAGS.save_path))

    return running_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../toy_data', type=str, help='path to the data folder.')
    parser.add_argument('--guide_csv', default='./doc/toy/test_reg.csv', type=str, help='path to the guide.csv which includes dataset splits.')
    parser.add_argument('--save_path', default='./model/toy_test_reg', type=str, help='path to save trained models.')
    parser.add_argument('--ckpt_path', default='./model/toy_test_reg', type=str, help='ckpt_path path for continuing training.')
  
    parser.add_argument('--model_name', default='densenet', type=str, help='the model architecture defined in model_zoo.py.')
    parser.add_argument('--attention_type', default='None', type=str,
                        help='the attention type defined in model_zoo.py.')
    parser.add_argument('--in_channels', default=1, type=int, help='number of samples during batch training.')
    parser.add_argument('--out_channels', default=1, type=int, help='num_workers for prefetch image data')
    
    parser.add_argument('--batch_size', default=1, type=int, help='number of samples during batch training.')
    parser.add_argument('--num_workers', default=32, type=int, help='num_workers for prefetch image data')
    parser.add_argument('--img_size', default=224, type=int, help='image size')
    parser.add_argument('--img_depth', default=96, type=int, help='image depth')
    parser.add_argument('--img_mode', default=3, type=int, help='input image mode')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu id used for training, only support single GPU for now.')
    parser.add_argument('--multi_gpu', default=False, type=bool, help='the gpu id used for training, only support single GPU for now.')
    
    parser.add_argument('--start_epoch', default=-1, type=int, help='start epoch for training.')
    parser.add_argument('--num_epochs', default=100, type=int, help='total training epochs.')
    parser.add_argument('--start_check', default=10, type=int, help='how many epoches to valid') 
    parser.add_argument('--check_period', default=2, type=int, help='how many epoches to valid')
    
    parser.add_argument('--opt', default='Adam', type=str, help='optimizer used for training.')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate for model training.')
    parser.add_argument('--weight_decay', default=0, type=float, help='initial learning rate for model training.')
    parser.add_argument('--milestones', default=[50, 80], type=list, help='initial learning rate for model training.')
    parser.add_argument('--gamma', default=0.1, type=float, help='initial learning rate for model training.')
    parser.add_argument('--loss_name', default='L1', type=str, help='loss function used for training NNs.')
    parser.add_argument('--eps', default=1e-6, type=float, help='to avoid dividing 0')

    parser.add_argument('--k_folds', default=4, type=int, help='k-folds train/val')
    parser.add_argument('--Break', default=False, type=bool, help='decide whether only to train one of the k_folds')
    parser.add_argument('--seed', default=0, type=int, help='set random seeds to reproduce results')
    FLAGS = parser.parse_args()
    print(FLAGS)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    result_df = pd.read_csv(FLAGS.guide_csv)

    for i in range(FLAGS.k_folds):
        if FLAGS.ckpt_path is not None:
            for f in os.listdir(FLAGS.ckpt_path):
                if re.search('Fold{}_best'.format(i), f) != None:
                    print(f)
                    ckpt = torch.load(os.path.join(FLAGS.ckpt_path, f))['model']
                    pred_label = main(FLAGS, ckpt, i)
                    result_df[str(i)] = pred_label
        else:
            raise ValueError("No checkpoint!")

    result_df.to_csv(os.path.join(FLAGS.save_path, 'test_result.csv'), index=False)