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

def main(FLAGS, ckpt):


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

            logging.info('Instance:[{} / {}], loss:{:.2f}, predict time:{:.2f}'.format(i, len(testLoader), loss.item(),pred.item()))

    running_pred = np.concatenate(running_pred, axis=0)
    running_label = np.concatenate(running_label, axis=0)

    runningLoss /= len(testLoader)
    runningMAE, runningRMSE, runningR2, runningP = eval_metrics_regression(running_label, running_pred)

    logging.info('Test complete, loss:{:.2f}, MAE:{:.2f}, RMSE:{:.2f}, R2:{:.2f}, P:{:.2f}' \
                 .format(runningLoss, runningMAE, runningRMSE, runningR2, runningP))
    logging.info('Test complete, result save at {}.'.format(FLAGS.save_path))

    return running_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../toy_data', type=str, help='path to the test data folder.')
    parser.add_argument('--guide_csv', default='./doc/toy/test_reg.csv', type=str,
                        help='path to the test.csv which includes dataset splits.')
    parser.add_argument('--save_path', default='./test_results/reg', type=str, help='path to save test results.')
    parser.add_argument('--ckpt_path', default='./model/reg.pth', type=str, help='ckpt_path path for the trained model.')
    parser.add_argument('--model_name', default='densenet', type=str, help='the model architecture defined in model_zoo.py.')

    parser.add_argument('--attention_type', default='None', type=str,
                        help='the attention type defined in model_zoo.py.')

    parser.add_argument('--batch_size', default=1, type=int, help='number of samples for dataloader.')
    parser.add_argument('--num_workers', default=32, type=int, help='num_workers for prefetch image data')
    parser.add_argument('--img_size', default=224, type=int, help='image size')
    parser.add_argument('--img_depth', default=96, type=int, help='image depth')
    parser.add_argument('--img_mode', default=3, type=int, help='input image mode')
    parser.add_argument('--eps', default=1e-6, type=float, help='to avoid dividing 0')

    FLAGS = parser.parse_args()
    print(FLAGS)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    result_df = pd.read_csv(FLAGS.guide_csv)
    ckpt = torch.load(FLAGS.ckpt_path)['model']
    pred_reg = main(FLAGS, ckpt)
    result_df['reg'] = pred_reg
    result_df.to_csv(os.path.join(FLAGS.save_path, 'reg_result.csv'), index=False)