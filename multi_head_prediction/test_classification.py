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
        model = model_zoo.generate_model(model_depth=121, n_input_channels=1, num_classes=1, attention_type=FLAGS.attention_type)
    else:
        raise Exception('Undefined model')

    state_dict = {k.replace('module.', ''):v for k, v in ckpt.items()}
    model.load_state_dict(state_dict, True)
    model = model.cuda()
    model.eval()

    lossFunction = nn.BCEWithLogitsLoss()

    runningLoss, runningTP, runningFP, runningTN, runningFN = 0, 0, 0, 0, 0
    predList, pred4ROC, label4ROC = [], [], []

    with torch.no_grad():
        for i, (img, label, lenth) in enumerate(testLoader):
            img, label = img.cuda(), label.cuda()
            b, t, c, h, w = img.shape
            img = torch.transpose(img, 1, 2)

            label = label.view(b, 1)
            pred = model(img)

            predList.append(pred.clone().detach().cpu())
            loss = lossFunction(pred, label)

            runningLoss += loss.item()
            pred4ROC.append(nn.Sigmoid()(pred).view(-1).clone().detach().cpu())
            label4ROC.append(label.view(-1).clone().detach().cpu())
            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            zeros = torch.zeros_like(label)
            TP = torch.sum(torch.where(pred == 1, label, zeros)).item()
            FP = torch.sum(torch.where(pred == 1, 1 - label, zeros)).item()
            TN = torch.sum(torch.where(pred == 0, 1 - label, zeros)).item()
            FN = torch.sum(torch.where(pred == 0, label, zeros)).item()
            runningTP += TP
            runningFP += FP
            runningTN += TN
            runningFN += FN
            accuracy = (TP + TN) / (TP + FP + TN + FN + FLAGS.eps)
            precision = TP / (TP + FP + FLAGS.eps)
            recall = TP / (TP + FN + FLAGS.eps)
            F1 = 2 * precision * recall / (precision + recall + FLAGS.eps)

            logging.info('Instance:[{} / {}], loss:{:.2f}, accuracy:{:.2f}, precision:{:.2f}, recall:{:.2f}, F1:{:.2f}' \
                         .format(i, len(testLoader), loss.item(), accuracy, precision, recall, F1))

    runningLoss /= len(testLoader)
    label4ROC = torch.cat(label4ROC, dim=0)
    pred4ROC = torch.cat(pred4ROC, dim=0)
    accuracy, precision, recall, specificity, F1, PPV, NPV, ROC = eval_metrics_classification(label4ROC, pred4ROC)
    logging.info('Test complete, loss:{:.2f}, accuracy:{:.2f}, precision:{:.2f}, recall:{:.2f}, '
                 'specificity:{:.2f}, F1:{:.2f}, PPV:{:.2f}, NPV:{:.2f}, ROC:{:.2f}' \
                 .format(runningLoss, accuracy, precision, recall, specificity, F1, PPV, NPV, ROC))
    logging.info('Test complete, result save at {}.'.format(FLAGS.save_path))
    return pred4ROC.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='../toy_data', type=str, help='path to the test data folder.')
    parser.add_argument('--guide_csv', default='./doc/toy/test_cls.csv', type=str, help='path to the test.csv which includes dataset splits.')
    parser.add_argument('--save_path', default='./test_results/8h', type=str, help='path to save test results.')
    parser.add_argument('--ckpt_path', default='./model/8h.pth', type=str, help='ckpt_path path for the trained model.')

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
    pred_label = main(FLAGS, ckpt)
    result_df['predict'] = pred_label
    result_df.to_csv(os.path.join(FLAGS.save_path, 'cls_result.csv'), index=False)