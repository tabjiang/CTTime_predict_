from sklearn.metrics import *
import logging
import os
import re
import math
import random
import numpy as np
import cv2
import torch
from scipy import ndimage
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def calculateROC(y_true, y_pred):
    try:
        ROC = roc_auc_score(y_true, y_pred)
    except:
        ROC = 0
        logging.warning("Label is all 0 or 1 while calculating ROC, set ROC = 0.")
    return ROC


def findXXXInDir(path, results, pattern):
    files = os.listdir(path)
    for f in files:
        if os.path.isdir(os.path.join(path, f)):
            findXXXInDir(os.path.join(path, f), results, pattern)
        else:
            if re.search(pattern, f) is not None:
                results.append(os.path.join(path, f))


def saveCkpt(epoch, modelStateDict, optimizerStateDict, fileName):
    ckpt = dict(
        epoch=epoch,
        model=modelStateDict,
        optimizer=optimizerStateDict
    )
    torch.save(ckpt, fileName)


def loadCkpt(ckpt_path, model, optimizer, strict=True):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['model']
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}

    print('load_dict: ' + ckpt_path)
    print('loading...')
    print(state_dict.keys())

    if sorted(list(state_dict.keys())) == sorted(list(model_dict.keys())):
        print('all params update')
    else:
        print('part of params update')

    model_dict.update(state_dict)
    # model.module.load_state_dict(model_dict, strict=strict)
    model.load_state_dict(model_dict, strict=strict)
    # optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']


def resetWindow(img, windowCenter, windowWidth):
    img_copy = img.copy()
    imgMin = windowCenter - windowWidth // 2
    imgMax = windowCenter + windowWidth // 2
    img_copy[img_copy < imgMin] = imgMin
    img_copy[img_copy > imgMax] = imgMax
    img_copy = (img_copy - imgMin) / (imgMax - imgMin) * 255
    return img_copy.astype(np.uint8)

def generate_k_fold(ct_list, k, seed):
    patient_list = sorted(list(set([i.split('-')[0] for i in ct_list])))
    random.Random(seed).shuffle(patient_list)
    n_fold = len(patient_list) / k
    folds = []
    for i in range(k):
        if i < (k-1):
            valid_patients = patient_list[int(n_fold*i): int(n_fold*(i+1))]
            train_patients = patient_list[:int(n_fold*i)] + patient_list[int(n_fold*(i+1)):]
        else:
            valid_patients = patient_list[int(n_fold*i):]
            train_patients = patient_list[:int(n_fold*i)]
        train_index_list = []
        valid_index_list = []
        for j in range(len(ct_list)):
            ct_name = ct_list[j]
            if ct_name.split('-')[0] in train_patients:
                train_index_list.append(j)
            elif ct_name.split('-')[0] in valid_patients:
                valid_index_list.append(j)
            else:
                raise Exception('Error during kfold split!')
        folds.append([train_index_list, valid_index_list])
    return folds


def generate_weak_fold(weak_ct_list, val_ct_list):
    patient_list = sorted(list(set([i.split('-')[0] for i in val_ct_list])))
    train_index_list = []
    valid_index_list = []
    for i in range(len(weak_ct_list)):
        ct_name = weak_ct_list[i]
        if ct_name.split('-')[0] in patient_list:
            valid_index_list.append(i)
        else:
            train_index_list.append(i)
    return train_index_list, valid_index_list


def eval_metrics_regression(true_label, predict_label):
    mae = mean_absolute_error(true_label, predict_label)
    rmse = mean_squared_error(true_label, predict_label, squared=False)
    total_vaiance = explained_variance_score(true_label, predict_label)
    p = pearsonr(true_label, predict_label)[0]
    return mae, rmse, total_vaiance, p


def eval_metrics_classification(label, pred):
    ROC = roc_auc_score(label, pred)
    fpr, tpr, thresholds = roc_curve(label, pred)
    youden_index = np.argmax(tpr - fpr)
    thresh = thresholds[youden_index]
    pred = (pred >= thresh).type(torch.float32)
    confusion = confusion_matrix(label,  pred, labels=[1,0])
    TP = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]
    TN = confusion[1,1]
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    Spe = TN / (FP + TN + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    PPV = TP / float(TP + FP + 1e-6)
    NPV = TN / float(FN + TN + 1e-6)
    return accuracy, precision, recall, Spe, F1, PPV, NPV, ROC


def get_pred_label(label, pred):
    fpr, tpr, thresholds = roc_curve(label, pred)
    youden_index = np.argmax(tpr - fpr)
    thresh = thresholds[youden_index]
    pred = (pred >= thresh).type(torch.float32)

    return pred


def draw_roc_curve(label, pred, save_path):
    fpr, tpr, thresholds = roc_curve(label, pred)
    auc_value = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
