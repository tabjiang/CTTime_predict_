import os
import sys

sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import nibabel
from utils import *
import logging
from random import random
import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO)


class HEDataSet(torch.utils.data.Dataset):
    def __init__(self, basePath, df, mode=0, imgSize=512, img_depth=48, isTrain=True, cropBrain=False):
        """
        params:
        basePath: The path of data set
        df: The data frame of this dataset
        mode: The mode of dataset, if mode == 0, return 3-channel image composed of three different window using image_t-1, image_t and image_t+1
        imgSize: The size of output image
        isTrain: Whether on training mode
        """
        super().__init__()
        self.basePath = basePath
        self.df = df
        assert (mode <= 4)
        self.mode = mode
        self.imgSize = imgSize
        self.isTrain = isTrain
        self.cropBrain = cropBrain
        self.input_D = img_depth
        self.input_H = imgSize
        self.input_W = imgSize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        line = self.df.iloc[index]

        name = line['ct_name']

        label = np.array([line['time_label']])

        try:
            # img = sitk.ReadImage(os.path.join(self.basePath, 'ori_BET_1', name + '.nii.gz'))
            img = sitk.ReadImage(os.path.join(self.basePath, name + '.nii.gz'))
        except:
            # img = sitk.ReadImage(os.path.join(self.basePath, 'ori_BET_1', name + '.nii'))
            img = sitk.ReadImage(os.path.join(self.basePath, name + '.nii'))
        # seg = sitk.ReadImage(os.path.join(self.basePath, 'seg', name + '-seg.nii.gz'))
        oriImg = sitk.GetArrayFromImage(img)

        # segImg = sitk.GetArrayFromImage(seg)
        if int(img.GetMetaData('sform_code')) > 0:
            if float(img.GetMetaData('srow_x').split(' ')[0]) > 0:
                oriImg = np.flip(oriImg, axis=2)
                # segImg = np.flip(segImg, axis=2)
            if float(img.GetMetaData('srow_y').split(' ')[1]) > 0:
                oriImg = np.flip(oriImg, axis=1)
                # segImg = np.flip(segImg, axis=1)
        elif int(img.GetMetaData('qform_code')) > 0:
            quatern_b = float(img.GetMetaData('quatern_b'))
            quatern_c = float(img.GetMetaData('quatern_c'))
            quatern_d = float(img.GetMetaData('quatern_d'))
            R11 = (1 - 2 * quatern_c ** 2 - 2 * quatern_d ** 2) * float(img.GetMetaData('pixdim[1]'))
            R22 = (1 - 2 * quatern_b ** 2 - 2 * quatern_d ** 2) * float(img.GetMetaData('pixdim[2]'))
            if R11 > 0:
                oriImg = np.flip(oriImg, axis=2)
                # segImg = np.flip(segImg, axis=2)
            if R22 > 0:
                oriImg = np.flip(oriImg, axis=1)
                # segImg = np.flip(segImg, axis=1)

        datas = oriImg
        # masks = segImg
        datas = resetWindow(datas, 45, 90)
        # datas = datas.reshape(datas.shape[0], datas.shape[1], datas.shape[2])
        # masks = masks.reshape(masks.shape[0], masks.shape[1], masks.shape[2])
        # datas = np.transpose(datas, (2, 0, 1))
        # masks = np.transpose(masks, (2, 0, 1))

        if self.isTrain:
            # datas, masks = self.__drop_invalid_range__(datas, masks)
            datas = self.__drop_invalid_range__(datas)
            # datas, masks = self.__crop_data__(datas, masks)
            # datas, masks = self.__random_flip__(datas, masks)
            datas = self.__crop_data__(datas)
            datas = self.__random_flip__(datas)
            datas = self.__resize_data__(datas)
            datas = self.__itensity_normalize_one_volume__(datas)
        else:
            datas = self.__drop_invalid_range__(datas)
            datas = self.__resize_data__(datas)
            datas = self.__itensity_normalize_one_volume__(datas)

        datas = datas.reshape((self.input_D, 1, self.imgSize, self.imgSize))
        datas = torch.from_numpy(datas)
        labels = torch.from_numpy(label)
        lenth = np.array([oriImg.shape[0]])
        lenth = torch.from_numpy(lenth)

        return datas.type(torch.float32), labels.type(torch.float32), lenth.type(torch.float32)

    def __random_flip__(self, volume, label=None):
        def flip_axis(img_numpy, axis):
            img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
            img_numpy = img_numpy[::-1, ...]
            img_numpy = img_numpy.swapaxes(0, axis)
            return img_numpy

        if random() < 0.5:
            volume = flip_axis(volume, 2)
            if label is not None:
                label = flip_axis(label, 2)
        if label is not None:
            return volume, label
        else:
            return volume

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __crop_data__(self, data, label=None):
        """
        Random crop with different methods:
        """
        # random center crop
        if label is not None:
            data, label = self.__random_center_crop__(data, label)
            return data, label
        else:
            data = self.__random_center_crop__(data)
            return data

    def __random_center_crop__(self, data, label=None):
        """
        Random crop
        """
        if random() < 0.5:
            [img_d, img_h, img_w] = data.shape
            crop_d_rate = np.random.uniform(low=0.75, high=1.0)
            crop_h_rate = np.random.uniform(low=0.75, high=1.0)
            crop_w_rate = np.random.uniform(low=0.75, high=1.0)
            crop_d = int(crop_d_rate * img_d)
            crop_h = int(crop_h_rate * img_h)
            crop_w = int(crop_w_rate * img_w)
            crop_tl_dmax = img_d - crop_d
            crop_tl_hmax = img_h - crop_h
            crop_tl_wmax = img_w - crop_w
            crop_img_dmin = np.random.randint(low=0, high=crop_tl_dmax)
            crop_img_hmin = np.random.randint(low=0, high=crop_tl_hmax)
            crop_img_wmin = np.random.randint(low=0, high=crop_tl_wmax)
            crop_img_dmax = crop_img_dmin + crop_d
            crop_img_hmax = crop_img_hmin + crop_h
            crop_img_wmax = crop_img_wmin + crop_w
            if label is not None:
                return data[crop_img_dmin: crop_img_dmax, crop_img_hmin: crop_img_hmax,
                       crop_img_wmin: crop_img_wmax], label[crop_img_dmin: crop_img_dmax, crop_img_hmin: crop_img_hmax,
                                                      crop_img_wmin: crop_img_wmax]
            else:
                return data[crop_img_dmin: crop_img_dmax, crop_img_hmin: crop_img_hmax, crop_img_wmin: crop_img_wmax]
        else:
            if label is not None:
                return data, label
            else:
                return data

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out


def load_HEdataset(path, guide, isTrain=True, batch_size=96, num_workers=32, img_mode=3, img_size=512, img_depth=48,
                   n_th=0, k_fold=4):
    # load guide file
    df = pd.read_csv(guide)

    if isTrain:
        ct_list = np.array(df['ct_name'].to_list())
        time_list = np.array(df['time_label'].to_list())
        fold = generate_k_fold(ct_list, k_fold, seed=100)[n_th]
        train_index, valid_index = fold
        ct_list_train = ct_list[train_index]
        ct_list_valid = ct_list[valid_index]
        time_list_train = time_list[train_index]
        time_list_valid = time_list[valid_index]
        dfTrain = pd.DataFrame({'ct_name': ct_list_train, 'time_label': time_list_train})
        dfValid = pd.DataFrame({'ct_name': ct_list_valid, 'time_label': time_list_valid})

        trainSet = HEDataSet(path, dfTrain, img_mode, img_size, img_depth, True)
        valSet = HEDataSet(path, dfValid, img_mode, img_size, img_depth, False)

        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size, shuffle=True, num_workers=num_workers,
                                                  pin_memory=True)
        valLoader = torch.utils.data.DataLoader(valSet, batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=True)

        return trainLoader, valLoader
    else:
        dfTest = df
        testSet = HEDataSet(path, dfTest, img_mode, img_size, img_depth, False)

        testLoader = torch.utils.data.DataLoader(testSet, batch_size, shuffle=False, num_workers=num_workers,
                                                 pin_memory=True)
        return testLoader, dfTest


if __name__ == '__main__':
    train, val = load_HEdataset('', './data_filter/guide_010_train_S.csv')

