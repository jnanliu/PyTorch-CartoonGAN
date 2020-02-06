# -*- coding : utf-8 -*-
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_tfs = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def get_video_frame(video_path, interval, save_dir) :
    '''
    Capture pictures by frame from video
    :param video_path: path of video
    :param interval: interval of frames
    :param save_dir: path of images
    '''
    cap = cv2.VideoCapture(video_path)
    # total frames
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    index, flag, pbar = 0, 0, tqdm(total=total // interval)
    if not os.path.exists(save_dir) :
        os.mkdir(save_dir)
    while cap.isOpened() :
        # seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        cv2.imwrite(save_dir + "/{}.jpg".format(flag + 681), frame)
        index, flag = index + interval, flag + 1
        pbar.update(1)
        if index > total :
            break
    pbar.close()

def resize_img(dataset_dir, img_h, img_w) :
    '''
    resize images
    :param dataset_dir: dirname of images
    :param img_h: height of images
    :param img_w: width of images
    :return:
    '''
    file_list = os.listdir(dataset_dir)
    for file in tqdm(file_list) :
        c_path = os.path.join(dataset_dir, file)
        img = cv2.imread(c_path)
        img = cv2.resize(img, (img_h, img_w))
        cv2.imwrite(c_path, img)

def crop_img(dataset_dir, crop_size) :
    file_list = os.listdir(dataset_dir)
    for file in tqdm(file_list) :
        c_path = os.path.join(dataset_dir, file)
        img: np.ndarray = cv2.imread(c_path)
        img = img.transpose((2, 0, 1))
        c, h, w = img.shape
        h_crop_nums, w_crop_nums = h // crop_size[0], w // crop_size[1]
        h_border, w_border = (h - h_crop_nums * crop_size[0]) // 2, (w - w_crop_nums * crop_size[1]) // 2
        img_idx = 0
        for i in range(h_crop_nums) :
            for j in range(w_crop_nums) :
                crop_img = img[:, h_border + i * crop_size[0] : h_border + (i + 1) * crop_size[0],
                           w_border + j * crop_size[1] : w_border + (j + 1) * crop_size[1]]
                cv2.imwrite(os.path.join(dataset_dir, "{}-{}.jpg".format(file.split('.')[0], img_idx)),
                                crop_img.transpose(1, 2, 0))
                img_idx += 1

def make_edge_smooth(dataset_dir, save_dir) :
    '''
    modified cartoon images with edges smoothed out
    :param dataset_dir: dirname of cartoon images
    :param save_dir: dirname of smoothed images
    '''
    file_list = os.listdir(dataset_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # gauss kernel
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1
    # refer to https://github.com/znxlwm/pytorch-CartoonGAN/blob/master
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(dataset_dir, f))
        gray_img = cv2.imread(os.path.join(dataset_dir, f), 0)
        rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0).item()):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        result = np.concatenate((rgb_img, gauss_img), axis=1)

        cv2.imwrite(os.path.join(save_dir, f), result)
        n += 1

def load_data(folder_name, batch_size, shuffle=False, drop_last=True) :
    '''
    load dataset
    :param folder_name: dirname of datasets
    :param batch_size:
    :param shuffle:
    :param drop_last:
    '''
    if folder_name != 'test' :
        dataset = ImageFolder(root="./data", transform=train_tfs)
    else :
        dataset = ImageFolder(root="./data", transform=test_tfs)
    select_ind = dataset.class_to_idx[folder_name]
    idx = 0
    # refer to https://github.com/znxlwm/pytorch-CartoonGAN/blob/master
    for i in range(dataset.__len__()) :
        if dataset.imgs[idx][1] != select_ind :
            del dataset.imgs[idx]
            idx -= 1
        idx += 1
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=4)

def print_info(s) :
    flag_len = (80 - len(s)) // 2
    print(">" * flag_len + s + "<" * flag_len)

if __name__ == '__main__' :
    import sys
    try :
        dataset_name = sys.argv[1]
        make_edge_smooth(dataset_dir=os.path.join(BASE_DIR, 'data', dataset_name),
                            save_dir=os.path.join(BASE_DIR, 'data', dataset_name))
    except Exception as e  :
        print('转换失败', e)
