# -*- coding: utf-8 -*-
# @Time    : 2023/4/12 15:23
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_process.py
# @Software: PyCharm
# 给定图像文件夹image_dir,掩码文件夹mask_dir
# 按顺序进行合成
import os
import cv2
import numpy as np

image_dir = 'Cloud_For_Test/files2'  # directory where the image is stored
mask_dir = 'Cloud_For_Test/mask2'  # directory where the mask is stored
save_dir = 'Cloud_For_Test/outputs2'  # directory where the image is saved
for image, mask in zip(os.listdir(image_dir), os.listdir(mask_dir)):
    image_name = image
    image_path = os.path.join(image_dir, image)
    mask_path = os.path.join(mask_dir, mask)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # image and mask should have the same size
    # merge image and mask
    # repeat the mask three times to match the image size
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    image = image * np.uint8(mask != 0) + (
        np.uint8(mask == 0) * 255)
    # if save_dir is is not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save image
    cv2.imwrite(os.path.join(save_dir, image_name), image)
