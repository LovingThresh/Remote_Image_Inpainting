# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 15:05
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Remote.py
# @Software: PyCharm
import os
import cv2
import numpy as np

# 抽取十一张图像
files_path = r'C:\Users\liuye\Desktop\T\files/'
mask_path = r'C:\Users\liuye\Desktop\T\outputs\attachments/'
files = os.listdir(files_path)

for m in range(12):
    cloud_image = cv2.imread(files_path + files[m])
    mask_image = cv2.imread(mask_path + files[m][:-4] + '.png', cv2.IMREAD_GRAYSCALE)
    for i in range(4):
        for j in range(4):
            x_0 = 256 * i
            x_1 = 256 * (i + 1)
            y_0 = 256 * j
            y_1 = 256 * (j + 1)
            crop_image = cloud_image[x_0:x_1, y_0:y_1, :]
            if mask_image is None:
                print(1)
            crop_mask = mask_image[x_0:x_1, y_0:y_1]
            # cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
            if not os.path.exists('O:/Project/Remote_Image_Inpainting/Test/crop_image/{}_{}'.format(i, j)):
                os.mkdir('O:/Project/Remote_Image_Inpainting/Test/crop_image/{}_{}'.format(i, j))
            cv2.imwrite('O:/Project/Remote_Image_Inpainting/Test/crop_image/{}_{}/{}.png'.format(i, j, files[m][:-4]), crop_image)
            if not os.path.exists('O:/Project/Remote_Image_Inpainting/Test/crop_mask/{}_{}'.format(i, j)):
                os.mkdir('O:/Project/Remote_Image_Inpainting/Test/crop_mask/{}_{}'.format(i, j))
            cv2.imwrite('O:/Project/Remote_Image_Inpainting/Test/crop_mask/{}_{}/{}.png'.format(i, j, files[m][:-4]), crop_mask)


