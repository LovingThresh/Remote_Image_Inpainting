# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 15:05
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Remote.py
# @Software: PyCharm
import os
import cv2
import numpy as np

# 抽取十二张图像
files_path = r'O:\Project\Remote_Image_Inpainting\Cloud_For_Test\files/'
mask_path = r'O:\Project\Remote_Image_Inpainting\Cloud_For_Test\outputs\attachments/'
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
            if not os.path.exists('O:/Project/Remote_Image_Inpainting/Test/crop_image_cloud/{}_{}'.format(i, j)):
                os.mkdir('O:/Project/Remote_Image_Inpainting/Test/crop_image_cloud/{}_{}'.format(i, j))
            cv2.imwrite('O:/Project/Remote_Image_Inpainting/Test/crop_image_cloud/{}_{}/{}.png'.format(i, j, files[m][:-4]), crop_image)
            if not os.path.exists('O:/Project/Remote_Image_Inpainting/Test/crop_mask_cloud/{}_{}'.format(i, j)):
                os.mkdir('O:/Project/Remote_Image_Inpainting/Test/crop_mask_cloud/{}_{}'.format(i, j))
            cv2.imwrite('O:/Project/Remote_Image_Inpainting/Test/crop_mask_cloud/{}_{}/{}.png'.format(i, j, files[m][:-4]), crop_mask)


cloud_image = np.ones((1024, 1024, 3), dtype=np.uint8)
for i in range(4):
    for j in range(4):
        m = i * 4 + (j + 1)
        crop_image = cv2.imread(
            r'E:\BJM\Video_inpainting\2022-10-11-16-27-22.629233\test_fig\{}_predictions\5.png'.format(m))
        x_0 = 256 * i
        x_1 = 256 * (i + 1)
        y_0 = 256 * j
        y_1 = 256 * (j + 1)
        cloud_image[x_0:x_1, y_0:y_1, :] =  crop_image
cv2.imwrite('image_2.png', cloud_image)


