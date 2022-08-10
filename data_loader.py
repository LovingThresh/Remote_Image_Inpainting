# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 10:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import cv2
import math
import numpy as np


def get_stripe_noise(image, angle, strength=0.2):
    image_w, image_h, c = image.shape

    beta = 255 * np.random.rand(1)
    g_col = np.random.normal(0, beta, int(image.shape[1] * math.sqrt(2)))
    g_noise = np.tile(g_col, (int((image.shape[0] * math.sqrt(2))), 1))
    g_noise = np.reshape(g_noise, (int(image.shape[0] * math.sqrt(2)), int(image.shape[1] * math.sqrt(2))))
    g_noise = (g_noise - np.min(g_noise)) / (np.max(g_noise) - np.min(g_noise))
    g_noise = g_noise * 255 * strength
    w, h = g_noise.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_noise = cv2.warpAffine(g_noise, M, (w, h))

    crop_noise = rotated_noise[center[0] - int(image_w / 2):center[0] + int(image_w / 2),
                                center[1] - int(image_h / 2):center[1] + int(image_h / 2)]

    crop_noise = np.expand_dims(crop_noise, axis=-1).repeat(3, axis=-1).astype(np.uint8)
    assert crop_noise.shape == (image_w, image_h, c)

    noised_image = image + crop_noise

    return noised_image


def get_light_spot(image, center=(200, 200), radius_scale=0.05):
    IMAGE_WIDTH, IMAGE_HEIGHT, c = image.shape
    assert (center[0] > IMAGE_WIDTH / 10) & (IMAGE_WIDTH - center[0] > IMAGE_WIDTH / 10)
    assert (center[1] > IMAGE_HEIGHT / 10) & (IMAGE_HEIGHT - center[1] > IMAGE_HEIGHT / 10)

    center_x = center[0]
    center_y = center[1]

    R = np.sqrt((IMAGE_WIDTH // 2) ** 2 + (IMAGE_HEIGHT // 2) ** 2) * radius_scale

    noised_image = np.zeros_like(image, dtype=np.uint8)

    # 利用 for 循环 实现
    # 这个有点慢，需要思考怎么提升速度
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            for m in range(c):
                dis = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                if np.exp(-0.5 * dis / R) * 255 + image[i, j, m] > 255:
                    noised_image[i, j, m] = 255
                else:
                    noised_image[i, j, m] = np.exp(-0.5 * dis / R) * 255 + image[i, j, m]

    return noised_image
