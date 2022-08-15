# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 10:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import cv2
import math
import random
import numpy as np
from PIL import Image, ImageDraw


# 生成条纹噪音
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


# 生成光斑
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


# 生成遮挡掩膜
# From Free-Form-Video-Inpainting
def get_video_masks_by_moving_random_stroke(
        video_len, imageWidth=320, imageHeight=180, nStroke=5, nVertexBound=[10, 30], maxHeadSpeed=15,
        maxHeadAcceleration=(15, 0.5), brushWidthBound=(5, 20), boarderGap=None, nMovePointRatio=0.5, maxPointMove=10,
        maxLineAcceleration=5, maxInitSpeed=5):
    """
        Get video masks by random strokes which move randomly between each
        frame, including the whole stroke and its control points

        Parameters
        ----------
        video_len : Video Length
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawn lines
        nVertexBound: Lower/upper bound of number of control points for each line
        maxHeadSpeed: Max head speed when creating control points
        maxHeadAcceleration: Max acceleration applying on the current head point (a head point and its velocity decides the next point)
        brushWidthBound (min, max): Bound of width for each stroke
        boarderGap: The minimum gap between image boarder and drawn lines
        nMovePointRatio: The ratio of control points to move for next frames
        maxPointMove: The magnitude of movement for control points for next frames
        maxLineAcceleration: The magnitude of acceleration for the whole l
    """
    assert (video_len >= 1)
    # Initialize a set of control points to draw the first mask
    # create a new canvas
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
    control_points_set = []
    for i in range(nStroke):
        brushWidth = np.random.randint(brushWidthBound[0], brushWidthBound[1])
        Xs, Ys, velocity = get_random_stroke_control_points(
            imageWidth=imageWidth, imageHeight=imageHeight,
            nVertexBound=nVertexBound, maxHeadSpeed=maxHeadSpeed,
            maxHeadAcceleration=maxHeadAcceleration, boarderGap=boarderGap,
            maxInitSpeed=maxInitSpeed
        )
        control_points_set.append((Xs, Ys, velocity, brushWidth))
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)

    # Generate the following masks by randomly move strokes and their control points
    masks = [mask]
    for i in range(video_len - 1):
        mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
        for j in range(len(control_points_set)):
            Xs, Ys, velocity, brushWidth = control_points_set[j]
            new_Xs, new_Ys = random_move_control_points(
                Xs, Ys, velocity, nMovePointRatio, maxPointMove,
                maxLineAcceleration, boarderGap
            )
            control_points_set[j] = (new_Xs, new_Ys, velocity, brushWidth)
        for Xs, Ys, velocity, brushWidth in control_points_set:
            draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)
        masks.append(mask)

    return masks


def get_random_stroke_control_points(imageWidth, imageHeight, nVertexBound=(10, 10), maxHeadSpeed=10, maxHeadAcceleration=(5, 0.5),
                                     boarderGap=20, maxInitSpeed=10):
    """
    Implementation the free-form training masks generating algorithm
    :param imageWidth:
    :param imageHeight:
    :param nVertexBound:
    :param maxHeadSpeed:
    :param maxHeadAcceleration:
    :param boarderGap:
    :param maxInitSpeed:
    :return:
    """
    startX = np.random.randint(imageWidth)
    startY = np.random.randint(imageHeight)
    Xs = [startX]
    Ys = [startY]

    numVertex = np.random.randint(nVertexBound[0], nVertexBound[1])

    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0, maxHeadSpeed)

    for i in range(numVertex):
        # 设置Acceleration加速度
        speed, angle = random_accelerate((speed, angle), maxHeadAcceleration)
        speed = np.clip(speed, 0, maxHeadSpeed)

        nextX = startX + speed * np.cos(angle)
        nextY = startY + speed * np.sin(angle)

        # 保证mask在一定的边界内
        if boarderGap is None:
            nextX = np.clip(nextX, boarderGap, imageWidth - boarderGap)
            nextY = np.clip(nextY, boarderGap, imageHeight - boarderGap)

        startX, startY = nextX, nextY
        Xs.append(nextX)
        Ys.append(nextY)

    velocity = get_random_velocity(maxInitSpeed, dist='guassian')

    return np.array(Xs), np.array(Ys), velocity


def get_random_velocity(max_speed, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    angle = np.random.uniform(0, 2 * np.pi)

    return speed, angle


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration

    if dist == 'uniform':
        speed += np.random.randint(-d_speed, d_speed)
        angle += np.random.randint(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(-d_speed, d_speed)
        angle += np.random.normal(-d_angle, d_angle)
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return speed, angle


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):

    radius = brushWidth // 2 + 1

    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i-1], Ys[i-1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)

    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    return mask


def random_move_control_points(Xs, Ys, lineVelocity, nMovePointRatio, maxPiontMove, maxLineAcceleration, boarderGap=15):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()

    # move the whole line and accelerate
    speed, angle = lineVelocity
    new_Xs += int(speed * np.cos(angle))
    new_Ys += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity, maxLineAcceleration, dist='guassian')

    chosen = np.arange(len(Xs))
    np.random.shuffle(chosen)
    for i in chosen:
        new_Xs[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        new_Ys[i] += np.random.randint(-maxPiontMove, maxPiontMove)
    return new_Xs, new_Ys


def get_random_walk_mask(imageWidth=320, imageHeight=180, length=None):
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    canvas = np.zeros((imageHeight, imageWidth)).astype("i")
    if length is None:
        length = imageWidth * imageHeight
    x = random.randint(0, imageHeight - 1)
    y = random.randint(0, imageWidth - 1)
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=imageHeight - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=imageWidth - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 1
    return Image.fromarray(canvas * 255).convert('1')


def get_masked_ratio(mask):
    """
    Calculate the masked ratio.
    mask: Expected a binary PIL image, where 0 and 1 represent
          masked(invalid) and valid pixel values.
    """
    hist = mask.histogram()
    return hist[0] / np.prod(mask.size)
