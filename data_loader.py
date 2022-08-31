# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 10:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import os
import cv2
import math
import random
import numpy as np
from PIL import Image, ImageDraw

from torch.utils.data.dataset import Dataset

OUTPUT_ROOT_DIR_NAMES = [
    'masked_frames',
    'result_frames',
    'optical_flows'
]


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
    video_len, imageWidth=320, imageHeight=180, nStroke=5,
    nVertexBound=[10, 30], maxHeadSpeed=15, maxHeadAcceleration=(15, 0.5),
    brushWidthBound=(5, 20), boarderGap=None, nMovePointRatio=0.5, maxPointMove=10,
    maxLineAcceleration=5, maxInitSpeed=5
):
    '''
    Get video masks by random strokes which move randomly between each
    frame, including the whole stroke and its control points

    Parameters
    ----------
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawed lines
        nVertexBound: Lower/upper bound of number of control points for each line
        maxHeadSpeed: Max head speed when creating control points
        maxHeadAcceleration: Max acceleration applying on the current head point (
            a head point and its velosity decides the next point)
        brushWidthBound (min, max): Bound of width for each stroke
        boarderGap: The minimum gap between image boarder and drawed lines
        nMovePointRatio: The ratio of control points to move for next frames
        maxPointMove: The magnitude of movement for control points for next frames
        maxLineAcceleration: The magnitude of acceleration for the whole line

    Examples
    ----------
        object_like_setting = {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (15, 3.14),
            "brushWidthBound": (30, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 10,
        }
        rand_curve_setting = {
            "nVertexBound": [10, 30],
            "maxHeadSpeed": 20,
            "maxHeadAcceleration": (15, 0.5),
            "brushWidthBound": (3, 10),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 3,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 6
        }
        get_video_masks_by_moving_random_stroke(video_len=5, nStroke=3, **object_like_setting)
    '''
    assert(video_len >= 1)

    # Initialize a set of control points to draw the first mask
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


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration

    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return (speed, angle)


def random_move_control_points(Xs, Ys, lineVelocity, nMovePointRatio, maxPiontMove, maxLineAcceleration, boarderGap=15):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()

    # move the whole line and accelerate
    speed, angle = lineVelocity
    new_Xs += int(speed * np.cos(angle))
    new_Ys += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity, maxLineAcceleration, dist='guassian')

    # choose points to move
    chosen = np.arange(len(Xs))
    np.random.shuffle(chosen)
    chosen = chosen[:int(len(Xs) * nMovePointRatio)]
    for i in chosen:
        new_Xs[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        new_Ys[i] += np.random.randint(-maxPiontMove, maxPiontMove)
    return new_Xs, new_Ys


def get_random_stroke_control_points(
    imageWidth, imageHeight,
    nVertexBound=(10, 30), maxHeadSpeed=10, maxHeadAcceleration=(5, 0.5), boarderGap=20,
    maxInitSpeed=10
):
    '''
    Implementation the free-form training masks generating algorithm
    proposed by JIAHUI YU et al. in "Free-Form Image Inpainting with Gated Convolution"
    '''
    startX = np.random.randint(imageWidth)
    startY = np.random.randint(imageHeight)
    Xs = [startX]
    Ys = [startY]

    numVertex = np.random.randint(nVertexBound[0], nVertexBound[1])

    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0, maxHeadSpeed)

    for i in range(numVertex):
        speed, angle = random_accelerate((speed, angle), maxHeadAcceleration)
        speed = np.clip(speed, 0, maxHeadSpeed)

        nextX = startX + speed * np.cos(angle)
        nextY = startY + speed * np.sin(angle)

        if boarderGap is not None:
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


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):
    radius = brushWidth // 2 - 1
    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i - 1], Ys[i - 1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)
    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return mask


# modified from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/generate_data.py
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


# Dataset Module
def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_dir_under_root(root_dir, name):
    full_dir_name = os.path.join(root_dir, name)
    make_dirs(full_dir_name)
    return full_dir_name


def read_dirnames_under_root(root_dir, skip_list=None):
    if skip_list is None:
        skip_list = []
    dirnames = [
        name for i, name in enumerate(sorted(os.listdir(root_dir)))
        if (os.path.isdir(os.path.join(root_dir, name))
            and name not in skip_list
            and i not in skip_list)
    ]
    print(f"Reading directories under {root_dir}, exclude {skip_list}, num: {len(dirnames)}")
    return dirnames


class RootInputDirectories:

    def __init__(
            self,
            root_videos_dir,
            root_masks_dir,
            video_names_filename=None
    ):
        self.root_videos_dir = root_videos_dir
        self.root_masks_dir = root_masks_dir
        if video_names_filename is not None:
            with open(video_names_filename, 'r') as fin:
                self.video_dirnames = [
                    os.path.join(root_videos_dir, line.split()[0])
                    for line in fin.readlines()
                ]

        else:
            self.video_dirnames = read_dirnames_under_root(root_videos_dir)
        self.mask_dirnames = read_dirnames_under_root(root_masks_dir)

    def __len__(self):
        return len(self.video_dirnames)


class RootOutputDirectories:

    def __init__(
            self, root_outputs_dir,
    ):
        self.output_root_dirs = {}
        for name in OUTPUT_ROOT_DIR_NAMES:
            self.output_root_dirs[name] = \
                make_dir_under_root(root_outputs_dir, name)

    def __getattr__(self, attr):
        if attr in self.output_root_dirs:
            return self.output_root_dirs[attr]
        else:
            raise KeyError(
                f"{attr} not in root_dir_names {self.output_root_dirs}")


# Get Dataset
class FrameAndMaskDataset(Dataset):
    def __init__(self,
                 rids: RootInputDirectories,
                 rods: RootOutputDirectories,
                 args: dict):
        self.rids = rids
        self.video_dirnames = rids.video_dirnames
        self.mask_dirnames = rids.mask_dirnames
        self.rods = rods

        self.sample_length = args['sample_length']
        self.random_sample = args['random_sample']
        self.random_sample_mask = args['random_sample_mask']
        self.random_sample_period_max = args.get('random_sample_period_max', 1)

        self.guidance = args.get('guidance', 'none')
        self.sigma = args.get('edge_sigma', 2)

        self.size = self.w, self.h = args['w'], args['h']
        self.mask_type = args['mask_type']

        self.do_augment = args.get('do_augment', False)
        self.skip_last = args.get('skip_last', False)

        self.mask_dilation = args.get('mask_dilation', 0)


# 先进行数据划分
# 先确定数据划分的原则
# 首先对一张1024*1023的图像进行插值，使其变为1024*1024，随机按照时间抽取12张图像进行抽取，每一轮抽取24组
# 思考两种策略×两种策略：①全部受损；②50%受损—————①随机图；②随机线 我们只需要256×256的mask即可

object_like_setting = {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (15, 3.14),
            "brushWidthBound": (30, 50),
            "nMovePointRatio": 0.5,
            "maxPointMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 10,
        }

rand_curve_setting = {
    "nVertexBound": [10, 30],
    "maxHeadSpeed": 20,
    "maxHeadAcceleration": (15, 0.5),
    "brushWidthBound": (3, 10),
    "nMovePointRatio": 0.5,
    "maxPointMove": 3,
    "maxLineAcceleration": (5, 0.5),
    "boarderGap": 20,
    "maxInitSpeed": 6
}


def get_mask_schedule_All():
    for j in os.listdir(r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_train\train'):
        os.makedirs(
            r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_train\train/' + j + r'/choice_images_mask_ALL')
        for i in range(24):
            save_path = r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_train\train/' + j + r'/choice_images_mask_ALL\{}/'.format(i)
            os.makedirs(save_path)
            choice_path = r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_train\train/' + j + r'\choice_images/' + str(i)
            choice_file_list = os.listdir(choice_path)
            seed = np.random.uniform(0, 1, 1)
            nStroke_seed = np.random.randint(1, 5)
            for m in range(12):
                if seed > 0.5:
                    mask = get_video_masks_by_moving_random_stroke(video_len=1, imageWidth=256, imageHeight=256, nStroke=nStroke_seed, **object_like_setting)
                else:
                    mask = get_video_masks_by_moving_random_stroke(video_len=1, nStroke=nStroke_seed, **rand_curve_setting)
                mask[0].save(save_path + choice_file_list[m][:-4] + '.png')


# get_mask_schedule_All()

for j in os.listdir(r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_test_public\test_public'):
    path = r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_test_public\test_public/' + j + r'\images_masked/'
    file_list = os.listdir(path)
    os.makedirs(
        r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_test_public\test_public/' + j + r'/choice_images')
    for i in range(24):
        save_path = r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_test_public\test_public/' + j + r'/choice_images\{}/'.format(i)
        os.makedirs(save_path)

        choice_from_list = np.random.choice(file_list, 12, replace=False)
        choice_from_list = list(choice_from_list)
        choice_from_list.sort(key=file_list.index)
        for file in choice_from_list:
            img = cv2.imread(path + file)
            img = cv2.resize(img, (1024, 1024))
            cv2.imwrite(save_path + file, img)

