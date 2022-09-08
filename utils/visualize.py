# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 23:09
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : visualize.py
# @Software: PyCharm
import os

import cv2
# import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt


def dim_to_numpy(x: torch.Tensor or np.array):
    if (x.ndim == 4) & (torch.is_tensor(x)):
        x = x.squeeze(0)
        x = x.cpu().numpy()
        x = x.transpose(1, 2, 0)
    elif (x.ndim == 3) & (torch.is_tensor(x)):
        x = x.cpu().numpy()
    return np.uint8(x)


def plot(x: np.array or torch.Tensor, size=(10, 10)):
    x = dim_to_numpy(x)
    assert x.ndim == 3 or (x.ndim == 2)
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(x)
    plt.show()


def visualize_model(model: torch.nn.Module, image, image_pair=False):
    model.eval()
    assert image.ndim == 4
    prediction = model(image)
    if image_pair:
        image, prediction = dim_to_numpy(image), dim_to_numpy(prediction)
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(prediction)
        plt.show()
    else:
        plot(prediction)


def visualize_pair(train_loader, input_size, plot_switch=True, mode='image'):

    a = next(iter(train_loader))

    input_tensor_numpy = a['input_tensors'][0][0:1].numpy()
    input_tensor_numpy = input_tensor_numpy.transpose(0, 2, 3, 1)
    input_tensor_numpy = input_tensor_numpy.reshape(input_size[0], input_size[1], 3)
    input_tensor_numpy = (input_tensor_numpy + 1) / 2
    input_tensor_numpy = np.uint8(input_tensor_numpy * 255)
    if plot_switch:
        plot(input_tensor_numpy)
    if mode == 'image':
        output_tensor_numpy = a['gt_tensors'][0][0:1].numpy()
    else:
        output_tensor_numpy = a['mask_tensors'][0][0:1].numpy()
    output_tensor_numpy = output_tensor_numpy.transpose(0, 2, 3, 1)
    if output_tensor_numpy.shape[-1] != 3:
        output_tensor_numpy = output_tensor_numpy[:, :, :, 0:].repeat(3, axis=-1)
    output_tensor_numpy = output_tensor_numpy.reshape(input_size[0], input_size[1], 3)
    if mode == 'image':
        output_tensor_numpy = (output_tensor_numpy + 1) / 2
    output_tensor_numpy = np.uint8(output_tensor_numpy * 255)
    if plot_switch:
        plot(output_tensor_numpy)

    return input_tensor_numpy, output_tensor_numpy


def visualize_save_pair(val_model: torch.nn.Module, train_loader, save_path, epoch, num=0, mode='image'):

    a = next(iter(train_loader))  # dict: B T C H W
    i = 1
    input_tensor = a['input_tensors'][0].cpu().numpy()
    output_tensor = a['gt_tensors'][0].cpu().numpy()
    mask_tensor_numpy = a['mask_tensors'][0].cpu().numpy()

    input_size = (input_tensor.shape[2], input_tensor.shape[3])
    crop_size  = (output_tensor.shape[2], output_tensor.shape[3])

    input_tensor_numpy_Temporal = input_tensor
    os.mkdir('{}/{}_input'.format(save_path, epoch + num))
    for number, input_tensor_numpy in enumerate(input_tensor_numpy_Temporal):
        input_tensor_numpy = input_tensor_numpy.transpose(1, 2, 0)
        input_tensor_numpy = input_tensor_numpy.reshape(input_size[0], input_size[1], 3)
        input_tensor_numpy = cv2.cvtColor(input_tensor_numpy, cv2.COLOR_BGR2RGB)
        input_tensor_numpy = (input_tensor_numpy + 1) / 2
        cv2.imwrite('{}/{}_input/{}.png'.format(save_path, epoch + num, number), np.uint8(input_tensor_numpy * 255))

    output_tensor_numpy = output_tensor
    output_tensor_numpy_Temporal = output_tensor_numpy.transpose(0, 2, 3, 1)
    os.mkdir('{}/{}_outputs'.format(save_path, epoch + num))
    for number, output_tensor_numpy in enumerate(output_tensor_numpy_Temporal):

        output_tensor_numpy = output_tensor_numpy.reshape(crop_size[0], crop_size[1], 3)
        output_tensor_numpy = cv2.cvtColor(output_tensor_numpy, cv2.COLOR_BGR2RGB)
        output_tensor_numpy = (output_tensor_numpy + 1) / 2
        cv2.imwrite('{}/{}_outputs/{}.png'.format(save_path, epoch + num, number), np.uint8(output_tensor_numpy * 255))

    mask_tensor_numpy_Temporal = mask_tensor_numpy.transpose(0, 2, 3, 1)
    os.mkdir('{}/{}_masks'.format(save_path, epoch + num))
    for number, mask_tensor_numpy in enumerate(mask_tensor_numpy_Temporal):
        mask_tensor_numpy = mask_tensor_numpy.reshape(crop_size[0], crop_size[1])
        mask_tensor_numpy = mask_tensor_numpy
        cv2.imwrite('{}/{}_masks/{}.png'.format(save_path, epoch + num, number), np.uint8(mask_tensor_numpy * 255))

    val_model.train(True)
    os.mkdir('{}/{}_predictions'.format(save_path, epoch + num))
    predict_tensor = val_model(a['input_tensors'][0:].cuda(), a['mask_tensors'][0:].cuda(), a['guidances'][0:].cuda())
    predict_tensor_numpy = predict_tensor['outputs'][0].detach().cpu().numpy()
    predict_tensor_numpy_Temporal = predict_tensor_numpy.transpose(0, 2, 3, 1)
    for number, predict_tensor_numpy in enumerate(predict_tensor_numpy_Temporal):
        if predict_tensor_numpy.shape[-1] == 2:
            predict_tensor_numpy = predict_tensor_numpy[:, :, :, 1:].repeat(3, axis=-1)
        predict_tensor_numpy = predict_tensor_numpy.reshape(crop_size[0], crop_size[1], 3)
        if mode == 'image':
            predict_tensor_numpy = cv2.cvtColor(predict_tensor_numpy, cv2.COLOR_RGB2BGR)
            predict_tensor_numpy = (predict_tensor_numpy + 1) / 2
        cv2.imwrite('{}/{}_predictions/{}.png'.format(save_path, epoch + num, number), np.uint8(predict_tensor_numpy * 255))


def image2tensor(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.transpose(0, 2)
    image = image.transpose(1, 2)

    return image.unsqueeze(0)


def tensor2array(tensor):

    tensor = tensor.squeeze(0)
    tensor = tensor.transpose(0, 2)
    tensor = tensor.transpose(0, 1)
    array = tensor.numpy()

    return array
