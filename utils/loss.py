# -*- coding: utf-8 -*-
# @Time    : 2022/9/3 10:17
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : loss.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from model.CannyEdgePytorch import Net as CannyEdgeNet


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


device = torch.device("cuda")
vgg = Vgg16(requires_grad=False).to(device)


class ReconLoss(nn.Module):
    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        if self.masked:
            masks = data_input['masks']
            return self.loss_fn(outputs * (1 - masks), targets * (1 - masks))
        else:
            return self.loss_fn(outputs, targets)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def vgg_loss(self, output, target):
        output_feature = vgg(output)
        target_feature = vgg(target)
        loss = (
                self.l1_loss(output_feature.relu2_2, target_feature.relu2_2)
                + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3)
                + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        )
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        # Note: It can be batch-sized
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(
                self.vgg_loss(outputs[:, frame_idx], targets[:, frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class StyleLoss(nn.Module):
    def __init__(self, original_channel_norm=True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.original_channel_norm = original_channel_norm

    # From https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    # Implement "Image Inpainting for Irregular Holes Using Partial Convolutions", Liu et al., 2018
    def style_loss(self, output, target):
        output_features = vgg(output)
        target_features = vgg(target)
        layers = ['relu2_2', 'relu3_3', 'relu4_3']  # n_channel: 128 (=2 ** 7), 256 (=2 ** 8), 512 (=2 ** 9)
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)  # original design (avoid too small loss)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        # Note: It can be batch-sized
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(
                self.style_loss(outputs[:, frame_idx], targets[:, frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


device = torch.device("cuda")
canny_edge_net = CannyEdgeNet(threshold=2.0, use_cuda=True).to(device)
canny_edge_net.eval()


def get_edge(tensor):
    with torch.no_grad():
        blurred_img, grad_mag, grad_orientation, thin_edges, threshold, early_threshold = \
            canny_edge_net(tensor)
    return threshold


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.current_output_edges = object
        self.current_output_edges = object

    def edge_loss(self, output, target):

        output_edge = get_edge(output)
        gt_edge = get_edge(target)
        loss = self.l1_loss(output_edge, gt_edge)
        return loss, output_edge, gt_edge

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        mean_image_loss = []
        output_edges = []
        target_edges = []
        for batch_idx in range(targets.size(0)):
            edges_o = []
            edges_t = []
            for frame_idx in range(targets.size(1)):
                loss, output_edge, target_edge = self.edge_loss(
                    outputs[batch_idx, frame_idx:frame_idx + 1],
                    targets[batch_idx, frame_idx:frame_idx + 1]
                )
                mean_image_loss.append(loss)
                edges_o.append(output_edge)
                edges_t.append(target_edge)
            output_edges.append(torch.cat(edges_o, dim=0))
            target_edges.append(torch.cat(edges_t, dim=0))

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        self.current_output_edges = output_edges
        self.current_target_edges = target_edges
        return mean_image_loss
