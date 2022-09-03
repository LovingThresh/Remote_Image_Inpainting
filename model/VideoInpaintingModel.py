# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 14:37
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : VideoInpaintingModel.py
# @Software: PyCharm
# Based on https://github.com/avalonstrel/GatedConvolution_pytorch/
import torch
import logging
import numpy as np
import torch.nn as nn
from Conv_Blocks import (
    GatedConv, GatedDeconv,
    PartialConv, PartialDeconv,
    VanillaConv, VanillaDeconv)


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'partial':
            self.ConvBlock = PartialConv
            self.DeconvBlock = PartialDeconv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv


class CoarseNet(nn.Module):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection: bool = False):
        super().__init__()
        self.conv_type = conv_type
        self.downsample_module = DownSampleModule(
            nc_in, nf, use_bias, norm, conv_by, conv_type)
        self.upsample_module = UpSampleModule(
            nf * 4, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)

    def preprocess(self, masked_imgs, masks, guidances):
        # B, L, C, H, W = masked.shape
        masked_imgs = masked_imgs.transpose(1, 2)
        masks = masks.transpose(1, 2)
        if self.conv_type == 'partial':
            if guidances is not None:
                raise NotImplementedError('Partial convolution does not support guidance')
            # the input and output of partial convolution are both tuple (imgs, mask)
            inp = (masked_imgs, masks)
        elif self.conv_type == 'gated' or self.conv_type == 'vanilla':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances.transpose(1, 2)
            inp = torch.cat([masked_imgs, masks, guidances], dim=1)
        else:
            raise NotImplementedError(f"{self.conv_type} not implemented")

        return inp

    def postprocess(self, masked_imgs, masks, c11):
        if self.conv_type == 'partial':
            inpainted = c11[0].transpose(1, 2) * (1 - masks)
        else:
            inpainted = c11.transpose(1, 2) * (1 - masks)

        out = inpainted + masked_imgs
        return out

    def forward(self, masked_imgs, masks, guidances=None):
        # B, L, C, H, W = masked.shape
        inp = self.preprocess(masked_imgs, masks, guidances)

        encoded_features = self.downsample_module(inp)

        c11 = self.upsample_module(encoded_features)

        out = self.postprocess(masked_imgs, masks, c11)

        return out


class Generator(nn.Module):
    def __init__(
            self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_refine=False,
            use_skip_connection: bool = False,
    ):
        super().__init__()
        self.coarse_net = CoarseNet(
            nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection
        )
        self.use_refine = use_refine
        if self.use_refine:
            self.refine_net = RefineNet(
                nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection
            )

    def forward(self, masked_imgs, masks, guidances=None):
        coarse_outputs = self.coarse_net(masked_imgs, masks, guidances)
        if self.use_refine:
            refined_outputs, offset_flows = self.refine_net(coarse_outputs, masks, guidances)
            return {
                "outputs": refined_outputs,
                "offset_flows": offset_flows,
                "coarse_outputs": coarse_outputs
            }
        else:
            return {"outputs": coarse_outputs}


class UpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type,
                 use_skip_connection=False):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in * 2 if use_skip_connection else nc_in,
            nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            nf * 4 if use_skip_connection else nf * 2,
            nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv10 = self.ConvBlock(
            nf * 1, nf // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(
            nf // 2, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)
        self.use_skip_connection = use_skip_connection

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c4, c2 = inp
        if self.use_skip_connection:
            d1 = self.deconv1(self.concat_feature(c8, c4))
            c9 = self.conv9(d1)
            d2 = self.deconv2(self.concat_feature(c9, c2))
        else:
            d1 = self.deconv1(c8)
            c9 = self.conv9(d1)
            d2 = self.deconv2(c9)

        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11


class DownSampleModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 2, 2))
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 4, 4))
        self.dilated_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 8, 8))
        self.dilated_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 16, 16))
        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)
        a3 = self.dilated_conv3(a2)
        a4 = self.dilated_conv4(a3)

        c7 = self.conv7(a4)
        c8 = self.conv8(c7)
        return c8, c4, c2  # For skip connection


class AttentionDownSampleModule(DownSampleModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(nc_in, nf, use_bias, norm, conv_by, conv_type)


class RefineNet(CoarseNet):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection: bool = False):
        super().__init__(nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.upsample_module = UpSampleModule(
            nf * 16, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.attention_downsample_module = AttentionDownSampleModule(
            nc_in, nf, use_bias, norm, conv_by, conv_type)

    def forward(self, coarse_output, masks, guidances=None):
        inp = self.preprocess(coarse_output, masks, guidances)

        encoded_features = self.downsample_module(inp)

        attention_features, offset_flow = self.attention_downsample_module(inp)

        deconv_inp = torch.cat([encoded_features, attention_features], dim=2)

        c11 = self.upsample_module(deconv_inp)

        out = self.postprocess(coarse_output, masks, c11)
        return out, offset_flow


class SNTemporalPatchGANDiscriminator(BaseModule):
    def __init__(
            self, nc_in, nf=64, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
            conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=None, activation=None,
            conv_by=conv_by
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        # B, L, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 1, 2)
        c1 = self.conv1(xs_t)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        if self.use_sigmoid:
            c6 = torch.sigmoid(c6)
        out = torch.transpose(c6, 1, 2)
        return out


class VideoInpaintingModel_G(BaseModel):
    def __init__(self, opts, nc_in=5, nc_out=3):
        super().__init__()

        nf = opts['nf']
        norm = opts['norm']
        use_bias = opts['bias']

        # warning: if 2d convolution is used in generator, settings (e.g. stride,
        # kernel_size, padding) on the temporal axis will be discarded
        self.conv_by = opts['conv_by'] if 'conv_by' in opts else '3d'
        self.conv_type = opts['conv_type'] if 'conv_type' in opts else 'gated'

        self.use_refine = opts['use_refine'] if 'use_refine' in opts else False
        use_skip_connection = opts.get('use_skip_connection', False)

        self.opts = opts

        ######################
        # Convolution layers #
        ######################
        self.generator = Generator(
            nc_in, nc_out, nf, use_bias, norm, self.conv_by, self.conv_type,
            use_refine=self.use_refine, use_skip_connection=use_skip_connection)

    def forward(self, imgs, masks, guidances=None):
        # imgs: [B, L, C=3, H, W]
        # masks: [B, L, C=1, H, W]
        # guidances: [B, L, C=1, H, W]

        masked_imgs = imgs * masks
        output = self.generator(masked_imgs, masks, guidances)

        return output


class VideoInpaintingModel_T(BaseModel):
    def __init__(self, d_t_args=None):
        super().__init__()

        #################
        # Discriminator #
        #################

        self.temporal_discriminator = SNTemporalPatchGANDiscriminator(nc_in=5, **self.d_t_args)

    def forward(self, imgs, masks, guidances=None):
        # imgs: [B, L, C=3, H, W]
        # masks: [B, L, C=1, H, W]
        # guidances: [B, L, C=1, H, W]

        guidances = torch.full_like(masks, 0.) if guidances is None else guidances
        input_imgs = torch.cat([imgs, masks, guidances], dim=2)
        output = self.temporal_discriminator(input_imgs)

        return output


class VideoInpaintingModel_S(BaseModel):
    def __init__(self, d_s_args=None):
        super().__init__()

        #################
        # Discriminator #
        #################

        self.spatial_discriminator = SNTemporalPatchGANDiscriminator(nc_in=5, conv_type='2d', **self.d_s_args)

    def forward(self, imgs, masks, guidances=None):
        # imgs: [B, L, C=3, H, W]
        # masks: [B, L, C=1, H, W]
        # guidances: [B, L, C=1, H, W]

        guidances = torch.full_like(masks, 0.) if guidances is None else guidances
        input_imgs = torch.cat([imgs, masks, guidances], dim=2)
        # merge temporal dimension to batch dimension
        in_shape = list(input_imgs.shape)
        input_imgs = input_imgs.view([in_shape[0] * in_shape[1]] + in_shape[2:])
        output = self.spatial_discriminator(input_imgs)
        # split batch and temporal dimension
        output = output.view(in_shape[0], in_shape[1], -1)

        return output
