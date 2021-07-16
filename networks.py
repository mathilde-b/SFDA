#!/usr/bin/env python3.6

import math
from typing import Iterable, Any

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch import Tensor

from utils import compose_acc
from layers import upSampleConv, conv_block_1, conv_block_3_3, conv_block_Asym
from layers import conv_block, conv_block_3, maxpool, conv_decod_block
from layers import convBatch, residualConv  # Imports for UNEt



class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        # self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        # self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x)
        return x


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Dummy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.down = nn.Conv2d(in_dim, 10, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(10, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, input: Tensor) -> Tensor:
        return self.up(self.down(input))


Dimwit = Dummy


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super().__init__()

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
        # Secondary branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        maxpool_output, indices = self.maxpool0(input)

        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        # Zero padding the feature maps from the main branch
        depth_to_pad = abs(maxpool_output.shape[1] - do.shape[1])
        padding = torch.zeros(maxpool_output.shape[0], depth_to_pad, maxpool_output.shape[2],
                              maxpool_output.shape[3], device=maxpool_output.device)
        maxpool_output_pad = torch.cat((maxpool_output, padding), 1)
        output = maxpool_output_pad + do

        # _, c, _, _ = maxpool_output.shape
        # output = do
        # output[:, :c, :, :] += maxpool_output

        final_output = self.PReLU3(output)

        return final_output, indices


class BottleNeckDownSamplingDilatedConv(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConv, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = input + do
        output = self.PReLU3(output)

        return output


class BottleNeckDownSamplingDilatedConvLast(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConvLast, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):

        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = self.conv_out(input) + do
        output = self.PReLU3(output)

        return output


class BottleNeckNormal(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_3_3(int(in_dim / projectionFactor), int(in_dim / projectionFactor))
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckNormal_Asym(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal_Asym, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_Asym(int(in_dim / projectionFactor), int(in_dim / projectionFactor), 5)
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super(BottleNeckUpSampling, self).__init__()
        # Main branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        return do


class ENet(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.projecting_factor = 4
        self.n_kernels = 16

        # Initial
        self.conv0 = nn.Conv2d(nin, 15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.n_kernels, self.projecting_factor, self.n_kernels * 4)
        self.bottleNeck1_1 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.n_kernels * 4, self.projecting_factor, self.n_kernels * 8)
        self.bottleNeck2_1 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 4)
        self.bottleNeck2_5 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 16)

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 4)
        self.bottleNeck3_5 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.n_kernels * 8, self.projecting_factor,
                                                                   self.n_kernels * 4, 16)

        # ### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)

        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.n_kernels * 8, self.projecting_factor,
                                                      self.n_kernels * 4)
        self.PReLU_Up_1 = nn.PReLU()

        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor,
                                                  0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels, self.projecting_factor, 0.1)

        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.n_kernels * 2, self.projecting_factor, self.n_kernels)
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.n_kernels, self.n_kernels, self.projecting_factor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()

        # Unpooling Last
        self.deconv3 = upSampleConv(self.n_kernels, self.n_kernels)

        self.out_025 = nn.Conv2d(self.n_kernels * 8, nout, kernel_size=3, stride=1, padding=1)
        self.out_05 = nn.Conv2d(self.n_kernels, nout, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(self.n_kernels, nout, kernel_size=1)

    def forward(self, input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0, indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)

        # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)

        # #### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8, indices_2)

        # bn_up_1_0 = self.bottleNeck_Up_1_0(unpool_0) # Not concatenate
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0, bn1_4), dim=1))  # concatenate

        up_block_1 = self.PReLU_Up_1(unpool_0 + bn_up_1_0)

        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)

        #  Second block #
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        # bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1) # Not concatenate
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1, outputInitial), dim=1))  # concatenate

        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1 + bn_up_2_2)

        unpool_12 = self.deconv3(up_block_1)

        return self.final(unpool_12)


class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class ResidualUNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32):
        super().__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        # Encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # Decoder
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_decod_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # Params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        print(f"Initialized {self.__class__.__name__} succesfully")

    def forward(self, input):
        # Encoding path

        down_1 = self.down_1(input)  # This will go as res in deconv path
        down_2 = self.down_2(self.pool_1(down_1))
        down_3 = self.down_3(self.pool_2(down_2))
        down_4 = self.down_4(self.pool_3(down_3))

        bridge = self.bridge(self.pool_4(down_4))

        # Decoding path
        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2  # Residual connection
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2  # Residual connection
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2  # Residual connection
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2  # Residual connection
        up_4 = self.up_4(skip_4)

        return self.out(up_4)


class UNet(nn.Module):
    #def __init__(self, nin, nout, nG=64):
    def __init__(self, nin, nout, nG=32): # m changed the default from 64 to 32
        super().__init__()

        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        bridge = self.bridge(x2)

        y0 = self.deconv1(bridge)
        y1 = self.deconv2(self.conv5(torch.cat((y0, x2), dim=1)))
        y2 = self.deconv3(self.conv6(torch.cat((y1, x1), dim=1)))
        y3 = self.conv7(torch.cat((y2, x0), dim=1))

        return self.final(y3)


class fcn8s(nn.Module):
    def __init__(self, nin, nout, learned_billinear=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = nout

        self.conv1 = nn.Sequential(
            nn.Conv2d(nin, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)

        self.forward_path: Iterable[Any] = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.classifier]

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

    def apply(self, _) -> None:
        print("Override default apply fn, call its own weight init instead")
        vgg16 = models.vgg16(pretrained=True)
        self.init_vgg16_params(vgg16)

    def forward(self, x):
        _, _, _, conv3, conv4, _, score = compose_acc(self.forward_path, x)

        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.interpolate(score, size=score_pool4.size()[2:], mode="bilinear", align_corners=True)
        score += score_pool4
        score = F.interpolate(score, size=score_pool3.size()[2:], mode="bilinear", align_corners=True)
        score += score_pool3
        out = F.interpolate(score, size=x.size()[2:], mode="bilinear", align_corners=True)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv1,
                  self.conv2,
                  self.conv3,
                  self.conv4,
                  self.conv5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


# Densenet Things
class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class DenseNet(nn.Module):
    def __init__(self, depth=22, block=Bottleneck,
                 dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        # planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


# ResNext things
class BottleneckC(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super().__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.dropout(out, p=0.5, training=self.training)
        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super().__init__()
        block = BottleneckC

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, 1)
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext101(baseWidth, cardinality, n_class=2):
    """
    Construct ResNeXt-101.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 23, 3], num_classes=n_class)
    return model
