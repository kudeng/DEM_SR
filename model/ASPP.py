from functools import partial
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

BATCH_NORM_MOMENTUM = 0.005
ENABLE_BIAS = True
activation_fn = nn.ELU()

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class AtrousBlock(nn.Module):
    def __init__(self, input_filters, filters, dilation, apply_initial_bn=True):
        super(AtrousBlock, self).__init__()

        self.initial_bn = nn.BatchNorm2d(input_filters, BATCH_NORM_MOMENTUM)
        self.apply_initial_bn = apply_initial_bn

        self.conv1 = nn.Conv2d(input_filters, filters*2, 1, 1, 0, bias=False)
        self.norm1 = nn.BatchNorm2d(filters*2, BATCH_NORM_MOMENTUM)

        self.atrous_conv = nn.Conv2d(filters*2, filters, 3, 1, dilation, dilation, bias=False)
        self.norm2 = nn.BatchNorm2d(filters, BATCH_NORM_MOMENTUM)

    def forward(self, input):
        if self.apply_initial_bn:
            input = self.initial_bn(input)

        input = self.conv1(input.relu())
        input = self.norm1(input)
        input = self.atrous_conv(input.relu())
        input = self.norm2(input)
        return input


class ASSPBlock(nn.Module):
    def __init__(self, input_filters=256, cat_filters=448, atrous_filters=128):
        super(ASSPBlock, self).__init__()

        self.atrous_conv_r3 = AtrousBlock(input_filters, atrous_filters, 3, apply_initial_bn=False)
        self.atrous_conv_r6 = AtrousBlock(cat_filters + atrous_filters, atrous_filters, 6)
        self.atrous_conv_r12 = AtrousBlock(cat_filters + atrous_filters*2, atrous_filters, 12)
        self.atrous_conv_r18 = AtrousBlock(cat_filters + atrous_filters*3, atrous_filters, 18)
        self.atrous_conv_r24 = AtrousBlock(cat_filters + atrous_filters*4, atrous_filters, 24)

        self.conv = nn.Conv2d(5 * atrous_filters + cat_filters, atrous_filters, 3, 1, 1, bias=ENABLE_BIAS)

    def forward(self, input):
        input, cat = input
        layer1_out = self.atrous_conv_r3(input)
        concat1 = torch.cat((cat, layer1_out), 1)

        layer2_out = self.atrous_conv_r6(concat1)
        concat2 = torch.cat((concat1, layer2_out), 1)

        layer3_out = self.atrous_conv_r12(concat2)
        concat3 = torch.cat((concat2, layer3_out), 1)

        layer4_out = self.atrous_conv_r18(concat3)
        concat4 = torch.cat((concat3, layer4_out), 1)

        layer5_out = self.atrous_conv_r24(concat4)
        concat5 = torch.cat((concat4, layer5_out), 1)

        features = activation_fn(self.conv(concat5))
        return features