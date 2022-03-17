import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.downsample = nn.Conv2d(channel, channel // 2, 1, 1, 0)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = self.downsample(x)
        x = x * self.spatialattention(x)
        return x


class HA(nn.Module):
    def __init__(self, channels, scale=1):
        super(HA, self).__init__()
        head = []
        for _ in range(int(math.log2(scale))):
            head.append(ConvBlock(channels, channels))
            head.append(ConvBlock(channels, channels, 4, 2, 1))
        head.append(ConvBlock(channels, channels))
        self.ha_head = nn.Sequential(*head)
        # AvgPool
        self.ha_blur = nn.Sequential(
            nn.AvgPool2d(2),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1)
        )
        # act
        self.act = torch.nn.PReLU()
        self.attention = CBAM(channels * 2)

    def forward(self, de_feature, sr_feature):
        x = self.ha_head(de_feature)
        blured = self.ha_blur(x)
        acted = self.act(x - blured)
        high_freq_feature = x + acted * x
        concated = torch.cat([high_freq_feature, sr_feature], 1)
        out = self.attention(concated)
        return out


class CG(nn.Module):
    def __init__(self, channels, scale=1, conv=default_conv):
        super(CG, self).__init__()
        self.cg_head_de = conv(channels, channels, 3)
        self.cg_head_sr = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Upsample(scale_factor=scale),
            ConvBlock(channels, channels)
        )
        self.cg_head_sr_2 = ConvBlock(channels, channels)

        # Conv + softmax, scored
        self.scored = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Softmax(dim=1)
        )
        self.attention = CBAM(channels * 2)

    def forward(self, de_feature, sr_feature):
        sr_feature = self.cg_head_sr(sr_feature)
        x = self.cg_head_de(de_feature)
        y = self.cg_head_sr_2(sr_feature)
        structure_score = self.scored(x - y)
        sr_feature = sr_feature + y * structure_score
        concated = torch.cat([sr_feature, de_feature], 1)
        out = self.attention(concated)
        return out


import torch.nn as nn


class BridgeNetV3(nn.Module):
    def __init__(self, in_channels=1, img_channels=3, n_resblocks=4, n_feats=128, scale=4, conv=default_conv):
        super(BridgeNetV3, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale = scale

        # define head module
        self.sr_head = conv(in_channels, n_feats, kernel_size)
        self.de_head = conv(img_channels, n_feats, kernel_size)

        # define body module
        for i in range(1, 5):
            sr_blocks = [ResBlock(conv, n_feats, kernel_size) for _ in range(n_resblocks)]
            de_blocks = [ResBlock(conv, n_feats, kernel_size) for _ in range(n_resblocks)]
            sr_body = nn.Sequential(*sr_blocks)
            de_body = nn.Sequential(*de_blocks)
            setattr(self, f'sr_body_{i}', sr_body)
            setattr(self, f'de_body_{i}', de_body)

        self.sr_body_last = conv(n_feats, n_feats, kernel_size)
        self.de_body_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        assert scale & (scale - 1) == 0, "scale doesn't equal to 2^n has not realised."
        for i in range(1, int(math.log2(scale)) + 1):
            de_tail = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size)
            )
            sr_tail = nn.Sequential(
                conv(n_feats, 4 * n_feats, 3),
                nn.PixelShuffle(2)
            )
            setattr(self, f'sr_tail_{i}', sr_tail)
            setattr(self, f'de_tail_{i}', de_tail)

        self.sr_out = conv(n_feats, in_channels, kernel_size)
        self.de_out = conv(n_feats, in_channels, kernel_size)

        for i in range(1, 5):
            setattr(self, f'correction_{i}', CG(n_feats, scale=scale))

        for i in range(1, int(math.log2(scale)) + 1):
            setattr(self, f'guidance_{i}', HA(n_feats, scale=(scale >> i)))

    def forward(self, img, lr):
        de_fea = self.de_head(img)
        sr_fea = self.sr_head(lr)

        sr_1 = self.sr_body_1(sr_fea)
        sr_2 = self.sr_body_2(sr_1)
        sr_3 = self.sr_body_3(sr_2)
        sr_4 = self.sr_body_4(sr_3)
        sr_fea = sr_fea + self.sr_body_last(sr_4)

        de_1 = self.de_body_1(de_fea)
        de_2 = self.de_body_1(self.correction_1(de_1, sr_1))
        de_3 = self.de_body_1(self.correction_2(de_2, sr_2))
        de_4 = self.de_body_1(self.correction_3(de_3, sr_3))
        de_fea = de_fea + self.de_body_last(self.correction_4(de_4, sr_4))

        sr_up_1 = self.sr_tail_1(sr_fea)
        de_up_1 = self.de_tail_1(de_fea)
        sr_up_2 = self.sr_tail_2(self.guidance_1(de_up_1, sr_up_1))
        de_up_2 = self.de_tail_2(de_up_1)

        if self.scale == 4:
            sr_out = self.sr_out(self.guidance_2(de_up_2, sr_up_2))
            de_out = self.de_out(de_up_2)
        else:
            sr_up_3 = self.sr_tail_3(self.guidance_2(de_up_2, sr_up_2))
            de_up_3 = self.de_tail_3(de_up_2)
            sr_out = self.sr_out(self.guidance_3(de_up_3, sr_up_3))
            de_out = self.de_out(de_up_3)
        return de_out, sr_out