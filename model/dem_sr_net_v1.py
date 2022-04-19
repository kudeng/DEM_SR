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
    def __init__(self, in_channels, out_channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
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
        self.attention = CBAM(160, 128)

    def forward(self, de_feature, sr_feature):
        x = self.ha_head(de_feature)
        blured = self.ha_blur(x)
        acted = self.act(x - blured)
        high_freq_feature = x + acted * x
        concated = torch.cat([high_freq_feature, sr_feature], 1)
        out = self.attention(concated)
        return out



import torch.nn as nn

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)
        self.attention = CBAM(in_size, out_size)

    def forward(self, inputs1, inputs2, depth):
        if depth is None:
            outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        else:
            outputs = torch.cat([inputs1, self.up(inputs2), depth], 1)
        outputs = self.attention(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


from model.vgg import VGG16
from model.ASPP import ASPP
from math import log2


class DemSrNetV1(nn.Module):
    def __init__(self, in_channels=1, img_channels=1, n_resblocks=4, n_feats=128,
                 scale=4, conv=default_conv, pretrained = False):
        super(DemSrNetV1, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale = scale

        self.vgg = VGG16(pretrained=pretrained, in_channels=img_channels)
        in_filters = [128, 256, 512, 512]
        out_filters = [32, 64, 128, 256]
        self.aspp = ASPP(in_channels=256, atrous_rates=[2, 4, 8], out_channels=256)
        self.after_aspp = conv(256, 256, 3)
        # upsampling
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])



        # define head module
        self.sr_head = conv(in_channels, n_feats, kernel_size)

        # define body module
        for i in range(1, 5):
            sr_blocks = [ResBlock(conv, n_feats, kernel_size) for _ in range(n_resblocks)]
            sr_body = nn.Sequential(*sr_blocks)
            setattr(self, f'sr_body_{i}', sr_body)

        self.sr_body_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        assert scale & (scale - 1) == 0, "scale doesn't equal to 2^n has not realised."
        for i in range(1, int(math.log2(scale)) + 1):
            sr_tail = nn.Sequential(
                conv(n_feats, 4 * n_feats, 3),
                nn.PixelShuffle(2)
            )
            setattr(self, f'sr_tail_{i}', sr_tail)

        self.sr_out = conv(n_feats, in_channels, kernel_size)
        self.de_out = nn.Conv2d(out_filters[0], 1, 1)

        up_factor = [scale >> i for i in range(2, -1, -1)]
        for i in range(1, 4):
            setattr(self, f'refinement_{i}',
                    nn.Sequential(
                        nn.Upsample(scale_factor=up_factor[i-1]),
                        nn.Conv2d(n_feats, out_filters[3-i], 1, 1, 0)
                    ))

        for i in range(1, int(math.log2(scale)) + 1):
            setattr(self, f'guidance_{i}', HA(32, scale=(scale >> i)))

    def forward(self, img, lr):
        # DE encoder
        [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(img)
        aspp = self.aspp(feat5)
        feat5 = feat5 + self.after_aspp(aspp)

        # SR encoder
        sr_fea = self.sr_head(lr)
        sr_1 = self.sr_body_1(sr_fea)
        sr_2 = self.sr_body_2(sr_1)
        sr_3 = self.sr_body_3(sr_2)
        sr_4 = self.sr_body_4(sr_3)
        sr_fea = sr_fea + self.sr_body_last(sr_4)

        # DE decoder + Refinement
        de_up4 = self.up_concat4(feat4, feat5, None)
        de_up3 = self.up_concat3(feat3, de_up4, self.refinement_1(sr_fea))
        de_up2 = self.up_concat2(feat2, de_up3, self.refinement_2(sr_fea))
        de_up1 = self.up_concat1(feat1, de_up2, self.refinement_3(sr_fea))


        # multi-scale output
        sr_up_1 = self.sr_tail_1(sr_fea)
        last_up = sr_up_1
        init_num = 2
        for i in range(int(log2(self.scale)) - 1):
            last_guidance = getattr(self, f'guidance_{init_num + i - 1}')
            last_sr_tail = getattr(self, f'sr_tail_{init_num + i}')
            sr_out = last_sr_tail(last_guidance(de_up1, last_up))
            last_up = sr_out
        last_guidance = getattr(self, f'guidance_{int(log2(self.scale))}')
        sr_out = self.sr_out(last_guidance(de_up1, last_up))
        de_out = self.de_out(de_up1)

        return de_out, sr_out

