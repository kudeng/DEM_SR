import torch.nn.functional as F
from model.unet import Unet
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
        self.attention = CBAM(channels * 2, channels * 2)

    def forward(self, de_feature, sr_feature):
        x = self.ha_head(de_feature)
        blured = self.ha_blur(x)
        acted = self.act(x - blured)
        high_freq_feature = x + acted * x
        concated = torch.cat([high_freq_feature, sr_feature], 1)
        out = self.attention(concated)
        return out


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


class DemSrNet(nn.Module):
    def __init__(self, in_channels=1, img_channels=1, n_resblocks=16, n_feats=128,
                 scale=4, conv=default_conv, res_scale=1):
        super(DemSrNet, self).__init__()
        n_colors = img_channels
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale = scale

        # define DE branch (using u-net)
        self.de_net = Unet(num_classes=n_feats, backbone='resnet50', pretrained=False)

        # define SR branch (using edsr)
        m_head = [conv(n_colors, n_feats, kernel_size)]
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
        ]
        self.sr_head = nn.Sequential(*m_head)
        self.sr_body = nn.Sequential(*m_body)
        self.sr_up = nn.Sequential(*m_tail)

        # Guidance and fusion
        self.guidance = HA(n_feats, scale=1)
        self.sr_tail = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats * 2, n_feats * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats * 2, n_feats * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.de_out = nn.Conv2d(n_feats, in_channels, 3, 1, 1)
        self.sr_out = nn.Conv2d(n_feats * 2, in_channels, 3, 1, 1)

    def forward(self, img, lr):
        # SR encoder
        lr = self.sr_head(lr)
        lr_res = self.sr_body(lr)
        lr_res += lr

        # DE net
        de = self.de_net(img)
        de = de + nn.functional.interpolate(lr_res, scale_factor=self.scale, mode='bilinear')

        # guidance and fusion
        sr = self.sr_up(lr_res)
        concated = self.guidance(de, sr)
        concated = concated + self.sr_tail(concated)

        de_out = self.de_out(de)
        sr_out = self.sr_out(concated)
        return de_out, sr_out

