"""PraNet (Fan et al., MICCAI 2020) re-implemented from the published architecture description:
Res2Net-50 backbone + RFB-modified blocks + parallel partial decoder + three reverse-attention stages.

Used here as a polyp-specific baseline (R1.7). Differences from the public reference code, stated honestly:
  * the Res2Net-50 backbone is trained FROM SCRATCH (no ImageNet weights), like every other baseline in the paper;
  * in training mode the model returns (final map, reverse-attention map 3, reverse-attention map 4) so that the same
    deep-supervision loss as LiteMedSeg (LightweightLoss, 0.6 / 0.4) can be used; in eval mode only the final map is returned.
Outputs are logits at the input resolution.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=26, scale=4, stype="normal"):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = 1 if scale == 1 else scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False) for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype, self.scale, self.width = stype, scale, width

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if (i == 0 or self.stype == "stage") else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1:
            last = self.pool(spx[self.nums]) if self.stype == "stage" else spx[self.nums]
            out = torch.cat((out, last), 1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Res2Net50(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), base_width=26, scale=4):
        super().__init__()
        self.inplanes = 64
        self.base_width, self.scale = base_width, scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes * Bottle2neck.expansion:
            down = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * Bottle2neck.expansion, 1, bias=False),
                nn.BatchNorm2d(planes * Bottle2neck.expansion))
        layers = [Bottle2neck(self.inplanes, planes, stride, down, self.base_width, self.scale, "stage")]
        self.inplanes = planes * Bottle2neck.expansion
        layers += [Bottle2neck(self.inplanes, planes, 1, None, self.base_width, self.scale) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class RFBModified(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(cin, cout, 1))

        def br(k):
            p = k // 2
            return nn.Sequential(
                BasicConv2d(cin, cout, 1),
                BasicConv2d(cout, cout, (1, k), padding=(0, p)),
                BasicConv2d(cout, cout, (k, 1), padding=(p, 0)),
                BasicConv2d(cout, cout, 3, padding=k, dilation=k))
        self.branch1, self.branch2, self.branch3 = br(3), br(5), br(7)
        self.conv_cat = BasicConv2d(4 * cout, cout, 3, padding=1)
        self.conv_res = BasicConv2d(cin, cout, 1)

    def forward(self, x):
        cat = self.conv_cat(torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1))
        return self.relu(cat + self.conv_res(x))


class Aggregation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cu1 = BasicConv2d(c, c, 3, padding=1)
        self.cu2 = BasicConv2d(c, c, 3, padding=1)
        self.cu3 = BasicConv2d(c, c, 3, padding=1)
        self.cu4 = BasicConv2d(c, c, 3, padding=1)
        self.cu5 = BasicConv2d(2 * c, 2 * c, 3, padding=1)
        self.cc2 = BasicConv2d(2 * c, 2 * c, 3, padding=1)
        self.cc3 = BasicConv2d(3 * c, 3 * c, 3, padding=1)
        self.conv4 = BasicConv2d(3 * c, 3 * c, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * c, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.cu1(self.up(x1)) * x2
        x3_1 = self.cu2(self.up(self.up(x1))) * self.cu3(self.up(x2)) * x3
        x2_2 = self.cc2(torch.cat((x2_1, self.cu4(self.up(x1_1))), 1))
        x3_2 = self.cc3(torch.cat((x3_1, self.cu5(self.up(x2_2))), 1))
        return self.conv5(self.conv4(x3_2))


class PraNet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.resnet = Res2Net50()
        self.rfb2_1, self.rfb3_1, self.rfb4_1 = RFBModified(512, channel), RFBModified(1024, channel), RFBModified(2048, channel)
        self.agg1 = Aggregation(channel)
        self.ra4_conv1 = BasicConv2d(2048, 256, 1)
        self.ra4_conv2 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, 1)
        self.ra3_conv1 = BasicConv2d(1024, 64, 1)
        self.ra3_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, 3, padding=1)
        self.ra2_conv1 = BasicConv2d(512, 64, 1)
        self.ra2_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, 3, padding=1)

    @staticmethod
    def _rev(feat, guide):
        return (1 - torch.sigmoid(guide)).expand(-1, feat.shape[1], -1, -1) * feat

    def forward(self, x):
        size = x.shape[-2:]
        _, x2, x3, x4 = self.resnet(x)
        ra5 = self.agg1(self.rfb4_1(x4), self.rfb3_1(x3), self.rfb2_1(x2))
        crop4 = F.interpolate(ra5, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        t = self._rev(x4, crop4)
        t = F.relu(self.ra4_conv1(t)); t = F.relu(self.ra4_conv2(t)); t = F.relu(self.ra4_conv3(t)); t = F.relu(self.ra4_conv4(t))
        m4 = self.ra4_conv5(t) + crop4
        crop3 = F.interpolate(m4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        t = self._rev(x3, crop3)
        t = F.relu(self.ra3_conv1(t)); t = F.relu(self.ra3_conv2(t)); t = F.relu(self.ra3_conv3(t))
        m3 = self.ra3_conv4(t) + crop3
        crop2 = F.interpolate(m3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        t = self._rev(x2, crop2)
        t = F.relu(self.ra2_conv1(t)); t = F.relu(self.ra2_conv2(t)); t = F.relu(self.ra2_conv3(t))
        m2 = self.ra2_conv4(t) + crop2
        up = lambda z: F.interpolate(z, size=size, mode="bilinear", align_corners=False)
        if self.training:
            return up(m2), up(m3), up(m4)   # (main, aux, aux) -> LightweightLoss
        return up(m2)
