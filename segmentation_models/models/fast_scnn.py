"""
Fast-SCNN: Fast Semantic Segmentation Network (Poudel et al., BMVC 2019,
https://arxiv.org/abs/1902.04502), adapted to binary output for the revision's
added lightweight-baseline comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
    )


class DSConv(nn.Module):
    """Depthwise separable conv, stride configurable."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class LearningToDownsample(nn.Module):
    def __init__(self, in_ch=3, dims=(32, 48, 64)):
        super().__init__()
        c1, c2, c3 = dims
        self.conv = conv_bn_relu(in_ch, c1, 3, 2, 1)
        self.dsconv1 = DSConv(c1, c2, stride=2)
        self.dsconv2 = DSConv(c2, c3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [conv_bn_relu(in_ch, hidden, 1, 1, 0)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class PyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch, sizes=(1, 2, 3, 6)):
        super().__init__()
        branch_ch = in_ch // len(sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s), conv_bn_relu(in_ch, branch_ch, 1, 1, 0)) for s in sizes
        ])
        self.project = conv_bn_relu(in_ch + branch_ch * len(sizes), out_ch, 1, 1, 0)

    def forward(self, x):
        size = x.shape[2:]
        feats = [x] + [F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False) for stage in self.stages]
        return self.project(torch.cat(feats, dim=1))


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_ch=64, dims=(64, 96, 128), out_ch=128):
        super().__init__()
        c1, c2, c3 = dims
        self.bottleneck1 = nn.Sequential(
            InvertedResidual(in_ch, c1, 2, 6), InvertedResidual(c1, c1, 1, 6), InvertedResidual(c1, c1, 1, 6)
        )
        self.bottleneck2 = nn.Sequential(
            InvertedResidual(c1, c2, 2, 6), InvertedResidual(c2, c2, 1, 6), InvertedResidual(c2, c2, 1, 6)
        )
        self.bottleneck3 = nn.Sequential(
            InvertedResidual(c2, c3, 1, 6), InvertedResidual(c3, c3, 1, 6), InvertedResidual(c3, c3, 1, 6)
        )
        self.ppm = PyramidPooling(c3, out_ch)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return self.ppm(x)


class FeatureFusion(nn.Module):
    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Conv2d(low_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.high_dw = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 3, 1, 1, groups=high_ch, bias=False),
            nn.BatchNorm2d(high_ch), nn.ReLU(inplace=True),
        )
        self.high_proj = nn.Sequential(nn.Conv2d(high_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low, high):
        high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)
        high = self.high_proj(self.high_dw(high))
        low = self.low_proj(low)
        return self.relu(low + high)


class Classifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.dsconv1 = DSConv(in_ch, in_ch, stride=1)
        self.dsconv2 = DSConv(in_ch, in_ch, stride=1)
        self.conv = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return self.conv(x)


class FastSCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(in_channels, dims=(32, 48, 64))
        self.global_feature_extractor = GlobalFeatureExtractor(64, dims=(64, 96, 128), out_ch=128)
        self.fusion = FeatureFusion(low_ch=64, high_ch=128, out_ch=128)
        self.classifier = Classifier(128, num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        low = self.learning_to_downsample(x)
        high = self.global_feature_extractor(low)
        fused = self.fusion(low, high)
        out = self.classifier(fused)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def create_fast_scnn(num_classes=1, in_channels=3):
    return FastSCNN(in_channels=in_channels, num_classes=num_classes)
