"""
Ablation network for LiteMedSeg (Scientific Reports revision).

Every module has THREE modes, so "removed" and "replaced by a heavier standard module"
are never conflated (Reviewer 1.5):
  lsk : 'light' (LSK) | 'standard' (3-branch full-conv SK) | 'none' (single 3x3 depthwise-separable conv)
  eca : 'light' (ECA) | 'standard' (SE-style avg+max MLP)  | 'none' (identity)
  mpp : 'light' (MPP) | 'standard' (PSP, bins 1/2/3/6)      | 'none' (identity)
  hga : 'full' | 'gate_only' | 'attn_only' | 'std_gate_attn' | 'none' (identity)
        ('std_gate_attn' = CBAM-style spatial gate + MHSA = the 'standard' counterpart of HGA)
Unlike the old model_ablation_study/ code, no unused/dead parameters are instantiated
(the old StandardDeformableAttention stub returned its input unchanged).

Variant names:  Abl-<variant>_b<B_Ch>   e.g. Abl-wo_lsk_none_b12   (see VARIANTS)
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lite_med_seg import (LightweightSelectiveKernel, EfficientChannelAttention,
                           MobilePyramidPooling, MicroDeformableAttention, LightweightFusion)

class StandardSelectiveKernel(nn.Module):
    """Original SK implementation with full complexity"""
    def __init__(self, in_channels, out_channels, stride=1, M=3, r=16):
        super().__init__()
        d = max(in_channels // r, 32)  # Larger bottleneck
        self.M = M
        self.out_channels = out_channels
        
        # Full convolutions without grouping
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Standard attention with FC layers
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, out_channels * M)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Three branches
        feat1 = self.conv3x3(x)
        feat2 = self.conv5x5(x)
        feat3 = self.conv7x7(x)
        
        # Standard fuse-and-choose
        feats = torch.cat([feat1, feat2, feat3], dim=1)
        feats_sum = feat1 + feat2 + feat3
        
        # MPS-compatible global pooling
        if x.device.type == 'mps':
            feats_gap = torch.mean(feats_sum, dim=[2, 3])  # [B, C]
        else:
            feats_gap = F.adaptive_avg_pool2d(feats_sum, 1).view(batch_size, -1)
        
        attention = self.fc(feats_gap)
        attention = attention.view(batch_size, self.M, self.out_channels, 1, 1)
        attention = self.softmax(attention)
        
        feats = torch.stack([feat1, feat2, feat3], dim=1)
        output = torch.sum(feats * attention, dim=1)
        return output

class StandardPyramidPooling(nn.Module):
    """Original PPM implementation"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList()
        
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),  # Standard reduction
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * (in_channels // 4), in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            pyramid = stage(x)
            pyramid = F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(pyramid)

        return self.bottleneck(torch.cat(pyramids, dim=1))



class _DSConv(nn.Module):
    """Plain 3x3 depthwise-separable conv block (used when LSK is removed)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class _SEAttention(nn.Module):
    """Standard SE-style (avg+max) channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, r, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(r, channels, bias=False))

    def forward(self, x):
        b, c, _, _ = x.size()
        out = torch.sigmoid(self.fc(self.avg_pool(x).view(b, c)) + self.fc(self.max_pool(x).view(b, c)))
        return x * out.view(b, c, 1, 1)


def _make_lsk(mode, cin, cout):
    return {'light': lambda: LightweightSelectiveKernel(cin, cout),
            'standard': lambda: StandardSelectiveKernel(cin, cout),
            'none': lambda: _DSConv(cin, cout)}[mode]()


def _make_eca(mode, c):
    return {'light': lambda: EfficientChannelAttention(c),
            'standard': lambda: _SEAttention(c),
            'none': lambda: nn.Identity()}[mode]()


class AblResidualBlock(nn.Module):
    def __init__(self, cin, cout, lsk, eca, dropout=0.1):
        super().__init__()
        self.conv1 = _make_lsk(lsk, cin, cout)
        self.attention = _make_eca(eca, cout)
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, groups=cout, bias=False),
                                   nn.Conv2d(cout, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.downsample = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        idt = self.downsample(x)
        out = self.conv2(self.dropout(self.attention(self.conv1(x))))
        return F.relu(out + idt)


class AblationNet(nn.Module):
    def __init__(self, base_channels=12, lsk='light', eca='light', mpp='light', hga='full',
                 in_channels=3, out_channels=1, dropout=0.1):
        super().__init__()
        ch = [base_channels * m for m in (1, 2, 4, 8, 16)]
        blk = lambda i, o: AblResidualBlock(i, o, lsk, eca, dropout)
        self.initial = nn.Sequential(nn.Conv2d(in_channels, ch[0], 3, padding=1, bias=False),
                                     nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True))
        self.enc1, self.enc2 = blk(ch[0], ch[1]), blk(ch[1], ch[2])
        self.enc3, self.enc4 = blk(ch[2], ch[3]), blk(ch[3], ch[4])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = blk(ch[4], ch[4])
        self.deform_attn = nn.Identity() if hga == 'none' else MicroDeformableAttention(ch[4], num_heads=4, variant=hga)
        self.pyramid = {'light': lambda: MobilePyramidPooling(ch[4]),
                        'standard': lambda: StandardPyramidPooling(ch[4]),
                        'none': lambda: nn.Identity()}[mpp]()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion4, self.dec4 = LightweightFusion(ch[4], ch[4], ch[3]), blk(ch[3], ch[3])
        self.fusion3, self.dec3 = LightweightFusion(ch[3], ch[3], ch[2]), blk(ch[2], ch[2])
        self.fusion2, self.dec2 = LightweightFusion(ch[2], ch[2], ch[1]), blk(ch[1], ch[1])
        self.fusion1, self.dec1 = LightweightFusion(ch[1], ch[1], ch[0]), blk(ch[0], ch[0])
        self.out_conv = nn.Sequential(nn.Conv2d(ch[0], ch[0] // 2, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(ch[0] // 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(ch[0] // 2, out_channels, 1))
        self.aux_out4, self.aux_out3 = nn.Conv2d(ch[3], out_channels, 1), nn.Conv2d(ch[2], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(self.initial(x)); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.pyramid(self.deform_attn(self.bottleneck(self.pool(e4))))
        d4 = self.dec4(self.fusion4(self.up(b), e4)); d3 = self.dec3(self.fusion3(self.up(d4), e3))
        d2 = self.dec2(self.fusion2(self.up(d3), e2)); d1 = self.dec1(self.fusion1(self.up(d2), e1))
        main = self.out_conv(d1)
        if self.training:
            sz = x.shape[-2:]
            return (main, F.interpolate(self.aux_out4(d4), size=sz, mode='bilinear', align_corners=False),
                    F.interpolate(self.aux_out3(d3), size=sz, mode='bilinear', align_corners=False))
        return main


L, S, N = 'light', 'standard', 'none'
VARIANTS = {
    # --- proposed model and reference networks -----------------------------------------
    'full':          dict(lsk=L, eca=L, mpp=L, hga='full'),
    'bare':          dict(lsk=N, eca=N, mpp=N, hga=N),                    # all four removed
    'baseline_std':  dict(lsk=S, eca=S, mpp=S, hga='std_gate_attn'),      # all four heavier/standard
    # --- (a) genuine module REMOVAL from the full model --------------------------------
    'wo_lsk_none':   dict(lsk=N, eca=L, mpp=L, hga='full'),
    'wo_eca_none':   dict(lsk=L, eca=N, mpp=L, hga='full'),
    'wo_mpp_none':   dict(lsk=L, eca=L, mpp=N, hga='full'),
    'wo_hga_none':   dict(lsk=L, eca=L, mpp=L, hga=N),
    # --- (b) REPLACEMENT by the heavier standard counterpart ----------------------------
    'wo_lsk_std':    dict(lsk=S, eca=L, mpp=L, hga='full'),
    'wo_eca_std':    dict(lsk=L, eca=S, mpp=L, hga='full'),
    'wo_mpp_std':    dict(lsk=L, eca=L, mpp=S, hga='full'),
    'wo_hga_std':    dict(lsk=L, eca=L, mpp=L, hga='std_gate_attn'),
    # --- (c) single-module ADDITION to the bare network ---------------------------------
    'bare_lsk':      dict(lsk=L, eca=N, mpp=N, hga=N),
    'bare_eca':      dict(lsk=N, eca=L, mpp=N, hga=N),
    'bare_mpp':      dict(lsk=N, eca=N, mpp=L, hga=N),
    'bare_hga':      dict(lsk=N, eca=N, mpp=N, hga='full'),
    # --- (d) HGA decomposition (Reviewer 2.2): all other modules light ------------------
    'hga_gate_only': dict(lsk=L, eca=L, mpp=L, hga='gate_only'),
    'hga_attn_only': dict(lsk=L, eca=L, mpp=L, hga='attn_only'),
    'hga_std_gate':  dict(lsk=L, eca=L, mpp=L, hga='std_gate_attn'),
}


def create_ablation_model(name, **kw):
    """name like 'Abl-wo_lsk_none_b12' -> AblationNet."""
    m = re.fullmatch(r'Abl-(\w+?)_b(\d+)', name)
    if not m or m.group(1) not in VARIANTS:
        raise ValueError(f'Unknown ablation model {name}; variants: {list(VARIANTS)}')
    return AblationNet(base_channels=int(m.group(2)), **VARIANTS[m.group(1)], **kw)


def _n_params(variant, b):
    return sum(p.numel() for p in AblationNet(b, **VARIANTS[variant]).parameters())


def match_width(variant, target_params, widths=range(4, 68, 4)):
    """Parameter-matched comparison helper: base-channel width whose parameter count is closest
    to target_params (widths are multiples of 4 because of the grouped convolutions)."""
    best = min(widths, key=lambda b: abs(_n_params(variant, b) - target_params))
    return best, _n_params(variant, best)
