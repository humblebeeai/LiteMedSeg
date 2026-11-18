"""
MobileViT-based segmentation network (Mehta & Rastegari, "MobileViT: Light-weight,
General-purpose, and Mobile-friendly Vision Transformer", ICLR 2022,
https://arxiv.org/abs/2110.02178), adapted with a lightweight U-Net-style
decoder for binary segmentation. There is no single canonical "MobileViT-Seg"
checkpoint for this task, so this is our own compact encoder-decoder built on
the paper's MV2 + MobileViT-block design, added for the revision's lightweight
hybrid CNN/transformer baseline comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False), nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class MV2Block(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_act(in_ch, hidden, 1, 1, 0))
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """Local conv representation -> unfold to patches -> transformer -> fold -> fuse."""

    def __init__(self, channels, transformer_dim, depth=2, patch_size=2, heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.local_rep = nn.Sequential(
            conv_bn_act(channels, channels, 3, 1, 1),
            conv_bn_act(channels, transformer_dim, 1, 1, 0),
        )
        self.transformer = nn.Sequential(*[TransformerEncoderLayer(transformer_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(transformer_dim)
        self.proj = conv_bn_act(transformer_dim, channels, 1, 1, 0)
        self.fuse = conv_bn_act(2 * channels, channels, 3, 1, 1)

    def forward(self, x):
        p = self.patch_size
        B, C, H, W = x.shape
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        y = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x
        Hp, Wp = y.shape[2], y.shape[3]

        local = self.local_rep(y)
        Ct = local.shape[1]
        nh, nw = Hp // p, Wp // p
        # unfold into (B, num_patches, patch_area, Ct) then merge patch_area into batch-like seq
        patches = local.unfold(2, p, p).unfold(3, p, p)  # B,Ct,nh,nw,p,p
        patches = patches.contiguous().view(B, Ct, nh * nw, p * p).permute(0, 3, 2, 1)  # B,p*p,N,Ct
        patches = patches.reshape(B * p * p, nh * nw, Ct)

        patches = self.transformer(patches)
        patches = self.norm(patches)

        patches = patches.reshape(B, p * p, nh * nw, Ct).permute(0, 3, 2, 1)  # B,Ct,N,p*p
        patches = patches.reshape(B, Ct, nh, nw, p, p).permute(0, 1, 2, 4, 3, 5).contiguous()
        folded = patches.view(B, Ct, Hp, Wp)
        folded = self.proj(folded)
        if pad_h or pad_w:
            folded = folded[:, :, :H, :W]

        out = self.fuse(torch.cat([x, folded], dim=1))
        return out


class MobileViTSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, dims=(16, 24, 48, 64, 80), t_dims=(96, 120)):
        super().__init__()
        c0, c1, c2, c3, c4 = dims

        self.stem = conv_bn_act(in_channels, c0, 3, 2, 1)  # 1/2
        self.stage1 = MV2Block(c0, c1, stride=2, expand_ratio=4)  # 1/4
        self.stage2 = nn.Sequential(
            MV2Block(c1, c2, stride=2, expand_ratio=4),  # 1/8
            MV2Block(c2, c2, stride=1, expand_ratio=4),
        )
        self.stage3 = nn.Sequential(
            MV2Block(c2, c3, stride=2, expand_ratio=4),  # 1/16
            MobileViTBlock(c3, t_dims[0], depth=2, patch_size=2, heads=4),
        )
        self.stage4 = nn.Sequential(
            MV2Block(c3, c4, stride=2, expand_ratio=4),  # 1/32
            MobileViTBlock(c4, t_dims[1], depth=3, patch_size=2, heads=4),
        )

        self.dec4 = conv_bn_act(c4, c3, 3, 1, 1)
        self.dec3 = conv_bn_act(c3 + c3, c2, 3, 1, 1)
        self.dec2 = conv_bn_act(c2 + c2, c1, 3, 1, 1)
        self.dec1 = conv_bn_act(c1 + c1, c0, 3, 1, 1)
        self.dec0 = conv_bn_act(c0 + c0, c0, 3, 1, 1)
        self.head = nn.Conv2d(c0, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        s0 = self.stem(x)      # 1/2
        s1 = self.stage1(s0)   # 1/4
        s2 = self.stage2(s1)   # 1/8
        s3 = self.stage3(s2)   # 1/16
        s4 = self.stage4(s3)   # 1/32

        d4 = self.dec4(s4)
        d4 = F.interpolate(d4, size=s3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d4, s3], dim=1))
        d3 = F.interpolate(d3, size=s2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3, s2], dim=1))
        d2 = F.interpolate(d2, size=s1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2, s1], dim=1))
        d1 = F.interpolate(d1, size=s0.shape[2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([d1, s0], dim=1))

        out = self.head(d0)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def create_mobilevit_seg(num_classes=1, in_channels=3):
    return MobileViTSeg(in_channels=in_channels, num_classes=num_classes)
