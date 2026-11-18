"""
UNeXt: MLP-based Rapid Medical Image Segmentation Network
(Valanarasu & Patel, MICCAI 2022, https://arxiv.org/abs/2203.04967)

Compact reimplementation (conv stem + tokenized-MLP bottleneck stages with
shifted-MLP token mixing) for the revision's added lightweight-baseline
comparison. Matches the paper's reported ~1.47-2.05M parameter range.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


def shift_along(x, H, W, shift_size=5, dim=2):
    """Axial token shift: pad both spatial dims, roll channel-chunks along `dim`
    (2=height, 3=width) by different offsets, then crop back to (H, W)."""
    B, N, C = x.shape
    pad = shift_size // 2
    xs = x.transpose(1, 2).view(B, C, H, W)
    xs = F.pad(xs, (pad, pad, pad, pad), mode="constant", value=0)
    chunks = torch.chunk(xs, shift_size, dim=1)
    shifted = [torch.roll(c, shift, dims=dim) for shift, c in enumerate(chunks, start=-pad)]
    xs = torch.cat(shifted, dim=1)
    xs = xs[:, :, pad:pad + H, pad:pad + W]
    return xs.reshape(B, C, H * W).transpose(1, 2)


class ShiftedMLP(nn.Module):
    """Token-mixing MLP with axial token shifting, in place of self-attention."""

    def __init__(self, in_dim, hidden_dim, shift_size=5):
        super().__init__()
        self.shift_size = shift_size
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, H, W):
        x = shift_along(x, H, W, self.shift_size, dim=2)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = shift_along(x, H, W, self.shift_size, dim=3)
        x = self.fc2(x)
        return x


class TokMLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = ShiftedMLP(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        return x + self.mlp(self.norm(x), H, W)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNeXt(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, conv_dims=(16, 32, 128), tok_dims=(160, 256)):
        super().__init__()
        c1, c2, c3 = conv_dims
        t1, t2 = tok_dims

        self.enc1 = ConvBNReLU(in_channels, c1)
        self.enc2 = ConvBNReLU(c1, c2)
        self.enc3 = ConvBNReLU(c2, c3)

        self.patch3 = OverlapPatchEmbed(c3, t1, 3, 2)
        self.tok3 = TokMLPBlock(t1)
        self.norm3 = nn.LayerNorm(t1)

        self.patch4 = OverlapPatchEmbed(t1, t2, 3, 2)
        self.tok4 = TokMLPBlock(t2)
        self.norm4 = nn.LayerNorm(t2)

        self.dtok4 = TokMLPBlock(t1)
        self.dnorm4 = nn.LayerNorm(t1)
        self.dec4_proj = nn.Conv2d(t2, t1, 3, 1, 1)

        self.dtok3 = TokMLPBlock(c3)
        self.dnorm3 = nn.LayerNorm(c3)
        self.dec3_proj = nn.Conv2d(t1, c3, 3, 1, 1)

        self.dec2 = ConvBNReLU(c3, c2)
        self.dec1 = ConvBNReLU(c2, c1)
        self.final = nn.Conv2d(c1, out_channels, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        B = x.shape[0]

        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2)

        t3, H3, W3 = self.patch3(e3)
        t3 = self.tok3(t3, H3, W3)
        t3 = self.norm3(t3)
        t3_img = t3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2)

        t4, H4, W4 = self.patch4(t3_img)
        t4 = self.tok4(t4, H4, W4)
        t4 = self.norm4(t4)
        t4_img = t4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2)

        d4 = F.interpolate(t4_img, size=(H3, W3), mode="bilinear", align_corners=False)
        d4 = self.dec4_proj(d4)
        d4 = d4 + t3_img
        d4_seq = d4.flatten(2).transpose(1, 2)
        d4_seq = self.dtok4(d4_seq, H3, W3)
        d4_seq = self.dnorm4(d4_seq)
        d4_img = d4_seq.reshape(B, H3, W3, -1).permute(0, 3, 1, 2)

        d3 = F.interpolate(d4_img, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3_proj(d3)
        d3 = d3 + e3
        d3_seq = d3.flatten(2).transpose(1, 2)
        d3_seq = self.dtok3(d3_seq, e3.shape[2], e3.shape[3])
        d3_seq = self.dnorm3(d3_seq)
        d3_img = d3_seq.reshape(B, e3.shape[2], e3.shape[3], -1).permute(0, 3, 1, 2)

        d2 = F.interpolate(d3_img, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(d2) + e2

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(d1) + e1

        return self.final(d1)


def create_unext(num_classes=1, in_channels=3):
    return UNeXt(in_channels=in_channels, out_channels=num_classes)
