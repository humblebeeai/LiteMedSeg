import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class SpatialAwareChannelAttention(nn.Module):
    """Spatial-aware Channel Attention (SCA) Module"""
    def __init__(self, channels, reduction=16):
        super(SpatialAwareChannelAttention, self).__init__()
        self.channels = channels
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False)
        )
        
        # Spatial attention components
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # Cross-dimensional interaction
        self.conv_cross = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        
        # Apply channel attention
        x_channel = x * channel_attention
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)  # [B, 2, H, W]
        spatial_attention = self.sigmoid(self.conv_spatial(spatial_concat))  # [B, 1, H, W]
        
        # Apply spatial attention
        x_spatial = x_channel * spatial_attention
        
        # Cross-dimensional interaction for spatial-aware channel dependencies
        x_cross = self.conv_cross(x_spatial)
        output = x + x_cross  # Residual connection
        
        return output

class InceptionNeXtBlock(nn.Module):
    """Simplified InceptionNeXt Block"""
    def __init__(self, in_channels, out_channels):
        super(InceptionNeXtBlock, self).__init__()
        mid_channels = out_channels // 2
        
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 depthwise separable conv
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels // 2, kernel_size=3, padding=1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        
        # Concatenate branches
        concat = torch.cat([branch1, branch2], dim=1)
        
        # Fusion
        out = self.fusion(concat)
        
        # Residual connection
        out = out + self.shortcut(x)
        
        return out

class FSCA_Block(nn.Module):
    """Feature Fusion with SCA (FSCA) for skip connections"""
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super(FSCA_Block, self).__init__()
        self.encoder_conv = nn.Conv2d(encoder_channels, out_channels, kernel_size=1, bias=False)
        self.decoder_conv = nn.Conv2d(decoder_channels, out_channels, kernel_size=1, bias=False)
        self.sca = SpatialAwareChannelAttention(out_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, encoder_feat, decoder_feat):
        # Resize decoder feature to match encoder feature size
        decoder_feat = F.interpolate(decoder_feat, size=encoder_feat.shape[2:], 
                                   mode='bilinear', align_corners=False)
        
        # Channel alignment
        encoder_feat = self.encoder_conv(encoder_feat)
        decoder_feat = self.decoder_conv(decoder_feat)
        
        # Element-wise addition and apply SCA
        fused_feat = encoder_feat + decoder_feat
        attended_feat = self.sca(fused_feat)
        
        # Final fusion
        output = self.fusion_conv(attended_feat)
        
        return output

class SCA_InceptionUNeXt(nn.Module):
    """SCA-InceptionUNeXt: Lightweight Spatial-Channel-Attention-based network"""
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(SCA_InceptionUNeXt, self).__init__()
        
        # Initial conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder1 = InceptionNeXtBlock(base_channels, base_channels)
        self.encoder2 = InceptionNeXtBlock(base_channels, base_channels * 2)
        self.encoder3 = InceptionNeXtBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = InceptionNeXtBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck with SCA
        self.bottleneck = nn.Sequential(
            InceptionNeXtBlock(base_channels * 8, base_channels * 16),
            SpatialAwareChannelAttention(base_channels * 16)
        )
        
        # Decoder with upsampling
        self.decoder4 = InceptionNeXtBlock(base_channels * 8, base_channels * 8)
        self.decoder3 = InceptionNeXtBlock(base_channels * 4, base_channels * 4)
        self.decoder2 = InceptionNeXtBlock(base_channels * 2, base_channels * 2)
        self.decoder1 = InceptionNeXtBlock(base_channels, base_channels)
        
        # FSCA blocks for skip connections
        self.fsca4 = FSCA_Block(base_channels * 8, base_channels * 8, base_channels * 8)
        self.fsca3 = FSCA_Block(base_channels * 4, base_channels * 4, base_channels * 4)
        self.fsca2 = FSCA_Block(base_channels * 2, base_channels * 2, base_channels * 2)
        self.fsca1 = FSCA_Block(base_channels, base_channels, base_channels)
        
        # Upsampling layers
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 
                                     kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 
                                     kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                                     kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 
                                     kernel_size=2, stride=2)
        
        # Downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )
        
    def forward(self, x):
        # Initial processing
        x = self.input_conv(x)
        
        # Encoder path
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(self.pool(e1))  # 128 channels
        e3 = self.encoder3(self.pool(e2))  # 256 channels
        e4 = self.encoder4(self.pool(e3))  # 512 channels
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 1024 channels
        
        # Decoder path with FSCA skip connections
        d4 = self.up4(b)
        d4 = self.fsca4(e4, d4)
        d4 = self.decoder4(d4)
        
        d3 = self.up3(d4)
        d3 = self.fsca3(e3, d3)
        d3 = self.decoder3(d3)
        
        d2 = self.up2(d3)
        d2 = self.fsca2(e2, d2)
        d2 = self.decoder2(d2)
        
        d1 = self.up1(d2)
        d1 = self.fsca1(e1, d1)
        d1 = self.decoder1(d1)
        
        # Final output
        output = self.final_conv(d1)
        
        return output
