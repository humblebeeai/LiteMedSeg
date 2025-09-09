import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with two conv layers"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    """Encoder block with conv block and max pooling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip

class AttentionGate(nn.Module):
    """Attention Gate module"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal from lower resolution
        # x: feature map from encoder (higher resolution)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 dimensions if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class DecoderBlock(nn.Module):
    """Decoder block with upsampling, attention, and conv block"""
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(F_g=out_channels, F_l=skip_channels, F_int=out_channels//2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Apply attention gate
        skip = self.attention(g=x, x=skip)
        
        # Ensure spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class AttentionUNet(nn.Module):
    """
    Attention U-Net implementation
    
    Args:
        in_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 1)
        features (list): List of feature dimensions for each level (default: [64, 128, 256, 512])
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.features = features
        
        # Encoder (Contracting Path)
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        
        # Decoder (Expanding Path)
        self.decoder4 = DecoderBlock(features[3] * 2, features[3], features[3])
        self.decoder3 = DecoderBlock(features[3], features[2], features[2])
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        
        # Bottleneck
        bottleneck = self.bottleneck(x4)
        
        # Decoder with attention
        d4 = self.decoder4(bottleneck, skip4)
        d3 = self.decoder3(d4, skip3)
        d2 = self.decoder2(d3, skip2)
        d1 = self.decoder1(d2, skip1)
        
        # Final output
        output = self.final_conv(d1)
        
        return output

# Factory function to create different variants
def attention_unet_small(in_channels=3, out_channels=1):
    """Small Attention U-Net with fewer features"""
    return AttentionUNet(in_channels, out_channels, features=[32, 64, 128, 256])

def attention_unet_standard(in_channels=3, out_channels=1):
    """Standard Attention U-Net"""
    return AttentionUNet(in_channels, out_channels, features=[64, 128, 256, 512])

def attention_unet_large(in_channels=3, out_channels=1):
    """Large Attention U-Net with more features"""
    return AttentionUNet(in_channels, out_channels, features=[64, 128, 256, 512, 1024])

