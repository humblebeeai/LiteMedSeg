import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= LIGHTWEIGHT ADVANCED COMPONENTS =========================

class LightweightSelectiveKernel(nn.Module):
    """
    Lightweight Selective Kernel with reduced complexity
    """
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=4):
        super().__init__()
        d = max(in_channels // r, 8)  # Reduced bottleneck dimension
        self.M = M
        self.out_channels = out_channels
        
        # Lightweight convolutions with different kernel sizes
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=in_channels//4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride, 2, groups=in_channels//4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Lightweight attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, out_channels * M, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Two branches with different kernels
        feat1 = self.conv3x3(x)
        feat2 = self.conv5x5(x)
        
        # Feature fusion
        feats_sum = feat1 + feat2
        
        # Lightweight attention
        feats_gap = self.global_pool(feats_sum)
        attention = self.fc(feats_gap)
        attention = attention.view(batch_size, self.M, self.out_channels, 1, 1)
        attention = self.softmax(attention)
        
        # Apply attention
        feats = torch.stack([feat1, feat2], dim=1)
        feats_weight = feats * attention
        output = torch.sum(feats_weight, dim=1)
        
        return output

class EfficientChannelAttention(nn.Module):
    """Ultra-lightweight ECA with minimal parameters"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class LightweightDynamicConv(nn.Module):
    """
    Simplified dynamic convolution with reduced kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, K=2):
        super().__init__()
        self.K = K
        
        # Lightweight kernels using depthwise separable conv
        self.kernels = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(K)
        ])
        
        # Lightweight attention for kernel selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, K, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get attention weights
        attention = self.attention(x)  # (B, K, 1, 1)
        
        # Apply kernels
        outputs = []
        for i, kernel in enumerate(self.kernels):
            outputs.append(kernel(x))
        
        # Weighted combination
        outputs = torch.stack(outputs, dim=1)  # (B, K, C, H, W)
        attention = attention.unsqueeze(2)  # (B, K, 1, 1, 1)
        output = torch.sum(outputs * attention, dim=1)
        
        return F.relu(output)


class MicroDeformableAttention(nn.Module):
    """
    Simplified lightweight deformable attention 
    Uses offset information to modulate spatial attention
    """
    def __init__(self, dim, num_heads=4, kernel_size=3, dilation=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Number of sampling points (3x3 = 9 points)
        self.num_points = kernel_size * kernel_size
        
        # Simplified offset generation
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, self.num_points * 2, 3, padding=1)
        )
        
        # Standard multi-head attention components
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize offset conv to output zeros initially
        nn.init.constant_(self.offset_conv[-1].weight, 0.)
        nn.init.constant_(self.offset_conv[-1].bias, 0.)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate offset predictions
        offset = self.offset_conv(x)  # [B, 18, H, W]
        offset = offset.view(B, self.num_points, 2, H, W)  # [B, 9, 2, H, W]
        
        # Use offset magnitude as spatial attention weight
        offset_magnitude = torch.norm(offset, dim=2)  # [B, 9, H, W]
        spatial_weights = torch.mean(offset_magnitude, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_weights = torch.sigmoid(spatial_weights)  # Normalize to [0, 1]
        
        # Apply spatial modulation
        x_modulated = x * spatial_weights
        
        # Standard multi-head attention computation
        q = self.q_conv(x_modulated)  # [B, C, H, W]
        k = self.k_conv(x_modulated)  # [B, C, H, W]
        v = self.v_conv(x_modulated)  # [B, C, H, W]
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        k = k.view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        v = v.view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        
        # Transpose for attention computation
        q = q.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        
        # Reshape back to spatial format
        out = out.transpose(-2, -1)  # [B, num_heads, head_dim, H*W]
        out = out.contiguous().view(B, C, H, W)  # [B, C, H, W]
        
        # Final projection
        out = self.proj(out)
        
        # Residual connection
        return out + x
    
    
class MobilePyramidPooling(nn.Module):
    """Lightweight pyramid pooling with reduced channels"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.stages = nn.ModuleList()
        reduced_ch = in_channels // 8  # Much smaller reduction
        
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, reduced_ch, 1, bias=False),
                nn.BatchNorm2d(reduced_ch),
                nn.ReLU(inplace=True)
            ))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * reduced_ch, in_channels, 1, bias=False),
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
        
        output = torch.cat(pyramids, dim=1)
        output = self.bottleneck(output)
        return output

# Standard/Original implementations for comparison
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

class StandardChannelAttention(nn.Module):
    """Standard SE-style channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class StandardDeformableAttention(nn.Module):
    """More complex deformable attention (simplified but larger than micro)"""
    def __init__(self, dim, num_heads=8, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.num_points = kernel_size * kernel_size  # 25 points for 5x5
        
        # More complex offset generation
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, self.num_points * 2, 3, padding=1)
        )
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        nn.init.constant_(self.offset_conv[-1].weight, 0.)
        nn.init.constant_(self.offset_conv[-1].bias, 0.)
    
    def forward(self, x):
        # Simplified implementation - similar to your micro version but with 5x5 grid
        # This would be more complex in practice
        return x  # Placeholder - would implement full 5x5 deformable sampling


# ========================= LIGHTWEIGHT BUILDING BLOCKS =========================

class LightweightResidualBlock(nn.Module):
    """Highly optimized residual block with minimal parameters"""
    def __init__(self, in_ch, out_ch, dropout_rate=0.1, use_sk=True, use_eca=True):
        super().__init__()
        
        # Use lightweight components
        if use_sk:
            self.conv1 = LightweightSelectiveKernel(in_ch, out_ch)
        else:
            # Fallback to standard SK
            self.conv1 = StandardSelectiveKernel(in_ch, out_ch)
       
        if use_eca:
            self.attention = EfficientChannelAttention(out_ch)
        else:
            # Fallback to standard channel attention
            self.attention = StandardChannelAttention(out_ch)
        
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Second conv (always lightweight)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        self.downsample = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.attention(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        return F.relu(out + identity)

class LightweightFusion(nn.Module):
    """Minimal parameter fusion module"""
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(high_ch + low_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, high_feat, low_feat):
        return self.conv(torch.cat([high_feat, low_feat], dim=1))

# ========================= MAIN LIGHTWEIGHT MODEL =========================

class LightweightDeepSegNet(nn.Module):
    """
    Truly lightweight version with advanced components
    Target: <5M parameters while maintaining novelty
    """
    def __init__(self, in_channels=3, out_channels=1, dropout_rate=0.1, 
                 base_channels=8, use_sk=True, use_eca=True, use_mpp=True, use_mda=True):
        super().__init__()
        
        # Much smaller channel progression
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]  # [16, 32, 64, 128, 256]
        
        # Lightweight initial processing
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with progressive complexity
        self.enc1 = LightweightResidualBlock(ch[0], ch[1], dropout_rate, use_sk=use_sk, use_eca=use_eca)
        self.enc2 = LightweightResidualBlock(ch[1], ch[2], dropout_rate, use_sk=use_sk, use_eca=use_eca)
        self.enc3 = LightweightResidualBlock(ch[2], ch[3], dropout_rate, use_sk=use_sk, use_eca=use_eca)
        self.enc4 = LightweightResidualBlock(ch[3], ch[4], dropout_rate, use_sk=use_sk, use_eca=use_eca)

        self.pool = nn.MaxPool2d(2)

        # Lightweight bottleneck
        self.bottleneck = LightweightResidualBlock(ch[4], ch[4], dropout_rate)
        
        if use_mda:
            self.deform_attn = MicroDeformableAttention(ch[4], num_heads=4)
        else:
            # Fallback to standard deformable attention
            self.deform_attn = StandardDeformableAttention(ch[4], num_heads=8)

        if use_mpp:
            self.pyramid = MobilePyramidPooling(ch[4])
        else:
            # Fallback to standard pyramid pooling
            self.pyramid = StandardPyramidPooling(ch[4])
        
        # Lightweight decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion4 = LightweightFusion(ch[4], ch[4], ch[3])
        self.dec4 = LightweightResidualBlock(ch[3], ch[3], dropout_rate, use_sk=use_sk, use_eca=use_eca)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion3 = LightweightFusion(ch[3], ch[3], ch[2])
        self.dec3 = LightweightResidualBlock(ch[2], ch[2], dropout_rate, use_sk=use_sk, use_eca=use_eca)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion2 = LightweightFusion(ch[2], ch[2], ch[1])
        self.dec2 = LightweightResidualBlock(ch[1], ch[1], dropout_rate, use_sk=use_sk, use_eca=use_eca)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion1 = LightweightFusion(ch[1], ch[1], ch[0])
        self.dec1 = LightweightResidualBlock(ch[0], ch[0], dropout_rate, use_sk=use_sk, use_eca=use_eca)

        # Minimal output layers
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch[0], ch[0]//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0]//2, out_channels, 1)
        )
        
        # Lightweight auxiliary outputs
        self.aux_out4 = nn.Conv2d(ch[3], out_channels, 1)
        self.aux_out3 = nn.Conv2d(ch[2], out_channels, 1)

    def forward(self, x):
        # Initial processing
        x0 = self.initial(x)
        
        # Encoder
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        b = self.deform_attn(b)
        b = self.pyramid(b)

        # Decoder
        d4 = self.up4(b)
        d4 = self.fusion4(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.fusion3(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.fusion2(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.fusion1(d1, e1)
        d1 = self.dec1(d1)

        # Output
        main_out = self.out_conv(d1)
        
        # Auxiliary outputs for training
        if self.training:
            aux4 = F.interpolate(self.aux_out4(d4), size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_out3(d3), size=x.shape[-2:], mode='bilinear', align_corners=False)
            return main_out, aux4, aux3
        else:
            return main_out


def create_lightweight_deepsegnet(config='standard', **kwargs):
    """
    Create lightweight model variants
    
    Args:
        config: 'tiny', 'standard', 'advanced'
        base_channels: Base number of channels (16 for ~1-2M params, 24 for ~3-4M params)
    """
    configs = {
        'tiny': {
            'base_channels': 12,
            'dropout_rate': 0.1
        },
        'standard': {
            'base_channels': 16,
            'dropout_rate': 0.1
        },
        'advanced': {
            'base_channels': 32,
            'dropout_rate': 0.1
        }
    }
    
    config_params = configs.get(config, configs['standard'])
    config_params.update(kwargs)
    
    return LightweightDeepSegNet(**config_params)