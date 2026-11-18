import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# class LightweightDynamicConv(nn.Module):
#     """
#     Simplified dynamic convolution with reduced kernels
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, K=2):
#         super().__init__()
#         self.K = K
        
#         # Lightweight kernels using depthwise separable conv
#         self.kernels = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False),
#                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             ) for _ in range(K)
#         ])
        
#         # Lightweight attention for kernel selection
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, K, 1),
#             nn.Softmax(dim=1)
#         )
    
#     def forward(self, x):
#         # Get attention weights
#         attention = self.attention(x)  # (B, K, 1, 1)
        
#         # Apply kernels
#         outputs = []
#         for i, kernel in enumerate(self.kernels):
#             outputs.append(kernel(x))
        
#         # Weighted combination
#         outputs = torch.stack(outputs, dim=1)  # (B, K, C, H, W)
#         attention = attention.unsqueeze(2)  # (B, K, 1, 1, 1)
#         output = torch.sum(outputs * attention, dim=1)
        
#         return F.relu(output)


class MicroDeformableAttention(nn.Module):
    """
    Hadamard-Gated Attention (HGA). (Class name kept for checkpoint/import compatibility.)
    The 18-channel output of `descriptor_conv` was called 'offset' in earlier versions; renamed because it is never used for sampling.

    A conv branch predicts 9 paired (dx, dy) descriptors per pixel; their L2 magnitude,
    averaged over the 9 points, is mapped to a spatial gate g = sigmoid(a * m + b) that
    multiplies the input (Hadamard product) before multi-head self-attention + residual.
    The descriptors are NOT used as sampling offsets (no deformable sampling).

    FIX (revision): the previous version zero-initialised the last descriptor conv (then called
    'offset_conv'), so ||descriptors||_2 = 0 and its gradient was exactly 0 -> the gate was frozen at sigmoid(0)=0.5
    and the branch never trained. The last conv now has small non-zero random init and the
    gate has a learnable scale/bias, so gradients reach the whole branch.

    variant (for the HGA decomposition ablation):
      'full'          : magnitude gate + MHSA            (proposed HGA)
      'gate_only'     : magnitude gate, no self-attention
      'attn_only'     : MHSA only, no gate
      'std_gate_attn' : standard CBAM-style spatial gate (7x7 conv on [mean,max]) + MHSA
    """
    def __init__(self, dim, num_heads=4, kernel_size=3, dilation=1, variant='full'):
        super().__init__()
        assert variant in ('full', 'gate_only', 'attn_only', 'std_gate_attn')
        self.variant = variant
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_points = kernel_size * kernel_size

        if variant in ('full', 'gate_only'):
            self.descriptor_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, self.num_points * 2, 3, padding=1)
            )
            nn.init.normal_(self.descriptor_conv[-1].weight, std=0.02)   # was zeros (dead gradient)
            nn.init.constant_(self.descriptor_conv[-1].bias, 0.1)
            self.gate_scale = nn.Parameter(torch.tensor(1.0))
            self.gate_bias = nn.Parameter(torch.tensor(0.0))
        elif variant == 'std_gate_attn':
            self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)

        if variant != 'gate_only':
            self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
            self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
            self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
            self.dropout = nn.Dropout(0.1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints written before the rename used 'offset_conv' (same tensors, only the name changed)
        for k in [k for k in state_dict if k.startswith(prefix + "offset_conv.")]:
            state_dict[prefix + "descriptor_conv." + k[len(prefix + "offset_conv."):]] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def spatial_gate(self, x):
        B, C, H, W = x.shape
        if self.variant in ('full', 'gate_only'):
            desc = self.descriptor_conv(x).view(B, self.num_points, 2, H, W)   # paired spatial descriptors
            mag = torch.norm(desc, dim=2).mean(dim=1, keepdim=True)        # [B,1,H,W]
            return torch.sigmoid(self.gate_scale * mag + self.gate_bias)
        if self.variant == 'std_gate_attn':
            pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
            return torch.sigmoid(self.spatial_conv(pooled))
        return None

    def forward(self, x):
        B, C, H, W = x.shape
        gate = self.spatial_gate(x)
        x_mod = x * gate if gate is not None else x

        if self.variant == 'gate_only':
            return self.proj(x_mod) + x

        q = self.q_conv(x_mod).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        k = self.k_conv(x_mod).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        v = self.v_conv(x_mod).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(-2, -1).contiguous().view(B, C, H, W)
        return self.proj(out) + x


HadamardGatedAttention = MicroDeformableAttention


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

# ========================= LIGHTWEIGHT BUILDING BLOCKS =========================

class LightweightResidualBlock(nn.Module):
    """Highly optimized residual block with minimal parameters"""
    def __init__(self, in_ch, out_ch, dropout_rate=0.1, use_sk=True):
        super().__init__()
        
        # Use lightweight components
        if use_sk:
            self.conv1 = LightweightSelectiveKernel(in_ch, out_ch)
        # elif use_dynamic:
        #     self.conv1 = LightweightDynamicConv(in_ch, out_ch)
        else:
            # Standard depthwise separable
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.attention = EfficientChannelAttention(out_ch)
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
                 base_channels=12, use_deformable=True, hga_variant='full'):
        super().__init__()
        
        # Much smaller channel progression
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]  # [12, 24, 48, 96, 192]

        # Lightweight initial processing
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with progressive complexity
        self.enc1 = LightweightResidualBlock(ch[0], ch[1], dropout_rate, use_sk=True)
        self.enc2 = LightweightResidualBlock(ch[1], ch[2], dropout_rate, use_sk=True)
        self.enc3 = LightweightResidualBlock(ch[2], ch[3], dropout_rate, use_sk=True)
        self.enc4 = LightweightResidualBlock(ch[3], ch[4], dropout_rate, use_sk=True)

        self.pool = nn.MaxPool2d(2)

        # Lightweight bottleneck
        self.bottleneck = LightweightResidualBlock(ch[4], ch[4], dropout_rate)
        
        if use_deformable:
            self.deform_attn = MicroDeformableAttention(ch[4], num_heads=4, variant=hga_variant)
        else:
            self.deform_attn = nn.Identity()
            
        self.pyramid = MobilePyramidPooling(ch[4])
        
        # Lightweight decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion4 = LightweightFusion(ch[4], ch[4], ch[3])
        self.dec4 = LightweightResidualBlock(ch[3], ch[3], dropout_rate)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion3 = LightweightFusion(ch[3], ch[3], ch[2])
        self.dec3 = LightweightResidualBlock(ch[2], ch[2], dropout_rate)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion2 = LightweightFusion(ch[2], ch[2], ch[1])
        self.dec2 = LightweightResidualBlock(ch[1], ch[1], dropout_rate)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fusion1 = LightweightFusion(ch[1], ch[1], ch[0])
        self.dec1 = LightweightResidualBlock(ch[0], ch[0], dropout_rate)

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

# ========================= LOSS FUNCTIONS =========================

def safe_model_forward(model, images, training=True):
    """
    Safe forward pass that handles training/inference mode properly
    """
    if training:
        model.train()
        outputs = model(images)
        return outputs  # Could be tuple or single tensor
    else:
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            return outputs  # Always single tensor in eval mode

# Example training loop helper
def train_step(model, images, masks, criterion, optimizer):
    """
    Single training step with proper handling of model outputs
    """
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss (handles both tuple and single tensor)
    loss = criterion(outputs, masks)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example validation step helper  
def val_step(model, images, masks, criterion):
    """
    Single validation step
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # For metrics calculation, use main output
        if isinstance(outputs, tuple):
            main_output = outputs[0]
        else:
            main_output = outputs
            
        return loss.item(), main_output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ========================= MODEL FACTORY =========================

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
            'use_deformable': True,
            'dropout_rate': 0.05
        },
        'standard': {
            'base_channels': 16,
            'use_deformable': True,
            'dropout_rate': 0.1
        },
        'advanced': {
            'base_channels': 32,
            'use_deformable': True,
            'dropout_rate': 0.1
        }
    }
    
    config_params = configs.get(config, configs['standard'])
    config_params.update(kwargs)
    
    return LightweightDeepSegNet(**config_params)
