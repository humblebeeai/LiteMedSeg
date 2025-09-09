import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers and skip connection
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResidualBottleneckBlock(nn.Module):
    """
    Residual bottleneck block with 1x1 -> 3x3 -> 1x1 convolutions
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, dropout_rate=0.0):
        super(ResidualBottleneckBlock, self).__init__()
        
        bottleneck_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResidualEncoder(nn.Module):
    """
    Residual encoder block with downsampling
    """
    def __init__(self, in_channels, out_channels, num_blocks=2, block_type='basic', 
                 stride=2, dropout_rate=0.0):
        super(ResidualEncoder, self).__init__()
        
        if block_type == 'basic':
            block = ResidualBlock
        elif block_type == 'bottleneck':
            block = ResidualBottleneckBlock
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        layers = []
        
        # First block with potential downsampling
        layers.append(block(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class ResidualDecoder(nn.Module):
    """
    Residual decoder block with upsampling and skip connections
    """
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=2, 
                 block_type='basic', upsample_mode='bilinear', dropout_rate=0.0):
        super(ResidualDecoder, self).__init__()
        
        if block_type == 'basic':
            block = ResidualBlock
        elif block_type == 'bottleneck':
            block = ResidualBottleneckBlock
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        self.upsample_mode = upsample_mode
        
        # Upsampling layer
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upsample = None
        
        # Residual blocks
        layers = []
        # First block after concatenation
        layers.append(block(in_channels + skip_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, skip=None):
        # Upsample
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
        
        # Concatenate with skip connection
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Apply residual blocks
        x = self.layers(x)
        return x

class ResUNet(nn.Module):
    """
    ResU-Net: Residual U-Net implementation
    
    Args:
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1)
        features (list): List of feature dimensions for each level
        num_blocks (list): Number of residual blocks per level
        block_type (str): Type of residual block ('basic' or 'bottleneck')
        upsample_mode (str): Upsampling mode ('bilinear' or 'transpose')
        dropout_rate (float): Dropout rate (default: 0.0)
        deep_supervision (bool): Whether to use deep supervision
    """
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512, 1024],
                 num_blocks=[2, 2, 2, 2, 2], block_type='basic', upsample_mode='bilinear',
                 dropout_rate=0.0, deep_supervision=False):
        super(ResUNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.deep_supervision = deep_supervision
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        
        # First encoder (no downsampling)
        self.encoders.append(
            ResidualEncoder(features[0], features[0], num_blocks[0], block_type, 
                          stride=1, dropout_rate=dropout_rate)
        )
        
        # Remaining encoders with downsampling
        for i in range(1, len(features)):
            self.encoders.append(
                ResidualEncoder(features[i-1], features[i], num_blocks[i], block_type,
                              stride=2, dropout_rate=dropout_rate)
            )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 1):
            decoder_idx = len(features) - 2 - i
            self.decoders.append(
                ResidualDecoder(
                    features[decoder_idx + 1],  # in_channels
                    features[decoder_idx],      # skip_channels
                    features[decoder_idx],      # out_channels
                    num_blocks[decoder_idx],
                    block_type,
                    upsample_mode,
                    dropout_rate
                )
            )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Deep supervision layers
        if deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for i in range(1, len(features) - 1):
                self.deep_supervision_outputs.append(
                    nn.Conv2d(features[len(features) - 1 - i], num_classes, kernel_size=1)
                )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Decoder
        deep_supervision_outputs = []
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(encoder_outputs) - 2 - i
            skip = encoder_outputs[skip_idx] if skip_idx >= 0 else None
            x = decoder(x, skip)
            
            # Deep supervision
            if self.deep_supervision and self.training and i < len(self.deep_supervision_outputs):
                ds_output = self.deep_supervision_outputs[i](x)
                # Upsample to input size
                if ds_output.shape[2:] != encoder_outputs[0].shape[2:]:
                    ds_output = F.interpolate(ds_output, size=encoder_outputs[0].shape[2:], 
                                            mode='bilinear', align_corners=False)
                deep_supervision_outputs.append(ds_output)
        
        # Final output
        output = self.final_conv(x)
        
        if self.deep_supervision and self.training:
            return [output] + deep_supervision_outputs
        else:
            return output

# Factory functions for different configurations
def resunet_basic(in_channels=3, num_classes=1, features=None):
    """Basic ResU-Net with basic residual blocks"""
    if features is None:
        features = [64, 128, 256, 512, 1024]
    return ResUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        features=features,
        num_blocks=[2, 2, 2, 2, 2],
        block_type='basic'
    )

def resunet_bottleneck(in_channels=3, num_classes=1, features=None):
    """ResU-Net with bottleneck residual blocks"""
    if features is None:
        features = [64, 128, 256, 512, 1024]
    return ResUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        features=features,
        num_blocks=[2, 2, 2, 2, 2],
        block_type='bottleneck'
    )

def resunet_small(in_channels=3, num_classes=1):
    """Small ResU-Net for faster training"""
    return ResUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[32, 64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2, 2],
        block_type='basic'
    )

def resunet_large(in_channels=3, num_classes=1):
    """Large ResU-Net with more blocks"""
    return ResUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[64, 128, 256, 512, 1024],
        num_blocks=[3, 3, 4, 6, 3],
        block_type='basic'
    )

def resunet_deep_supervision(in_channels=3, num_classes=1):
    """ResU-Net with deep supervision"""
    return ResUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[64, 128, 256, 512, 1024],
        num_blocks=[2, 2, 2, 2, 2],
        block_type='basic',
        deep_supervision=True
    )

# Utility function to create model
def create_resunet(config='basic', in_channels=3, num_classes=1):
    """
    Create ResU-Net model with specified configuration
    
    Args:
        config (str): Model configuration ('small', 'basic', 'bottleneck', 'large', 'deep_supervision')
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
    
    Returns:
        model: ResU-Net model
    """
    if config == 'small':
        return resunet_small(in_channels, num_classes)
    elif config == 'basic':
        return resunet_basic(in_channels, num_classes)
    elif config == 'bottleneck':
        return resunet_bottleneck(in_channels, num_classes)
    elif config == 'large':
        return resunet_large(in_channels, num_classes)
    elif config == 'deep_supervision':
        return resunet_deep_supervision(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown config: {config}")

def get_model_configs():
    """Get different model configurations for various use cases"""
    return {
        'small': {
            'features': [32, 64, 128, 256, 512],
            'blocks': [1, 2, 2, 2, 2],
            'description': 'Lightweight version for fast training'
        },
        'basic': {
            'features': [32, 64, 128, 256, 512],
            'blocks': [2, 2, 2, 2, 2],
            'description': 'Standard ResU-Net with basic blocks'
        },
        'bottleneck': {
            'features': [64, 128, 256, 512, 1024],
            'blocks': [2, 2, 2, 2, 2],
            'description': 'ResU-Net with bottleneck blocks (more efficient)'
        },
        'large': {
            'features': [64, 128, 256, 512, 1024],
            'blocks': [3, 3, 4, 6, 3],
            'description': 'Deeper network with more residual blocks'
        },
        'deep_supervision': {
            'features': [64, 128, 256, 512, 1024],
            'blocks': [2, 2, 2, 2, 2],
            'description': 'Standard with deep supervision training'
        }
    }


