import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block with two conv layers and skip connection
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

class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Squeeze
        y = self.global_pool(x).view(b, c)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Scale
        y = y.view(b, c, 1, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0, use_se=True):
        super(ResidualSEBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.se = SqueezeExcitationBlock(out_channels) if use_se else None
        
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
        
        if self.se is not None:
            out = self.se(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Image pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        # Upsample image pooling result
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class NestedBlock(nn.Module):
    """
    Nested block for dense connections (U-Net++ style)
    """
    def __init__(self, in_channels_list, out_channels, num_blocks=2, use_se=True, dropout_rate=0.0):
        super(NestedBlock, self).__init__()
        
        total_in_channels = sum(in_channels_list)
        
        layers = []
        # First block
        layers.append(ResidualSEBlock(total_in_channels, out_channels, stride=1, 
                                     dropout_rate=dropout_rate, use_se=use_se))
        
        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualSEBlock(out_channels, out_channels, stride=1,
                                         dropout_rate=dropout_rate, use_se=use_se))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, inputs):
        # Concatenate all inputs
        if isinstance(inputs, list):
            # Ensure all inputs have the same spatial dimensions
            target_size = inputs[0].shape[2:]
            aligned_inputs = []
            for inp in inputs:
                if inp.shape[2:] != target_size:
                    inp = F.interpolate(inp, size=target_size, mode='bilinear', align_corners=False)
                aligned_inputs.append(inp)
            x = torch.cat(aligned_inputs, dim=1)
        else:
            x = inputs
        
        return self.layers(x)

class ResUNetPlusPlus(nn.Module):
    """
    ResU-Net++: Enhanced ResU-Net with nested skip connections and attention mechanisms
    
    Args:
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1)
        features (list): List of feature dimensions for each level
        num_blocks (list): Number of residual blocks per level
        use_se (bool): Whether to use Squeeze-and-Excitation blocks
        use_aspp (bool): Whether to use ASPP in the bottleneck
        dropout_rate (float): Dropout rate (default: 0.0)
        deep_supervision (bool): Whether to use deep supervision
        nested_connections (bool): Whether to use nested skip connections (U-Net++ style)
    """
    def __init__(self, in_channels=3, num_classes=1, features=[32, 64, 128, 256, 512],
                 num_blocks=[2, 2, 2, 2, 2], use_se=True, use_aspp=True,
                 dropout_rate=0.0, deep_supervision=False, nested_connections=True):
        super(ResUNetPlusPlus, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.num_blocks = num_blocks
        self.deep_supervision = deep_supervision
        self.nested_connections = nested_connections
        self.num_levels = len(features)
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        
        # First encoder (no downsampling)
        layers = []
        for _ in range(num_blocks[0]):
            layers.append(ResidualSEBlock(features[0], features[0], stride=1, 
                                         dropout_rate=dropout_rate, use_se=use_se))
        self.encoders.append(nn.Sequential(*layers))
        
        # Remaining encoders with downsampling
        for i in range(1, self.num_levels):
            layers = []
            # First block with downsampling
            layers.append(ResidualSEBlock(features[i-1], features[i], stride=2,
                                         dropout_rate=dropout_rate, use_se=use_se))
            # Additional blocks
            for _ in range(num_blocks[i] - 1):
                layers.append(ResidualSEBlock(features[i], features[i], stride=1,
                                             dropout_rate=dropout_rate, use_se=use_se))
            self.encoders.append(nn.Sequential(*layers))
        
        # Bottleneck with ASPP
        if use_aspp:
            self.bottleneck = ASPP(features[-1], features[-1])
        else:
            self.bottleneck = nn.Identity()
        
        # Nested decoder blocks (U-Net++ style)
        if nested_connections:
            self.nested_blocks = nn.ModuleDict()
            
            for i in range(self.num_levels - 1):
                for j in range(i + 1):
                    level_idx = self.num_levels - 2 - i  # Current level index
                    
                    if j == 0:
                        # First column - upsampling + skip connection
                        # Input: upsampled from lower level + skip from encoder
                        in_channels_list = [features[level_idx + 1], features[level_idx]]
                    else:
                        # Dense connections from previous columns + skip connection
                        in_channels_list = []
                        # Add channels from all previous columns at this level
                        for k in range(j):
                            in_channels_list.append(features[level_idx])
                        # Add upsampled from previous level
                        in_channels_list.append(features[level_idx + 1])
                        # Add skip connection from encoder
                        in_channels_list.append(features[level_idx])
                    
                    block_name = f"nested_{i}_{j}"
                    self.nested_blocks[block_name] = NestedBlock(
                        in_channels_list, features[level_idx],
                        num_blocks=2, use_se=use_se, dropout_rate=dropout_rate
                    )
        
        else:
            # Standard decoder
            self.decoders = nn.ModuleList()
            for i in range(self.num_levels - 1):
                decoder_idx = self.num_levels - 2 - i
                layers = []
                # First block after upsampling and concatenation
                layers.append(ResidualSEBlock(features[decoder_idx + 1] + features[decoder_idx],
                                             features[decoder_idx], stride=1,
                                             dropout_rate=dropout_rate, use_se=use_se))
                # Additional blocks
                for _ in range(num_blocks[decoder_idx] - 1):
                    layers.append(ResidualSEBlock(features[decoder_idx], features[decoder_idx],
                                                 stride=1, dropout_rate=dropout_rate, use_se=use_se))
                self.decoders.append(nn.Sequential(*layers))
        
        # Final classification layers
        if deep_supervision and nested_connections:
            self.final_convs = nn.ModuleList()
            for i in range(self.num_levels - 1):
                self.final_convs.append(nn.Conv2d(features[i], num_classes, kernel_size=1))
        else:
            self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
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
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        if self.nested_connections:
            # Nested decoder (U-Net++ style)
            nested_outputs = {}
            
            for i in range(self.num_levels - 1):
                for j in range(i + 1):
                    level_idx = self.num_levels - 2 - i  # Current decoder level
                    
                    if j == 0:
                        # First column - direct upsampling + skip connection
                        if i == 0:
                            # From bottleneck
                            upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                        else:
                            # From previous level's first column
                            prev_output = nested_outputs[f"nested_{i-1}_0"]
                            upsampled = F.interpolate(prev_output, scale_factor=2, mode='bilinear', align_corners=False)
                        
                        # Skip connection from encoder
                        skip = encoder_outputs[level_idx]
                        if upsampled.shape[2:] != skip.shape[2:]:
                            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='bilinear', align_corners=False)
                        
                        inputs = [upsampled, skip]
                    
                    else:
                        # Dense connections from previous columns
                        inputs = []
                        
                        # Add all previous columns at this level
                        for k in range(j):
                            inputs.append(nested_outputs[f"nested_{i}_{k}"])
                        
                        # Add upsampled from previous level
                        if i > 0:
                            prev_output = nested_outputs[f"nested_{i-1}_{j-1}"]
                            upsampled = F.interpolate(prev_output, scale_factor=2, mode='bilinear', align_corners=False)
                            # Ensure spatial dimensions match
                            target_size = nested_outputs[f"nested_{i}_0"].shape[2:]
                            if upsampled.shape[2:] != target_size:
                                upsampled = F.interpolate(upsampled, size=target_size, mode='bilinear', align_corners=False)
                            inputs.append(upsampled)
                        
                        # Add skip connection from encoder
                        skip = encoder_outputs[level_idx]
                        # Ensure spatial dimensions match
                        target_size = nested_outputs[f"nested_{i}_0"].shape[2:]
                        if skip.shape[2:] != target_size:
                            skip = F.interpolate(skip, size=target_size, mode='bilinear', align_corners=False)
                        inputs.append(skip)
                    
                    block_name = f"nested_{i}_{j}"
                    nested_outputs[block_name] = self.nested_blocks[block_name](inputs)
            
            # Generate outputs
            if self.deep_supervision and self.training:
                outputs = []
                for i in range(self.num_levels - 1):
                    # Use the last column of each level
                    feature = nested_outputs[f"nested_{i}_{i}"]
                    output = self.final_convs[i](feature)
                    # Upsample to input size
                    output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
                    outputs.append(output)
                return outputs
            else:
                # Use the output from the top level
                feature = nested_outputs[f"nested_{self.num_levels-2}_{self.num_levels-2}"]
                output = self.final_convs[0](feature) if hasattr(self, 'final_convs') else self.final_conv(feature)
                output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
                return output
        
        else:
            # Standard decoder
            for i, decoder in enumerate(self.decoders):
                # Upsample
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                
                # Skip connection
                skip_idx = self.num_levels - 2 - i
                skip = encoder_outputs[skip_idx]
                
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
                # Concatenate and decode
                x = torch.cat([x, skip], dim=1)
                x = decoder(x)
            
            # Final output
            output = self.final_conv(x)
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
            return output

# Factory functions for different configurations
def resunetpp_standard(in_channels=3, num_classes=1):
    """Standard ResU-Net++ configuration"""
    return ResUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[32, 64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2, 2],
        use_se=True,
        use_aspp=True,
        nested_connections=True
    )

def resunetpp_light(in_channels=3, num_classes=1):
    """Lightweight ResU-Net++ configuration"""
    return ResUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[16, 32, 64, 128, 256],
        num_blocks=[1, 2, 2, 2, 2],
        use_se=False,
        use_aspp=False,
        nested_connections=True
    )

def resunetpp_deep(in_channels=3, num_classes=1):
    """Deep ResU-Net++ configuration"""
    return ResUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[64, 128, 256, 512, 1024],
        num_blocks=[3, 3, 4, 6, 3],
        use_se=True,
        use_aspp=True,
        nested_connections=True
    )

def resunetpp_deep_supervision(in_channels=3, num_classes=1):
    """ResU-Net++ with deep supervision"""
    return ResUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[32, 64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2, 2],
        use_se=True,
        use_aspp=True,
        nested_connections=True,
        deep_supervision=True
    )

def resunetpp_no_nested(in_channels=3, num_classes=1):
    """ResU-Net++ without nested connections (standard ResU-Net with SE and ASPP)"""
    return ResUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        features=[32, 64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2, 2],
        use_se=True,
        use_aspp=True,
        nested_connections=False
    )

# Utility function to create model
def create_resunetpp(config='standard', in_channels=3, num_classes=1):
    """
    Create ResU-Net++ model with specified configuration
    
    Args:
        config (str): Model configuration ('light', 'standard', 'deep', 'deep_supervision', 'no_nested')
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
    
    Returns:
        model: ResU-Net++ model
    """
    if config == 'light':
        return resunetpp_light(in_channels, num_classes)
    elif config == 'standard':
        return resunetpp_standard(in_channels, num_classes)
    elif config == 'deep':
        return resunetpp_deep(in_channels, num_classes)
    elif config == 'deep_supervision':
        return resunetpp_deep_supervision(in_channels, num_classes)
    elif config == 'no_nested':
        return resunetpp_no_nested(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown config: {config}")

def get_model_configs():
    """Get different model configurations for various use cases"""
    return {
        'light': {
            'features': [16, 32, 64, 128, 256],
            'blocks': [1, 2, 2, 2, 2],
            'se': False,
            'aspp': False,
            'description': 'Lightweight version for fast training'
        },
        'standard': {
            'features': [32, 64, 128, 256, 512],
            'blocks': [2, 2, 2, 2, 2],
            'se': True,
            'aspp': True,
            'description': 'Standard ResU-Net++ with all features'
        },
        'deep': {
            'features': [64, 128, 256, 512, 1024],
            'blocks': [3, 3, 4, 6, 3],
            'se': True,
            'aspp': True,
            'description': 'Deeper network with more residual blocks'
        },
        'deep_supervision': {
            'features': [32, 64, 128, 256, 512],
            'blocks': [2, 2, 2, 2, 2],
            'se': True,
            'aspp': True,
            'description': 'Standard with deep supervision training'
        },
        'no_nested': {
            'features': [32, 64, 128, 256, 512],
            'blocks': [2, 2, 2, 2, 2],
            'se': True,
            'aspp': True,
            'description': 'ResU-Net with SE and ASPP, no nested connections'
        }
    }