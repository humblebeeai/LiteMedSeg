import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SeparableConv2d(nn.Module):
    """Separable convolution as used in DeepLabV3+"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ASPPConv(nn.Module):
    """Atrous convolution with batch norm and ReLU"""
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ASPPPooling(nn.Module):
    """Image-level features for ASPP"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=256):
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
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Image pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
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
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ Decoder"""
    def __init__(self, num_classes, low_level_channels=256, low_level_key='low_level'):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.low_level_key = low_level_key
        
        # Low-level feature projection
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            SeparableConv2d(304, 256, 3, padding=1, bias=False),  # 256 + 48 = 304
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeparableConv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, features):
        low_level_feature = features[self.low_level_key]
        high_level_feature = features['out']
        
        # Project low-level features
        low_level_feature = self.project(low_level_feature)
        
        # Upsample high-level features
        high_level_feature = F.interpolate(high_level_feature, size=low_level_feature.shape[2:], 
                                         mode='bilinear', align_corners=False)
        
        # Concatenate features
        concat_features = torch.cat([high_level_feature, low_level_feature], dim=1)
        
        # Final classification
        x = self.classifier(concat_features)
        
        return x


class ResNetBackbone(nn.Module):
    """ResNet backbone with dilated convolutions"""
    def __init__(self, backbone_name='resnet50', pretrained=True, output_stride=16):
        super(ResNetBackbone, self).__init__()

        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Helper: first 3x3 conv in the block where stride is applied
        from torchvision.models.resnet import BasicBlock, Bottleneck
        def set_block_stride(layer, s):
            b0 = layer[0]
            if isinstance(b0, BasicBlock):
                # stride is on conv1 for BasicBlock
                b0.conv1.stride = (s, s)
            elif isinstance(b0, Bottleneck):
                # stride is on conv2 for Bottleneck
                b0.conv2.stride = (s, s)
            else:
                raise TypeError("Unknown ResNet block type")
            if b0.downsample is not None and isinstance(b0.downsample[0], nn.Conv2d):
                b0.downsample[0].stride = (s, s)

        # Helper: apply dilation to all 3x3 convolutions in a layer
        def set_layer_dilation(layer, d):
            for m in layer.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    m.dilation = (d, d)
                    m.padding  = (d, d)

        if output_stride == 16:
            # Keep layer1, layer2, layer3 default (layer3 downsamples).
            # Stop downsampling at layer4 and use dilation=2 there.
            set_block_stride(backbone.layer4, 1)
            set_layer_dilation(backbone.layer4, 2)

        elif output_stride == 8:
            # Stop downsampling at layer3 and layer4; use dilation=2 and 4.
            set_block_stride(backbone.layer3, 1)
            set_block_stride(backbone.layer4, 1)
            set_layer_dilation(backbone.layer3, 2)
            set_layer_dilation(backbone.layer4, 4)

        else:
            raise ValueError("output_stride must be 8 or 16")

        self.backbone = backbone

        # Channel counts
        if backbone_name in ['resnet50', 'resnet101']:
            self.low_level_channels = 256   # layer1
            self.high_level_channels = 2048 # layer4
        else:  # resnet18/34
            self.low_level_channels = 64
            self.high_level_channels = 512

    def forward(self, x):
        features = {}
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x); 
        x = self.backbone.relu(x); 
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        features['low_level'] = x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        features['out'] = x
        return features



class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model implementation
    
    Args:
        num_classes (int): Number of output classes
        backbone (str): Backbone architecture ('resnet50', 'resnet101', 'resnet34', 'resnet18')
        pretrained (bool): Whether to use pretrained backbone
        output_stride (int): Output stride for backbone (8 or 16)
        aspp_dilations (list): Dilation rates for ASPP
    """
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True, 
                 output_stride=16, aspp_dilations=[6, 12, 18]):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone
        self.backbone = ResNetBackbone(backbone, pretrained, output_stride)
        
        # ASPP
        self.aspp = ASPP(self.backbone.high_level_channels, aspp_dilations)
        
        # Decoder
        self.decoder = DeepLabV3PlusDecoder(num_classes, self.backbone.low_level_channels)
        
        # Initialize weights
        if not pretrained:
            self._init_weights()
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Extract features
        features = self.backbone(x)
        
        # Apply ASPP
        features['out'] = self.aspp(features['out'])
        
        # Decode
        x = self.decoder(features)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
    
    def _init_weights(self):
        """Initialize weights for non-pretrained components"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Factory functions for different configurations
def deeplabv3plus_resnet50(num_classes=1, pretrained=True, output_stride=16):
    """DeepLabV3+ with ResNet-50 backbone"""
    return DeepLabV3Plus(num_classes, 'resnet50', pretrained, output_stride)

def deeplabv3plus_resnet101(num_classes=1, pretrained=True, output_stride=16):
    """DeepLabV3+ with ResNet-101 backbone"""
    return DeepLabV3Plus(num_classes, 'resnet101', pretrained, output_stride)

def deeplabv3plus_resnet34(num_classes=1, pretrained=True, output_stride=16):
    """DeepLabV3+ with ResNet-34 backbone (lighter)"""
    return DeepLabV3Plus(num_classes, 'resnet34', pretrained, output_stride)

def deeplabv3plus_resnet18(num_classes=1, pretrained=True, output_stride=16):
    """DeepLabV3+ with ResNet-18 backbone (lightest)"""
    return DeepLabV3Plus(num_classes, 'resnet18', pretrained, output_stride)

# Utility function to create model
def create_deeplabv3plus(num_classes=1, backbone='resnet50', pretrained=True, output_stride=16):
    """
    Create DeepLabV3+ model with specified configuration
    
    Args:
        num_classes (int): Number of output classes
        backbone (str): Backbone architecture
        pretrained (bool): Use pretrained weights
        output_stride (int): Output stride (8 or 16)
    
    Returns:
        model: DeepLabV3+ model
    """
    return DeepLabV3Plus(num_classes, backbone, pretrained, output_stride)

# Model configuration examples
def get_model_configs():
    """Get different model configurations for various use cases"""
    return {
        'lightweight': {
            'backbone': 'resnet18',
            'output_stride': 16,
            'description': 'Fast inference, lower accuracy'
        },
        'balanced': {
            'backbone': 'resnet50',
            'output_stride': 16,
            'description': 'Good balance of speed and accuracy'
        },
        'high_accuracy': {
            'backbone': 'resnet101',
            'output_stride': 8,
            'description': 'Best accuracy, slower inference'
        }
    }