import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768, dropout_rate=0.1):
        super(Embeddings, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Ensure img_size is divisible by patch_size
        assert img_size % patch_size == 0, f"img_size {img_size} must be divisible by patch_size {patch_size}"
        
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        x = self.patch_embeddings(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(-1, -2)  # (B, num_patches, embed_dim)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout_rate=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.proj_dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.hidden_size = embed_dim
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = MLP(embed_dim, mlp_dim, dropout_rate)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class VisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768, 
                 num_heads=12, num_layers=12, mlp_dim=3072, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        self.embeddings = Embeddings(img_size, patch_size, in_channels, embed_dim, dropout_rate)
        self.transformer = Transformer(embed_dim, num_heads, num_layers, mlp_dim, dropout_rate)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.transformer(x)
        return x

class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super(DecoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super(SegmentationHead, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x

class TransUNet(nn.Module):
    """
    TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
    
    Args:
        img_size (int): Input image size (default: 256)
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1)
        patch_size (int): Patch size for Vision Transformer (default: 16)
        embed_dim (int): Embedding dimension (default: 768)
        num_heads (int): Number of attention heads (default: 12)
        num_layers (int): Number of transformer layers (default: 12)
        mlp_dim (int): Hidden dimension in MLP (default: 3072)
        dropout_rate (float): Dropout rate (default: 0.1)
        decoder_channels (list): Decoder channel configuration
    """
    def __init__(self, img_size=256, in_channels=3, num_classes=1, patch_size=16,
                 embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, 
                 dropout_rate=0.1, decoder_channels=(256, 128, 64, 16)):
        super(TransUNet, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        
        # CNN Encoder (first few layers)
        self.conv_more = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks for skip connections
        self.encoder1 = ConvBlock(16, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Vision Transformer
        # After 3 pooling operations: 256//8 = 32
        self.feature_size = img_size // 8  # 32 for 256x256 input
        self.transformer = VisionTransformer(
            img_size=self.feature_size,
            patch_size=2,  # Use 2x2 patches for 32x32 feature maps
            in_channels=256,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate
        )
        
        # Patch to embedding projection for decoder
        self.patch_to_embedding = nn.Linear(embed_dim, decoder_channels[0])
        
        # Decoder
        self.decoder4 = DecoderBlock(decoder_channels[0], decoder_channels[0], skip_channels=256)
        self.decoder3 = DecoderBlock(decoder_channels[0], decoder_channels[1], skip_channels=128)
        self.decoder2 = DecoderBlock(decoder_channels[1], decoder_channels[2], skip_channels=64)
        self.decoder1 = DecoderBlock(decoder_channels[2], decoder_channels[3], skip_channels=16)
        
        # Final segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[3],
            out_channels=num_classes,
            kernel_size=3,
            upsampling=1
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Initial conv
        x = self.conv_more(x)  # (B, 16, 256, 256)
        skip_connections.append(x)
        
        # Encoder path
        x = self.pool(x)  # (B, 16, 128, 128)
        x = self.encoder1(x)  # (B, 64, 128, 128)
        skip_connections.append(x)
        
        x = self.pool(x)  # (B, 64, 64, 64)
        x = self.encoder2(x)  # (B, 128, 64, 64)
        skip_connections.append(x)
        
        x = self.pool(x)  # (B, 128, 32, 32)
        x = self.encoder3(x)  # (B, 256, 32, 32)
        skip_connections.append(x)
        
        # Transformer encoding
        B, C, H, W = x.shape  # Should be (B, 256, 32, 32)
        x = self.transformer(x)  # (B, n_patches, embed_dim)
        
        # Calculate correct dimensions for reshaping
        patch_size = 2  # We're using 2x2 patches
        n_patches_h = H // patch_size  # 32 // 2 = 16
        n_patches_w = W // patch_size  # 32 // 2 = 16
        
        # Reshape back to feature map
        x = self.patch_to_embedding(x)  # (B, n_patches, decoder_channels[0])
        x = x.transpose(-1, -2)  # (B, decoder_channels[0], n_patches)
        x = x.view(B, self.decoder_channels[0], n_patches_h, n_patches_w)  # (B, 256, 16, 16)
        
        # Decoder path with skip connections
        # decoder4: 16x16 -> 32x32, concat with skip_connections[3] (256, 32, 32)
        x = self.decoder4(x, skip_connections[3])  # (B, 256, 32, 32)
        
        # decoder3: 32x32 -> 64x64, concat with skip_connections[2] (128, 64, 64)  
        x = self.decoder3(x, skip_connections[2])  # (B, 128, 64, 64)
        
        # decoder2: 64x64 -> 128x128, concat with skip_connections[1] (64, 128, 128)
        x = self.decoder2(x, skip_connections[1])  # (B, 64, 128, 128)
        
        # decoder1: 128x128 -> 256x256, concat with skip_connections[0] (16, 256, 256)
        x = self.decoder1(x, skip_connections[0])  # (B, 16, 256, 256)
        
        # Final prediction
        logits = self.segmentation_head(x)  # (B, num_classes, 256, 256)
        
        return logits

# Factory functions for different configurations
def transunet_small(num_classes=1, img_size=256):
    """Small TransUNet configuration"""
    return TransUNet(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        mlp_dim=1536,
        decoder_channels=(128, 64, 32, 16)
    )

def transunet_base(num_classes=1, img_size=256):
    """Base TransUNet configuration"""
    return TransUNet(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        decoder_channels=(256, 128, 64, 16)
    )

def transunet_large(num_classes=1, img_size=256):
    """Large TransUNet configuration"""
    return TransUNet(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=1024,
        num_heads=16,
        num_layers=24,
        mlp_dim=4096,
        decoder_channels=(512, 256, 128, 64)
    )


# Utility function to create model
def create_transunet(config='base', num_classes=1, img_size=256):
    """
    Create TransUNet model with specified configuration
    
    Args:
        config (str): Model configuration ('small', 'base', 'large')
        num_classes (int): Number of output classes
        img_size (int): Input image size
    
    Returns:
        model: TransUNet model
    """
    if config == 'small':
        return transunet_small(num_classes, img_size)
    elif config == 'base':
        return transunet_base(num_classes, img_size)
    elif config == 'large':
        return transunet_large(num_classes, img_size)
    else:
        raise ValueError(f"Unknown config: {config}")

def get_model_info():
    """Get information about different TransUNet configurations"""
    return {
        'small': {
            'params': '~23M',
            'embed_dim': 384,
            'layers': 6,
            'description': 'Lightweight version for fast training'
        },
        'base': {
            'params': '~105M', 
            'embed_dim': 768,
            'layers': 12,
            'description': 'Standard configuration, good accuracy'
        },
        'large': {
            'params': '~304M',
            'embed_dim': 1024,
            'layers': 24,
            'description': 'Highest accuracy, requires more resources'
        }
    }