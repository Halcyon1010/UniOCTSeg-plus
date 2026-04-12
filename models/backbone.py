import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from segmentation_models_pytorch.encoders import get_encoder
from monai.networks.blocks import UnetUpBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from models.attention import ViT

class TransUNet(nn.Module):
    """
    TransUNet architecture
    """
    def __init__(self, spatial_dim=2, in_channels=1, hidden_size=768):
        super(TransUNet, self).__init__()
        
        self.spatial_dim = spatial_dim
        
        # --- 1. CNN Encoder ---
        # Uses ResNet34 to extract a pyramid of features
        self.encoder = get_encoder(
            name='resnet34',
            in_channels=in_channels
        )
        # Access the channel counts of the encoder outputs for building the decoder
        base_channels = self.encoder.out_channels 

        # --- 2. Transformer Bottleneck Preparation ---
        # 1x1 or 2x2 Conv to project CNN features to Transformer hidden dimension
        self.patch_embeddings = nn.Conv2d(
            in_channels=base_channels[-1],
            out_channels=hidden_size,
            kernel_size=2,
            stride=2
        )
        
        # Learnable position embeddings (1, Sequence_Length, Hidden_Size)
        # NOTE: Hardcoded '64' assumes a specific input resolution and downsampling factor.
        # Ensure your input image size results in an 8x8 feature map at this stage (8*8=64).
        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, hidden_size))
        self.drop = nn.Dropout(0.1)
        
        # Vision Transformer (ViT) module
        self.transformer = ViT()

        # --- 3. Hybrid Decoder (U-Net Style) ---
        # Step-by-step upsampling combined with skip connections from the CNN encoder
        
        # Decoder 1: Upsamples Transformer output and fuses with last Encoder feature
        self.decoder1 = UnetUpBlock(
            spatial_dims=self.spatial_dim,
            in_channels=hidden_size,
            out_channels=base_channels[-1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            stride=1
        )
        
        # Decoder 2
        self.decoder2 = UnetUpBlock(
            spatial_dims=self.spatial_dim,
            in_channels=base_channels[-1],
            out_channels=base_channels[-2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            stride=1
        )
        
        # Decoder 3
        self.decoder3 = UnetUpBlock(
            spatial_dims=self.spatial_dim,
            in_channels=base_channels[-2],
            out_channels=base_channels[-3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            stride=1
        )
        
        # Decoder 4
        self.decoder4 = UnetUpBlock(
            spatial_dims=self.spatial_dim,
            in_channels=base_channels[-3],
            out_channels=base_channels[-4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            stride=1
        )
        
        # Decoder 5
        self.decoder5 = UnetUpBlock(
            spatial_dims=self.spatial_dim,
            in_channels=base_channels[-4],
            out_channels=base_channels[-5],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            stride=1
        )

        # --- 4. Final Upsampling / Reconstruction ---
        # A sequential block to refine the output and restore spatial resolution
        self.up_conv = nn.Sequential(
            get_conv_layer(
                spatial_dims=2,
                in_channels=base_channels[-5],
                out_channels=base_channels[-5],
                is_transposed=True, # Upsampling
                kernel_size=3,
                stride=2,
                act='ReLu'
            ),
            get_conv_layer(
                spatial_dims=2,
                in_channels=base_channels[-4], # WARNING: Check if this matches the output of previous layer
                out_channels=base_channels[-5],
                kernel_size=1,
                stride=1,
                act=None
            )
        )

    def forward(self, x):
        # --- CNN Encoder Pass ---
        # Returns a list of feature maps at different resolutions
        f_list = self.encoder(x)

        # --- Transformer Bridge ---
        # Get the deepest feature map from CNN
        x_1 = self.patch_embeddings(f_list[-1])
        
        # Flatten and Transpose for Transformer: (B, C, H, W) -> (B, N, C)
        B, C, H, W = x_1.shape
        x_1 = x_1.flatten(2)
        x_1 = x_1.transpose(-1, -2)
        
        # Add position embeddings
        embeddings = x_1 + self.position_embeddings
        embeddings = self.drop(embeddings)

        # Pass through Transformer
        # Assuming ViT returns (features, weights)
        enc_feature, _ = self.transformer(embeddings)
        
        # Reshape back to image format: (B, N, C) -> (B, C, H, W)
        enc_feature = enc_feature.permute(0, 2, 1).view(B, C, H, W)

        # --- Decoder Pass (Cascaded Upsampling) ---
        x_1 = self.decoder1(enc_feature, f_list[-1]) # Skip connection 1
        x_2 = self.decoder2(x_1, f_list[-2])         # Skip connection 2
        x_3 = self.decoder3(x_2, f_list[-3])         # Skip connection 3
        x_4 = self.decoder4(x_3, f_list[-4])         # Skip connection 4
        x_5 = self.decoder5(x_4, f_list[-5])         # Skip connection 5
        
        # Final refinement
        x_6 = self.up_conv(x_5)

        # Return the transformer feature (bottleneck) and list of decoder outputs
        return enc_feature, [x_1, x_2, x_3, x_4, x_5, x_6]